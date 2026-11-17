//! Bounded request admission for one engine. No task or thread is spawned.
use std::fmt;
use std::sync::{Arc, OnceLock};
use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore, TryAcquireError};

/// Failed admission. Queue overflow is distinct from permanent shutdown.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdmissionError {
    Full,
    Closed,
}

impl fmt::Display for AdmissionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Full => "local engine request queue is full",
            Self::Closed => "local engine request queue is closed",
        })
    }
}
impl std::error::Error for AdmissionError {}

impl From<TryAcquireError> for AdmissionError {
    fn from(error: TryAcquireError) -> Self {
        match error {
            TryAcquireError::Closed => Self::Closed,
            TryAcquireError::NoPermits => Self::Full,
        }
    }
}

/// Exclusive engine access, including cancellation cleanup and model operations.
///
/// Keep this alive until the engine request has finished or cancellation is
/// acknowledged. Cancelling a caller must not drop the lease while its upstream
/// work can still execute. Dropping the lease grants the next waiting request.
/// A lease cannot be cloned, and dropping the queue does not invalidate it.
#[must_use = "retain the lease until engine work and cancellation cleanup finish"]
pub struct Lease {
    _permit: OwnedSemaphorePermit,
    // Drop after the permit so a notified guest can try to acquire the engine.
    _notify_guest: NotifyGuest,
}

struct Guests {
    // Waiting guests cannot occupy the last slot reserved for owner admission.
    slots: Semaphore,
    turn: Semaphore,
    changed: Notify,
}

struct NotifyGuest(Arc<OnceLock<Guests>>);

impl Drop for NotifyGuest {
    fn drop(&mut self) {
        if let Some(guests) = self.0.get() {
            guests.changed.notify_one();
        }
    }
}

/// One active request plus at most `max_waiters` suspended acquisitions.
///
/// FIFO order begins when an acquisition future is first polled. Dropping a
/// waiting future cancels that ticket and returns its capacity immediately.
/// A cancelled request that retries joins the back of the queue. The underlying
/// Tokio semaphore supplies owner FIFO admission and cancellation-safe wakeups.
/// Guests wait in their own FIFO and take the same engine permit once no owner
/// holds a grant. An owner never interrupts an active guest.
///
/// Clones refer to the same engine. Use the same queue for model operations and
/// generation; snapshots of `waiting()` are for status, never idle admission.
#[derive(Clone)]
pub struct RequestQueue {
    engine: Arc<Semaphore>,
    slots: Arc<Semaphore>,
    max_waiters: usize,
    guests: Arc<OnceLock<Guests>>,
}

impl RequestQueue {
    /// Panics if the capacity exceeds Tokio's platform-specific permit limit.
    /// Zero capacity permits immediate work but refuses all waiting requests.
    pub fn new(max_waiters: usize) -> Self {
        Self {
            engine: Arc::new(Semaphore::new(1)),
            slots: Arc::new(Semaphore::new(max_waiters)),
            max_waiters,
            guests: Arc::new(OnceLock::new()),
        }
    }

    fn lease(&self, permit: OwnedSemaphorePermit) -> Lease {
        Lease {
            _permit: permit,
            _notify_guest: NotifyGuest(self.guests.clone()),
        }
    }

    /// Acquires immediately, preserving the place of already queued requests.
    /// Failure to acquire an occupied engine is reported as Full.
    pub fn try_acquire(&self) -> Result<Lease, AdmissionError> {
        self.engine
            .clone()
            .try_acquire_owned()
            .map(|permit| self.lease(permit))
            .map_err(AdmissionError::from)
    }

    /// No polling timer is used: release, shutdown and cancellation drive wakes.
    pub async fn acquire(&self) -> Result<Lease, AdmissionError> {
        match self.try_acquire() {
            Ok(lease) => return Ok(lease),
            Err(AdmissionError::Closed) => return Err(AdmissionError::Closed),
            Err(AdmissionError::Full) => (),
        }
        // Capacity is taken only by work that actually has to wait. This guard
        // lives inside the acquisition future and is dropped on cancellation.
        let slot = self.slots.clone().try_acquire_owned()?;
        // If cancellation returns a grant before we can build a Lease, the
        // waiting guest must also see that the engine permit became available.
        let _notify_guest = NotifyGuest(self.guests.clone());
        let permit = self
            .engine
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| AdmissionError::Closed)?;
        drop(slot);
        Ok(self.lease(permit))
    }

    /// Wait for the shared engine at guest priority without passing queued owners.
    pub async fn acquire_guest(&self) -> Result<Lease, AdmissionError> {
        let guests = self.guests.get_or_init(|| Guests {
            slots: Semaphore::new(self.max_waiters.saturating_sub(1)),
            turn: Semaphore::new(1),
            changed: Notify::new(),
        });
        let turn = match guests.turn.try_acquire() {
            Ok(turn) => {
                match self.try_acquire() {
                    Ok(lease) => return Ok(lease),
                    Err(AdmissionError::Closed) => return Err(AdmissionError::Closed),
                    Err(AdmissionError::Full) => (),
                }
                Some(turn)
            }
            Err(TryAcquireError::Closed) => return Err(AdmissionError::Closed),
            Err(TryAcquireError::NoPermits) => None,
        };
        let _slot = self.slots.clone().try_acquire_owned()?;
        let _guest_slot = guests.slots.try_acquire()?;
        let _turn = match turn {
            Some(turn) => turn,
            None => guests.turn.acquire().await.map_err(|_| AdmissionError::Closed)?,
        };

        loop {
            let mut changed = std::pin::pin!(guests.changed.notified());
            changed.as_mut().enable();
            match self.try_acquire() {
                Ok(lease) => return Ok(lease),
                Err(AdmissionError::Closed) => return Err(AdmissionError::Closed),
                Err(AdmissionError::Full) => changed.await,
            }
        }
    }

    /// Rejects new work and wakes all waiters. An active lease remains owned
    /// until its holder completes cleanup. This queue cannot be reopened.
    pub fn close(&self) {
        self.slots.close();
        self.engine.close();
        if let Some(guests) = self.guests.get() {
            guests.slots.close();
            guests.turn.close();
            guests.changed.notify_waiters();
        }
    }

    /// Approximate status, including a granted ticket not yet resumed by its
    /// caller. Never use this count to decide whether unload is safe: take a lease.
    pub fn waiting(&self) -> usize {
        self.max_waiters - self.slots.available_permits()
    }
}
