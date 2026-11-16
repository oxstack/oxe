//! A bounded FIFO of requests for one engine. No task or thread is spawned.
use std::fmt;
use std::sync::Arc;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, TryAcquireError};

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

/// Exclusive engine access, including cancellation cleanup and model operations.
///
/// Keep this alive until the engine request has finished or cancellation is
/// acknowledged. Cancelling a caller must not drop the lease while its upstream
/// work can still execute. Dropping the lease grants the next waiting request.
/// A lease cannot be cloned, and dropping the queue does not invalidate it.
#[must_use = "retain the lease until engine work and cancellation cleanup finish"]
pub struct Lease {
    _permit: OwnedSemaphorePermit,
}

/// One active request plus at most `max_waiters` suspended acquisitions.
///
/// FIFO order begins when an acquisition future is first polled. Dropping a
/// waiting future cancels that ticket and returns its capacity immediately.
/// A cancelled request that retries joins the back of the queue. The underlying
/// Tokio semaphore supplies FIFO admission and cancellation-safe wakeups.
///
/// Clones refer to the same engine. Use the same queue for model operations and
/// generation; snapshots of `waiting()` are for status, never idle admission.
#[derive(Clone)]
pub struct RequestQueue {
    engine: Arc<Semaphore>,
    slots: Arc<Semaphore>,
    max_waiters: usize,
}

impl RequestQueue {
    /// Panics if the capacity exceeds Tokio's platform-specific permit limit.
    /// Zero capacity permits immediate work but refuses all waiting requests.
    pub fn new(max_waiters: usize) -> Self {
        Self {
            engine: Arc::new(Semaphore::new(1)),
            slots: Arc::new(Semaphore::new(max_waiters)),
            max_waiters,
        }
    }

    /// Acquires immediately, preserving the place of already queued requests.
    /// Failure to acquire an occupied engine is reported as Full.
    pub fn try_acquire(&self) -> Result<Lease, AdmissionError> {
        self.engine
            .clone()
            .try_acquire_owned()
            .map(|permit| Lease { _permit: permit })
            .map_err(|error| match error {
                TryAcquireError::Closed => AdmissionError::Closed,
                TryAcquireError::NoPermits => AdmissionError::Full,
            })
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
        let slot = self
            .slots
            .clone()
            .try_acquire_owned()
            .map_err(|error| match error {
                TryAcquireError::Closed => AdmissionError::Closed,
                TryAcquireError::NoPermits => AdmissionError::Full,
            })?;
        let permit = self
            .engine
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| AdmissionError::Closed)?;
        drop(slot);
        Ok(Lease { _permit: permit })
    }

    /// Rejects new work and wakes all waiters. An active lease remains owned
    /// until its holder completes cleanup. This queue cannot be reopened.
    pub fn close(&self) {
        self.slots.close();
        self.engine.close();
    }

    /// Approximate status, including a granted ticket not yet resumed by its
    /// caller. Never use this count to decide whether unload is safe: take a lease.
    pub fn waiting(&self) -> usize {
        self.max_waiters - self.slots.available_permits()
    }
}
