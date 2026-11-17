use oxc_local::queue::{AdmissionError, RequestQueue};
use std::future::Future;
use std::pin::Pin;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::task::{Context, Poll, Wake, Waker};

fn poll<F: Future>(future: Pin<&mut F>) -> Poll<F::Output> {
    future.poll(&mut Context::from_waker(Waker::noop()))
}

fn ready<T>(result: Poll<T>) -> T {
    match result {
        Poll::Ready(value) => value,
        Poll::Pending => panic!("expected ready"),
    }
}

#[test]
fn one_active_lease_and_bounded_waiters() {
    let q = RequestQueue::new(2);
    let active = q.try_acquire().unwrap();
    let mut a = Box::pin(q.acquire());
    let mut b = Box::pin(q.acquire());
    assert!(poll(a.as_mut()).is_pending());
    assert!(poll(b.as_mut()).is_pending());
    assert_eq!(q.waiting(), 2);
    let mut excess = Box::pin(q.acquire());
    assert!(matches!(
        poll(excess.as_mut()),
        Poll::Ready(Err(AdmissionError::Full))
    ));
    drop(active);
    let first = ready(poll(a.as_mut())).unwrap();
    assert_eq!(q.waiting(), 1);
    assert!(poll(b.as_mut()).is_pending());
    assert!(matches!(q.try_acquire(), Err(AdmissionError::Full)));
    drop(first);
    let second = ready(poll(b.as_mut())).unwrap();
    assert_eq!(q.waiting(), 0);
    assert!(matches!(q.try_acquire(), Err(AdmissionError::Full)));
    drop(second);
    assert!(q.try_acquire().is_ok());
}

#[test]
fn fifo_and_cancellation_do_not_let_newcomers_jump_a_waiter() {
    let q = RequestQueue::new(3);
    let active = q.try_acquire().unwrap();
    let mut a = Box::pin(q.acquire());
    let mut cancelled = Box::pin(q.acquire());
    let mut b = Box::pin(q.acquire());
    for future in [&mut a, &mut cancelled, &mut b] {
        assert!(poll(future.as_mut()).is_pending());
    }
    drop(cancelled);
    assert_eq!(q.waiting(), 2);
    drop(active);
    // a owns the grant even before it is polled again.
    assert!(matches!(q.try_acquire(), Err(AdmissionError::Full)));
    assert!(poll(b.as_mut()).is_pending());
    let mut newcomer = Box::pin(q.acquire());
    assert!(poll(newcomer.as_mut()).is_pending());
    drop(ready(poll(a.as_mut())).unwrap());
    assert!(poll(newcomer.as_mut()).is_pending());
    drop(ready(poll(b.as_mut())).unwrap());
    assert!(ready(poll(newcomer.as_mut())).is_ok());
}

#[test]
fn cancellation_after_grant_transfers_the_permit_to_the_next_waiter() {
    let q = RequestQueue::new(2);
    let active = q.try_acquire().unwrap();
    let mut a = Box::pin(q.acquire());
    let mut b = Box::pin(q.acquire());
    assert!(poll(a.as_mut()).is_pending());
    assert!(poll(b.as_mut()).is_pending());
    drop(active);
    drop(a); // grant exists, but the owner never resumes its acquisition future
    assert_eq!(q.waiting(), 1);
    assert!(ready(poll(b.as_mut())).is_ok());
    assert_eq!(q.waiting(), 0);
}

#[test]
fn cancelled_ticket_releases_capacity_immediately_and_retry_goes_last() {
    let q = RequestQueue::new(2);
    let active = q.try_acquire().unwrap();
    let mut a = Box::pin(q.acquire());
    let mut b = Box::pin(q.acquire());
    assert!(poll(a.as_mut()).is_pending());
    assert!(poll(b.as_mut()).is_pending());
    drop(a);
    let mut retried = Box::pin(q.acquire());
    assert!(poll(retried.as_mut()).is_pending());
    drop(active);
    assert!(poll(retried.as_mut()).is_pending());
    drop(ready(poll(b.as_mut())).unwrap());
    assert!(ready(poll(retried.as_mut())).is_ok());
}

#[test]
fn lazy_future_does_not_reserve_a_slot_and_zero_waiters_still_allows_work() {
    let q = RequestQueue::new(0);
    let _unpolled = q.acquire();
    let active = q.try_acquire().unwrap();
    assert_eq!(q.waiting(), 0);
    let mut refused = Box::pin(q.acquire());
    assert!(matches!(
        poll(refused.as_mut()),
        Poll::Ready(Err(AdmissionError::Full))
    ));
    drop(active);
    let mut admitted = Box::pin(q.acquire());
    assert!(ready(poll(admitted.as_mut())).is_ok());
}

#[test]
fn close_wakes_waiters_and_does_not_release_active_work() {
    let q = RequestQueue::new(2);
    let clone = q.clone();
    let active = q.try_acquire().unwrap();
    let mut waiting = Box::pin(q.acquire());
    assert!(poll(waiting.as_mut()).is_pending());
    clone.close();
    assert!(matches!(
        poll(waiting.as_mut()),
        Poll::Ready(Err(AdmissionError::Closed))
    ));
    assert_eq!(q.waiting(), 0);
    assert!(matches!(q.try_acquire(), Err(AdmissionError::Closed)));
    drop(active);
    let mut later = Box::pin(clone.acquire());
    assert!(matches!(
        poll(later.as_mut()),
        Poll::Ready(Err(AdmissionError::Closed))
    ));
}

struct Wakes(AtomicUsize);
impl Wake for Wakes {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn release_and_shutdown_wake_suspended_tasks_without_a_timer() {
    let q = RequestQueue::new(2);
    let wakes = Arc::new(Wakes(AtomicUsize::new(0)));
    let waker = Waker::from(wakes.clone());
    let mut cx = Context::from_waker(&waker);
    let active = q.try_acquire().unwrap();
    let mut waiting = Box::pin(q.acquire());
    assert!(waiting.as_mut().poll(&mut cx).is_pending());
    assert_eq!(wakes.0.load(Ordering::SeqCst), 0);
    drop(active);
    assert_eq!(wakes.0.load(Ordering::SeqCst), 1);
    let granted = ready(waiting.as_mut().poll(&mut cx)).unwrap();
    let mut next = Box::pin(q.acquire());
    assert!(next.as_mut().poll(&mut cx).is_pending());
    q.close();
    assert_eq!(wakes.0.load(Ordering::SeqCst), 2);
    assert!(matches!(
        next.as_mut().poll(&mut cx),
        Poll::Ready(Err(AdmissionError::Closed))
    ));
    drop(granted);
}

#[test]
fn queue_clones_serialize_concurrent_engine_work() {
    struct ThreadWake(std::thread::Thread);
    impl Wake for ThreadWake {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
        fn wake_by_ref(self: &Arc<Self>) {
            self.0.unpark();
        }
    }
    fn wait<F: Future>(future: F) -> F::Output {
        let waker = Waker::from(Arc::new(ThreadWake(std::thread::current())));
        let mut cx = Context::from_waker(&waker);
        let mut future = std::pin::pin!(future);
        loop {
            if let Poll::Ready(value) = future.as_mut().poll(&mut cx) {
                return value;
            }
            std::thread::park();
        }
    }
    let queue = RequestQueue::new(8);
    let live = AtomicUsize::new(0);
    let completed = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..8 {
            let queue = queue.clone();
            let live = &live;
            let completed = &completed;
            scope.spawn(move || {
                for _ in 0..32 {
                    let lease = wait(queue.acquire()).unwrap();
                    assert_eq!(live.fetch_add(1, Ordering::SeqCst), 0);
                    std::thread::yield_now();
                    assert_eq!(live.fetch_sub(1, Ordering::SeqCst), 1);
                    completed.fetch_add(1, Ordering::SeqCst);
                    drop(lease);
                }
            });
        }
    });
    assert_eq!(live.load(Ordering::SeqCst), 0);
    assert_eq!(completed.load(Ordering::SeqCst), 256);
    assert_eq!(queue.waiting(), 0);
    assert!(queue.try_acquire().is_ok());
}

#[test]
fn queued_owner_runs_before_guest_without_preempting_active_guest() {
    let queue = RequestQueue::new(3);
    let active_guest = ready(poll(Box::pin(queue.acquire_guest()).as_mut())).unwrap();
    let mut guest = Box::pin(queue.acquire_guest());
    let mut first_owner = Box::pin(queue.acquire());
    let mut second_owner = Box::pin(queue.acquire());

    assert!(poll(guest.as_mut()).is_pending());
    assert!(poll(first_owner.as_mut()).is_pending());
    assert!(poll(second_owner.as_mut()).is_pending());

    drop(active_guest);
    assert!(poll(guest.as_mut()).is_pending());
    assert!(poll(second_owner.as_mut()).is_pending());
    let first = ready(poll(first_owner.as_mut())).unwrap();
    drop(first);
    assert!(poll(guest.as_mut()).is_pending());
    let second = ready(poll(second_owner.as_mut())).unwrap();
    drop(second);
    assert!(ready(poll(guest.as_mut())).is_ok());
}

#[test]
fn guests_preserve_owner_capacity_and_fifo_after_cancellation() {
    let queue = RequestQueue::new(3);
    let active = queue.try_acquire().unwrap();
    let mut first_guest = Box::pin(queue.acquire_guest());
    let mut second_guest = Box::pin(queue.acquire_guest());
    assert!(poll(first_guest.as_mut()).is_pending());
    assert!(poll(second_guest.as_mut()).is_pending());
    assert!(matches!(
        poll(Box::pin(queue.acquire_guest()).as_mut()),
        Poll::Ready(Err(AdmissionError::Full))
    ));
    let mut owner = Box::pin(queue.acquire());
    assert!(poll(owner.as_mut()).is_pending());
    assert_eq!(queue.waiting(), 3);

    drop(first_guest);
    let mut retry = Box::pin(queue.acquire_guest());
    assert!(poll(retry.as_mut()).is_pending());
    drop(active);
    assert!(poll(second_guest.as_mut()).is_pending());
    drop(owner); // Return the semaphore grant without constructing an owner lease.
    assert!(poll(retry.as_mut()).is_pending());
    let guest = ready(poll(second_guest.as_mut())).unwrap();
    assert!(poll(retry.as_mut()).is_pending());
    drop(guest);
    assert!(ready(poll(retry.as_mut())).is_ok());
}

#[test]
fn guest_shutdown_wakes_both_the_head_and_other_waiters() {
    let queue = RequestQueue::new(3);
    let active = queue.try_acquire().unwrap();
    let mut head = Box::pin(queue.acquire_guest());
    let mut tail = Box::pin(queue.acquire_guest());
    assert!(poll(head.as_mut()).is_pending());
    assert!(poll(tail.as_mut()).is_pending());
    queue.close();
    assert!(matches!(
        poll(head.as_mut()),
        Poll::Ready(Err(AdmissionError::Closed))
    ));
    assert!(matches!(
        poll(tail.as_mut()),
        Poll::Ready(Err(AdmissionError::Closed))
    ));
    assert_eq!(queue.waiting(), 0);
    drop(active);
}

#[test]
fn cancellation_of_granted_owner_wakes_guest_without_polling() {
    let queue = RequestQueue::new(2);
    let active = queue.try_acquire().unwrap();
    let wakes = Arc::new(Wakes(AtomicUsize::new(0)));
    let waker = Waker::from(wakes.clone());
    let mut context = Context::from_waker(&waker);
    let mut guest = Box::pin(queue.acquire_guest());
    let mut owner = Box::pin(queue.acquire());

    assert!(guest.as_mut().poll(&mut context).is_pending());
    assert!(poll(owner.as_mut()).is_pending());
    drop(active);
    assert!(guest.as_mut().poll(&mut context).is_pending());

    let observed = wakes.0.load(Ordering::SeqCst);
    drop(owner);
    assert!(wakes.0.load(Ordering::SeqCst) > observed);
    assert!(ready(guest.as_mut().poll(&mut context)).is_ok());
}
