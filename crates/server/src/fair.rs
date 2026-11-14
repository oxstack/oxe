//! Fair time-slicing of the single engine slot between generations (`--fair-quantum-ms N`, env
//! `OXE_FAIR_QUANTUM_MS`; 0 or unset = off: every request takes today's path, `Worker::generate`
//! under the worker lock, first come first served).
//!
//! With a quantum, every generation takes a ticket in one FIFO turn queue; the ticket at the
//! front has the turn and takes the worker. A running generation gives the engine away at a token
//! boundary once its slice lasted the effective quantum and another generation waits: its whole
//! session (KV, recurrent and MTP state, checkpoints) is held in RAM outside the engine
//! (`PrefixCache::hold_live`), the worker lock is released, the ticket goes to the back of the
//! queue and the next one runs. When its turn comes back the live session (a generation that ended
//! meanwhile) is put away as a session switch puts it away, and the held session is loaded back
//! bit for bit (`restore_held`), never re-prefilled (prefill and decode numerics differ). Output is
//! therefore identical to an unsliced run: same tokens, text, stop, probability rows and cache
//! state. See `Worker::generate_slice` for the yield point.
//!
//! - Effective quantum = max(configured, 8 x the last measured suspend + resume of this run), so a
//!   switch costs at most ~12 % of a slice.
//! - Not sliced: prefill (a waiting request waits for a running prefill), ephemeral (LAN share)
//!   and prefill-only (`max_tokens: 0`) requests. These take a ticket too (FIFO) but never yield.
//! - A run whose session cannot be held (no park budget, a state larger than it, or other
//!   suspended runs already hold that RAM) does not yield: it keeps the engine to its end.
//! - Cancel: a queued request leaves the queue at once; a suspended one is dropped and its session
//!   joins the RAM parked sessions (`Worker::generate_abort`). SIGTERM raises every ticket's cancel
//!   flag ([`Fair::cancel_all`]) so the bounded shutdown write gets the worker.
//! - Cache endpoints (`/state-cache/clear|budget`, session purges) answer 409 while a generation
//!   is suspended, as they do while one runs: a clear can never be undone by a later resume. The
//!   preload route sees the suspended request as work in flight (busy).
//! - Post-reply jobs (state write, idle re-prefill) of a generation: when another generation is
//!   queued at its end they are skipped and that generation's switch path writes the state when it
//!   puts it away (as for an overtaken request); when nobody is queued and it was sliced they run
//!   although newer requests were counted while it was suspended ([`Fair::post_reply_req`]).
//!
//! Lock order (outer first): `State::persist_gate` -> `State::worker` -> `Fair::q`. A generation
//! never waits for its turn while it holds the worker; the cache endpoints only `try_lock` the
//! worker and read the suspended count under it; `note_suspend` / `note_resume` are called with
//! the worker held, so that count is consistent for anyone holding the worker.

use crate::engine::Engine;
use crate::generate::{GenError, GenParams, GenResult, Partial};
use crate::service::State;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};
use tokenizer::Token;

/// A switch may cost at most 1/SWITCH_FACTOR of a slice (~12 %).
pub const SWITCH_FACTOR: f64 = 8.0;

/// How often a waiting request looks at its cancel flag (ms).
const POLL_MS: u64 = 20;

/// A request number no request has (`State::reqs` starts at 1): post-reply jobs compared with it
/// never run.
pub const SKIP: u64 = 0;

#[derive(Default)]
struct Queue {
    /// tickets in turn order; the front has the turn
    order: VecDeque<u64>,
    next: u64,
    /// every ticket's cancel flag (SIGTERM)
    flags: Vec<(u64, Arc<AtomicBool>)>,
}

pub struct Fair {
    quantum_ms: AtomicU64,
    q: Mutex<Queue>,
    cv: Condvar,
    /// generations suspended right now (changed only with the worker held)
    suspended: AtomicUsize,
    /// (request number at the start of a sliced generation, request number at its end) for its
    /// post-reply jobs; the newest few
    rebased: Mutex<VecDeque<(u64, u64)>>,
}

/// A place in the turn queue; leaves it when dropped.
pub struct Ticket<'a> {
    fair: &'a Fair,
    id: u64,
}

impl Drop for Ticket<'_> {
    fn drop(&mut self) {
        let mut q = self.fair.q.lock().unwrap_or_else(|p| p.into_inner());
        q.order.retain(|&x| x != self.id);
        q.flags.retain(|x| x.0 != self.id);
        drop(q);
        self.fair.cv.notify_all();
    }
}

impl Fair {
    pub fn new(quantum_ms: u64) -> Fair {
        Fair { quantum_ms: AtomicU64::new(quantum_ms), q: Mutex::new(Queue::default()), cv: Condvar::new(), suspended: AtomicUsize::new(0), rebased: Mutex::new(VecDeque::new()) }
    }

    /// `OXE_FAIR_QUANTUM_MS` (unset, unparsable or 0: off).
    pub fn from_env() -> Fair {
        Fair::new(std::env::var("OXE_FAIR_QUANTUM_MS").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(0).min(600_000))
    }

    /// The configured quantum; None = fair slicing off.
    pub fn quantum(&self) -> Option<Duration> {
        match self.quantum_ms.load(Ordering::Relaxed) {
            0 => None,
            ms => Some(Duration::from_millis(ms)),
        }
    }

    pub fn set_quantum_ms(&self, ms: u64) {
        self.quantum_ms.store(ms, Ordering::Relaxed);
    }

    /// max(configured, SWITCH_FACTOR x the last switch of this run)
    pub fn effective_quantum(configured: Duration, switch_ms: f64) -> Duration {
        configured.max(Duration::from_secs_f64((switch_ms * SWITCH_FACTOR / 1e3).max(0.0)))
    }

    /// Join the back of the queue.
    pub fn enter(&self, cancel: Arc<AtomicBool>) -> Ticket<'_> {
        let mut q = self.q.lock().unwrap_or_else(|p| p.into_inner());
        q.next += 1;
        let id = q.next;
        q.order.push_back(id);
        q.flags.push((id, cancel));
        Ticket { fair: self, id }
    }

    /// Wait until `t` is at the front. False when `cancel` rose first.
    pub fn wait_turn(&self, t: &Ticket, cancel: &AtomicBool) -> bool {
        let mut q = self.q.lock().unwrap_or_else(|p| p.into_inner());
        loop {
            if cancel.load(Ordering::SeqCst) {
                return false;
            }
            if q.order.front() == Some(&t.id) {
                return true;
            }
            q = self.cv.wait_timeout(q, Duration::from_millis(POLL_MS)).unwrap_or_else(|p| p.into_inner()).0;
        }
    }

    /// Another ticket is in the queue (queued or suspended).
    pub fn others_waiting(&self, t: &Ticket) -> bool {
        let q = self.q.lock().unwrap_or_else(|p| p.into_inner());
        q.order.iter().any(|&x| x != t.id)
    }

    /// `t` goes to the back of the queue (it just gave the engine away).
    pub fn rotate(&self, t: &Ticket) {
        let mut q = self.q.lock().unwrap_or_else(|p| p.into_inner());
        q.order.retain(|&x| x != t.id);
        q.order.push_back(t.id);
        drop(q);
        self.cv.notify_all();
    }

    /// The tickets in turn order (tests, status).
    pub fn order(&self) -> Vec<u64> {
        self.q.lock().unwrap_or_else(|p| p.into_inner()).order.iter().copied().collect()
    }

    pub fn ticket_id(t: &Ticket) -> u64 {
        t.id
    }

    /// SIGTERM: cancel every queued, suspended and running generation.
    pub fn cancel_all(&self) {
        let q = self.q.lock().unwrap_or_else(|p| p.into_inner());
        for (_, f) in &q.flags {
            f.store(true, Ordering::SeqCst);
        }
        drop(q);
        self.cv.notify_all();
    }

    /// Generations suspended now (read with the worker held for a consistent answer).
    pub fn suspended(&self) -> usize {
        self.suspended.load(Ordering::SeqCst)
    }

    pub(crate) fn note_suspend(&self) {
        self.suspended.fetch_add(1, Ordering::SeqCst);
    }

    pub(crate) fn note_resume(&self) {
        self.suspended.fetch_sub(1, Ordering::SeqCst);
    }

    /// The post-reply jobs of request `from` compare `to` with `State::reqs` instead of `from`: the
    /// current number when it was sliced and ended with nobody queued (the newer requests counted
    /// meanwhile all ended before it), [`SKIP`] when it ended with another generation queued.
    pub fn rebase(&self, from: u64, to: u64) {
        let mut r = self.rebased.lock().unwrap_or_else(|p| p.into_inner());
        r.retain(|x| x.0 != from);
        r.push_back((from, to));
        while r.len() > 64 {
            r.pop_front();
        }
    }

    /// The request number a post-reply job of request `req_no` compares with `State::reqs` (every
    /// job of the request asks: the state write and the idle re-prefill).
    pub fn post_reply_req(&self, req_no: u64) -> u64 {
        let r = self.rebased.lock().unwrap_or_else(|p| p.into_inner());
        r.iter().rev().find(|x| x.0 == req_no).map_or(req_no, |x| x.1)
    }
}

/// Clears the running-generation cancel hook (SIGTERM) on every exit of a slice.
struct Running<'a>(&'a Mutex<Option<Arc<AtomicBool>>>);

impl<'a> Running<'a> {
    fn set(m: &'a Mutex<Option<Arc<AtomicBool>>>, f: &Arc<AtomicBool>) -> Running<'a> {
        *m.lock().unwrap_or_else(|p| p.into_inner()) = Some(f.clone());
        Running(m)
    }
}

impl Drop for Running<'_> {
    fn drop(&mut self) {
        *self.0.lock().unwrap_or_else(|p| p.into_inner()) = None;
    }
}

/// The fair driver (`service::run_blocking` with a quantum, on a blocking thread): ticket, turn,
/// worker, then slices until the generation ends, giving the worker and the turn away between them.
/// `gp.cancel` must be set (run_blocking does).
pub(crate) fn run_sliced<E: Engine>(st: &State<E>, prompt: &[Token], gp: &mut GenParams, queued: Instant, on_partial: &mut dyn FnMut(Partial) -> bool) -> Result<GenResult, GenError> {
    let fair = &st.fair;
    let flag = gp.cancel.clone().unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
    // the request number this generation's post-reply jobs are armed with (see `post_reply_req`)
    let my_req = gp.req_no.unwrap_or_else(|| st.reqs.load(Ordering::SeqCst));
    if let Some(pr) = &gp.progress {
        pr.fair.store(true, Ordering::Relaxed);
    }
    let ticket = fair.enter(flag.clone());
    if !fair.wait_turn(&ticket, &flag) {
        return Err(GenError::Cancelled); // the client left while queued
    }
    let mut w = st.worker.lock().unwrap_or_else(|p| p.into_inner());
    if gp.cancelled() {
        return Err(GenError::Cancelled);
    }
    // queue time: until the generation first holds the worker (the later suspensions are fair_wait_ms)
    gp.queue_ms = crate::generate::ms_since(queued);
    if gp.ephemeral || gp.n_predict == 0 {
        let _running = Running::set(&st.running, &flag);
        return w.generate(&st.tok, prompt, gp, on_partial);
    }
    let mut run = {
        let _running = Running::set(&st.running, &flag);
        w.generate_start(prompt, gp)?
    };
    loop {
        let r = {
            let _running = Running::set(&st.running, &flag);
            let q = fair.quantum();
            let eff = q.map(|q| Fair::effective_quantum(q, run.switch_ms()));
            let t0 = Instant::now();
            let mut want = || eff.is_some_and(|e| t0.elapsed() >= e) && fair.others_waiting(&ticket);
            w.generate_slice(&st.tok, prompt, gp, &mut run, on_partial, &mut want)?
        };
        if let Some(r) = r {
            if fair.others_waiting(&ticket) {
                // the next generation takes the engine before this one's post-reply jobs could:
                // they are skipped and that generation's switch path writes this state (with this
                // request's hint) when it puts it away, as for an overtaken request
                fair.rebase(my_req, SKIP);
            } else if r.timings.fair_yields > 0 {
                // the newer requests counted while it was suspended all ended before it
                fair.rebase(my_req, st.reqs.load(Ordering::SeqCst));
            }
            return Ok(r);
        }
        // suspended: counted while the worker is still held (the cache endpoints read the count
        // under the worker lock), then the worker and the turn go to the next generation
        fair.note_suspend();
        if let Some(pr) = &gp.progress {
            pr.suspended.store(true, Ordering::Relaxed);
        }
        drop(w);
        fair.rotate(&ticket);
        let turn = fair.wait_turn(&ticket, &flag);
        if turn {
            // as a starting request does (not the request counter: no newer request came)
            st.preempt_jobs();
        }
        w = st.worker.lock().unwrap_or_else(|p| p.into_inner());
        fair.note_resume();
        if let Some(pr) = &gp.progress {
            pr.suspended.store(false, Ordering::Relaxed);
        }
        if !turn || gp.cancelled() {
            w.generate_abort(run);
            return Err(GenError::Cancelled);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_tickets_rotate_round_robin_and_leave_on_drop() {
        let f = Fair::new(10);
        let c = || Arc::new(AtomicBool::new(false));
        let (a, b, d) = (f.enter(c()), f.enter(c()), f.enter(c()));
        let ids = [Fair::ticket_id(&a), Fair::ticket_id(&b), Fair::ticket_id(&d)];
        assert_eq!(f.order(), ids);
        let never = AtomicBool::new(false);
        assert!(f.wait_turn(&a, &never));
        assert!(f.others_waiting(&a));
        let mut seen = Vec::new();
        for _ in 0..6 {
            let front = f.order()[0];
            seen.push(front);
            let t = [&a, &b, &d].into_iter().find(|t| Fair::ticket_id(t) == front).unwrap();
            assert!(f.wait_turn(t, &never));
            f.rotate(t);
        }
        assert_eq!(seen, [ids[0], ids[1], ids[2], ids[0], ids[1], ids[2]], "round robin");
        drop(b);
        assert_eq!(f.order(), [ids[0], ids[2]]);
        drop(a);
        assert!(!f.others_waiting(&d));
        assert!(f.wait_turn(&d, &never));
    }

    #[test]
    fn a_queued_ticket_sees_its_cancel_and_cancel_all_raises_every_flag() {
        let f = Arc::new(Fair::new(10));
        let first = f.enter(Arc::new(AtomicBool::new(false)));
        let flag = Arc::new(AtomicBool::new(false));
        let (f2, flag2) = (f.clone(), flag.clone());
        let h = std::thread::spawn(move || {
            let t = f2.enter(flag2.clone());
            f2.wait_turn(&t, &flag2)
        });
        while f.order().len() < 2 {
            std::thread::sleep(Duration::from_millis(1));
        }
        f.cancel_all();
        assert!(!h.join().unwrap(), "cancelled while waiting");
        assert!(flag.load(Ordering::SeqCst));
        assert_eq!(f.order(), [Fair::ticket_id(&first)], "the cancelled ticket left the queue");
    }

    #[test]
    fn effective_quantum_and_post_reply_rebase() {
        let q = Duration::from_millis(50);
        assert_eq!(Fair::effective_quantum(q, 1.0), q);
        assert_eq!(Fair::effective_quantum(q, 20.0), Duration::from_millis(160));
        let f = Fair::new(0);
        assert!(f.quantum().is_none());
        assert_eq!(f.post_reply_req(7), 7);
        f.rebase(7, 9);
        assert_eq!(f.post_reply_req(7), 9);
        assert_eq!(f.post_reply_req(7), 9, "every job of the request asks");
        f.rebase(8, SKIP);
        assert_eq!(f.post_reply_req(8), SKIP);
    }
}
