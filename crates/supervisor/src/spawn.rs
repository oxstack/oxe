//! Engine child creation: one process-lifetime spawner thread plus the parent-death contract.
//!
//! `PR_SET_PDEATHSIG` follows the *thread* that called `fork`, not the process that owns it: the
//! kernel signals the child when that thread exits. A child spawned from a short-lived caller (a
//! Tokio blocking-pool worker, say) would therefore be killed whenever that worker is parked,
//! long before its owner exits. Every child is instead created by one dedicated thread that lives
//! until the process does, so the parent-death signal really means "the owning process is gone".
//! Keep it that way: `Supervisor` spawns from arbitrary caller threads, including blocking
//! workers of `oxe serve`.
use std::io;
use std::process::{Child, Command};
use std::sync::mpsc::{self, SyncSender};
use std::sync::OnceLock;

type Job = Box<dyn FnOnce() + Send + 'static>;

/// The spawning thread. Started on first use and never joined: it must outlive every child it
/// creates, so it is parked on this channel for the rest of the process.
fn jobs() -> &'static SyncSender<Job> {
    static JOBS: OnceLock<SyncSender<Job>> = OnceLock::new();
    JOBS.get_or_init(|| {
        let (sender, receiver) = mpsc::sync_channel::<Job>(0);
        std::thread::Builder::new()
            .name("oxe-engine-spawn".into())
            .spawn(move || {
                while let Ok(job) = receiver.recv() {
                    job();
                }
            })
            .expect("spawn the process-lifetime engine spawner thread");
        sender
    })
}

/// Creates the child on the process-lifetime thread with its parent-death signal armed.
pub(crate) fn spawn(mut command: Command) -> io::Result<Child> {
    arm_parent_death(&mut command);
    let (reply, answer) = mpsc::channel();
    jobs()
        .send(Box::new(move || {
            let _ = reply.send(command.spawn());
        }))
        .map_err(|_| io::Error::other("engine spawner thread exited"))?;
    answer
        .recv()
        .map_err(|_| io::Error::other("engine spawner thread dropped the spawn"))?
}

/// Arms `PR_SET_PDEATHSIG(SIGKILL)` in the child and closes the window in which the owner could
/// have died between `fork` and `prctl`: if `getppid()` inside the child is no longer us, the
/// child refuses to exec instead of becoming an orphan.
#[cfg(target_os = "linux")]
#[allow(unsafe_code)] // `CommandExt::pre_exec` is unsafe by contract; the body is nix's safe API.
fn arm_parent_death(command: &mut Command) {
    use nix::errno::Errno;
    use nix::sys::prctl::set_pdeathsig;
    use nix::sys::signal::Signal;
    use nix::unistd::{getpid, getppid};
    use std::os::unix::process::CommandExt;

    let parent = getpid().as_raw();
    // SAFETY: the closure runs in the forked child before `execve`, so it must stay
    // async-signal-safe. It only makes the `prctl` and `getppid` syscalls through nix's safe
    // wrappers, allocates nothing and shares no state with the parent's other threads.
    unsafe {
        command.pre_exec(move || {
            set_pdeathsig(Signal::SIGKILL).map_err(io::Error::from)?;
            if getppid().as_raw() != parent {
                // The owner is already gone; `std` reports this as a failed spawn, never an orphan.
                return Err(io::Error::from(Errno::ESRCH));
            }
            Ok(())
        });
    }
}

#[cfg(not(target_os = "linux"))]
fn arm_parent_death(_command: &mut Command) {}
