//! Hybrid-aware CPU placement selector (engine N2, issue #913).
//!
//! **Contract, not a default.** Nothing on the serving path calls this yet: `oxe-server` still
//! builds its pools from `--oxe-cpus` (default `0-7`), `--oxe-prefill-cpus` and
//! [`crate::clamp_to_allowed`] exactly as before, and `engine/.cargo/config.toml` still generates
//! Oryon-1 code for `aarch64-unknown-linux-gnu`. This module is the isolated, pure selector the N2
//! contract in `engine/docs/CPU-TARGET-AND-PLACEMENT.md` specifies, so the placement policy can be
//! reviewed and fixture-tested before any owner-qualified default change. The doc is the contract;
//! this file is its implementation and [`crate::tests_placement`] its fixture goldens.
//!
//! Rules (the reasons are in the doc):
//!
//! - An explicit user CPU list is never reordered, extended or re-classified: it keeps the caller's
//!   order and defines its own size (`threads` is ignored for it). CPUs in it that the platform does
//!   not report online, or that are outside the process affinity mask, are dropped with a note —
//!   today's `clamp_to_allowed` behaviour — and a list with nothing left is an error.
//! - With no explicit list, capacity decides only when *every* candidate reports one: the
//!   highest-capacity class is selected, at most `threads` of it, ties in ascending CPU id.
//!   Equal known capacities therefore keep the current ascending-id order.
//! - Missing capacity (absent, empty, unparsable or `0`) degrades the whole machine to the
//!   equal-capacity order with a note. Capacity is never inferred from CPU numbering.
//! - Conflicting data for one CPU id (two entries that disagree) makes that id a non-candidate with
//!   a note.
//! - The selector never returns a CPU outside `online ∩ allowed`; `allowed = None` means the mask
//!   could not be read and every online CPU counts as allowed (the existing `clamp_to_allowed`
//!   fallback). No CPU is ever added to fill `threads`.
//!
//! Only std is used (`std::fs` for the read-only sysfs reader); the caller injects the data, so the
//! selector itself is pure and testable without a real `/sys` or a real affinity mask.

use std::collections::BTreeSet;
use std::path::Path;

/// One CPU's platform data, injected by the caller: sysfs `online` / `cpu_capacity` (see
/// [`read_cpu_dir`]), an override, or a test fixture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpuData {
    pub id: usize,
    /// `false` only when the platform reports the CPU offline (`cpuN/online` is `0`).
    pub online: bool,
    /// Relative capacity (sysfs `cpu_capacity`), or `None` when it is unknown.
    pub capacity: Option<u32>,
}

/// A documented, stable degradation of the selection. Rendered by `Display` so callers can print
/// exactly one line each; the fixture goldens compare that text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlacementNote {
    /// Two or more entries disagreed about one CPU id; that id is not a candidate.
    DataInconsistent(Vec<usize>),
    /// Auto selection: CPUs the platform reports offline.
    CandidatesOffline(Vec<usize>),
    /// Auto selection: online CPUs outside the process affinity mask.
    CandidatesNotAllowed(Vec<usize>),
    /// An explicit list named CPUs the platform does not report online.
    RequestedOffline(Vec<usize>),
    /// An explicit list named online CPUs outside the process affinity mask.
    RequestedNotAllowed(Vec<usize>),
    /// At least one candidate has no capacity: the equal-capacity (ascending id) order was used.
    CapacityMissing(Vec<usize>),
}

impl std::fmt::Display for PlacementNote {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlacementNote::DataInconsistent(ids) => write!(
                f,
                "platform data for {ids:?} is inconsistent; those cpus are not candidates"
            ),
            PlacementNote::CandidatesOffline(ids) => {
                write!(f, "cpus {ids:?} are offline and were not candidates")
            }
            PlacementNote::CandidatesNotAllowed(ids) => write!(
                f,
                "cpus {ids:?} are outside the process affinity mask and were not candidates"
            ),
            PlacementNote::RequestedOffline(ids) => {
                write!(f, "cpus {ids:?} in the requested list are not online")
            }
            PlacementNote::RequestedNotAllowed(ids) => write!(
                f,
                "cpus {ids:?} in the requested list are outside the process affinity mask"
            ),
            PlacementNote::CapacityMissing(ids) => write!(
                f,
                "cpu capacity is unknown for {ids:?}; using the equal-capacity (ascending id) order"
            ),
        }
    }
}

/// Why no usable list could be returned. Conservative by design: the caller keeps its current
/// explicit default instead of guessing, and the notes gathered before the failure are dropped.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlacementError {
    /// No CPU is both online and inside the process affinity mask.
    NoUsableCpu,
    /// Auto selection with a thread budget of zero.
    NoThreads,
    /// Every CPU in the explicit list is unavailable; `dropped` is the list as given.
    RequestedUnavailable { dropped: Vec<usize> },
}

impl std::fmt::Display for PlacementError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlacementError::NoUsableCpu => {
                write!(
                    f,
                    "no cpu is both online and inside the process affinity mask"
                )
            }
            PlacementError::NoThreads => write!(f, "threads must be at least 1"),
            PlacementError::RequestedUnavailable { dropped } => write!(
                f,
                "none of the requested cpus {dropped:?} is available to this process"
            ),
        }
    }
}

/// The chosen CPU list, in the order the pools will use it, plus the degradations that applied.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placement {
    pub cpus: Vec<usize>,
    pub notes: Vec<PlacementNote>,
}

/// Choose the CPUs for one pool from injected platform data.
///
/// `requested` is an explicit user list (`--oxe-cpus`) in the user's order, or `None` for auto
/// selection. `allowed` is the process affinity mask as a CPU list, or `None` when it could not be
/// read. `threads` is the caller's current thread count; auto selection never exceeds it and never
/// pads up to it.
pub fn select(
    data: &[CpuData],
    requested: Option<&[usize]>,
    allowed: Option<&[usize]>,
    threads: usize,
) -> Result<Placement, PlacementError> {
    // Merge duplicate ids. Two entries that disagree make the id a non-candidate: the sysfs tree
    // reader cannot produce this, but a caller that merges an override can.
    let mut ids: Vec<usize> = data.iter().map(|d| d.id).collect();
    ids.sort_unstable();
    ids.dedup();
    let mut merged: Vec<(usize, bool, Option<u32>)> = Vec::with_capacity(ids.len());
    let mut inconsistent: Vec<usize> = Vec::new();
    for id in ids {
        let first = *data.iter().find(|d| d.id == id).expect("id came from data");
        let agree = data
            .iter()
            .all(|d| d.id != id || (d.online == first.online && d.capacity == first.capacity));
        if agree {
            merged.push((id, first.online, first.capacity));
        } else {
            inconsistent.push(id);
            merged.push((id, false, None));
        }
    }

    let allowed_set: Option<BTreeSet<usize>> = allowed.map(|a| a.iter().copied().collect());
    let is_allowed = |id: usize| allowed_set.as_ref().is_none_or(|s| s.contains(&id));
    let is_inconsistent = |id: usize| inconsistent.contains(&id);
    let is_online = |id: usize| merged.iter().any(|&(i, on, _)| i == id && on);
    let capacity = |id: usize| {
        merged
            .iter()
            .find(|&&(i, _, _)| i == id)
            .and_then(|&(_, _, c)| c)
    };
    let usable = |id: usize| is_online(id) && is_allowed(id) && !is_inconsistent(id);

    let mut notes = Vec::new();
    if !inconsistent.is_empty() {
        notes.push(PlacementNote::DataInconsistent(inconsistent.clone()));
    }

    if let Some(req) = requested {
        let kept: Vec<usize> = req.iter().copied().filter(|&id| usable(id)).collect();
        if kept.is_empty() {
            return Err(PlacementError::RequestedUnavailable {
                dropped: req.to_vec(),
            });
        }
        let offline: Vec<usize> = req
            .iter()
            .copied()
            .filter(|&id| !usable(id) && !is_inconsistent(id) && !is_online(id))
            .collect();
        let not_allowed: Vec<usize> = req
            .iter()
            .copied()
            .filter(|&id| !usable(id) && !is_inconsistent(id) && is_online(id))
            .collect();
        if !offline.is_empty() {
            notes.push(PlacementNote::RequestedOffline(offline));
        }
        if !not_allowed.is_empty() {
            notes.push(PlacementNote::RequestedNotAllowed(not_allowed));
        }
        return Ok(Placement { cpus: kept, notes });
    }

    if threads == 0 {
        return Err(PlacementError::NoThreads);
    }

    let offline: Vec<usize> = merged
        .iter()
        .filter(|&&(id, on, _)| !on && !is_inconsistent(id))
        .map(|&(id, _, _)| id)
        .collect();
    let not_allowed: Vec<usize> = merged
        .iter()
        .filter(|&&(id, on, _)| on && !is_allowed(id) && !is_inconsistent(id))
        .map(|&(id, _, _)| id)
        .collect();
    if !offline.is_empty() {
        notes.push(PlacementNote::CandidatesOffline(offline));
    }
    if !not_allowed.is_empty() {
        notes.push(PlacementNote::CandidatesNotAllowed(not_allowed));
    }

    let candidates: Vec<usize> = merged
        .iter()
        .filter(|&&(id, on, _)| on && is_allowed(id) && !is_inconsistent(id))
        .map(|&(id, _, _)| id)
        .collect();
    if candidates.is_empty() {
        return Err(PlacementError::NoUsableCpu);
    }

    let unknown: Vec<usize> = candidates
        .iter()
        .copied()
        .filter(|&id| capacity(id).is_none())
        .collect();
    let cpus: Vec<usize> = if unknown.is_empty() {
        let fastest = candidates
            .iter()
            .map(|&id| capacity(id).expect("no unknown candidate here"))
            .max()
            .expect("candidates is not empty");
        candidates
            .iter()
            .copied()
            .filter(|&id| capacity(id) == Some(fastest))
            .take(threads)
            .collect()
    } else {
        notes.push(PlacementNote::CapacityMissing(unknown));
        candidates.iter().copied().take(threads).collect()
    };
    Ok(Placement { cpus, notes })
}

/// Read one CPU's platform data from a sysfs-like directory: `/sys/devices/system/cpu/cpuN`, or a
/// fixture tree under `tests/fixtures/cpu/` in tests. Read-only.
///
/// - `online`: the trimmed value, `false` only for `0`; a missing or unreadable file counts as
///   online (some platforms have no per-CPU switch).
/// - `cpu_capacity`: the trimmed integer; a missing, empty, unparsable or `0` value is unknown
///   (`None`), never an error — the conservative direction is "no capacity data".
pub fn read_cpu_dir(id: usize, dir: &Path) -> CpuData {
    let online = match std::fs::read_to_string(dir.join("online")) {
        Ok(s) => s.trim() != "0",
        Err(_) => true,
    };
    let capacity = std::fs::read_to_string(dir.join("cpu_capacity"))
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
        .filter(|&c| c > 0);
    CpuData {
        id,
        online,
        capacity,
    }
}

/// Scan `root` for `cpuN` directories and read each with [`read_cpu_dir`], ascending by id.
/// Anything that is not `cpu<digits>` is ignored, so a fixture directory may hold a golden file
/// next to the tree.
pub fn read_cpu_tree(root: &Path) -> std::io::Result<Vec<CpuData>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if let Some(rest) = name.strip_prefix("cpu") {
            if let Ok(id) = rest.parse::<usize>() {
                if entry.file_type()?.is_dir() {
                    out.push(read_cpu_dir(id, &entry.path()));
                }
            }
        }
    }
    out.sort_by_key(|d| d.id);
    Ok(out)
}
