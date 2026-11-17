//! Fixture goldens for the N2 placement contract (`placement.rs`, issue #913).
//!
//! Every case is a sysfs-like tree under `tests/fixtures/cpu/<case>/` plus a `golden.txt` holding
//! the exact rendered outcome. The trees are synthetic and read-only; nothing here touches the real
//! `/sys`, the host affinity mask or a pool.

use super::placement::*;
use std::path::PathBuf;

const ALLOWED_CPUSET: &[usize] = &[2, 3, 6, 7];
const ALLOWED_03: &[usize] = &[0, 3];
const REQUEST_0_1: &[usize] = &[0, 1];
const REQUEST_PARTLY: &[usize] = &[2, 0, 8, 3];
const REQUEST_NONE: &[usize] = &[8, 9];

/// One fixture case: tree name, injected affinity mask, requested list, thread count.
type Case = (
    &'static str,
    Option<&'static [usize]>,
    Option<&'static [usize]>,
    usize,
);

/// (case, allowed, requested, threads)
const CASES: &[Case] = &[
    ("equal-12-threads8", None, None, 8),
    ("equal-8", None, None, 8),
    ("mixed-4e4p", None, None, 8),
    ("mixed-4e4p-threads2", None, None, 2),
    ("mixed-2e8p-threads4", None, None, 4),
    ("mixed-inverted-threads3", None, None, 3),
    ("id-gaps", None, None, 8),
    ("cpuset", Some(ALLOWED_CPUSET), None, 8),
    ("missing-capacity", None, None, 4),
    ("malformed", None, None, 4),
    ("offline", None, None, 8),
    ("explicit-override", None, Some(REQUEST_0_1), 8),
    (
        "explicit-partly-unavailable",
        Some(ALLOWED_03),
        Some(REQUEST_PARTLY),
        8,
    ),
    (
        "explicit-none-available",
        Some(ALLOWED_03),
        Some(REQUEST_NONE),
        8,
    ),
    ("inconsistent-tree", None, None, 8),
    ("empty-tree", None, None, 8),
    ("all-offline", None, None, 8),
];

fn fixture_dir(case: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/cpu")
        .join(case)
}

/// The golden format: one `cpus = [...]` line and a `notes =` section, or one `error = ...` line.
fn render(result: &Result<Placement, PlacementError>) -> String {
    match result {
        Ok(p) => {
            let mut s = format!("cpus = {:?}\nnotes =\n", p.cpus);
            for note in &p.notes {
                s.push_str(&format!("{note}\n"));
            }
            s
        }
        Err(e) => format!("error = {e}\n"),
    }
}

#[test]
fn fixture_trees_match_goldens() {
    for &(case, allowed, requested, threads) in CASES {
        let dir = fixture_dir(case);
        let data = read_cpu_tree(&dir).unwrap_or_else(|e| panic!("{case}: {e}"));
        let got = render(&select(&data, requested, allowed, threads));
        let want = std::fs::read_to_string(dir.join("golden.txt"))
            .unwrap_or_else(|e| panic!("{case}: {e}"));
        assert_eq!(got, want, "fixture {case} (left: got, right: golden)");
    }
}

#[test]
fn loader_reads_online_and_capacity_conservatively() {
    // "abc", empty, "0" are all unknown capacity; a positive integer is kept.
    let malformed = read_cpu_tree(&fixture_dir("malformed")).unwrap();
    assert_eq!(
        malformed
            .iter()
            .map(|d| (d.id, d.capacity))
            .collect::<Vec<_>>(),
        vec![(0, None), (1, None), (2, None), (3, Some(1024)),]
    );
    // online = 0 is offline; every other value (or no file at all) is online.
    let offline = read_cpu_tree(&fixture_dir("offline")).unwrap();
    assert!(!offline.iter().find(|d| d.id == 6).unwrap().online);
    assert!(offline.iter().find(|d| d.id == 0).unwrap().online);
    let equal8 = read_cpu_tree(&fixture_dir("equal-8")).unwrap();
    assert!(equal8.iter().all(|d| d.online)); // no `online` file in this tree
                                              // A tree with no cpuN entries is not an error; the selector refuses it.
    assert!(read_cpu_tree(&fixture_dir("empty-tree"))
        .unwrap()
        .is_empty());
}

#[test]
fn auto_selection_refuses_zero_threads() {
    let data = read_cpu_tree(&fixture_dir("equal-8")).unwrap();
    assert_eq!(select(&data, None, None, 0), Err(PlacementError::NoThreads));
}

#[test]
fn explicit_list_defines_its_own_size() {
    // An explicit list is the caller's decision: `threads` does not truncate or pad it, and
    // capacity never reorders it (cpu0/cpu1 are the slow class in this tree).
    let data = read_cpu_tree(&fixture_dir("mixed-4e4p")).unwrap();
    let p = select(&data, Some(REQUEST_0_1), None, 0).unwrap();
    assert_eq!(p.cpus, vec![0, 1]);
    assert!(p.notes.is_empty());
}

#[test]
fn regular_files_named_like_cpus_do_not_invent_online_processors() {
    let data = read_cpu_tree(&fixture_dir("regular-file")).unwrap();
    assert_eq!(data.iter().map(|cpu| cpu.id).collect::<Vec<_>>(), vec![1]);
    assert_eq!(select(&data, None, None, 1).unwrap().cpus, vec![1]);
    assert_eq!(data[0].capacity, Some(1024));
}
