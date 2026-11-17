//! Fixtures for the opt-in context-length decode-team policy (issue #917). Pure: no pool, no
//! thread, no environment, no timing. Every value here is fixture data, not a proposed threshold.
use crate::decode_team::*;

const POOL: u32 = 8;

fn cpus(n: usize) -> Vec<usize> {
    (0..n).collect()
}

fn policy(table: &str, override_team: Option<TeamSize>) -> TeamPolicy {
    TeamPolicy::new(
        POOL,
        cpus(POOL as usize),
        TeamTable::parse(table).expect("fixture table"),
        override_team,
    )
    .expect("fixture policy")
}

fn plan(table: &str, override_team: Option<TeamSize>, ctx: u64) -> TeamPlan {
    policy(table, override_team)
        .select(ctx, |_| true)
        .expect("fixture plan")
}

/// The default: an empty table (and no override) is off and selects the whole pool, exactly the
/// behaviour before this policy existed.
#[test]
fn empty_table_is_off_and_selects_the_whole_pool() {
    let p = policy("", None);
    assert!(!p.is_enabled());
    assert!(p.table().is_empty());
    for ctx in [0u64, 1, 4096, 1 << 20, u64::MAX] {
        let pl = p.select(ctx, |_| true).unwrap();
        assert_eq!(pl.team, POOL, "ctx {ctx}");
        assert_eq!(pl.cpus, cpus(POOL as usize), "ctx {ctx}");
        assert_eq!(pl.source, PlanSource::Default, "ctx {ctx}");
        assert_eq!(pl.bucket_min_context, None, "ctx {ctx}");
    }
    // blank is the same as empty: whitespace never turns a policy on
    assert!(!policy("   \t ", None).is_enabled());
}

/// Exact bucket edges: below, at and above every boundary, plus the open-ended last row.
#[test]
fn bucket_edges_are_below_at_and_above_each_boundary() {
    let table = "0=pool,4096=6,16384=4";
    let cases: [(u64, u32, PlanSource, Option<u64>); 10] = [
        (0, POOL, PlanSource::Table, Some(0)),
        (1, POOL, PlanSource::Table, Some(0)),
        (4095, POOL, PlanSource::Table, Some(0)),
        (4096, 6, PlanSource::Table, Some(4096)),
        (4097, 6, PlanSource::Table, Some(4096)),
        (16383, 6, PlanSource::Table, Some(4096)),
        (16384, 4, PlanSource::Table, Some(16384)),
        (16385, 4, PlanSource::Table, Some(16384)),
        (1 << 20, 4, PlanSource::Table, Some(16384)),
        (u64::MAX, 4, PlanSource::Table, Some(16384)),
    ];
    for (ctx, team, source, bucket) in cases {
        let pl = plan(table, None, ctx);
        assert_eq!(pl.team, team, "ctx {ctx}");
        assert_eq!(pl.cpus, cpus(team as usize), "ctx {ctx}");
        assert_eq!(pl.source, source, "ctx {ctx}");
        assert_eq!(pl.bucket_min_context, bucket, "ctx {ctx}");
    }
}

/// An explicit user override wins over the table at every context, and `pool` is a valid override
/// that turns a table off.
#[test]
fn explicit_override_wins_over_the_table() {
    let table = "0=pool,4096=6,16384=4";
    let p = policy(table, Some(TeamSize::Threads(3)));
    assert!(p.is_enabled());
    for ctx in [0u64, 4096, 1 << 20] {
        let pl = p.select(ctx, |_| true).unwrap();
        assert_eq!(pl.team, 3, "ctx {ctx}");
        assert_eq!(pl.cpus, cpus(3), "ctx {ctx}");
        assert_eq!(pl.source, PlanSource::Override, "ctx {ctx}");
        assert_eq!(pl.bucket_min_context, None, "ctx {ctx}");
    }
    // override = the whole pool: the table is ignored but the plan is still an override
    let pl = plan(table, Some(TeamSize::Pool), 1 << 20);
    assert_eq!(pl.team, POOL);
    assert_eq!(pl.cpus, cpus(POOL as usize));
    assert_eq!(pl.source, PlanSource::Override);
    assert_eq!(pl.bucket_min_context, None);
}

/// Unavailable CPUs: a team that cannot be staffed here is refused, not silently shrunk, and the
/// full-pool plan is refused only when the pool itself is not staffable.
#[test]
fn unavailable_cpus_refuse_the_team_instead_of_shrinking_it() {
    let table = "0=pool,4096=6,16384=4";
    let p = policy(table, None);
    // cpu 3 is not usable: the 4-thread plan needs 4 of the first 4 cpus, so it is refused
    let err = p.select(16384, |c| c != 3).unwrap_err();
    assert_eq!(
        err,
        PolicyError::CpusShort {
            team: 4,
            allowed: 3
        }
    );
    // the 6-thread plan needs the first 6, and cpu 5 is missing: refused with 5 allowed
    let err = p.select(4096, |c| c != 5).unwrap_err();
    assert_eq!(
        err,
        PolicyError::CpusShort {
            team: 6,
            allowed: 5
        }
    );
    // an unavailable cpu beyond the team does not matter
    let pl = p.select(16384, |c| c != 7).unwrap();
    assert_eq!(pl.team, 4);
    assert_eq!(pl.cpus, cpus(4));
    // the whole pool cannot be staffed either: same refusal, no fallback inside `select`
    let err = p.select(0, |c| c != 0).unwrap_err();
    assert_eq!(
        err,
        PolicyError::CpusShort {
            team: POOL,
            allowed: POOL - 1
        }
    );
}

/// Invalid and overlapping tables, and bad overrides, are refused where they are built: no
/// uncovered range, no duplicate edge, no zero or above-pool team, no mismatched CPU list.
#[test]
fn invalid_tables_and_overrides_are_refused() {
    // first row not at 0 leaves a range uncovered
    assert_eq!(
        TeamTable::parse("4096=6").unwrap_err(),
        PolicyError::FirstBucketNotZero(4096)
    );
    // duplicate and out-of-order edges
    assert_eq!(
        TeamTable::parse("0=8,0=4").unwrap_err(),
        PolicyError::UnsortedOrDuplicate(0)
    );
    assert_eq!(
        TeamTable::parse("0=8,4096=6,4096=4").unwrap_err(),
        PolicyError::UnsortedOrDuplicate(4096)
    );
    assert_eq!(
        TeamTable::parse("0=8,4096=6,2048=4").unwrap_err(),
        PolicyError::UnsortedOrDuplicate(2048)
    );
    // malformed rows: no '=', empty part, non-numeric, zero team, negative, extra '='
    for row in ["0", "0=8,", ",0=8", "x=8", "0=x", "0=0", "0=-1", "0=8=9"] {
        assert!(
            matches!(TeamTable::parse(row), Err(PolicyError::BadRow(_))),
            "row {row:?} must be refused"
        );
    }
    // a team above the pool would need new threads
    assert_eq!(
        TeamPolicy::new(
            POOL,
            cpus(POOL as usize),
            TeamTable::parse("0=9").unwrap(),
            None
        )
        .unwrap_err(),
        PolicyError::TeamAbovePool {
            team: 9,
            pool: POOL
        }
    );
    assert_eq!(
        TeamPolicy::new(
            POOL,
            cpus(POOL as usize),
            TeamTable::empty(),
            Some(TeamSize::Threads(9))
        )
        .unwrap_err(),
        PolicyError::TeamAbovePool {
            team: 9,
            pool: POOL
        }
    );
    // the CPU list is one entry per pool worker
    assert_eq!(
        TeamPolicy::new(4, cpus(3), TeamTable::empty(), None).unwrap_err(),
        PolicyError::CpuCountMismatch { cpus: 3, pool: 4 }
    );
    // a zero pool is not a pool
    assert!(TeamPolicy::new(0, Vec::new(), TeamTable::empty(), None).is_err());
}

/// The pool is never resized inside a job: a plan for another team waits for the next boundary,
/// and the same team (or no running job) applies immediately.
#[test]
fn a_different_team_waits_for_the_next_job_boundary() {
    let table = "0=pool,4096=4";
    let small = plan(table, None, 4096);
    let big = plan(table, None, 0);
    assert_eq!(admit(None, &small), Admission::Apply);
    assert_eq!(admit(Some(4), &small), Admission::Apply);
    assert_eq!(admit(Some(POOL), &small), Admission::HoldUntilJobBoundary);
    assert_eq!(admit(Some(POOL), &big), Admission::Apply);
    assert_eq!(admit(Some(4), &big), Admission::HoldUntilJobBoundary);
}
