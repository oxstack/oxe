//! OPT-IN context-length decode-team policy (issue #917): pure bucket selection only.
//!
//! The decode team is a fixed pool today: `OxeEngine` builds one [`crate::Pool`] from
//! `--oxe-cpus` and every decode step runs on all of it. Issue #917 asks for an *optional* rule
//! that picks a decode team from the active context length, with the default unchanged and no
//! invented crossover threshold. This module is the pure half of that design:
//!
//! - [`TeamTable`] is the configurable, owner-supplied mapping from active-context ranges to team
//!   sizes. The default is the empty table: the policy is off and [`TeamPolicy::select`] returns
//!   the whole pool, exactly today's behaviour.
//! - [`TeamPolicy::select`] is a pure function of (active context, allowed CPUs, table, explicit
//!   override). It never touches a pool, never spawns a thread and never blocks.
//! - [`admit`] encodes the "never resize a pool inside a job" rule as data.
//!
//! What this module deliberately does NOT do (owner seam, not implemented here): run a team
//! smaller than the pool through the existing [`crate::Barrier`], whose membership is fixed at
//! `Pool::new`. A team below the pool size needs the owner-approved scheduler seam; until that
//! seam exists nothing in the engine calls [`TeamPolicy`] and the decode team stays the pool.
//! The design, the identity gates and the deterministic fixture plan are in
//! `engine/docs/DECODE-TEAM-POLICY.md`.

use std::fmt;

/// A decode team size.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TeamSize {
    /// Every worker of the pool (the unchanged behaviour, and the only size the current pool can
    /// run without a new scheduler seam).
    Pool,
    /// Exactly `n` threads, `1 <= n <= pool threads`. Never more than the pool: the policy never
    /// creates threads per token.
    Threads(u32),
}

/// One table row: the team for active contexts `[min_context, next row)`; the last row is open
/// ended. Rows are ordered by `min_context` and the first row starts at 0, so the table is a total
/// function of the active context (no uncovered range, no overlap).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Bucket {
    pub min_context: u64,
    pub team: TeamSize,
}

/// A validated, ordered table of context buckets. Empty = the policy is off.
#[derive(Clone, Default, PartialEq, Eq, Debug)]
pub struct TeamTable {
    buckets: Vec<Bucket>,
}

/// A selected decode team: the size, the CPU mask and where the choice came from.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TeamPlan {
    pub team: u32,
    /// The first `team` CPUs of the pool, in pool order (worker `i` -> `cpus[i]`).
    pub cpus: Vec<usize>,
    pub source: PlanSource,
    /// `min_context` of the table row the active context fell in; `None` for the default plan and
    /// for an explicit override (the source says which).
    pub bucket_min_context: Option<u64>,
}

/// Which rule produced a [`TeamPlan`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PlanSource {
    /// No table, no override: the whole pool (today's behaviour).
    Default,
    /// A table row matched the active context.
    Table,
    /// The explicit user override won over the table.
    Override,
}

/// Whether a plan may take effect now.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Admission {
    Apply,
    /// A different team is running: the plan waits for the next job boundary. The pool is never
    /// resized inside a job, and a barrier's membership never changes while it is in use.
    HoldUntilJobBoundary,
}

/// A refused table, plan or override. Every variant is a configuration or availability fault; none
/// of them changes a default, a numerical flag or an admission limit.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum PolicyError {
    /// A row is not `MIN=TEAM` (or `TEAM` is not `pool` / a positive decimal).
    BadRow(String),
    /// Two rows share a `min_context`, or a row is out of order.
    UnsortedOrDuplicate(u64),
    /// The first row does not start at 0: the table would leave a range uncovered.
    FirstBucketNotZero(u64),
    /// A team larger than the pool (would need new threads) or a zero team.
    TeamAbovePool { team: u32, pool: u32 },
    /// The CPU list is not one entry per pool worker.
    CpuCountMismatch { cpus: usize, pool: u32 },
    /// Fewer than `team` of the plan's CPUs are available here.
    CpusShort { team: u32, allowed: u32 },
}

impl fmt::Display for PolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PolicyError::BadRow(r) => write!(f, "decode-team row {r:?}: want MIN=TEAM"),
            PolicyError::UnsortedOrDuplicate(m) => {
                write!(f, "decode-team rows must ascend: {m} repeats or precedes")
            }
            PolicyError::FirstBucketNotZero(m) => {
                write!(f, "decode-team table must start at 0 (got {m})")
            }
            PolicyError::TeamAbovePool { team, pool } => {
                write!(f, "decode-team {team} > pool {pool}")
            }
            PolicyError::CpuCountMismatch { cpus, pool } => {
                write!(f, "decode-team: {cpus} cpus for {pool} pool workers")
            }
            PolicyError::CpusShort { team, allowed } => {
                write!(
                    f,
                    "decode-team {team} needs {team} available cpus, {allowed} are here"
                )
            }
        }
    }
}

impl std::error::Error for PolicyError {}

impl TeamSize {
    /// `n` for `Threads(n)`, the pool size for `Pool`.
    pub fn resolve(self, pool_threads: u32) -> u32 {
        match self {
            TeamSize::Pool => pool_threads,
            TeamSize::Threads(n) => n,
        }
    }
}

impl TeamTable {
    pub fn empty() -> TeamTable {
        TeamTable {
            buckets: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.buckets.is_empty()
    }

    pub fn buckets(&self) -> &[Bucket] {
        &self.buckets
    }

    /// Parse `"MIN=TEAM,MIN=TEAM,..."`, where `TEAM` is `pool` or a positive decimal. An empty or
    /// blank string is the empty table (the policy off). The grammar is deliberately tiny: no
    /// ranges, no arithmetic, nothing that could invent a threshold the owner did not write.
    pub fn parse(s: &str) -> Result<TeamTable, PolicyError> {
        let s = s.trim();
        if s.is_empty() {
            return Ok(TeamTable::empty());
        }
        let mut buckets: Vec<Bucket> = Vec::new();
        for row in s.split(',') {
            let row = row.trim();
            let (min, team) = row
                .split_once('=')
                .ok_or_else(|| PolicyError::BadRow(row.to_string()))?;
            let min_context: u64 = min
                .trim()
                .parse()
                .map_err(|_| PolicyError::BadRow(row.to_string()))?;
            let team = match team.trim() {
                "pool" => TeamSize::Pool,
                t => match t.parse::<u32>() {
                    Ok(0) | Err(_) => return Err(PolicyError::BadRow(row.to_string())),
                    Ok(n) => TeamSize::Threads(n),
                },
            };
            match buckets.last() {
                Some(last) if min_context <= last.min_context => {
                    return Err(PolicyError::UnsortedOrDuplicate(min_context))
                }
                None if min_context != 0 => {
                    return Err(PolicyError::FirstBucketNotZero(min_context))
                }
                _ => {}
            }
            buckets.push(Bucket { min_context, team });
        }
        Ok(TeamTable { buckets })
    }

    /// The row whose `[min_context, next row)` range contains `active_context`, if the table is
    /// not empty. The table starts at 0, so a non-empty table always matches.
    pub fn team_for(&self, active_context: u64) -> Option<Bucket> {
        self.buckets
            .iter()
            .rev()
            .find(|b| b.min_context <= active_context)
            .copied()
    }
}

/// The pure context-length-to-team rule. Constructing it validates the table and the override; it
/// holds no pool, so it can be built and exercised without any thread or CPU.
#[derive(Clone, Debug)]
pub struct TeamPolicy {
    pool_threads: u32,
    cpus: Vec<usize>,
    table: TeamTable,
    override_team: Option<TeamSize>,
}

impl TeamPolicy {
    /// `cpus` is the pool's worker->CPU list (`Pool::cpus`), one entry per worker.
    pub fn new(
        pool_threads: u32,
        cpus: Vec<usize>,
        table: TeamTable,
        override_team: Option<TeamSize>,
    ) -> Result<TeamPolicy, PolicyError> {
        if pool_threads == 0 {
            return Err(PolicyError::TeamAbovePool { team: 0, pool: 0 });
        }
        if cpus.len() != pool_threads as usize {
            return Err(PolicyError::CpuCountMismatch {
                cpus: cpus.len(),
                pool: pool_threads,
            });
        }
        for b in table.buckets() {
            if let TeamSize::Threads(n) = b.team {
                if n == 0 || n > pool_threads {
                    return Err(PolicyError::TeamAbovePool {
                        team: n,
                        pool: pool_threads,
                    });
                }
            }
        }
        if let Some(TeamSize::Threads(n)) = override_team {
            if n == 0 || n > pool_threads {
                return Err(PolicyError::TeamAbovePool {
                    team: n,
                    pool: pool_threads,
                });
            }
        }
        Ok(TeamPolicy {
            pool_threads,
            cpus,
            table,
            override_team,
        })
    }

    /// The empty table and no override: the policy does nothing.
    pub fn is_enabled(&self) -> bool {
        self.override_team.is_some() || !self.table.is_empty()
    }

    pub fn table(&self) -> &TeamTable {
        &self.table
    }

    pub fn override_team(&self) -> Option<TeamSize> {
        self.override_team
    }

    /// The team for `active_context` (positions the job will have written after this step),
    /// restricted to the CPUs `allowed` reports as usable here. Pure: same inputs, same plan.
    ///
    /// Precedence: explicit override, then the table, then the whole pool. A plan that cannot be
    /// staffed on this host (`CpusShort`) is refused rather than silently shrunk: the caller's
    /// documented fallback is the unchanged default (the whole pool), and the refusal is what a
    /// later step reports.
    pub fn select(
        &self,
        active_context: u64,
        allowed: impl Fn(usize) -> bool,
    ) -> Result<TeamPlan, PolicyError> {
        let (team, source, bucket_min_context) = match self.override_team {
            Some(TeamSize::Pool) => (self.pool_threads, PlanSource::Override, None),
            Some(TeamSize::Threads(n)) => (n, PlanSource::Override, None),
            None => match self.table.team_for(active_context) {
                Some(b) => (
                    b.team.resolve(self.pool_threads),
                    PlanSource::Table,
                    Some(b.min_context),
                ),
                None => (self.pool_threads, PlanSource::Default, None),
            },
        };
        let cpus: Vec<usize> = self.cpus[..team as usize].to_vec();
        let allowed_n = cpus.iter().filter(|&&c| allowed(c)).count() as u32;
        if allowed_n < team {
            return Err(PolicyError::CpusShort {
                team,
                allowed: allowed_n,
            });
        }
        Ok(TeamPlan {
            team,
            cpus,
            source,
            bucket_min_context,
        })
    }
}

/// Whether `plan` may take effect while a job runs on `active_team` (`None` = no job running).
/// The pool is never resized inside a job, so only a plan for the running team applies; any other
/// plan waits for the next job boundary.
pub fn admit(active_team: Option<u32>, plan: &TeamPlan) -> Admission {
    match active_team {
        Some(t) if t != plan.team => Admission::HoldUntilJobBoundary,
        _ => Admission::Apply,
    }
}
