//! Linux host and inherited no-swap cgroup sampling. Never opens models or credentials.
use crate::lifecycle::MemorySample;
use std::{
    fs, io,
    path::{Component, PathBuf},
};

#[derive(Clone)]
pub struct MemorySource {
    proc_root: PathBuf,
    cgroup_root: PathBuf,
}
impl Default for MemorySource {
    fn default() -> Self {
        Self {
            proc_root: "/proc".into(),
            cgroup_root: "/sys/fs/cgroup".into(),
        }
    }
}
impl MemorySource {
    #[cfg(test)]
    pub(crate) fn test_roots(proc_root: PathBuf, cgroup_root: PathBuf) -> Self {
        Self {
            proc_root,
            cgroup_root,
        }
    }
    pub fn sample(&self) -> io::Result<MemorySample> {
        self.sample_process("self")
    }
    pub fn sample_child(&self, pid: u32) -> io::Result<MemorySample> {
        self.sample_process(&pid.to_string())
    }
    pub(crate) fn child_in_scope(&self, pid: u32, unit: &str) -> io::Result<bool> {
        let membership = fs::read_to_string(self.proc_root.join(pid.to_string()).join("cgroup"))?;
        Ok(membership
            .lines()
            .any(|line| line.starts_with("0::/") && line.ends_with(&format!("/{unit}"))))
    }
    fn sample_process(&self, process: &str) -> io::Result<MemorySample> {
        let membership = fs::read_to_string(self.proc_root.join(process).join("cgroup"))?;
        let relative = membership
            .lines()
            .find_map(|line| line.strip_prefix("0::"))
            .ok_or_else(invalid)?;
        let relative = relative.strip_prefix('/').ok_or_else(invalid)?;
        if PathBuf::from(relative)
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
        {
            return Err(invalid());
        }
        let group = self.cgroup_root.join(relative);
        let meminfo = fs::read_to_string(self.proc_root.join("meminfo"))?;
        let pressure = fs::read_to_string(self.proc_root.join("pressure/memory"))?;
        let full = pressure
            .lines()
            .find(|l| l.starts_with("full "))
            .and_then(|line| {
                line.split_whitespace()
                    .find_map(|field| field.strip_prefix("avg10="))
            })
            .ok_or_else(invalid)?;
        let psi_full_avg10: f64 = full.parse().map_err(|_| invalid())?;
        if !psi_full_avg10.is_finite() || !(0.0..=100.0).contains(&psi_full_avg10) {
            return Err(invalid());
        }
        let events = fs::read_to_string(group.join("memory.events"))?;
        let oom_events = number(&events, "oom")?
            .checked_add(number(&events, "oom_kill")?)
            .and_then(|v| v.checked_add(number(&events, "oom_group_kill").ok()?))
            .ok_or_else(invalid)?;
        Ok(MemorySample {
            available_bytes: kib(&meminfo, "MemAvailable:")?,
            swap_free_bytes: kib(&meminfo, "SwapFree:")?,
            psi_full_avg10,
            swap_current_bytes: scalar(&fs::read_to_string(group.join("memory.swap.current"))?)?,
            swap_limit_zero: fs::read_to_string(group.join("memory.swap.max"))?.trim() == "0",
            oom_events,
            host_oom_kills: number(&fs::read_to_string(self.proc_root.join("vmstat"))?, "oom_kill")?,
            pids_max_events: number(&fs::read_to_string(group.join("pids.events"))?, "max")?,
        })
    }
}
fn invalid() -> io::Error {
    io::Error::other("local memory metadata is missing or invalid; use a no-swap cgroup")
}
fn scalar(text: &str) -> io::Result<u64> {
    let text = text.trim();
    if text.is_empty() || !text.bytes().all(|b| b.is_ascii_digit()) {
        return Err(invalid());
    }
    text.parse().map_err(|_| invalid())
}
fn number(text: &str, key: &str) -> io::Result<u64> {
    let mut rows = text.lines().filter_map(|line| {
        let mut fields = line.split_whitespace();
        (fields.next()? == key).then_some(fields)
    });
    let mut row = rows.next().ok_or_else(invalid)?;
    let value = scalar(row.next().ok_or_else(invalid)?)?;
    if row.next().is_some() || rows.next().is_some() {
        return Err(invalid());
    }
    Ok(value)
}
fn kib(text: &str, key: &str) -> io::Result<u64> {
    let mut rows = text.lines().filter_map(|line| {
        let mut fields = line.split_whitespace();
        (fields.next()? == key).then_some(fields)
    });
    let mut row = rows.next().ok_or_else(invalid)?;
    let value = scalar(row.next().ok_or_else(invalid)?)?;
    if row.next() != Some("kB") || row.next().is_some() || rows.next().is_some() {
        return Err(invalid());
    }
    value.checked_mul(1024).ok_or_else(invalid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxa_fake::TestEnv;
    #[test]
    fn kernel_samples_are_complete_and_fail_closed_on_malformed_or_missing_data() {
        let env = TestEnv::new();
        env.write("work/proc/self/cgroup", "0::/engine\n");
        env.write(
            "work/proc/meminfo",
            "MemAvailable: 8000000 kB\nSwapFree: 100 kB\n",
        );
        env.write("work/proc/pressure/memory", "some avg10=1.0\nfull avg10=21.0\n");
        env.write("work/proc/vmstat", "oom_kill 7\n");
        env.write(
            "work/cgroup/engine/memory.events",
            "oom 1\noom_kill 2\noom_group_kill 3\n",
        );
        env.write("work/cgroup/engine/memory.swap.current", "0\n");
        env.write("work/cgroup/engine/memory.swap.max", "0\n");
        env.write("work/cgroup/engine/pids.events", "max 4\n");
        let source = MemorySource {
            proc_root: env.work().join("proc"),
            cgroup_root: env.work().join("cgroup"),
        };
        let sample = source.sample().unwrap();
        assert_eq!(sample.available_bytes, 8000000 * 1024);
        assert_eq!(sample.oom_events, 6);
        assert!(sample.swap_limit_zero);
        assert_eq!(sample.host_oom_kills, 7);
        assert_eq!(sample.pids_max_events, 4);
        env.write("work/proc/pressure/memory", "full avg10=NaN\n");
        assert!(source.sample().is_err());
        env.write("work/proc/pressure/memory", "full avg10=0\n");
        env.write("work/proc/self/cgroup", "0::/../outside\n");
        assert!(source.sample().is_err());
        env.write("work/proc/self/cgroup", "0::/missing\n");
        assert!(source.sample().is_err());
    }
}
