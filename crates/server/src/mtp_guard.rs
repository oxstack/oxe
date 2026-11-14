//! MTP acceptance guard (v0.2.0): a pure policy that turns speculative drafting off while the
//! draft acceptance is below a ratio, and probes again after a cooldown. It holds no engine
//! state; the decode loop calls [`MtpGuard::allow`] before each verify and [`MtpGuard::record`]
//! after it. Output is identical at any draft depth (verification is exact), so the guard changes
//! speed only. Default off: see [`from_env`].
//!
//! Acceptance is taken over a sliding window of the last `window` drafted tokens, one flag per
//! drafted token (accepted tokens are a prefix of each verify, so a verify with `a` of `d` accepted
//! records `a` hits and `d - a` misses).

use std::collections::VecDeque;

/// Drafted tokens in the window when none is given.
pub const DEFAULT_WINDOW: usize = 256;
/// Decoded tokens without speculation after a trip when none is given.
pub const DEFAULT_COOLDOWN: usize = 512;

#[derive(Debug, Clone)]
pub struct MtpGuard {
    min_accept: f32,
    window: usize,
    cooldown: usize,
    /// one flag per drafted token in the window: accepted or not (oldest first)
    flags: VecDeque<bool>,
    hits: usize,
    /// calls of `allow` still answered false
    cooldown_left: usize,
    trips: u64,
}

impl MtpGuard {
    /// A guard that trips when the acceptance over the last `window` drafted tokens is below
    /// `min_accept`, then refuses speculation for `cooldown` calls of [`MtpGuard::allow`].
    /// A window of 0 is read as 1.
    pub fn new(min_accept: f32, window: usize, cooldown: usize) -> MtpGuard {
        MtpGuard { min_accept, window: window.max(1), cooldown, flags: VecDeque::new(), hits: 0, cooldown_left: 0, trips: 0 }
    }

    /// `OXE_MTP_GUARD` (read at each call; the startup check has already refused a malformed
    /// value). Unset, `off` or `0` = None (no guard). A malformed value also gives None, with a
    /// warning on stderr.
    pub fn from_env() -> Option<MtpGuard> {
        let v = std::env::var("OXE_MTP_GUARD").ok()?;
        match parse(&v) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("oxe: {e}: the MTP guard stays off");
                None
            }
        }
    }

    /// Parse `off` | `0` | `RATIO` | `RATIO:WINDOW:COOLDOWN` (RATIO in (0, 1], WINDOW >= 1, COOLDOWN
    /// >= 0; defaults 256 and 512). An empty value is read as unset. Errors name the reason only.
    pub fn parse(s: &str) -> Result<Option<MtpGuard>, String> {
        parse(s)
    }

    /// Before each new verify: false = decode this token without speculation (this call counts
    /// the cooldown down; after `cooldown` false answers the window is probed again).
    pub fn allow(&mut self) -> bool {
        if self.cooldown_left == 0 {
            return true;
        }
        self.cooldown_left -= 1;
        false
    }

    /// After a verify is fully consumed or dropped: how many tokens were drafted and how many of
    /// them were accepted (values above `drafted` are clamped). Trips when the window is full and
    /// its acceptance is below the minimum. The window is cleared on a trip (calls of `record`
    /// during a cooldown do not happen in the decode loop).
    pub fn record(&mut self, drafted: usize, accepted: usize) {
        let accepted = accepted.min(drafted);
        for i in 0..drafted {
            let hit = i < accepted;
            self.flags.push_back(hit);
            self.hits += hit as usize;
            if self.flags.len() > self.window && self.flags.pop_front() == Some(true) {
                self.hits -= 1;
            }
        }
        if self.flags.len() == self.window && (self.hits as f32) < self.min_accept * self.window as f32 {
            self.trips += 1;
            self.cooldown_left = self.cooldown;
            self.flags.clear();
            self.hits = 0;
        }
    }

    /// How many times the guard has tripped.
    pub fn trips(&self) -> u64 {
        self.trips
    }
}

/// See [`MtpGuard::parse`].
pub fn parse(s: &str) -> Result<Option<MtpGuard>, String> {
    let t = s.trim();
    if t.is_empty() || t == "off" || t == "0" {
        return Ok(None);
    }
    let parts: Vec<&str> = t.split(':').collect();
    if parts.len() > 3 {
        return Err("expected RATIO or RATIO:WINDOW:COOLDOWN".into());
    }
    let ratio: f32 = parts[0].trim().parse().map_err(|_| "the ratio is not a number (off | RATIO | RATIO:WINDOW:COOLDOWN)".to_string())?;
    if !(ratio.is_finite() && ratio > 0.0 && ratio <= 1.0) {
        return Err(format!("the ratio {} is outside (0, 1]", parts[0].trim()));
    }
    let window = match parts.get(1) {
        None => DEFAULT_WINDOW,
        Some(w) => match w.trim().parse::<usize>() {
            Ok(n) if n >= 1 => n,
            _ => return Err(format!("the window {:?} is not an integer >= 1", w.trim())),
        },
    };
    let cooldown = match parts.get(2) {
        None => DEFAULT_COOLDOWN,
        Some(c) => c.trim().parse::<usize>().map_err(|_| format!("the cooldown {:?} is not a non-negative integer", c.trim()))?,
    };
    Ok(Some(MtpGuard::new(ratio, window, cooldown)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn never_trips_at_or_above_the_ratio() {
        let mut g = MtpGuard::new(0.5, 8, 4);
        for _ in 0..100 {
            // exactly 50 % over every window: not below the ratio
            g.record(4, 2);
            assert!(g.allow());
        }
        assert_eq!(g.trips(), 0);
        // all accepted
        for _ in 0..10 {
            g.record(3, 3);
        }
        assert_eq!(g.trips(), 0);
    }

    #[test]
    fn trips_exactly_when_the_full_window_falls_below() {
        let mut g = MtpGuard::new(0.5, 8, 4);
        // 7 drafted, all rejected: the window is not full yet
        g.record(7, 0);
        assert_eq!(g.trips(), 0);
        // the 8th drafted token (rejected): full window, 0 of 8 < 4: trip
        g.record(1, 0);
        assert_eq!(g.trips(), 1);
        // a full window at exactly the ratio does not trip
        let mut h = MtpGuard::new(0.5, 8, 4);
        h.record(8, 4);
        assert_eq!(h.trips(), 0);
        // one hit fewer than the ratio does
        let mut k = MtpGuard::new(0.5, 8, 4);
        k.record(8, 3);
        assert_eq!(k.trips(), 1);
    }

    #[test]
    fn window_slides_over_drafted_tokens() {
        let mut g = MtpGuard::new(0.5, 4, 0);
        g.record(4, 4); // full, 4/4
        g.record(2, 0); // window now 2 hits + 2 misses = 2/4: below 0.5? 2 < 2 is false: no trip
        assert_eq!(g.trips(), 0);
        g.record(1, 0); // 1 hit + 3 misses = 1/4 < 2: trip
        assert_eq!(g.trips(), 1);
    }

    #[test]
    fn cooldown_is_exact_then_the_window_is_probed_again() {
        let mut g = MtpGuard::new(0.5, 4, 3);
        g.record(4, 0);
        assert_eq!(g.trips(), 1);
        // exactly `cooldown` refusals
        assert!(!g.allow());
        assert!(!g.allow());
        assert!(!g.allow());
        assert!(g.allow());
        assert!(g.allow());
        // the window was cleared: a new bad window is needed to trip again
        g.record(3, 0);
        assert_eq!(g.trips(), 1, "window not full yet");
        g.record(1, 0);
        assert_eq!(g.trips(), 2);
    }

    #[test]
    fn zero_cooldown_never_refuses() {
        let mut g = MtpGuard::new(1.0, 2, 0);
        g.record(2, 0);
        assert_eq!(g.trips(), 1);
        assert!(g.allow());
    }

    #[test]
    fn accepted_above_drafted_is_clamped() {
        let mut g = MtpGuard::new(0.5, 4, 0);
        g.record(2, 9);
        g.record(2, 2);
        assert_eq!(g.trips(), 0);
    }

    #[test]
    fn parse_forms() {
        for off in ["", "off", "0", "  off ", " 0 "] {
            assert!(parse(off).unwrap().is_none(), "{off:?}");
        }
        let g = parse("0.45").unwrap().unwrap();
        assert_eq!((g.window, g.cooldown), (DEFAULT_WINDOW, DEFAULT_COOLDOWN));
        let g = parse("1").unwrap().unwrap();
        assert!((g.min_accept - 1.0).abs() < 1e-6);
        let g = parse(" 0.5:128:64 ").unwrap().unwrap();
        assert_eq!((g.window, g.cooldown), (128, 64));
        let g = parse("0.5:16:0").unwrap().unwrap();
        assert_eq!(g.cooldown, 0);
    }

    #[test]
    fn parse_errors() {
        for bad in ["-0.1", "1.5", "nan", "inf", "x", "0.5:0:1", "0.5:x:1", "0.5:8:-1", "0.5:8:1:9", "0.5:8:"] {
            assert!(parse(bad).is_err(), "{bad:?} must be refused");
        }
    }
}
