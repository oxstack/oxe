//! Token sampling. Greedy (temperature <= 0) is exact argmax with lowest-id tie break, which
//! is what the llama.cpp comparison runs use. The stochastic path follows llama.cpp's default
//! chain order (top_k -> top_p -> min_p -> temperature -> dist) but is not bit-compatible with
//! llama.cpp's RNG.

#[derive(Clone, Debug)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub seed: u64,
    /// OpenAI `presence_penalty` (-2..=2): subtracted once from the logit of every token already
    /// generated in this reply. 0 = off.
    pub presence_penalty: f32,
    /// OpenAI `frequency_penalty` (-2..=2): subtracted per occurrence in this reply. 0 = off.
    pub frequency_penalty: f32,
    /// OpenAI `logit_bias`: (token id, bias) added to the logit before sampling; sorted by id.
    pub logit_bias: Vec<(u32, f32)>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        // llama.cpp common_params_sampling defaults
        SamplingParams { temperature: 0.8, top_k: 40, top_p: 0.95, min_p: 0.05, seed: 0xFFFF_FFFF, presence_penalty: 0.0, frequency_penalty: 0.0, logit_bias: Vec::new() }
    }
}

impl SamplingParams {
    /// Penalties or a logit bias change the logits the sampler sees. Without them the sampler reads
    /// the engine's row as it is (bit-identical output, the greedy fast path stays on).
    pub fn adjusts_logits(&self) -> bool {
        self.presence_penalty != 0.0 || self.frequency_penalty != 0.0 || !self.logit_bias.is_empty()
    }
}

pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best as u32
}

/// Softmax over all logits, top `n` (id, p) by probability (llama-server `get_token_probabilities`).
pub fn top_probs(logits: &[f32], n: usize) -> Vec<(u32, f32)> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f64;
    for &l in logits {
        sum += ((l - max) as f64).exp();
    }
    let n = n.min(logits.len());
    // partial selection of the n largest
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    if n < idx.len() && n > 0 {
        idx.select_nth_unstable_by(n - 1, |&a, &b| logits[b as usize].total_cmp(&logits[a as usize]).then(a.cmp(&b)));
        idx.truncate(n);
    }
    idx.sort_by(|&a, &b| logits[b as usize].total_cmp(&logits[a as usize]).then(a.cmp(&b)));
    idx.truncate(n);
    idx.into_iter().map(|i| (i, (((logits[i as usize] - max) as f64).exp() / sum) as f32)).collect()
}

/// Probability of one token under the full softmax.
pub fn token_prob(logits: &[f32], tok: u32) -> f32 {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = logits.iter().map(|&l| ((l - max) as f64).exp()).sum();
    (((logits[tok as usize] - max) as f64).exp() / sum) as f32
}

pub struct Sampler {
    pub p: SamplingParams,
    rng: u64,
    /// occurrences of each generated token (penalties only)
    counts: std::collections::HashMap<u32, u32>,
    adjusted: Vec<f32>,
}

impl Sampler {
    pub fn new(p: SamplingParams) -> Sampler {
        let seed = if p.seed == 0xFFFF_FFFF {
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1)
        } else {
            p.seed
        };
        Sampler { p, rng: seed ^ 0x9E37_79B9_7F4A_7C15, counts: Default::default(), adjusted: Vec::new() }
    }

    fn next_f64(&mut self) -> f64 {
        // splitmix64
        self.rng = self.rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Record a generated token (the penalties count them). Cheap no-op without penalties.
    pub fn accept(&mut self, t: u32) {
        if self.p.presence_penalty != 0.0 || self.p.frequency_penalty != 0.0 {
            *self.counts.entry(t).or_insert(0) += 1;
        }
    }

    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        if !self.p.adjusts_logits() {
            return self.sample_row(logits);
        }
        let mut row = std::mem::take(&mut self.adjusted);
        row.clear();
        row.extend_from_slice(logits);
        for (&t, &n) in &self.counts {
            if let Some(l) = row.get_mut(t as usize) {
                *l -= self.p.presence_penalty + self.p.frequency_penalty * n as f32;
            }
        }
        for &(t, b) in &self.p.logit_bias {
            if let Some(l) = row.get_mut(t as usize) {
                *l += b;
            }
        }
        let t = self.sample_row(&row);
        self.adjusted = row;
        t
    }

    /// The row the last [`Sampler::sample`] chose from: the adjusted copy when penalties or a bias
    /// apply, else `logits` itself. Log-probabilities are reported from this row (the distribution
    /// the token was actually drawn from).
    pub fn effective_row<'a>(&'a self, logits: &'a [f32]) -> &'a [f32] {
        if self.p.adjusts_logits() && self.adjusted.len() == logits.len() {
            &self.adjusted
        } else {
            logits
        }
    }

    fn sample_row(&mut self, logits: &[f32]) -> u32 {
        if self.p.temperature <= 0.0 {
            return argmax(logits);
        }
        let k = if self.p.top_k <= 0 { logits.len() } else { (self.p.top_k as usize).min(logits.len()) };
        let mut cand = top_probs(logits, k); // sorted, softmax over all (renormalized below)
        if cand.is_empty() {
            return argmax(logits);
        }
        let s: f32 = cand.iter().map(|c| c.1).sum();
        for c in cand.iter_mut() {
            c.1 /= s;
        }
        if self.p.top_p < 1.0 {
            let mut acc = 0.0;
            let mut keep = cand.len();
            for (i, c) in cand.iter().enumerate() {
                acc += c.1;
                if acc >= self.p.top_p {
                    keep = i + 1;
                    break;
                }
            }
            cand.truncate(keep.max(1));
        }
        if self.p.min_p > 0.0 {
            let thr = cand[0].1 * self.p.min_p;
            cand.retain(|c| c.1 >= thr);
        }
        // total: the request checks keep min_p <= 1, but NaN probabilities or a direct caller can
        // still empty the list (REVIEW-v0.1.1 B2: this was an unwrap panic = abort in release)
        if cand.is_empty() {
            return argmax(logits);
        }
        // temperature on the remaining logits
        let lg: Vec<f64> = cand.iter().map(|c| (logits[c.0 as usize] / self.p.temperature) as f64).collect();
        let m = lg.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let w: Vec<f64> = lg.iter().map(|l| (l - m).exp()).collect();
        let tot: f64 = w.iter().sum();
        if !(tot.is_finite() && tot > 0.0) {
            return cand[0].0; // degenerate weights (temperature ~1e-38 gives NaN): most probable
        }
        let mut r = self.next_f64() * tot;
        for (i, x) in w.iter().enumerate() {
            r -= x;
            if r <= 0.0 {
                return cand[i].0;
            }
        }
        cand[cand.len() - 1].0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn greedy_and_probs() {
        let l = [0.0, 3.0, 1.0, 3.0];
        assert_eq!(argmax(&l), 1);
        let p = top_probs(&l, 2);
        assert_eq!(p[0].0, 1);
        assert_eq!(p[1].0, 3);
        assert!((p[0].1 - p[1].1).abs() < 1e-6);
        let mut s = Sampler::new(SamplingParams { temperature: 0.0, ..Default::default() });
        assert_eq!(s.sample(&l), 1);
        let mut s = Sampler::new(SamplingParams { seed: 7, ..Default::default() });
        let t = s.sample(&l);
        assert!(t == 1 || t == 3 || t == 2);
    }

    /// REVIEW-v0.1.1 B2: min_p > 1 removed every candidate and `cand.last().unwrap()` panicked
    /// (abort in release). The sampler is total now: an empty list gives the argmax.
    #[test]
    fn min_p_above_one_falls_back_to_argmax() {
        let mut s = Sampler::new(SamplingParams { temperature: 1.0, min_p: 2.0, seed: 1, ..Default::default() });
        assert_eq!(s.sample(&[0.0, 1.0]), 1);
        let mut s = Sampler::new(SamplingParams { temperature: 1.0, min_p: f32::INFINITY, top_k: 0, top_p: 1.0, seed: 1, ..Default::default() });
        assert_eq!(s.sample(&[0.5, 2.0, 1.0]), 1);
    }

    /// NaN logits or a vanishing temperature (REVIEW N7) never panic and pick a candidate.
    #[test]
    fn degenerate_inputs_do_not_panic() {
        let mut s = Sampler::new(SamplingParams { temperature: 1e-38, seed: 3, ..Default::default() });
        assert_eq!(s.sample(&[0.0, 5.0, 1.0]), 1);
        let mut s = Sampler::new(SamplingParams { temperature: 1.0, seed: 3, ..Default::default() });
        let t = s.sample(&[f32::NAN, f32::NAN, f32::NAN]);
        assert!(t < 3);
    }

    /// OpenAI penalties and logit_bias: absent = the row is read as it is (identical picks for a
    /// fixed seed); a bias steers greedy; penalties count this reply's tokens.
    #[test]
    fn penalties_and_logit_bias() {
        let l = [0.0f32, 3.0, 2.9, 1.0];
        let base = SamplingParams { seed: 11, ..Default::default() };
        assert!(!base.adjusts_logits());
        let (mut a, mut b) = (Sampler::new(base.clone()), Sampler::new(base.clone()));
        for _ in 0..50 {
            let t = a.sample(&l);
            b.accept(t);
            assert_eq!(t, b.sample(&l), "no adjustment: same draws");
        }
        let g = SamplingParams { temperature: 0.0, logit_bias: vec![(3, 5.0)], ..Default::default() };
        assert_eq!(Sampler::new(g).sample(&l), 3, "bias +5 makes token 3 the argmax");
        let g = SamplingParams { temperature: 0.0, logit_bias: vec![(1, -100.0)], ..Default::default() };
        assert_eq!(Sampler::new(g).sample(&l), 2);
        let g = SamplingParams { temperature: 0.0, presence_penalty: 0.5, ..Default::default() };
        let mut s = Sampler::new(g);
        assert_eq!(s.sample(&l), 1);
        s.accept(1);
        assert_eq!(s.sample(&l), 2, "1 was generated: 3.0 - 0.5 < 2.9");
        let g = SamplingParams { temperature: 0.0, frequency_penalty: 0.06, ..Default::default() };
        let mut s = Sampler::new(g);
        s.accept(1);
        assert_eq!(s.sample(&l), 1, "one occurrence: 3.0 - 0.06 > 2.9");
        s.accept(1);
        assert_eq!(s.sample(&l), 2, "two: 3.0 - 0.12 < 2.9");
    }

    /// Codex HOLD (1): the reported distribution is the one the token was drawn from.
    #[test]
    fn effective_row_is_the_adjusted_row() {
        let l = [0.0f32, 2.0];
        let mut s = Sampler::new(SamplingParams { temperature: 0.0, logit_bias: vec![(0, 10.0)], ..Default::default() });
        assert_eq!(s.sample(&l), 0);
        let p = token_prob(s.effective_row(&l), 0);
        assert!((p - 0.99966).abs() < 1e-4, "{p}");
        assert_eq!(top_probs(s.effective_row(&l), 1)[0].0, 0);
        let mut s = Sampler::new(SamplingParams { temperature: 0.0, ..Default::default() });
        s.sample(&l);
        assert!(std::ptr::eq(s.effective_row(&l).as_ptr(), l.as_ptr()), "no adjustment: the engine row itself");
    }
}
