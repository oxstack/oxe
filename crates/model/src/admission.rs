//! Admission report: the memory bounds oxa's native local admission needs, computed from the GGUF
//! header alone (no weights are mapped or read, no device session is opened).
//!
//! Every number comes from the allocation code the engine itself uses: [`ctx_mem`] (the
//! engine's context-dependent buffers, held equal to the real allocation by the long-context
//! tests), [`lane_shape`] (the recurrent slots and lanes of an engine, shared with
//! `Engine::with_options`) and [`Plan::budget`] (bytes per token). Nothing here measures RSS.
//!
//! Split of the engine's context-dependent bytes `T = ctx_mem(...).total(decode, prefill)` of the
//! launch as oxe-server builds it (oxa adds the weights file itself):
//!
//! * `kv_bytes_per_position`: K+V bytes per position of every attention layer, the MTP
//!   layer included once (`CtxMem::kv_per_pos`). oxa multiplies it by the 256-padded context.
//! * `scratch_bytes`: everything else of the engine WITHOUT MTP: the fixed V^T row pad, the
//!   recurrent state of one slot, decode scratch, per-decode-thread rows and the retained prefill
//!   teams.
//! * `mtp_state_bytes`: what MTP adds beyond its KV share: extra recurrent slots, extra lanes and
//!   the pad of its attention layer, plus the size of a separate sidecar head file when one is
//!   used (it is mapped in addition to the model file).
//! * `margin_bytes`: the documented margin rule, see [`MARGIN_RULE`].
//!
//! So `kv_bytes_per_position * capacity + scratch_bytes + mtp_state_bytes` equals `T` (plus the
//! sidecar head file size), exactly.

use crate::desc::ModelDesc;
use crate::engine::{ctx_mem, lane_shape, validate_n_ctx, CtxMem, KvType};
use crate::plan::{Budget, Plan};
use gguf::Header;

/// Schema tag of the machine-readable report.
pub const SCHEMA: &str = "oxe-admission/1";

/// Default of `margin_bytes`: 1 GiB of host headroom. The engine terms above are exact, so the
/// margin only covers what the report does not model: the HTTP server and tokenizer, the
/// optional prefix/state RAM caches, the per-lane working rows carved in the state arena and the
/// operating system. It is a documented default, not a measurement; the owner overrides it with
/// `--margin-bytes`.
pub const DEFAULT_MARGIN_BYTES: u64 = 1 << 30;

/// The margin rule, stated in every report.
pub const MARGIN_RULE: &str = "margin_bytes = --margin-bytes if given, else 1073741824 (1 GiB): host headroom for what the report does not model (HTTP server, tokenizer, optional prefix/state RAM caches, per-lane working rows, the operating system). The engine terms are exact; the margin is a documented default the owner may override.";

/// A sidecar MTP head GGUF (`--mtp-head`, or `<model>.mtp-head.gguf`).
pub struct HeadSource<'a> {
    pub header: &'a Header,
    pub file_bytes: u64,
}

/// The launch profile the budget is for.
pub struct Request<'a> {
    /// `--ctx-size`
    pub n_ctx: usize,
    /// `--cache-type-k/-v` mapped to the engine's KV type
    pub kv: KvType,
    /// MTP draft depth the server would build the engine for (0 = off)
    pub mtp_depth: usize,
    pub head: Option<HeadSource<'a>>,
    /// decode threads (`--oxe-cpus`)
    pub decode_threads: usize,
    /// prefill threads (`--oxe-prefill-cpus`; equal to `decode_threads` when not separate)
    pub prefill_threads: usize,
    pub margin_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AllocationBudget {
    pub kv_bytes_per_position: u64,
    pub scratch_bytes: u64,
    pub mtp_state_bytes: u64,
    pub margin_bytes: u64,
}

#[derive(Clone, Debug)]
pub struct Report {
    pub architecture: String,
    pub n_ctx_requested: usize,
    /// the 256-padded context capacity the engine allocates
    pub n_ctx_capacity: usize,
    /// GGUF `context_length`
    pub n_ctx_train: usize,
    pub kv: KvType,
    /// MTP depth in effect (0 when off or when no MTP block is available)
    pub mtp_depth: usize,
    /// set when MTP was asked for but the server would run without it
    pub mtp_fallback: Option<String>,
    pub recurrent_slots: usize,
    pub lanes: usize,
    pub decode_threads: usize,
    pub prefill_threads: usize,
    pub attention_layers: usize,
    pub recurrent_layers: usize,
    pub budget: AllocationBudget,
    pub mtp_head_file_bytes: u64,
    /// engine without MTP at this context
    pub base: CtxMem,
    /// engine as launched (equal to `base` when MTP is off)
    pub engine: CtxMem,
    /// unique bytes moved per token (from `Plan::budget`)
    pub bytes_per_token: Budget,
}

/// The MTP block the server loads (`MtpWeights::load`): `blk.{n_layer}` attention + MoE + nextn.
pub fn has_mtp_block(h: &Header, n_layer: usize) -> bool {
    [
        "attn_q.weight",
        "ffn_gate_exps.weight",
        "nextn.eh_proj.weight",
    ]
    .iter()
    .all(|s| h.tensor(&format!("blk.{n_layer}.{s}")).is_some())
}

/// The model's own `context_length`: the context an admission report uses when none is asked for.
pub fn trained_context(h: &Header) -> Result<usize, String> {
    match ModelDesc::from_header(h)?.n_ctx_train {
        0 => Err("the model declares no context_length; pass --ctx".into()),
        n => Ok(n),
    }
}

fn to_u64(v: usize) -> u64 {
    v as u64
}

/// Compute the report. Every refusal carries its reason; nothing is guessed.
pub fn report(h: &Header, rq: &Request) -> Result<Report, String> {
    if rq.decode_threads == 0 || rq.prefill_threads == 0 {
        return Err("thread counts must be at least 1".into());
    }
    if rq.margin_bytes == 0 {
        return Err("margin_bytes must be positive".into());
    }
    if rq.mtp_depth > 3 {
        return Err("MTP depth must be 0..3".into());
    }
    // unknown architecture, missing metadata or tensors: the descriptor says why
    let desc = ModelDesc::from_header(h)?;
    let capacity = validate_n_ctx(rq.n_ctx, desc.n_ctx_train)?;
    // the same plan the loader builds (and its coverage check), for the bytes-per-token section
    let plan = Plan::build(h, &desc, rq.decode_threads, capacity)?;
    plan.check_coverage()?;
    let bytes_per_token = plan.budget(&h.tensors);

    let mut depth = rq.mtp_depth;
    let mut mtp_fallback = None;
    let mut head_bytes = 0;
    if depth > 0 {
        match &rq.head {
            Some(src) if has_mtp_block(src.header, desc.n_layer) => head_bytes = src.file_bytes,
            Some(_) => {
                mtp_fallback = Some(format!(
                    "the sidecar head has no usable MTP block blk.{}",
                    desc.n_layer
                ))
            }
            None if has_mtp_block(h, desc.n_layer) => {}
            None => {
                mtp_fallback = Some(format!(
                    "the model has no MTP block blk.{} and no sidecar head",
                    desc.n_layer
                ))
            }
        }
        if mtp_fallback.is_some() {
            depth = 0;
        }
    }

    let (dec, pf) = (rq.decode_threads, rq.prefill_threads);
    let n_att = desc.n_attention();
    let base = ctx_mem(&desc, capacity, rq.kv, n_att, 1, 1);
    let (slots, lanes, engine) = if depth > 0 {
        let (slots, lanes) = lane_shape(true, depth + 1);
        (
            slots,
            lanes,
            ctx_mem(&desc, capacity, rq.kv, n_att + 1, lanes, slots),
        )
    } else {
        (1, 1, base)
    };
    let non_kv = |m: &CtxMem| -> Result<u64, String> {
        m.total(dec, pf)
            .checked_sub(m.n_ctx * m.kv_per_pos)
            .map(to_u64)
            .ok_or_else(|| "internal: KV larger than the engine total".to_string())
    };
    let scratch = non_kv(&base)?;
    let mtp_extra = non_kv(&engine)?
        .checked_sub(scratch)
        .ok_or("internal: MTP engine smaller than the plain engine")?;
    Ok(Report {
        architecture: h.architecture().unwrap_or_default().to_string(),
        n_ctx_requested: rq.n_ctx,
        n_ctx_capacity: capacity,
        n_ctx_train: desc.n_ctx_train,
        kv: rq.kv,
        mtp_depth: depth,
        mtp_fallback,
        recurrent_slots: slots,
        lanes,
        decode_threads: dec,
        prefill_threads: pf,
        attention_layers: n_att,
        recurrent_layers: desc.n_recurrent(),
        budget: AllocationBudget {
            kv_bytes_per_position: to_u64(engine.kv_per_pos),
            scratch_bytes: scratch,
            mtp_state_bytes: mtp_extra
                .checked_add(head_bytes)
                .ok_or("mtp_state_bytes overflow")?,
            margin_bytes: rq.margin_bytes,
        },
        mtp_head_file_bytes: head_bytes,
        base,
        engine,
        bytes_per_token,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{mini_model_ext, GgufWriter};

    fn header(tag: &str, mtp: bool) -> (std::path::PathBuf, gguf::Gguf) {
        let dir = std::env::temp_dir().join(format!("oxe-admission-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mini.gguf");
        mini_model_ext(&path, mtp);
        let g = gguf::Gguf::open(&path).unwrap();
        (dir, g)
    }

    fn rq(n_ctx: usize, depth: usize) -> Request<'static> {
        Request {
            n_ctx,
            kv: KvType::F16,
            mtp_depth: depth,
            head: None,
            decode_threads: 2,
            prefill_threads: 2,
            margin_bytes: DEFAULT_MARGIN_BYTES,
        }
    }

    #[test]
    fn components_add_up_to_the_engine_total() {
        for mtp in [false, true] {
            let (dir, g) = header("sum", mtp);
            for kv in [KvType::F16, KvType::KV8] {
                let r = report(
                    &g.header,
                    &Request {
                        kv,
                        ..rq(1000, mtp as usize * 2)
                    },
                )
                .unwrap();
                let b = r.budget;
                assert_eq!(
                    b.kv_bytes_per_position * r.n_ctx_capacity as u64
                        + b.scratch_bytes
                        + b.mtp_state_bytes,
                    r.engine.total(2, 2) as u64
                );
                assert_eq!(r.n_ctx_capacity, 1024);
            }
            std::fs::remove_dir_all(dir).ok();
        }
    }

    #[test]
    fn mtp_without_a_block_falls_back_like_the_server() {
        let (dir, g) = header("fb", false);
        let r = report(&g.header, &rq(256, 2)).unwrap();
        assert_eq!(r.mtp_depth, 0);
        assert!(r.mtp_fallback.unwrap().contains("no MTP block"));
        assert_eq!(r.budget.mtp_state_bytes, 0);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn refuses_zero_margin_and_context_beyond_training() {
        let (dir, g) = header("ref", false);
        let mut q = rq(256, 0);
        q.margin_bytes = 0;
        assert!(report(&g.header, &q).unwrap_err().contains("margin"));
        let e = report(&g.header, &rq(1 << 20, 0)).unwrap_err();
        assert!(e.contains("trained context"), "{e}");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn unknown_architecture_names_the_reason() {
        let dir = std::env::temp_dir().join(format!("oxe-admission-arch-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("other.gguf");
        let mut w = GgufWriter {
            kv: vec![],
            n_kv: 0,
            tensors: vec![],
        };
        w.str("general.architecture", "llama");
        w.write(&path);
        let g = gguf::Gguf::open(&path).unwrap();
        let e = report(&g.header, &rq(256, 0)).unwrap_err();
        assert!(e.contains("architecture llama is not supported"), "{e}");
        std::fs::remove_dir_all(dir).ok();
    }
}
