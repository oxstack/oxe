//! `oxe admission --model M [--ctx N] [oxe-server launch arguments] [--margin-bytes B] --json`
//!
//! Prints, as JSON, the allocation bounds oxa's native local admission needs for one model and
//! launch profile. Only the GGUF header is read (the weights are not mapped, no thread pool or
//! device session is created). All numbers come from `model::admission`, which reuses the
//! engine's own allocation code; see that module for the split and the margin rule.
//!
//! The launch arguments are the oxe-server names (`--ctx-size`/`-c`, `--oxe-cpus`,
//! `--oxe-prefill-cpus`, `--spec-type`, `--spec-draft-n-max`/`--draft-max`, `--cache-type-k/-v`,
//! `--mtp-head`, `--engine oxe`) resolved by the same rules as oxe-server (default depth per model,
//! sidecar head, KV type mapping). An option this command does not understand is an error: it
//! never guesses whether an option changes allocations. `arguments` in the output is the complete
//! resolved list the budget is valid for.

use model::admission::{self, HeadSource, Request, DEFAULT_MARGIN_BYTES, MARGIN_RULE, SCHEMA};
use model::engine::KvType;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

const HELP: &str = "\
Usage: oxe admission --model M [--ctx N] [oxe-server launch options] [--margin-bytes B] --json

Print JSON allocation bounds from the model header without loading model weights.
  --model, -m M                   Read the GGUF header from M
  --ctx, --ctx-size, -c N         Size allocations for N context positions
  --oxe-cpus LIST                  Set the decode CPU list
  --oxe-prefill-cpus LIST          Set the prefill CPU list
  --engine oxe                    Use the native oxe engine
  --spec-type TYPE                 Select draft-mtp or none
  --spec-draft-n-max N             Set the draft depth
  --draft, --draft-max, --draft-n N  Enable MTP and set its draft depth
  --cache-type-k, -ctk TYPE         Set the key cache format (f16 or q8_0)
  --cache-type-v, -ctv TYPE         Set the value cache format (f16 or q8_0)
  --mtp-head PATH                  Select a separate MTP head
  --mtp-tune PATH                  Accept a server MTP tuning file
  --margin-bytes B                 Set the extra allocation margin in bytes
  --json                           Request JSON output (the default)
  -h, --help                       Show this help

Also accepted from server launch arguments: --alias/-a, --host, --port,
--ubatch-size/-ub, --oxe-weights, --mtp-max-ctx, --mlock, and --mtp-sampled.
";

/// Options that change nothing the report computes (value arity 1), accepted so that a real
/// oxe-server argument list can be passed through unchanged.
const NEUTRAL_WITH_VALUE: &[&str] = &[
    "--alias",
    "-a",
    "--host",
    "--port",
    "--ubatch-size",
    "-ub",
    "--oxe-weights",
    "--mtp-max-ctx",
];
const NEUTRAL_FLAGS: &[&str] = &["--mlock", "--mtp-sampled", "--json"];

#[derive(Default)]
struct Parsed {
    model: Option<String>,
    ctx: Option<usize>,
    cpus: Option<String>,
    prefill_cpus: Option<String>,
    spec_mtp: Option<bool>,
    draft_n_max: Option<usize>,
    cache_k: Option<String>,
    cache_v: Option<String>,
    mtp_head: Option<String>,
    mtp_tune: Option<String>,
    margin: Option<u64>,
}

fn number<T: std::str::FromStr>(flag: &str, v: &str) -> Result<T, String> {
    v.parse().map_err(|_| format!("{flag} {v}: not a number"))
}

fn parse(args: &[String]) -> Result<Parsed, String> {
    let mut p = Parsed::default();
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        if NEUTRAL_FLAGS.contains(&flag) {
            i += 1;
            continue;
        }
        let value = || {
            args.get(i + 1)
                .cloned()
                .ok_or_else(|| format!("{flag} needs a value"))
        };
        if NEUTRAL_WITH_VALUE.contains(&flag) {
            value()?;
            i += 2;
            continue;
        }
        let v = value().map_err(|e| {
            format!("{e} (or {flag} is not an option `oxe admission` understands; it never guesses whether an option changes allocations)")
        })?;
        match flag {
            "--model" | "-m" => p.model = Some(v),
            "--ctx" | "--ctx-size" | "-c" => p.ctx = Some(number(flag, &v)?),
            "--oxe-cpus" => p.cpus = Some(v),
            "--oxe-prefill-cpus" => p.prefill_cpus = Some(v),
            "--engine" => {
                if v != "oxe" {
                    return Err(format!(
                        "--engine {v}: admission is defined for --engine oxe only (the stub allocates no model; flash has its own streaming admission)"
                    ));
                }
            }
            "--spec-type" => {
                for t in v.split(',') {
                    match t {
                        "draft-mtp" => p.spec_mtp = Some(true),
                        "none" => p.spec_mtp = Some(false),
                        x => return Err(format!("--spec-type {x}: draft-mtp or none")),
                    }
                }
            }
            "--spec-draft-n-max" => p.draft_n_max = Some(number(flag, &v)?),
            "--draft" | "--draft-max" | "--draft-n" => {
                p.draft_n_max = Some(number(flag, &v)?);
                p.spec_mtp = Some(true);
            }
            "--cache-type-k" | "-ctk" => p.cache_k = Some(v),
            "--cache-type-v" | "-ctv" => p.cache_v = Some(v),
            "--mtp-head" => p.mtp_head = Some(v),
            "--mtp-tune" => p.mtp_tune = Some(v),
            "--margin-bytes" => p.margin = Some(number(flag, &v)?),
            other => {
                return Err(format!(
                    "unknown option {other}: `oxe admission` takes --model, --ctx, --oxe-cpus, --oxe-prefill-cpus, --spec-type, --spec-draft-n-max, --cache-type-k/-v, --mtp-head, --margin-bytes and --json; it never guesses whether another option changes allocations"
                ))
            }
        }
        i += 2;
    }
    Ok(p)
}

/// oxe-server's mapping of the two cache types to the engine's KV type.
fn kv_type(k: &str, v: &str) -> Result<KvType, String> {
    Ok(match (k, v) {
        ("f16", "f16") => KvType::F16,
        ("q8_0", "f16") => KvType::K8,
        ("f16", "q8_0") => KvType::V8,
        ("q8_0", "q8_0") => KvType::KV8,
        (k, v) => {
            return Err(format!(
                "--cache-type-k {k} / --cache-type-v {v}: oxe supports f16 and q8_0"
            ))
        }
    })
}

fn cpu_count(flag: &str, list: &str) -> Result<usize, String> {
    let cpus = sched::parse_cpu_list(list).map_err(|e| format!("{flag}: {e}"))?;
    let (kept, _) = sched::clamp_to_allowed(&cpus);
    if kept.is_empty() {
        return Err(format!(
            "{flag}: none of the CPUs {cpus:?} is available to this process"
        ));
    }
    Ok(kept.len())
}

/// The whole command on an argument list (after `admission`): the report as JSON, or why not.
pub fn report_json(args: &[String]) -> Result<Value, String> {
    let p = parse(args)?;
    let model = PathBuf::from(p.model.ok_or("--model is required")?);
    let cpus_arg = p.cpus.unwrap_or_else(|| "0-7".to_string());
    let decode = cpu_count("--oxe-cpus", &cpus_arg)?;
    let prefill = match &p.prefill_cpus {
        // a separate prefill team exists only when its list differs (oxe_engine::open_cache)
        Some(l) if *l != cpus_arg => cpu_count("--oxe-prefill-cpus", l)?,
        _ => decode,
    };
    let cache_k = p.cache_k.unwrap_or_else(|| "f16".into());
    let cache_v = p.cache_v.unwrap_or_else(|| "f16".into());
    let kv = kv_type(&cache_k, &cache_v)?;

    let file_bytes = std::fs::metadata(&model)
        .map_err(|e| format!("{}: {e}", model.display()))?
        .len();
    let g = gguf::Gguf::open(&model).map_err(|e| format!("{}: {e}", model.display()))?;
    // no --ctx: the model's own context_length (stated in the report), never a made-up size
    let (ctx, ctx_source) = match p.ctx {
        Some(c) => (c, "requested"),
        None => (
            admission::trained_context(&g.header)?,
            "model context_length",
        ),
    };

    // MTP depth, as oxe-server resolves it
    let head_path = model::mtp::mtp_head_path(&model, p.mtp_head.as_deref().map(Path::new));
    let model_default = model::mtp::default_mtp_depth(&model, head_path.as_deref());
    let tune = p
        .mtp_tune
        .map(PathBuf::from)
        .or_else(|| Some(model.with_extension("mtp-tune.json")).filter(|t| t.exists()));
    if tune.is_some() && p.spec_mtp != Some(false) && p.draft_n_max.is_none() {
        return Err(format!(
            "an MTP tune table ({}) decides the MTP depth inside oxe-server; pass --spec-draft-n-max N or --spec-type none so the budget is for a stated depth",
            tune.unwrap().display()
        ));
    }
    let depth = match p.spec_mtp {
        Some(false) => 0,
        Some(true) => p
            .draft_n_max
            .unwrap_or(if model_default > 0 { model_default } else { 2 }),
        None if model_default > 0 => p.draft_n_max.unwrap_or(model_default),
        None => 0,
    };
    if depth > 3 {
        return Err("--spec-draft-n-max must be 1..3 for draft-mtp".into());
    }
    let head = match (&head_path, depth > 0) {
        (Some(h), true) => {
            let hg = gguf::Gguf::open(h).map_err(|e| format!("MTP head {}: {e}", h.display()))?;
            let bytes = hg.file_bytes;
            Some((hg, bytes))
        }
        _ => None,
    };
    let rq = Request {
        n_ctx: ctx,
        kv,
        mtp_depth: depth,
        head: head.as_ref().map(|(hg, bytes)| HeadSource {
            header: &hg.header,
            file_bytes: *bytes,
        }),
        decode_threads: decode,
        prefill_threads: prefill,
        margin_bytes: p.margin.unwrap_or(DEFAULT_MARGIN_BYTES),
    };
    let r = admission::report(&g.header, &rq)?;

    let mut arguments: Vec<String> = ["--engine", "oxe", "--ctx-size"]
        .into_iter()
        .map(String::from)
        .collect();
    arguments.push(ctx.to_string());
    arguments.extend(["--oxe-cpus".into(), cpus_arg.clone()]);
    if let Some(l) = &p.prefill_cpus {
        arguments.extend(["--oxe-prefill-cpus".into(), l.clone()]);
    }
    if r.mtp_depth > 0 {
        arguments.extend([
            "--spec-type".into(),
            "draft-mtp".into(),
            "--spec-draft-n-max".into(),
            r.mtp_depth.to_string(),
        ]);
        if let Some(h) = &head_path {
            arguments.extend(["--mtp-head".into(), h.display().to_string()]);
        }
    } else {
        arguments.extend(["--spec-type".into(), "none".into()]);
    }
    if kv != KvType::F16 {
        arguments.extend([
            "--cache-type-k".into(),
            cache_k,
            "--cache-type-v".into(),
            cache_v,
        ]);
    }

    let b = r.budget;
    let e = &r.engine;
    let bt = &r.bytes_per_token;
    Ok(json!({
        "schema": SCHEMA,
        "engine": "oxe",
        "engine_version": env!("CARGO_PKG_VERSION"),
        "model": {
            "path": model.display().to_string(),
            "file_bytes": file_bytes,
            "architecture": r.architecture,
            "n_ctx_train": r.n_ctx_train,
        },
        "launch": {
            "context": r.n_ctx_requested,
            "context_source": ctx_source,
            "context_capacity": r.n_ctx_capacity,
            "kv_type": r.kv.name(),
            "mtp_depth": r.mtp_depth,
            "mtp_fallback": r.mtp_fallback,
            "mtp_head_file_bytes": r.mtp_head_file_bytes,
            "decode_threads": r.decode_threads,
            "prefill_threads": r.prefill_threads,
            "recurrent_slots": r.recurrent_slots,
            "lanes": r.lanes,
            "attention_layers_with_kv": r.attention_layers,
            "recurrent_layers": r.recurrent_layers,
        },
        "arguments": arguments,
        "limits": {
            "context_window": r.n_ctx_requested,
            "max_output": r.n_ctx_requested,
            "allocation_budget": {
                "kv_bytes_per_position": b.kv_bytes_per_position,
                "scratch_bytes": b.scratch_bytes,
                "mtp_state_bytes": b.mtp_state_bytes,
                "margin_bytes": b.margin_bytes,
            },
        },
        "max_output_basis": "the engine's own bound is the context length; narrower limits are the owner's choice",
        "margin_rule": MARGIN_RULE,
        "weights_note": "oxa adds the model file size (full residency) to the budget itself",
        "engine_allocation": {
            "kv_bytes_at_capacity": e.kv,
            "recurrent_bytes": e.recurrent,
            "decode_scratch_bytes": e.decode_scratch,
            "decode_per_thread_bytes": e.decode_per_thread,
            "prefill_per_thread_bytes": e.prefill_per_thread,
            "total_bytes": e.total(r.decode_threads, r.prefill_threads),
        },
        "bytes_per_token": {
            "weights": bt.weights(),
            "state_rw": bt.state_rw,
            "total_at_ctx": bt.total(),
        },
    }))
}

pub fn run(args: &[String]) -> ! {
    if matches!(args, [flag] if matches!(flag.as_str(), "-h" | "--help")) {
        print!("{HELP}");
        std::process::exit(0);
    }
    match report_json(args) {
        Ok(v) => {
            println!("{}", serde_json::to_string_pretty(&v).unwrap());
            std::process::exit(0)
        }
        Err(e) => {
            eprintln!("oxe admission: {e}");
            std::process::exit(2)
        }
    }
}
