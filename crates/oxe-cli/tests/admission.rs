//! `oxe admission`: the report matches the engine's own allocation code, scales with the context,
//! and refuses with a reason instead of guessing. Runs the real `oxe` binary on synthetic GGUFs
//! (header only is read: the fixtures' weights are random filler).

use model::desc::ModelDesc;
use model::engine::{ctx_mem, lane_shape, KvType};
use model::plan::Plan;
use model::testkit::{mini_model_ext, GgufWriter};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn fixture(tag: &str, mtp: bool) -> (PathBuf, PathBuf) {
    let dir = std::env::temp_dir().join(format!("oxe-admission-cli-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("mini.gguf");
    mini_model_ext(&path, mtp);
    (dir, path)
}

fn admission(model: &Path, extra: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_oxe"))
        .arg("admission")
        .arg("--model")
        .arg(model)
        .args(["--oxe-cpus", "0"])
        .args(extra)
        .arg("--json")
        .output()
        .unwrap()
}

fn report(model: &Path, extra: &[&str]) -> Value {
    let out = admission(model, extra);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).unwrap()
}

fn num(v: &Value) -> u64 {
    v.as_u64().unwrap()
}

#[test]
fn admission_help_lists_launch_options_without_a_model() {
    for flag in ["--help", "-h"] {
        let output = Command::new(env!("CARGO_BIN_EXE_oxe"))
            .args(["admission", flag])
            .output()
            .unwrap();
        assert!(output.status.success(), "{:?}", output);
        assert!(output.stderr.is_empty(), "{:?}", output);
        let help = std::str::from_utf8(&output.stdout).unwrap();
        assert!(help.contains("oxe admission --model M"), "{help}");
        for option in [
            "--ctx-size",
            "--spec-draft-n-max",
            "--mtp-tune",
            "--margin-bytes",
            "--json",
        ] {
            assert!(help.contains(option), "missing {option} in {help}");
        }
    }
}

#[test]
fn admission_report_matches_engine_allocation_plan() {
    for mtp in [false, true] {
        let (dir, path) = fixture(if mtp { "plan-mtp" } else { "plan" }, mtp);
        let depth = if mtp { 2 } else { 0 };
        let spec = if mtp { "draft-mtp" } else { "none" };
        let r = report(
            &path,
            &[
                "--ctx",
                "1000",
                "--spec-type",
                spec,
                "--spec-draft-n-max",
                "2",
                "--session-ram-bytes",
                "0",
            ],
        );
        // the engine's own numbers, computed here from the same header
        let g = gguf::Gguf::open(&path).unwrap();
        let desc = ModelDesc::from_header(&g.header).unwrap();
        let n_att = desc.n_attention();
        let base = ctx_mem(&desc, 1000, KvType::F16, n_att, 1, 1);
        let (full, lanes) = if mtp {
            let (slots, lanes) = lane_shape(true, depth + 1);
            (
                ctx_mem(&desc, 1000, KvType::F16, n_att + 1, lanes, slots),
                lanes,
            )
        } else {
            (base, 1)
        };
        let cap = 1024u64;
        let b = &r["limits"]["allocation_budget"];
        assert_eq!(num(&b["kv_bytes_per_position"]), full.kv_per_pos as u64);
        // independent of ctx_mem: K+V f16 rows of every KV head of every attention layer
        let layers = n_att + mtp as usize;
        assert_eq!(
            full.kv_per_pos,
            layers * desc.n_head_kv * 2 * 2 * desc.head_dim
        );
        let non_kv = |m: &model::engine::CtxMem| (m.total(1, 1) - m.n_ctx * m.kv_per_pos) as u64;
        assert_eq!(num(&b["scratch_bytes"]), non_kv(&base));
        assert_eq!(
            num(&b["mtp_state_bytes"]),
            non_kv(&full) - non_kv(&base),
            "mtp {mtp}"
        );
        assert_eq!(
            num(&b["kv_bytes_per_position"]) * cap
                + num(&b["scratch_bytes"])
                + num(&b["mtp_state_bytes"]),
            full.total(1, 1) as u64
        );
        assert_eq!(num(&b["margin_bytes"]), 1 << 30);
        assert_eq!(num(&r["launch"]["lanes"]), lanes as u64);
        assert_eq!(num(&r["limits"]["context_window"]), 1000);
        assert_eq!(num(&r["launch"]["context_capacity"]), cap);
        // bytes per token come from the loader's plan
        let plan = Plan::build(&g.header, &desc, 1, 1024).unwrap();
        let bt = plan.budget(&g.header.tensors);
        assert_eq!(num(&r["bytes_per_token"]["weights"]), bt.weights());
        assert_eq!(num(&r["bytes_per_token"]["state_rw"]), bt.state_rw);
        assert_eq!(num(&r["bytes_per_token"]["total_at_ctx"]), bt.total());
        // the resolved launch arguments are complete and carry the context
        let args: Vec<&str> = r["arguments"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| a.as_str().unwrap())
            .collect();
        assert!(args.windows(2).any(|w| w == ["--ctx-size", "1000"]));
        assert!(args.windows(2).any(|w| w == ["--spec-type", spec]));
        assert_eq!(
            num(&r["model"]["file_bytes"]),
            std::fs::metadata(&path).unwrap().len()
        );
        std::fs::remove_dir_all(dir).ok();
    }
}

#[test]
fn admission_report_scales_kv_with_ctx() {
    let (dir, path) = fixture("scale", false);
    let at = |ctx: &str| report(&path, &["--ctx", ctx]);
    let (a, b) = (at("1024"), at("2048"));
    let kv = |r: &Value| num(&r["limits"]["allocation_budget"]["kv_bytes_per_position"]);
    assert_eq!(
        kv(&a),
        kv(&b),
        "the coefficient does not depend on the context"
    );
    assert_eq!(
        num(&b["engine_allocation"]["kv_bytes_at_capacity"])
            - num(&a["engine_allocation"]["kv_bytes_at_capacity"]),
        kv(&a) * 1024
    );
    // padding to 256 positions: 1025 allocates 1280
    let c = at("1025");
    assert_eq!(num(&c["launch"]["context_capacity"]), 1280);
    assert_eq!(num(&c["limits"]["context_window"]), 1025);
    // context-sized scratch grows with the context
    let sc = |r: &Value| num(&r["limits"]["allocation_budget"]["scratch_bytes"]);
    assert!(sc(&b) > sc(&a));
    // q8_0 KV is smaller per position
    let q = report(
        &path,
        &[
            "--ctx",
            "1024",
            "--cache-type-k",
            "q8_0",
            "--cache-type-v",
            "q8_0",
        ],
    );
    assert!(kv(&q) < kv(&a));
    assert_eq!(q["launch"]["kv_type"], "kv8");
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn admission_report_refuses_unknown_architecture_with_reason() {
    let dir = std::env::temp_dir().join(format!("oxe-admission-cli-arch-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("other.gguf");
    let mut w = GgufWriter {
        kv: vec![],
        n_kv: 0,
        tensors: vec![],
    };
    w.str("general.architecture", "llama");
    w.write(&path);
    let out = admission(&path, &["--ctx", "4096"]);
    assert_eq!(out.status.code(), Some(2));
    assert!(out.stdout.is_empty());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("architecture llama is not supported"), "{err}");

    // other refusals are explained too: unknown option, context beyond training
    let (d2, mini) = fixture("refuse", false);
    for (extra, why) in [
        (
            vec!["--ctx", "4096", "--frobnicate", "1"],
            "unknown option --frobnicate",
        ),
        (vec!["--ctx", "999999"], "trained context"),
        (
            vec!["--ctx", "4096", "--engine", "stub"],
            "--engine oxe only",
        ),
    ] {
        let out = admission(&mini, &extra);
        assert_eq!(out.status.code(), Some(2), "{extra:?}");
        assert!(out.stdout.is_empty(), "{extra:?}");
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(err.contains(why), "{extra:?}: {err}");
    }
    // no --ctx: the model's own context_length, said so in the report (the fixture declares 4096)
    let r = report(&mini, &[]);
    assert_eq!(num(&r["limits"]["context_window"]), 4096);
    assert_eq!(r["launch"]["context_source"], "model context_length");
    assert_eq!(
        report(&mini, &["--ctx", "512"])["launch"]["context_source"],
        "requested"
    );
    std::fs::remove_dir_all(dir).ok();
    std::fs::remove_dir_all(d2).ok();
}

fn words(r: &Value) -> Vec<String> {
    r["arguments"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| a.as_str().unwrap().to_string())
        .collect()
}

fn value_of(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

#[test]
fn admission_launch_carries_cache_capacity_and_counts_session_ram() {
    let (dir, path) = fixture("cache", false);
    let r = report(&path, &["--ctx", "1024"]);
    let a = words(&r);
    assert_eq!(value_of(&a, "--state-cache").as_deref(), Some("default"));
    assert_eq!(value_of(&a, "--prefix-cache").as_deref(), Some("default"));
    assert_eq!(value_of(&a, "--ctx-checkpoints").as_deref(), Some("8"));
    let ram: u64 = value_of(&a, "--session-ram-bytes")
        .unwrap()
        .parse()
        .unwrap();
    assert!(ram > 0, "the native launch must give parked sessions RAM");
    assert_eq!(num(&r["launch"]["session_ram_bytes"]), ram);
    // the session RAM is in the budget: with it switched off the scratch term is smaller by exactly it
    let off = report(&path, &["--ctx", "1024", "--session-ram-bytes", "0"]);
    let scratch = |r: &Value| num(&r["limits"]["allocation_budget"]["scratch_bytes"]);
    assert_eq!(scratch(&r) - scratch(&off), ram);
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn admission_owner_cache_values_win() {
    let (dir, path) = fixture("cache-owner", false);
    let r = report(
        &path,
        &[
            "--ctx",
            "1024",
            "--session-ram-bytes",
            "123456789",
            "--ctx-checkpoints",
            "3",
            "--state-cache",
            "/tmp/owner-state",
            "--prefix-cache",
            "/tmp/owner-prefix",
        ],
    );
    let a = words(&r);
    assert_eq!(
        value_of(&a, "--session-ram-bytes").as_deref(),
        Some("123456789")
    );
    assert_eq!(value_of(&a, "--ctx-checkpoints").as_deref(), Some("3"));
    assert_eq!(
        value_of(&a, "--state-cache").as_deref(),
        Some("/tmp/owner-state")
    );
    assert_eq!(
        value_of(&a, "--prefix-cache").as_deref(),
        Some("/tmp/owner-prefix")
    );
    // each flag appears once
    for f in [
        "--session-ram-bytes",
        "--ctx-checkpoints",
        "--state-cache",
        "--prefix-cache",
    ] {
        assert_eq!(a.iter().filter(|w| *w == f).count(), 1, "{f}");
    }
    std::fs::remove_dir_all(dir).ok();
}
