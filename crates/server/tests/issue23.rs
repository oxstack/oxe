//! #23: oxe-server must not serve the stub silently. Without `--engine`, an absent or
//! unsupported architecture refuses before any weights are mapped; the stub stays available
//! only when asked for explicitly. Every fixture here is a synthetic GGUF header.
use model::testkit::GgufWriter;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

const BIN: &str = env!("CARGO_BIN_EXE_oxe-server");

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("oxe-issue23-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn unsupported_gguf(path: &Path) {
    let mut w = GgufWriter {
        kv: vec![],
        n_kv: 0,
        tensors: vec![],
    };
    w.str("general.architecture", "llama");
    w.write(path);
}

fn stub_gguf(path: &Path) {
    let mut w = GgufWriter {
        kv: vec![],
        n_kv: 0,
        tensors: vec![],
    };
    w.str("general.architecture", "llama");
    w.str("tokenizer.ggml.model", "gpt2");
    w.str("tokenizer.ggml.pre", "qwen2");
    let tokens = ["<unk>", "a", "b", "Ġ", "Ġa", "Ġb", "hello"];
    w.key("tokenizer.ggml.tokens", 9);
    w.kv.extend(8u32.to_le_bytes());
    w.kv.extend((tokens.len() as u64).to_le_bytes());
    for token in tokens {
        w.kv.extend((token.len() as u64).to_le_bytes());
        w.kv.extend(token.as_bytes());
    }
    w.key("tokenizer.ggml.merges", 9);
    w.kv.extend(8u32.to_le_bytes());
    w.kv.extend(0u64.to_le_bytes());
    w.key("tokenizer.ggml.eos_token_id", 5);
    w.kv.extend(2i32.to_le_bytes());
    w.write(path);
}

fn command(home: &Path, args: &[&str]) -> Command {
    let mut command = Command::new(BIN);
    command
        .env_clear()
        .env("HOME", home)
        .env("PATH", "/usr/bin:/bin")
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .args(args);
    command
}

#[test]
fn the_default_refuses_an_unsupported_architecture_by_name() {
    let dir = scratch("arch");
    let model = dir.join("llama.gguf");
    unsupported_gguf(&model);
    let output = command(&dir, &["--model", model.to_str().unwrap()])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("architecture llama is not supported"),
        "{stderr}"
    );
    assert!(stderr.contains("pass --engine stub"), "{stderr}");
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn the_default_refuses_a_missing_model_file() {
    let dir = scratch("missing");
    let output = command(&dir, &["--model", "/nonexistent/model.gguf"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("cannot read the model header"), "{stderr}");
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn the_stub_starts_only_when_explicitly_requested() {
    let dir = scratch("stub");
    let model = dir.join("stub.gguf");
    stub_gguf(&model);
    let mut child = command(
        &dir,
        &[
            "--model",
            model.to_str().unwrap(),
            "--engine",
            "stub",
            "--port",
            "0",
        ],
    )
    .spawn()
    .unwrap();
    // Wait past argument parsing and tokenizer load: the default would already have refused.
    std::thread::sleep(Duration::from_millis(500));
    let running = child.try_wait().unwrap().is_none();
    let _ = child.kill();
    let _ = child.wait();
    assert!(running, "an explicit --engine stub must start");
    std::fs::remove_dir_all(dir).ok();
}
