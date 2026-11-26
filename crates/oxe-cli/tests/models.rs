//! `oxe models add/list/rm` and `oxe import-oxa` without weights or inference. The add path
//! asks the engine's own header reader (`oxe admission`) for the launch limits, using the
//! synthetic mini GGUF from model::testkit.
use serde_json::json;
use std::path::{Path, PathBuf};
use std::process::Command;

const OXE: &str = env!("CARGO_BIN_EXE_oxe");

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "oxe-models-test-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .subsec_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn oxe(home: &Path, args: &[&str]) -> std::process::Output {
    Command::new(OXE)
        .env_clear()
        .env("HOME", home)
        .env("PATH", "/usr/bin:/bin")
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn add_asks_the_engine_for_arguments_and_limits_then_lists_and_removes() {
    let dir = scratch("add");
    let gguf = dir.join("synthetic.gguf");
    model::testkit::mini_model(&gguf);
    let config = dir.join("models.toml");
    let added = oxe(
        &dir,
        &[
            "models",
            "add",
            gguf.to_str().unwrap(),
            "--id",
            "demo",
            "--ctx",
            "1024",
            "--config",
            config.to_str().unwrap(),
        ],
    );
    assert!(
        added.status.success(),
        "{}",
        String::from_utf8_lossy(&added.stderr)
    );
    let text = std::fs::read_to_string(&config).unwrap();
    assert!(text.contains("[models.demo]"), "{text}");
    assert!(text.contains("--engine"), "{text}");
    assert!(text.contains("context_window = 1024"), "{text}");
    assert!(text.contains("kv_bytes_per_position"), "{text}");
    assert!(text.contains("margin_bytes"), "{text}");

    let listed = oxe(
        &dir,
        &["models", "list", "--config", config.to_str().unwrap()],
    );
    assert!(listed.status.success());
    assert!(String::from_utf8_lossy(&listed.stdout).contains("demo"));

    let removed = oxe(
        &dir,
        &["models", "rm", "demo", "--config", config.to_str().unwrap()],
    );
    assert!(
        removed.status.success(),
        "{}",
        String::from_utf8_lossy(&removed.stderr)
    );
    let text = std::fs::read_to_string(&config).unwrap();
    assert!(!text.contains("[models.demo]"), "{text}");
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn import_oxa_converts_models_limits_and_environment_without_touching_the_source() {
    let dir = scratch("import");
    let source = dir.join("settings.json");
    let settings = json!({
        "oxa": {
            "local": {
                "binary": "/opt/oxe/bin/oxe-server",
                "idle_unload_seconds": 300,
                "environment": {"HF_HUB_OFFLINE": "1"},
                "models": {
                    "ornith": {
                        "model": "/models/ornith.gguf",
                        "arguments": ["--engine", "oxe", "--oxe-cpus", "0-7"]
                    }
                },
                "limits": {
                    "ornith": {
                        "context_window": 131072,
                        "max_output": 4096,
                        "allocation_budget": {
                            "kv_bytes_per_position": 22528,
                            "scratch_bytes": 1048576,
                            "mtp_state_bytes": 1048576,
                            "margin_bytes": 1048576
                        }
                    }
                }
            }
        }
    });
    std::fs::write(&source, serde_json::to_vec_pretty(&settings).unwrap()).unwrap();
    let digest = std::fs::read(&source).unwrap();
    let config = dir.join("models.toml");

    let imported = oxe(
        &dir,
        &[
            "import-oxa",
            "--from",
            source.to_str().unwrap(),
            "--config",
            config.to_str().unwrap(),
        ],
    );
    assert!(
        imported.status.success(),
        "{}",
        String::from_utf8_lossy(&imported.stderr)
    );
    let text = std::fs::read_to_string(&config).unwrap();
    assert!(text.contains("[models.ornith]"), "{text}");
    assert!(text.contains("/models/ornith.gguf"), "{text}");
    assert!(text.contains("context_window = 131072"), "{text}");
    assert!(text.contains("kv_bytes_per_position = 22528"), "{text}");
    assert!(text.contains("HF_HUB_OFFLINE"), "{text}");
    assert!(text.contains("idle_unload_seconds = 300"), "{text}");
    // The source's binary path belongs to the other machine: not copied unless asked.
    assert!(!text.contains("binary ="), "{text}");
    assert!(!text.contains("/opt/oxe/bin/oxe-server"), "{text}");
    assert_eq!(
        std::fs::read(&source).unwrap(),
        digest,
        "source was modified"
    );

    let again = oxe(
        &dir,
        &[
            "import-oxa",
            "--from",
            source.to_str().unwrap(),
            "--config",
            config.to_str().unwrap(),
        ],
    );
    assert_eq!(again.status.code(), Some(1), "an existing id must refuse");
    assert!(String::from_utf8_lossy(&again.stderr).contains("--force"));

    let forced = oxe(
        &dir,
        &[
            "import-oxa",
            "--from",
            source.to_str().unwrap(),
            "--config",
            config.to_str().unwrap(),
            "--force",
        ],
    );
    assert!(forced.status.success());
    assert_eq!(
        std::fs::read(&source).unwrap(),
        digest,
        "source was modified"
    );
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn import_oxa_dry_run_prints_but_writes_nothing() {
    let dir = scratch("dry");
    let source = dir.join("settings.json");
    std::fs::write(
        &source,
        serde_json::to_vec(&json!({
            "oxa": {"local": {
                "models": {"m": {"model": "/m.gguf", "arguments": []}},
                "limits": {"m": {
                    "context_window": 2048,
                    "max_output": 128,
                    "allocation_budget": {
                        "kv_bytes_per_position": 2,
                        "scratch_bytes": 0,
                        "mtp_state_bytes": 0,
                        "margin_bytes": 4096
                    }
                }}
            }}
        }))
        .unwrap(),
    )
    .unwrap();
    let config = dir.join("models.toml");
    let dry = oxe(
        &dir,
        &[
            "import-oxa",
            "--from",
            source.to_str().unwrap(),
            "--config",
            config.to_str().unwrap(),
            "--dry-run",
        ],
    );
    assert_eq!(
        dry.status.code(),
        Some(0),
        "a successful dry run exits 0: {}",
        String::from_utf8_lossy(&dry.stderr)
    );
    assert!(String::from_utf8_lossy(&dry.stdout).contains("[models.m]"));
    assert!(!config.exists(), "a dry run must not write the config");
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn import_oxa_keeps_the_binary_only_with_keep_binary() {
    let dir = scratch("keep-binary");
    let source = dir.join("settings.json");
    std::fs::write(
        &source,
        serde_json::to_vec(&json!({
            "oxa": {"local": {
                "binary": "/opt/oxe/bin/oxe-server",
                "models": {"m": {"model": "/m.gguf", "arguments": []}},
                "limits": {"m": {
                    "context_window": 2048,
                    "max_output": 128,
                    "allocation_budget": {
                        "kv_bytes_per_position": 2,
                        "scratch_bytes": 0,
                        "mtp_state_bytes": 0,
                        "margin_bytes": 4096
                    }
                }}
            }}
        }))
        .unwrap(),
    )
    .unwrap();
    let config = dir.join("models.toml");
    let kept = oxe(
        &dir,
        &[
            "import-oxa",
            "--from",
            source.to_str().unwrap(),
            "--config",
            config.to_str().unwrap(),
            "--keep-binary",
        ],
    );
    assert!(
        kept.status.success(),
        "{}",
        String::from_utf8_lossy(&kept.stderr)
    );
    let text = std::fs::read_to_string(&config).unwrap();
    assert!(
        text.contains("binary = \"/opt/oxe/bin/oxe-server\""),
        "{text}"
    );
    std::fs::remove_dir_all(dir).ok();
}
