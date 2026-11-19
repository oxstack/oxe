// Record the engine's git commit (plus "-dirty") in the M0 report.
use std::process::Command;

fn main() {
    let rev = Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success());
    // --no-optional-locks: a plain `git status` refreshes and rewrites .git/index, which is
    // watched below, so every cargo call would see it changed and rerun this script.
    let dirty = Command::new("git")
        .args([
            "--no-optional-locks",
            "status",
            "--porcelain",
            "--untracked-files=no",
        ])
        .output()
        .ok()
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(false);
    if let Some(o) = rev {
        let mut s = String::from_utf8_lossy(&o.stdout).trim().to_string();
        if dirty {
            s.push_str("-dirty");
        }
        println!("cargo:rustc-env=OXE_GIT_COMMIT={s}");
    }
    // A watched path that does not exist (a git worktree has a .git file, not a directory)
    // counts as changed on every run, so watch only the files that exist.
    let manifest = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    for watched in ["../../../.git/HEAD", "../../../.git/index"] {
        if manifest.join(watched).exists() {
            println!("cargo:rerun-if-changed={watched}");
        }
    }
}
