// Record the engine's git commit (plus "-dirty") in the M0 report.
mod build_support;

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
    // Ask Git for the per-worktree metadata paths: in a linked worktree, `.git` is a
    // pointer file and HEAD/index live under the main repository's worktrees directory.
    let manifest = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    for watched in build_support::git_metadata_paths(&manifest) {
        println!("cargo:rerun-if-changed={}", watched.display());
    }
}
