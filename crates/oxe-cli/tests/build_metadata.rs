#[path = "../build_support.rs"]
mod build_support;

use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn git(current_dir: &Path, args: &[&str]) {
    let status = Command::new("git")
        .args(args)
        .current_dir(current_dir)
        .status()
        .unwrap();
    assert!(status.success(), "git {args:?} failed");
}

#[test]
fn watches_linked_worktree_head_and_index() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("oxe-build-metadata-{nonce}"));
    let repository = root.join("repository");
    let worktree = root.join("linked");
    fs::create_dir_all(&repository).unwrap();
    git(&repository, &["init", "-q"]);
    git(&repository, &["config", "user.name", "Test User"]);
    git(
        &repository,
        &["config", "user.email", "test@example.invalid"],
    );
    fs::write(repository.join("tracked"), "one\n").unwrap();
    git(&repository, &["add", "tracked"]);
    git(&repository, &["commit", "-qm", "initial"]);
    git(
        &repository,
        &["worktree", "add", "-q", worktree.to_str().unwrap()],
    );

    let manifest = worktree.join("engine/crates/oxe-cli");
    fs::create_dir_all(&manifest).unwrap();
    let watched = build_support::git_metadata_paths(&manifest);

    assert_eq!(watched.len(), 2);
    assert!(watched.iter().all(|path| path.is_file()));
    assert!(watched
        .iter()
        .all(|path| path.starts_with(repository.join(".git/worktrees"))));
    assert!(watched.iter().any(|path| path.ends_with("HEAD")));
    assert!(watched.iter().any(|path| path.ends_with("index")));

    fs::remove_dir_all(root).unwrap();
}
