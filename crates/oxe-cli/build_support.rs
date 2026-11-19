use std::path::{Path, PathBuf};
use std::process::Command;

pub fn git_metadata_paths(manifest_dir: &Path) -> Vec<PathBuf> {
    let output = Command::new("git")
        .args(["rev-parse", "--git-path", "HEAD", "--git-path", "index"])
        .current_dir(manifest_dir)
        .output()
        .ok()
        .filter(|output| output.status.success());

    output
        .map(|output| {
            String::from_utf8_lossy(&output.stdout)
                .lines()
                .map(PathBuf::from)
                .map(|path| {
                    if path.is_absolute() {
                        path
                    } else {
                        manifest_dir.join(path)
                    }
                })
                .filter(|path| path.exists())
                .collect()
        })
        .unwrap_or_default()
}
