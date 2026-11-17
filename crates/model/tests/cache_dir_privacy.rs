use model::weights::cache_dir;
use std::os::unix::fs::{symlink, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "oxe-cache-privacy-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
        Self(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn assert_private_refusal(path: &Path) {
    let error = cache_dir(Some(path)).unwrap_err();
    assert_eq!(
        error,
        format!(
            "unsafe cache dir {} (must be a directory owned by the current user, not group/other-writable)",
            path.display()
        )
    );
}

#[test]
fn unsafe_cache_refusal_preserves_guidance_without_numeric_owner() {
    let scratch = Scratch::new();
    let cache = scratch.0.join("cache");
    std::fs::create_dir(&cache).unwrap();
    for mode in [0o720, 0o702] {
        std::fs::set_permissions(&cache, std::fs::Permissions::from_mode(mode)).unwrap();
        assert_private_refusal(&cache);
    }
    std::fs::set_permissions(&cache, std::fs::Permissions::from_mode(0o700)).unwrap();
    let link = scratch.0.join("link");
    symlink(&cache, &link).unwrap();
    assert_private_refusal(&link);
    assert_eq!(cache_dir(Some(&cache)).unwrap(), cache);
}

#[test]
fn private_cache_is_created_and_reused() {
    let scratch = Scratch::new();
    let cache = scratch.0.join("new-cache");
    assert_eq!(cache_dir(Some(&cache)).unwrap(), cache);
    assert_eq!(
        std::fs::metadata(&cache).unwrap().permissions().mode() & 0o777,
        0o700
    );
    assert_eq!(cache_dir(Some(&cache)).unwrap(), cache);
}
