//! MEMLAT boot policy; all filesystem targets come from the owner policy.
pub mod image;
pub mod menu;
pub mod policy;
use sha2::{Digest, Sha256};
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
pub fn err<T>(s: &str) -> Result<T> {
    Err(s.into())
}
/// Match Python Path.read_text's universal-newline decoding.
pub fn read_text(path: &Path) -> Result<String> {
    Ok(fs::read_to_string(path)?
        .replace("\r\n", "\n")
        .replace('\r', "\n"))
}
pub fn sha256(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buf = vec![0; 4 * 1024 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
static NEXT: AtomicU64 = AtomicU64::new(0);
pub struct Temporary {
    pub path: PathBuf,
}
impl Temporary {
    pub fn directory(prefix: &str) -> Result<Self> {
        for _ in 0..100 {
            let path = std::env::temp_dir().join(format!(
                "{prefix}{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::DirBuilder::new().mode(0o700).create(&path) {
                Ok(()) => {
                    fs::set_permissions(&path, fs::Permissions::from_mode(0o700))?;
                    return Ok(Self { path });
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => (),
                Err(e) => return Err(e.into()),
            }
        }
        err("cannot create temporary directory")
    }
}
impl Drop for Temporary {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}
pub fn replace(path: &Path, before: &str, after: &str) -> Result<()> {
    let parent = path.parent().ok_or("missing parent")?;
    let tmp = parent.join(format!(
        ".memlat-default-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    replace_with_temporary(path, before, after, &tmp)
}
fn replace_with_temporary(path: &Path, before: &str, after: &str, tmp: &Path) -> Result<()> {
    let info = path.metadata()?;
    let parent = path.parent().ok_or("missing parent")?;
    let mut created = false;
    let result = (|| -> Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(tmp)?;
        created = true;
        file.write_all(after.as_bytes())?;
        file.flush()?;
        file.set_permissions(fs::Permissions::from_mode(info.mode() & 0o777))?;
        // SAFETY: fchown operates on the owned open temporary file; metadata is from target.
        unsafe {
            use std::os::fd::AsRawFd;
            if libc::geteuid() == 0 && libc::fchown(file.as_raw_fd(), info.uid(), info.gid()) != 0 {
                return Err(std::io::Error::last_os_error().into());
            }
        }
        file.sync_all()?;
        if read_text(path)? != before {
            return err("boot configuration changed concurrently");
        }
        fs::rename(tmp, path)?;
        fs::File::open(parent)?.sync_all()?;
        Ok(())
    })();
    if created {
        let _ = fs::remove_file(tmp);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn existing_temporary_is_never_removed_by_failed_replacement() {
        let dir = Temporary::directory("memlat-replace-test-").unwrap();
        let path = dir.path.join("config");
        let temporary = dir.path.join("existing");
        fs::write(&path, "original menu").unwrap();
        fs::write(&temporary, "other operation").unwrap();
        assert!(replace_with_temporary(&path, "original menu", "new menu", &temporary).is_err());
        assert_eq!(fs::read_to_string(&path).unwrap(), "original menu");
        assert_eq!(fs::read_to_string(&temporary).unwrap(), "other operation");
    }
}
