//! Fresh, short-lived child authentication. Never reads a credential file.
use nix::fcntl::{open, openat, AtFlags, OFlag};
use nix::sys::stat::{fstatat, mkdirat, Mode};
use nix::unistd::{geteuid, unlinkat, UnlinkatFlags};
use std::fs::File;
use std::io::{self, Read, Write};
use std::os::fd::{AsRawFd, OwnedFd};
use std::os::unix::fs::MetadataExt;
use std::path::{Component, Path, PathBuf};

pub(crate) struct ChildAuth {
    directory: OwnedFd,
    name: String,
    token: String,
    created: bool,
}
impl ChildAuth {
    pub(crate) fn create(path: &Path) -> io::Result<Self> {
        if !path.is_absolute() {
            return Err(invalid());
        }
        let flags = OFlag::O_RDONLY | OFlag::O_DIRECTORY | OFlag::O_CLOEXEC | OFlag::O_NOFOLLOW;
        let mut directory = open(Path::new("/"), flags, Mode::empty())?;
        for component in path.components() {
            match component {
                Component::RootDir => (),
                Component::Normal(name) => {
                    refuse_repository(&directory)?;
                    match mkdirat(&directory, name, Mode::S_IRWXU) {
                        Ok(()) | Err(nix::errno::Errno::EEXIST) => (),
                        Err(e) => return Err(e.into()),
                    }
                    directory = openat(&directory, name, flags, Mode::empty())?;
                }
                _ => return Err(invalid()),
            }
        }
        refuse_repository(&directory)?;
        let file = File::from(directory);
        let metadata = file.metadata()?;
        if metadata.uid() != geteuid().as_raw() || metadata.mode() & 0o077 != 0 {
            return Err(invalid());
        }
        let directory: OwnedFd = file.into();
        let mut random = [0u8; 48];
        File::open("/dev/urandom")?.read_exact(&mut random)?;
        let hex = |bytes: &[u8]| bytes.iter().map(|byte| format!("{byte:02x}")).collect::<String>();
        let name = format!("engine-{}.key", hex(&random[..16]));
        let token = hex(&random[16..]);
        let mut auth = Self {
            directory,
            name,
            token,
            created: false,
        };
        let result = (|| {
            let fd = openat(
                &auth.directory,
                auth.name.as_str(),
                OFlag::O_WRONLY | OFlag::O_CREAT | OFlag::O_EXCL | OFlag::O_CLOEXEC | OFlag::O_NOFOLLOW,
                Mode::S_IRUSR | Mode::S_IWUSR,
            )?;
            auth.created = true;
            let mut file = File::from(fd);
            writeln!(file, "{}", auth.token)?;
            Ok(())
        })();
        result.map(|()| auth)
    }
    pub(crate) fn path(&self) -> PathBuf {
        // Anchor the child read to the retained directory, even if its pathname
        // is replaced. No inherited FD or secret in argv/environment is needed.
        PathBuf::from(format!(
            "/proc/{}/fd/{}/{}",
            std::process::id(),
            self.directory.as_raw_fd(),
            self.name
        ))
    }
    pub(crate) fn token(&self) -> &str {
        &self.token
    }
}
impl std::fmt::Debug for ChildAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ChildAuth([redacted])")
    }
}
impl Drop for ChildAuth {
    fn drop(&mut self) {
        if self.created {
            let _ = unlinkat(&self.directory, self.name.as_str(), UnlinkatFlags::NoRemoveDir);
        }
    }
}
fn invalid() -> io::Error {
    io::Error::other("child authentication requires a private owner-only directory")
}
fn refuse_repository(directory: &OwnedFd) -> io::Result<()> {
    let exists = |name: &str| match fstatat(directory, name, AtFlags::AT_SYMLINK_NOFOLLOW) {
        Ok(_) => Ok(true),
        Err(nix::errno::Errno::ENOENT) => Ok(false),
        Err(error) => Err(io::Error::from(error)),
    };
    if exists(".git")? || (exists("HEAD")? && exists("objects")?) {
        return Err(invalid());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxa_fake::TestEnv;
    #[test]
    fn private_child_file_is_fresh_anchored_and_removed_without_reading_it() {
        let env = TestEnv::new();
        let path = env.work().join("private");
        let first = ChildAuth::create(&path).unwrap();
        let second = ChildAuth::create(&path).unwrap();
        assert_ne!(first.token(), second.token());
        assert!(!format!("{first:?}").contains(first.token()));
        assert_eq!(std::fs::metadata(first.path()).unwrap().mode() & 0o777, 0o600);
        let key = first.path();
        drop(first);
        assert!(!key.exists());
        assert!(second.path().exists());
    }
    #[test]
    fn symlink_or_shared_storage_is_refused() {
        use std::os::unix::fs::PermissionsExt;
        let env = TestEnv::new();
        std::os::unix::fs::symlink(env.work(), env.work().join("link")).unwrap();
        assert!(ChildAuth::create(&env.work().join("link/private")).is_err());
        std::fs::set_permissions(env.work(), std::fs::Permissions::from_mode(0o755)).unwrap();
        assert!(ChildAuth::create(&env.work()).is_err());
    }

    #[test]
    fn private_authentication_is_never_created_under_repository_metadata() {
        let env = TestEnv::new();
        std::fs::create_dir(env.work().join(".git")).unwrap();
        assert!(ChildAuth::create(&env.work().join("private")).is_err());
        assert!(!env.work().join("private").exists());
    }
}
