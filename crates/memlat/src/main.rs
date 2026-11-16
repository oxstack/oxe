use oxc_memlat::{
    image,
    policy::{self, Host},
    Result,
};
use serde_json::Value;
use std::{
    fs,
    io::{self, Write},
    os::fd::AsRawFd,
    path::Path,
    process::Command,
    time::{Duration, Instant},
};
struct System;
impl Host for System {
    fn build(&self, p: &Value, output: &Path) -> Result<Value> {
        image::build(p, output)
    }
    fn notify(&self, p: &Value, error: &str) {
        let message = format!(
            "MEMLAT update failed; stock Omarchy selected. {}",
            error.chars().take(300).collect::<String>()
        );
        println!("{message}");
        let _ = io::stdout().flush();
        let Some(user) = p.get("notification_user").and_then(Value::as_str) else {
            return;
        };
        let Some(uid) = p.get("notification_uid") else {
            return;
        };
        let uid = uid.as_str().map(str::to_owned).unwrap_or_else(|| uid.to_string());
        if let Ok(mut child) = Command::new("/usr/bin/runuser")
            .args([
                "-u",
                user,
                "--",
                "/usr/bin/env",
                &format!("DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/{uid}/bus"),
                "/usr/bin/notify-send",
                "-u",
                "critical",
                "Local AI: MEMLAT fallback",
                &message,
            ])
            .spawn()
        {
            let deadline = Instant::now() + Duration::from_secs(5);
            loop {
                match child.try_wait() {
                    Ok(Some(_)) => break,
                    Err(_) => {
                        let _ = child.kill();
                        let _ = child.wait();
                        break;
                    }
                    _ => (),
                }
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    break;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        }
    }
}
fn execute(action: &str, policy_path: &Path, no_lock: bool, dry: bool, force: bool) -> Result<Value> {
    let lock = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("/run/lock/boot-partition.lock")?;
    if !no_lock {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            // SAFETY: flock is passed a live descriptor and documented flag bits.
            if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
                break;
            }
            let error = io::Error::last_os_error();
            if error.kind() != io::ErrorKind::WouldBlock {
                return Err(error.into());
            }
            if Instant::now() >= deadline {
                return oxc_memlat::err("boot-partition mutex busy; retry after boot tooling finishes");
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    }
    let p: Value = serde_json::from_str(&fs::read_to_string(policy_path)?)?;
    if action == "rebuild" {
        policy::rebuild(policy_path, &p, dry, force, &System)
    } else {
        if dry || force {
            return oxc_memlat::err("--dry-run/--force are only valid for rebuild");
        }
        let result = policy::apply(&p, action, &System)?;
        if action == "fallback" {
            if let Some(pending) = p.get("pending").and_then(Value::as_str).filter(|s| !s.is_empty()) {
                match fs::remove_file(pending) {
                    Ok(()) => (),
                    Err(e) if e.kind() == io::ErrorKind::NotFound => (),
                    Err(e) => return Err(e.into()),
                }
            }
        }
        Ok(result)
    }
}
const USAGE: &str = "usage: local-ai-memlat-boot [-h] [--policy POLICY] [--no-lock] [--dry-run]\n                            [--force]\n                            {enable,fallback,check,prepare-update,rebuild}";
const HELP: &str = "usage: local-ai-memlat-boot [-h] [--policy POLICY] [--no-lock] [--dry-run]\n                            [--force]\n                            {enable,fallback,check,prepare-update,rebuild}\n\nExplicit MEMLAT default selection and fail-safe stock fallback (no reboot).\nInstall a root-owned copy and policy outside the checkout before selecting it.\nThe ALPM pre-transaction hook uses prepare-update, not check: the installed\nkernel has not changed yet. Post hooks rebuild and restore the approved\ndefault.\n\npositional arguments:\n  {enable,fallback,check,prepare-update,rebuild}\n\noptions:\n  -h, --help            show this help message and exit\n  --policy POLICY\n  --no-lock             only inside limine-entry-tool mutex\n  --dry-run             build and verify in temporary files; do not\n                        select/install\n  --force               explicitly rebuild even without an update marker";
fn usage_error(message: &str) -> ! {
    eprintln!("{USAGE}\nlocal-ai-memlat-boot: error: {message}");
    std::process::exit(2)
}
fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let mut action = None;
    let mut path = "/etc/local-ai/memlat-boot.json".to_string();
    let (mut no_lock, mut dry, mut force) = (false, false, false);
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                println!("{HELP}");
                return;
            }
            "--no-lock" => no_lock = true,
            "--dry-run" => dry = true,
            "--force" => force = true,
            "--policy" => {
                i += 1;
                path = args
                    .get(i)
                    .unwrap_or_else(|| usage_error("argument --policy: expected one argument"))
                    .clone();
            }
            s if s.starts_with("--policy=") => path = s[9..].into(),
            s if ["enable", "fallback", "check", "prepare-update", "rebuild"].contains(&s)
                && action.is_none() =>
            {
                action = Some(s.to_string())
            }
            s if !s.starts_with('-') && action.is_none() => usage_error(&format!("argument action: invalid choice: '{s}' (choose from 'enable', 'fallback', 'check', 'prepare-update', 'rebuild')")),
            s => usage_error(&format!("unrecognized arguments: {s}")),
        }
        i += 1;
    }
    let action = action.unwrap_or_else(|| usage_error("the following arguments are required: action"));
    match execute(&action, Path::new(&path), no_lock, dry, force) {
        Ok(value) => println!(
            "{}",
            policy::json_text(&value, false).expect("JSON serialization")
        ),
        Err(error) => {
            eprintln!("memlat-boot: {error}");
            std::process::exit(1);
        }
    }
}
