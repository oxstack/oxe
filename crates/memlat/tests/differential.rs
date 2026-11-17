//! Actual command results and on-disk bytes against the captured boot oracle.
#[test]
fn native_command_matches_captured_boot_fixtures() {
    let home = oxc_memlat::Temporary::directory("memlat-differential-home-").unwrap();
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let result = std::process::Command::new("python3")
        .arg(root.join("tests/differential.py"))
        .arg(env!("CARGO_BIN_EXE_local-ai-memlat-boot"))
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("HOME", &home.path)
        .output()
        .expect("execute boot fixture runner");
    assert!(
        result.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    print!("{}", String::from_utf8_lossy(&result.stdout));
}
