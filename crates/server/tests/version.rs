use std::process::Command;

/// What the binary was built to report: the release stamp when set, else the package version.
fn expected_version() -> &'static str {
    match option_env!("OX_BUILD_VERSION") {
        Some(stamp) if !stamp.is_empty() => stamp,
        _ => env!("CARGO_PKG_VERSION"),
    }
}

#[test]
fn version_without_model_or_port() {
    let output = Command::new(env!("CARGO_BIN_EXE_oxe-server"))
        .arg("--version")
        .env_clear()
        .output()
        .unwrap();
    assert!(output.status.success());
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        format!("oxe-server {}\n", expected_version())
    );
    assert!(output.stderr.is_empty());
}
