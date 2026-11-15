use std::process::Command;

#[test]
fn version_without_model() {
    let output = Command::new(env!("CARGO_BIN_EXE_oxe"))
        .arg("--version")
        .env_clear()
        .output()
        .unwrap();
    assert!(output.status.success());
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        format!("oxe {}\n", env!("CARGO_PKG_VERSION"))
    );
    assert!(output.stderr.is_empty());
}
