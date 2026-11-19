// The device backend is C (npu/oxn/host/oxn_host.c) built against the open-source FastRPC headers.
// Only compiled when the `device` feature is on; the default build touches no C at all.
use std::path::PathBuf;

fn main() {
    // Cargo treats a watched path that does not exist as changed on every run, which rebuilds
    // oxn and all its dependents on every cargo call. The default build therefore watches only
    // this script, and the device backend watches only the paths that exist.
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var_os("CARGO_FEATURE_DEVICE").is_none() {
        return;
    }
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    for watched in [
        "../../npu/oxn/host/oxn_host.c",
        "../../npu/oxn/host/oxn_host.h",
        "../../npu/oxn/common",
    ] {
        let path = root.join(watched);
        if path.exists() {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
    println!("cargo:rerun-if-env-changed=FASTRPC_INC");
    let inc = std::env::var("FASTRPC_INC").unwrap_or_else(|_| {
        panic!(
            "oxn: the `device` feature needs FASTRPC_INC=<dir> (the open-source FastRPC ABI include directory)"
        )
    });
    cc::Build::new()
        .file(root.join("../../npu/oxn/host/oxn_host.c"))
        .include(root.join("../../npu/oxn/common"))
        .include(&inc)
        .opt_level(2)
        .flag("-std=gnu11")
        .warnings(true)
        .compile("oxn_host");
    println!("cargo:rustc-link-lib=dl");
}
