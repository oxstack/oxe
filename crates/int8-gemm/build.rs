// The micro-kernels are C with NEON intrinsics (SMMLA needs +i8mm; std::arch's vmmlaq_s32 is not stable).
// -ffp-contract=off: the float fix-up order is part of the API contract (see src/lib.rs), so the
// compiler must not fuse a multiply and an add unless the code asks for it (vfmaq_f32).
const CPU_ARCH: &str = "-march=armv8.6-a+dotprod+i8mm+fp16+nosve+nosme";
// F-IMG-ATTN float attention is built on its own so +fp16fml (FMLAL) does not touch the int8
// kernels' flags.
const FATTN_ARCH: &str = "-march=armv8.6-a+dotprod+i8mm+fp16+fp16fml+nosve+nosme";

fn main() {
    // The kernels are aarch64 NEON/SMMLA code, so an -march=armv8.6-a compile is meaningless (and
    // fails) on any other target. Same guard as kernels-sys' build.rs: nothing is compiled here and
    // every entry point of the `raw` modules is a stub that refuses (src/lib.rs `kernel_decls!`), so
    // the crates that link int8-gemm still build for an x86_64 hosted check. aarch64: unchanged.
    if std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() != Ok("aarch64") {
        println!("cargo:rustc-env=OXE_INT8_GEMM_CFLAGS=none (not aarch64: kernels not built)");
        return;
    }
    let dir = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("csrc");
    cc::Build::new()
        .file(dir.join("i8g_kernels.c"))
        .file(dir.join("i8g_sinks.c"))
        .opt_level(3)
        .define("NDEBUG", None)
        .flag("-std=gnu11")
        .flag(CPU_ARCH)
        .flag("-ffp-contract=off")
        .flag("-fno-fast-math")
        .warnings(true)
        .compile("oxe_int8_gemm");
    cc::Build::new()
        .file(dir.join("i8g_fattn.c"))
        .opt_level(3)
        .define("NDEBUG", None)
        .flag("-std=gnu11")
        .flag(FATTN_ARCH)
        .flag("-ffp-contract=off")
        .flag("-fno-fast-math")
        .warnings(true)
        .compile("oxe_int8_gemm_fattn");
    // The flags above are part of the numerics; record them (asserted by the aarch64 test in
    // src/target_arch_tests.rs) so a silent change is visible in reports and in CI.
    println!(
        "cargo:rustc-env=OXE_INT8_GEMM_CFLAGS=i8g_kernels+i8g_sinks: -O3 -DNDEBUG -std=gnu11 {CPU_ARCH} -ffp-contract=off -fno-fast-math; \
i8g_fattn: -O3 -DNDEBUG -std=gnu11 {FATTN_ARCH} -ffp-contract=off -fno-fast-math"
    );
    println!("cargo:rerun-if-changed={}", dir.display());
}
