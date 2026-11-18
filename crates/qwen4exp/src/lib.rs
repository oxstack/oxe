//! qwen4exp: Qwen3.8-Flash-Next in the oxe engine. M0 = timing prototype only (see docs/lab-notes/M0-FLASH.md):
//! real weights and kernels, static per-thread plan, expert streaming from an R4 sidecar
//! through an io_uring cache, replayed routing; activations are dummy.

/// Declares kernel entry points of the Flash C/C++ side: `extern "C"` on aarch64 (the objects
/// `build.rs` builds and links); on any other target a stub of the same signature that refuses by
/// panicking, because `build.rs` compiles nothing there and a fabricated result would be a fake
/// successful kernel. Same guard as `kernels-sys` (`ffi_decls!`) and `int8-gemm`
/// (`kernel_decls!`), which this mirrors: same signatures, `extern "C"` on both sides so a Rust
/// function pointer taken to an entry point (`ffi::W::vec_dot`) still type-checks.
macro_rules! kernel_decls {
    ($( $(#[$m:meta])* $vis:vis fn $name:ident ( $($arg:ident : $ty:ty),* $(,)? ) $(-> $ret:ty)? ; )*) => {
        #[cfg(target_arch = "aarch64")]
        extern "C" {
            $( $(#[$m])* $vis fn $name($($arg: $ty),*) $(-> $ret)?; )*
        }
        $(
            #[cfg(not(target_arch = "aarch64"))]
            #[allow(unused_variables, dead_code, clippy::missing_safety_doc, clippy::too_many_arguments)]
            $(#[$m])*
            $vis unsafe extern "C" fn $name($($arg: $ty),*) $(-> $ret)? {
                panic!("qwen4exp: {} is an aarch64 kernel; this build has none", stringify!($name))
            }
        )*
    };
}

pub mod cache;
pub mod cli;
pub mod dense;
pub mod exec;
pub mod ffi;
pub mod graph;
pub mod io;
pub mod lookahead;
pub mod model;
pub mod mtp;
pub mod mtp_engine;
pub mod mtp_graph;
pub mod mtp_head;
pub mod mtp_vocab;
pub mod ngram;
pub mod pf_plan;
pub mod sidecar;
pub mod stream;
pub mod trace;
pub mod uring;

pub const KERNEL_CFLAGS: &str = env!("OXF_KERNEL_CFLAGS");
pub const ENGINE_COMMIT: &str = env!("OXE_GIT_COMMIT");

/// The production archive identity carried by Flash execution receipts.
pub fn ggml_prod_cpu_sha256() -> &'static str {
    #[cfg(target_arch = "aarch64")]
    {
        env!("OXF_GGML_PROD_CPU_SHA")
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        cli::die("the production Flash ggml archive is only available on aarch64")
    }
}
