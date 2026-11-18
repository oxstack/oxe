//! Target-architecture contract of the kernel entry points, the same one `kernels-sys` keeps
//! (`ffi_decls!`): on aarch64 the C kernels built by `build.rs` answer; on any other target every
//! entry point is a stub that refuses instead of returning a fabricated result, because the
//! micro-kernels are NEON/SMMLA C and a silent zero would corrupt the numerics.
//!
//! Model-free and device-free: the entry points tested here have static contracts, so the tests run
//! in hosted CI on both target families (aarch64 kernels, x86_64 stubs).

#[cfg(target_arch = "aarch64")]
#[test]
fn arm_kernels_record_their_compile_arguments() {
    // The flags are part of the numerics (see build.rs): the +i8mm/+fp16 extensions and FMA
    // contraction decide the machine code, and +fp16fml is deliberately kept off the int8 kernels.
    // build.rs records what it passed to cc; asserting the exact string means a flag edit cannot
    // land silently, and that the supported ARM path stayed as it was.
    assert_eq!(
        crate::KERNEL_CFLAGS,
        "i8g_kernels+i8g_sinks: -O3 -DNDEBUG -std=gnu11 \
-march=armv8.6-a+dotprod+i8mm+fp16+nosve+nosme -ffp-contract=off -fno-fast-math; \
i8g_fattn: -O3 -DNDEBUG -std=gnu11 \
-march=armv8.6-a+dotprod+i8mm+fp16+fp16fml+nosve+nosme -ffp-contract=off -fno-fast-math"
    );
}

#[cfg(not(target_arch = "aarch64"))]
mod not_aarch64 {
    //! A panic in an `extern "C"` stub terminates the process, so each entry point runs in its own
    //! copy of this test binary. The parent checks for that entry point's refusal message.
    //!
    //! Every helper below calls one entry point with null operands: the stub refuses before it would
    //! touch one, so no pointer is dereferenced.

    use crate::*;

    const SELECT: &str = "INT8_GEMM_REFUSAL_TEST";

    fn selected() -> Option<String> {
        std::env::var(SELECT).ok()
    }

    fn dot8x8() {
        let x = std::ptr::null::<i8>();
        let o = std::ptr::null_mut::<i32>();
        // SAFETY: null operands, and the callee is the refusing stub on this target.
        unsafe { raw::i8g_dot8x8(x, x, 1, o) }
    }

    fn udot8x8() {
        let x = std::ptr::null::<u8>();
        let w = std::ptr::null::<i8>();
        let o = std::ptr::null_mut::<i32>();
        // SAFETY: null operands; refusing stub.
        unsafe { raw::i8g_udot8x8(x, w, 1, o) }
    }

    fn tile8x8() {
        let p = std::ptr::null::<i8>();
        let f = std::ptr::null::<f32>();
        let y = std::ptr::null_mut::<f32>();
        // SAFETY: null operands; refusing stub.
        unsafe { raw::i8g_tile8x8(p, p, 1, 1, f, f, y, 0, 0, 0) }
    }

    fn utile8x8() {
        let p = std::ptr::null::<u8>();
        let w = std::ptr::null::<i8>();
        let f = std::ptr::null::<f32>();
        let y = std::ptr::null_mut::<f32>();
        // SAFETY: null operands; refusing stub.
        unsafe { raw::i8g_utile8x8(p, w, 1, 1, f, f, y, 0, 0) }
    }

    fn row_max() {
        let s = std::ptr::null::<f32>();
        // SAFETY: null operands; refusing stub.
        unsafe { raw::i8g_row_max(s, 4) };
    }

    fn softmax_pack() {
        let f = std::ptr::null::<f32>();
        let s = std::ptr::null_mut::<f32>();
        let c = std::ptr::null_mut::<u8>();
        // SAFETY: null operands; refusing stub.
        unsafe { raw::i8g_softmax_pack(s, 8, 0.0, f, c, 0, s, s) }
    }

    fn probe_sdot4x4() {
        let x = std::ptr::null::<i8>();
        let o = std::ptr::null_mut::<i32>();
        // SAFETY: null operands; refusing stub.
        unsafe { raw::i8g_probe_sdot4x4(x, x, 1, o) }
    }

    fn rows1() {
        let x = std::ptr::null::<i8>();
        let f = std::ptr::null::<f32>();
        let y = std::ptr::null_mut::<f32>();
        // SAFETY: null operands; refusing stub.
        unsafe { raw::i8g_rows1(x, x, 1, 1, f, f, y, 0) }
    }

    fn kernel_abi() {
        // SAFETY: refusing stub.
        unsafe { raw::i8g_kernel_abi() };
    }

    fn quant_chunk_raw() {
        let x = std::ptr::null::<f32>();
        let dst = std::ptr::null_mut::<i8>();
        // SAFETY: null operands; refusing stub.
        unsafe { sinks::raw::i8g_quant_chunk(x, 8, dst, 8) };
    }

    fn silu_gate_raw() {
        let row = std::ptr::null::<f32>();
        let out = std::ptr::null_mut::<f32>();
        // SAFETY: null operands; refusing stub.
        unsafe { sinks::raw::i8g_silu_gate(row, 8, out) }
    }

    fn rope_scale_raw() {
        let f = std::ptr::null::<f32>();
        let out = std::ptr::null_mut::<f32>();
        // SAFETY: null operands; refusing stub.
        unsafe { sinks::raw::i8g_rope_scale(f, 8, 1.0, f, f, out) }
    }

    fn fattn_qk_f32() {
        let q = std::ptr::null::<f32>();
        let s = std::ptr::null_mut::<f32>();
        // SAFETY: null operands; refusing stub.
        unsafe { fattn::raw::i8g_fattn_qk_f32(q, q, 8, 1, s, 8) }
    }

    fn fattn_qk_f16() {
        let q = std::ptr::null::<u16>();
        let s = std::ptr::null_mut::<f32>();
        // SAFETY: null operands; refusing stub.
        unsafe { fattn::raw::i8g_fattn_qk_f16(q, q, 8, 1, s, 8) }
    }

    fn fattn_softmax_f32() {
        let s = std::ptr::null::<f32>();
        let p = std::ptr::null_mut::<f32>();
        // SAFETY: null operands; refusing stub.
        unsafe { fattn::raw::i8g_fattn_softmax_f32(s, 8, 8, 0.0, p, 0) };
    }

    fn fattn_softmax_f16() {
        let s = std::ptr::null::<f32>();
        let p = std::ptr::null_mut::<u16>();
        // SAFETY: null operands; refusing stub.
        unsafe { fattn::raw::i8g_fattn_softmax_f16(s, 8, 8, 0.0, p, 0) };
    }

    fn fattn_pv_f32() {
        let p = std::ptr::null::<f32>();
        let o = std::ptr::null_mut::<f32>();
        // SAFETY: null operands; refusing stub.
        unsafe { fattn::raw::i8g_fattn_pv_f32(p, p, 8, 8, 8, o) }
    }

    fn fattn_pv_f16() {
        let p = std::ptr::null::<u16>();
        let o = std::ptr::null_mut::<f32>();
        // SAFETY: null operands; refusing stub.
        unsafe { fattn::raw::i8g_fattn_pv_f16(p, p, 8, 8, 8, o) }
    }

    fn fattn_to_f16() {
        let x = std::ptr::null::<f32>();
        let d = std::ptr::null_mut::<u16>();
        // SAFETY: null operands; refusing stub.
        unsafe { fattn::raw::i8g_fattn_to_f16(x, 8, d, 8) }
    }

    fn fattn_from_f16() {
        let s = std::ptr::null::<u16>();
        let x = std::ptr::null_mut::<f32>();
        // SAFETY: null operands; refusing stub.
        unsafe { fattn::raw::i8g_fattn_from_f16(s, 8, x) }
    }

    fn packed_from_f32() {
        // The safe API reaches the same stub; no block is fabricated.
        let rows = [1.0f32; 32];
        let _ = Packed::from_f32(&rows, 1, 32, 32);
    }

    fn safe_quant_chunk() {
        let x = [1.0f32; 8];
        let mut dst = [0i8; 8];
        let _ = sinks::quant_chunk(&x, &mut dst, 8);
    }

    const CASES: &[(&str, fn(), &str)] = &[
        ("dot8x8", dot8x8, "i8g_dot8x8"),
        ("udot8x8", udot8x8, "i8g_udot8x8"),
        ("tile8x8", tile8x8, "i8g_tile8x8"),
        ("utile8x8", utile8x8, "i8g_utile8x8"),
        ("row_max", row_max, "i8g_row_max"),
        ("softmax_pack", softmax_pack, "i8g_softmax_pack"),
        ("probe_sdot4x4", probe_sdot4x4, "i8g_probe_sdot4x4"),
        ("rows1", rows1, "i8g_rows1"),
        ("kernel_abi", kernel_abi, "i8g_kernel_abi"),
        ("quant_chunk_raw", quant_chunk_raw, "i8g_quant_chunk"),
        ("silu_gate_raw", silu_gate_raw, "i8g_silu_gate"),
        ("rope_scale_raw", rope_scale_raw, "i8g_rope_scale"),
        ("fattn_qk_f32", fattn_qk_f32, "i8g_fattn_qk_f32"),
        ("fattn_qk_f16", fattn_qk_f16, "i8g_fattn_qk_f16"),
        (
            "fattn_softmax_f32",
            fattn_softmax_f32,
            "i8g_fattn_softmax_f32",
        ),
        (
            "fattn_softmax_f16",
            fattn_softmax_f16,
            "i8g_fattn_softmax_f16",
        ),
        ("fattn_pv_f32", fattn_pv_f32, "i8g_fattn_pv_f32"),
        ("fattn_pv_f16", fattn_pv_f16, "i8g_fattn_pv_f16"),
        ("fattn_to_f16", fattn_to_f16, "i8g_fattn_to_f16"),
        ("fattn_from_f16", fattn_from_f16, "i8g_fattn_from_f16"),
        ("packed_from_f32", packed_from_f32, "i8g_quant_chunk"),
        ("safe_quant_chunk", safe_quant_chunk, "i8g_quant_chunk"),
    ];

    #[test]
    fn unsupported_target_refuses_every_entry_point() {
        assert!(
            KERNEL_CFLAGS.starts_with("none"),
            "build.rs must record that no kernel was built: {KERNEL_CFLAGS}"
        );
        if let Some(name) = selected() {
            let (_, run, _) = CASES
                .iter()
                .find(|(case, _, _)| *case == name.as_str())
                .expect("unknown refusal test case");
            run();
            panic!("{name}: the entry point returned on a target that has no kernels");
        }

        let exe = std::env::current_exe().expect("current_exe");
        let name = "target_arch_tests::not_aarch64::unsupported_target_refuses_every_entry_point";
        for &(case, _, symbol) in CASES {
            let out = std::process::Command::new(&exe)
                .args(["--exact", name, "--nocapture", "--test-threads=1"])
                .env(SELECT, case)
                .output()
                .expect("run this test binary");
            let text = format!(
                "{}{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
            // The child must have run this test (a renamed test would select nothing and exit 0).
            assert!(
                text.contains("running 1 test"),
                "{case}: the child ran no test; the name in this file is stale:\n{text}"
            );
            assert!(
                !out.status.success(),
                "{case}: an entry point answered on a target that has no kernels:\n{text}"
            );
            assert!(
                text.contains(&format!("int8-gemm: {symbol} is an aarch64 kernel")),
                "{case}: the refusal message is missing from the child's output:\n{text}"
            );
        }
    }
}
