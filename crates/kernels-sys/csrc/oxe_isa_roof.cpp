// v0.1.4 ISA-ROOF (V014-DIRECTION item 1): model-free microbenchmark kernels for `isa-roof`
// (crates/oxe-cli/src/bin/isa-roof.rs). Nothing here is used by the engine.
//
// 1. Register-only loops (oxe_isa_loop): 24 independent accumulators v8..v31 (no dependency
//    chain: 24 >= latency x pipes for any plausible core), sources v0/v1 (v2..v7 for LD1R), one
//    loop iteration = 24 instructions of the kind (the LD1R kind adds 12 LD1R). Operands start at
//    values that stay finite and normal (f16 1.0 x 0.125 saturates near 512, never inf / NaN).
//    The caller detects the CPU features at run time and never calls a kind the CPU lacks
//    (bf16 / i8mm / fhm), so the SIGILL never happens.
// 2. f32 exp: ggml_v_expf (the kernel the softmax of every prefill path uses) over an L1 buffer.
// 3. GEMM tiles (oxe_isa_tile_*): the candidate prefill-attention micro-kernels on packed
//    L1-resident operands (panels = 1) or with the 8-row B panel streamed from L2 (panels > 1, the
//    QK shape: Q^T in L1, K rows from L2):
//      f16 8x16 FMLA by element (the fa2x micro-kernel, fold 0 = pure f16 / 8 / 32 into f32),
//      f16 8x16 LD1R + FMLA (pre-broadcast B, packed [k][8]) fold 32,
//      bf16 8x8 BFMMLA (packed 2x4 blocks, f32 accumulation: no fold).
//    Each has a scalar self-test against f64 on the same rounded inputs.
// 4. Cycle counter: perf_event_open(PERF_COUNT_HW_CPU_CYCLES, user only) for the calling thread
//    (PMCCNTR_EL0 is not readable from EL0 on Linux); CNTFRQ_EL0 for the fixed-frequency timer.
#include "ggml-cpu/vec.h"

#include <arm_neon.h>
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace {

#define ISA_ARCH ".arch armv8.6-a+fp16+fp16fml+bf16+i8mm+dotprod\n\t"
#define R24(M) M(8) M(9) M(10) M(11) M(12) M(13) M(14) M(15) M(16) M(17) M(18) M(19) \
               M(20) M(21) M(22) M(23) M(24) M(25) M(26) M(27) M(28) M(29) M(30) M(31)
#define ISA_ZERO(n) "movi v" #n ".16b, #0\n\t"
#define ISA_CLOBBER "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", \
    "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", \
    "v29", "v30", "v31"

// f16 1.0 / 0.125, bf16 0.5 / 0.125, f32 1.0 / 0.125, int8 1 / 1
#define INIT_F16 "movi v0.8h, #0x3c, lsl #8\n\tmovi v1.8h, #0x30, lsl #8\n\t"
#define INIT_BF16 "movi v0.8h, #0x3f, lsl #8\n\tmovi v1.8h, #0x3e, lsl #8\n\t"
#define INIT_F32 "fmov v0.4s, #1.0\n\tfmov v1.4s, #0.125\n\t"
#define INIT_I8 "movi v0.16b, #1\n\tmovi v1.16b, #1\n\t"

#define ISA_LOOP(INIT, OP)                                                                              \
    __asm__ volatile(ISA_ARCH INIT R24(ISA_ZERO) "1:\n\t" R24(OP) "subs %x[n], %x[n], #1\n\tb.ne 1b\n\t" \
                     : [n] "+r"(n)                                                                      \
                     :                                                                                  \
                     : ISA_CLOBBER)

#define OP_F32_FMLA(n) "fmla v" #n ".4s, v0.4s, v1.4s\n\t"
#define OP_F16_FMLA(n) "fmla v" #n ".8h, v0.8h, v1.8h\n\t"
#define LANE_8 "0"
#define LANE_9 "1"
#define LANE_10 "2"
#define LANE_11 "3"
#define LANE_12 "4"
#define LANE_13 "5"
#define LANE_14 "6"
#define LANE_15 "7"
#define LANE_16 "0"
#define LANE_17 "1"
#define LANE_18 "2"
#define LANE_19 "3"
#define LANE_20 "4"
#define LANE_21 "5"
#define LANE_22 "6"
#define LANE_23 "7"
#define LANE_24 "0"
#define LANE_25 "1"
#define LANE_26 "2"
#define LANE_27 "3"
#define LANE_28 "4"
#define LANE_29 "5"
#define LANE_30 "6"
#define LANE_31 "7"
#define OP_F16_FMLA_ELEM(n) "fmla v" #n ".8h, v0.8h, v1.h[" LANE_##n "]\n\t"
#define OP_F16_FMLA_CHAIN(n) "fmla v8.8h, v0.8h, v1.8h\n\t"
#define OP_F16_FMLAL(n) "fmlal v" #n ".4s, v0.4h, v1.4h\n\t"
#define OP_BFDOT(n) "bfdot v" #n ".4s, v0.8h, v1.8h\n\t"
#define OP_BFMMLA(n) "bfmmla v" #n ".4s, v0.8h, v1.8h\n\t"
#define OP_BFMLALB(n) "bfmlalb v" #n ".4s, v0.8h, v1.8h\n\t"
#define OP_SDOT(n) "sdot v" #n ".4s, v0.16b, v1.16b\n\t"
#define OP_SMMLA(n) "smmla v" #n ".4s, v0.16b, v1.16b\n\t"
#define OP_FCVTL(n) "fcvtl v" #n ".4s, v0.4h\n\t"
#define OP_FCVTN(n) "fcvtn v" #n ".4h, v0.4s\n\t"
#define OP_BFCVTN(n) "bfcvtn v" #n ".4h, v0.4s\n\t"
#define OP_SHLL16(n) "shll v" #n ".4s, v0.4h, #16\n\t"

#define LD1R_PAIR(a, b, r) "ld1r {v" #r ".8h}, [%x[p]]\n\tfmla v" #a ".8h, v0.8h, v" #r ".8h\n\tfmla v" #b ".8h, v1.8h, v" #r ".8h\n\t"

// f16 helpers for the tiles
constexpr int T_MR = 8;
constexpr int T_NR = 16;

template <int L, bool FIRST>
static inline void t_step(float16x8_t (&acc)[T_MR][2], const float16x8_t (&b)[T_MR], const __fp16 * a) {
    const float16x8_t a0 = vld1q_f16(a);
    const float16x8_t a1 = vld1q_f16(a + 8);
    for (int i = 0; i < T_MR; i++) {
        if (FIRST) {
            acc[i][0] = vmulq_laneq_f16(a0, b[i], L);
            acc[i][1] = vmulq_laneq_f16(a1, b[i], L);
        } else {
            acc[i][0] = vfmaq_laneq_f16(acc[i][0], a0, b[i], L);
            acc[i][1] = vfmaq_laneq_f16(acc[i][1], a1, b[i], L);
        }
    }
}

// 8 consecutive k by element: B rows i (stride K) hold k contiguous, A rows k hold 16 columns
template <bool FIRST>
static inline void t_k8_elem(float16x8_t (&acc)[T_MR][2], const __fp16 * B, int K, const __fp16 * A, int k) {
    float16x8_t b[T_MR];
    for (int i = 0; i < T_MR; i++) {
        b[i] = vld1q_f16(B + (int64_t) i * K + k);
    }
    const __fp16 * a = A + (int64_t) k * T_NR;
    t_step<0, FIRST>(acc, b, a);
    t_step<1, false>(acc, b, a + 1 * T_NR);
    t_step<2, false>(acc, b, a + 2 * T_NR);
    t_step<3, false>(acc, b, a + 3 * T_NR);
    t_step<4, false>(acc, b, a + 4 * T_NR);
    t_step<5, false>(acc, b, a + 5 * T_NR);
    t_step<6, false>(acc, b, a + 6 * T_NR);
    t_step<7, false>(acc, b, a + 7 * T_NR);
}

// 8 consecutive k, B pre-broadcast with LD1R from a [k][8] packing
template <bool FIRST>
static inline void t_k8_dup(float16x8_t (&acc)[T_MR][2], const __fp16 * Bp, const __fp16 * A, int k) {
    for (int kk = 0; kk < 8; kk++) {
        const __fp16 * a = A + (int64_t) (k + kk) * T_NR;
        const __fp16 * b = Bp + (int64_t) (k + kk) * T_MR;
        const float16x8_t a0 = vld1q_f16(a);
        const float16x8_t a1 = vld1q_f16(a + 8);
        for (int i = 0; i < T_MR; i++) {
            const float16x8_t bi = vld1q_dup_f16(b + i);
            if (FIRST && kk == 0) {
                acc[i][0] = vmulq_f16(a0, bi);
                acc[i][1] = vmulq_f16(a1, bi);
            } else {
                acc[i][0] = vfmaq_f16(acc[i][0], a0, bi);
                acc[i][1] = vfmaq_f16(acc[i][1], a1, bi);
            }
        }
    }
}

// C[8][16] (f32) = sum_k B[i][k] A[k][c]: f16 chains of FOLD (0 = the whole K) folded into f32
template <int FOLD, bool DUP>
static void t_f16_8x16(int K, const __fp16 * B, const __fp16 * A, float * C) {
    const int fold = FOLD == 0 ? K : FOLD;
    for (int k0 = 0; k0 < K; k0 += fold) {
        float16x8_t acc[T_MR][2];
        if (DUP) {
            t_k8_dup<true>(acc, B, A, k0);
        } else {
            t_k8_elem<true>(acc, B, K, A, k0);
        }
        for (int k = k0 + 8; k < k0 + fold; k += 8) {
            if (DUP) {
                t_k8_dup<false>(acc, B, A, k);
            } else {
                t_k8_elem<false>(acc, B, K, A, k);
            }
        }
        for (int i = 0; i < T_MR; i++) {
            float * c = C + i * T_NR;
            for (int h = 0; h < 2; h++) {
                const float32x4_t lo = vcvt_f32_f16(vget_low_f16(acc[i][h]));
                const float32x4_t hi = vcvt_high_f32_f16(acc[i][h]);
                if (k0 == 0) {
                    vst1q_f32(c + 8 * h, lo);
                    vst1q_f32(c + 8 * h + 4, hi);
                } else {
                    vst1q_f32(c + 8 * h, vaddq_f32(vld1q_f32(c + 8 * h), lo));
                    vst1q_f32(c + 8 * h + 4, vaddq_f32(vld1q_f32(c + 8 * h + 4), hi));
                }
            }
        }
    }
}

// C[8][8] (f32) = sum_k A[i][k] B[j][k] with BFMMLA: A / B packed per k-block of 4 as 4 pairs of
// rows x 4 k (8 bf16 = one register per pair), f32 accumulation in 16 registers (2x2 each)
static void t_bf16_8x8(int K, const bfloat16_t * Ap, const bfloat16_t * Bp, float * C) {
    float32x4_t acc[4][4];
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            acc[r][c] = vdupq_n_f32(0.0f);
        }
    }
    for (int kb = 0; kb < K / 4; kb++) {
        const bfloat16_t * a = Ap + (int64_t) kb * 32;
        const bfloat16_t * b = Bp + (int64_t) kb * 32;
        const bfloat16x8_t a0 = vld1q_bf16(a), a1 = vld1q_bf16(a + 8), a2 = vld1q_bf16(a + 16), a3 = vld1q_bf16(a + 24);
        const bfloat16x8_t b0 = vld1q_bf16(b), b1 = vld1q_bf16(b + 8), b2 = vld1q_bf16(b + 16), b3 = vld1q_bf16(b + 24);
        const bfloat16x8_t av[4] = {a0, a1, a2, a3};
        const bfloat16x8_t bv[4] = {b0, b1, b2, b3};
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                acc[r][c] = vbfmmlaq_f32(acc[r][c], av[r], bv[c]);
            }
        }
    }
    // acc[r][c] lanes: (2r, 2c), (2r, 2c+1), (2r+1, 2c), (2r+1, 2c+1)
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            float v[4];
            vst1q_f32(v, acc[r][c]);
            C[(2 * r) * 8 + 2 * c] = v[0];
            C[(2 * r) * 8 + 2 * c + 1] = v[1];
            C[(2 * r + 1) * 8 + 2 * c] = v[2];
            C[(2 * r + 1) * 8 + 2 * c + 1] = v[3];
        }
    }
}

static inline uint16_t f32_to_bf16_rne(float x) {
    uint32_t u;
    memcpy(&u, &x, 4);
    u += 0x7fff + ((u >> 16) & 1);
    return (uint16_t) (u >> 16);
}
static inline float bf16_to_f32(uint16_t h) {
    uint32_t u = (uint32_t) h << 16;
    float x;
    memcpy(&x, &u, 4);
    return x;
}

enum TileKind { T_ELEM_NOFOLD = 0, T_ELEM_FOLD8 = 1, T_ELEM_FOLD32 = 2, T_DUP_FOLD32 = 3, T_BFMMLA = 4, T_KINDS = 5 };

struct IsaTile {
    int kind, K, panels;
    void * raw;
    __fp16 * A;       // [K][16] f16 (elem / dup)
    __fp16 * B;       // panels x (elem: [8][K], dup: [K][8]) f16
    bfloat16_t * Ab;  // panels x [K/4][4][8] bf16 (the streamed operand, as B above)
    bfloat16_t * Bb;  // [K/4][4][8] bf16 (the L1 operand)
    float * C;        // [8][16]
    float sink;
};

static uint64_t lcg_next(uint64_t & s) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return s >> 33;
}
static float lcg_f(uint64_t & s) {
    return (float) (lcg_next(s) % 20001) / 10000.0f - 1.0f;
}

}  // namespace

extern "C" {

// Register-only loop `kind` for `iters` iterations (see the header); returns 0, -1 unknown kind.
int oxe_isa_loop(int kind, uint64_t iters) {
    if (iters == 0) {
        return 0;
    }
    uint64_t n = iters;
    switch (kind) {
        case 0: ISA_LOOP(INIT_F32, OP_F32_FMLA); break;
        case 1: ISA_LOOP(INIT_F16, OP_F16_FMLA); break;
        case 2: ISA_LOOP(INIT_F16, OP_F16_FMLA_ELEM); break;
        case 3: {
            static const __fp16 src[8] = {(__fp16) 0.125f, (__fp16) 0.125f, (__fp16) 0.125f, (__fp16) 0.125f,
                                          (__fp16) 0.125f, (__fp16) 0.125f, (__fp16) 0.125f, (__fp16) 0.125f};
            const __fp16 * p = src;
            __asm__ volatile(ISA_ARCH INIT_F16 R24(ISA_ZERO) "1:\n\t"
                LD1R_PAIR(8, 9, 2) LD1R_PAIR(10, 11, 3) LD1R_PAIR(12, 13, 4) LD1R_PAIR(14, 15, 5)
                LD1R_PAIR(16, 17, 6) LD1R_PAIR(18, 19, 7) LD1R_PAIR(20, 21, 2) LD1R_PAIR(22, 23, 3)
                LD1R_PAIR(24, 25, 4) LD1R_PAIR(26, 27, 5) LD1R_PAIR(28, 29, 6) LD1R_PAIR(30, 31, 7)
                "subs %x[n], %x[n], #1\n\tb.ne 1b\n\t"
                : [n] "+r"(n)
                : [p] "r"(p)
                : ISA_CLOBBER);
            break;
        }
        case 4: ISA_LOOP(INIT_F16, OP_F16_FMLAL); break;
        case 5: ISA_LOOP(INIT_BF16, OP_BFDOT); break;
        case 6: ISA_LOOP(INIT_BF16, OP_BFMMLA); break;
        case 7: ISA_LOOP(INIT_BF16, OP_BFMLALB); break;
        case 8: ISA_LOOP(INIT_I8, OP_SDOT); break;
        case 9: ISA_LOOP(INIT_I8, OP_SMMLA); break;
        case 10: ISA_LOOP(INIT_F16, OP_FCVTL); break;
        case 11: ISA_LOOP(INIT_F32, OP_FCVTN); break;
        case 12: ISA_LOOP(INIT_F32, OP_BFCVTN); break;
        case 13: ISA_LOOP(INIT_BF16, OP_SHLL16); break;
        case 14: ISA_LOOP(INIT_F16, OP_F16_FMLA_CHAIN); break;
        default: return -1;
    }
    return 0;
}

// ggml_v_expf over a 256-float L1 buffer (inputs in [-20, 0]: the softmax range), `iters` times
// (256 exps each, 4 independent vectors in flight). Returns a checksum.
float oxe_isa_exp(uint64_t iters) {
    alignas(64) float in[256];
    alignas(64) float out[256];
    for (int i = 0; i < 256; i++) {
        in[i] = -20.0f * (float) i / 255.0f;
    }
    float32x4_t s = vdupq_n_f32(0.0f);
    for (uint64_t it = 0; it < iters; it++) {
        for (int i = 0; i < 256; i += 16) {
            const float32x4_t x0 = ggml_v_expf(vld1q_f32(in + i));
            const float32x4_t x1 = ggml_v_expf(vld1q_f32(in + i + 4));
            const float32x4_t x2 = ggml_v_expf(vld1q_f32(in + i + 8));
            const float32x4_t x3 = ggml_v_expf(vld1q_f32(in + i + 12));
            vst1q_f32(out + i, x0);
            vst1q_f32(out + i + 4, x1);
            vst1q_f32(out + i + 8, x2);
            vst1q_f32(out + i + 12, x3);
        }
        s = vaddq_f32(s, vld1q_f32(out + (it & 63) * 4));
        __asm__ volatile("" : : "r"(out) : "memory");
    }
    return vaddvq_f32(s);
}

// GEMM tile `kind` (TileKind) with depth K (multiple of 32) and `panels` streamed panels (1 = all
// operands L1-resident). NULL on a bad argument.
void * oxe_isa_tile_new(int kind, int K, int panels) {
    if (kind < 0 || kind >= T_KINDS || K < 32 || K % 32 || panels < 1 || panels > 4096) {
        return nullptr;
    }
    IsaTile * t = (IsaTile *) calloc(1, sizeof(IsaTile));
    if (!t) {
        return nullptr;
    }
    t->kind = kind;
    t->K = K;
    t->panels = panels;
    const size_t a16 = (size_t) K * T_NR * 2, b16 = (size_t) panels * K * T_MR * 2, abf = (size_t) panels * K * 8 * 2,
                 bbf = (size_t) K * 8 * 2, c = 8 * 16 * 4;
    const size_t total = a16 + b16 + abf + bbf + c + 5 * 64;
    t->raw = aligned_alloc(64, (total + 63) & ~(size_t) 63);
    if (!t->raw) {
        free(t);
        return nullptr;
    }
    uint8_t * p = (uint8_t *) t->raw;
    auto take = [&](size_t n) { uint8_t * r = p; p += (n + 63) & ~(size_t) 63; return r; };
    t->A = (__fp16 *) take(a16);
    t->B = (__fp16 *) take(b16);
    t->Ab = (bfloat16_t *) take(abf);
    t->Bb = (bfloat16_t *) take(bbf);
    t->C = (float *) take(c);
    uint64_t s = 0x15a7 + (uint64_t) kind;
    for (size_t i = 0; i < (size_t) K * T_NR; i++) {
        t->A[i] = (__fp16) lcg_f(s);
    }
    for (size_t i = 0; i < (size_t) panels * K * T_MR; i++) {
        t->B[i] = (__fp16) lcg_f(s);
    }
    uint16_t * ab = (uint16_t *) t->Ab;
    uint16_t * bb = (uint16_t *) t->Bb;
    for (size_t i = 0; i < (size_t) panels * K * 8; i++) {
        ab[i] = f32_to_bf16_rne(lcg_f(s));
    }
    for (size_t i = 0; i < (size_t) K * 8; i++) {
        bb[i] = f32_to_bf16_rne(lcg_f(s));
    }
    return t;
}

void oxe_isa_tile_free(void * h) {
    IsaTile * t = (IsaTile *) h;
    if (t) {
        free(t->raw);
        free(t);
    }
}

// Multiply-accumulates of one oxe_isa_tile_run repetition (every panel once).
double oxe_isa_tile_fma(void * h) {
    const IsaTile * t = (const IsaTile *) h;
    const double per = t->kind == T_BFMMLA ? 8.0 * 8.0 : 8.0 * 16.0;
    return per * t->K * t->panels;
}

static void tile_once(IsaTile * t, int panel) {
    const int K = t->K;
    const __fp16 * B = t->B + (size_t) panel * K * T_MR;
    switch (t->kind) {
        case T_ELEM_NOFOLD: t_f16_8x16<0, false>(K, B, t->A, t->C); break;
        case T_ELEM_FOLD8: t_f16_8x16<8, false>(K, B, t->A, t->C); break;
        case T_ELEM_FOLD32: t_f16_8x16<32, false>(K, B, t->A, t->C); break;
        case T_DUP_FOLD32: t_f16_8x16<32, true>(K, B, t->A, t->C); break;
        case T_BFMMLA: t_bf16_8x8(K, t->Ab + (size_t) panel * K * 8, t->Bb, t->C); break;
    }
}

// `reps` repetitions over every panel; returns a checksum.
float oxe_isa_tile_run(void * h, uint64_t reps) {
    IsaTile * t = (IsaTile *) h;
    for (uint64_t r = 0; r < reps; r++) {
        for (int p = 0; p < t->panels; p++) {
            tile_once(t, p);
            __asm__ volatile("" : : "r"(t->C) : "memory");
        }
    }
    t->sink += t->C[0];
    return t->sink;
}

// Self-test of tile `kind` (panel 0 of a fresh tile) against f64 on the same rounded inputs:
// max |C - ref| / max |ref|; -1 on a bad argument.
double oxe_isa_tile_selftest(int kind, int K) {
    IsaTile * t = (IsaTile *) oxe_isa_tile_new(kind, K, 2);
    if (!t) {
        return -1.0;
    }
    tile_once(t, 1);
    double err = 0.0, mag = 0.0;
    if (kind == T_BFMMLA) {
        const uint16_t * ab = (const uint16_t *) t->Ab + (size_t) K * 8;  // panel 1
        const uint16_t * bb = (const uint16_t *) t->Bb;
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                double r = 0.0;
                for (int k = 0; k < K; k++) {
                    // element (row i, k) at kb * 32 + (i / 2) * 8 + (i % 2) * 4 + k % 4
                    const size_t ai = (size_t) (k / 4) * 32 + (i / 2) * 8 + (i % 2) * 4 + k % 4;
                    const size_t bi = (size_t) (k / 4) * 32 + (j / 2) * 8 + (j % 2) * 4 + k % 4;
                    r += (double) bf16_to_f32(ab[ai]) * (double) bf16_to_f32(bb[bi]);
                }
                err = fmax(err, fabs((double) t->C[i * 8 + j] - r));
                mag = fmax(mag, fabs(r));
            }
        }
    } else {
        const __fp16 * B = t->B + (size_t) K * T_MR;  // panel 1
        for (int i = 0; i < 8; i++) {
            for (int c = 0; c < 16; c++) {
                double r = 0.0;
                for (int k = 0; k < K; k++) {
                    const double b = kind == T_DUP_FOLD32 ? (double) B[(size_t) k * 8 + i] : (double) B[(size_t) i * K + k];
                    r += b * (double) t->A[(size_t) k * 16 + c];
                }
                err = fmax(err, fabs((double) t->C[i * 16 + c] - r));
                mag = fmax(mag, fabs(r));
            }
        }
    }
    oxe_isa_tile_free(t);
    return mag > 0.0 ? err / mag : err;
}

// CPU cycles of the calling thread (user mode), or -errno when the PMU event is not available.
int oxe_isa_cycles_open(void) {
    struct perf_event_attr a;
    memset(&a, 0, sizeof a);
    a.type = PERF_TYPE_HARDWARE;
    a.size = sizeof a;
    a.config = PERF_COUNT_HW_CPU_CYCLES;
    a.exclude_kernel = 1;
    a.exclude_hv = 1;
    const long fd = syscall(__NR_perf_event_open, &a, 0, -1, -1, 0);
    return fd < 0 ? -errno : (int) fd;
}

int64_t oxe_isa_cycles_read(int fd) {
    uint64_t v = 0;
    if (fd < 0 || read(fd, &v, sizeof v) != (ssize_t) sizeof v) {
        return -1;
    }
    return (int64_t) v;
}

void oxe_isa_cycles_close(int fd) {
    if (fd >= 0) {
        close(fd);
    }
}

// CNTFRQ_EL0: the frequency of the generic timer (CNTVCT), NOT the core clock.
uint64_t oxe_isa_cntfrq(void) {
    uint64_t f;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(f));
    return f;
}

}  // extern "C"
