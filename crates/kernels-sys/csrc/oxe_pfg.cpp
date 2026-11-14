// v0.1.4 PFG path A ("pfg-f16", V014-DIRECTION section 1.6 A): the GEMM-shaped CPU prefill
// attention unit for the f16 KV cache (hand-written, not upstream). One call computes one unit:
// ncol query columns of ONE KV head (in the engine: a block of 16 tokens x the 8 query heads of
// the group = 128 columns, the GQA 8:1 reuse; every K / V^T element of a chunk is packed once and
// used by all ncol columns) against positions [0, last + 1), walking fixed absolute chunks of `ch`
// positions with the flash-attention-2 online softmax.
//
// Per unit: Q^T packed once into 16-column panels QT[cb][k][16] (f16, pad columns 0; k in the QK
// head-dim order below).
// Per chunk [s0, s0 + ch) with nr valid rows (positions <= last) and klen = round_up(nr, 8):
//   [pack] K rows -> KP (8-row panels), V^T rows -> VP (8-dim panels); rows >= nr are packed as
//          +0 (K and V), so nothing beyond the unit's last position is ever read, and a column's
//          result does not depend on how far the unit reaches (a zero weight times a zero value);
//   [QK]   S[cb][r][16] (f32) = K[s0 + r] . q_c                 (8x16 micro-kernel, k = head dim)
//   [soft] per column (vertical over rows, 4 columns per f32 vector): m' = max(m, max_r S*scale)
//          over the causal rows r <= pos_c - s0, P = f16(exp(S*scale - m')) (+0 when masked),
//          alpha = exp(m - m'), l = l * alpha + sum_r f32(P); a column with no valid row in the
//          chunk (pos_c < s0) keeps O, m and l untouched (select, not arithmetic)
//   [PV]   O[cb][d][16] = valid ? O * alpha + T : O,  T = V^T[d][s0 ..] . P[.][c]   (8x16 micro-kernel,
//          k = position)
// and out[c][d] = O * (1 / l) at the end.
//
// Micro-kernel (both products): T[i][c] = sum_k B[i][k] * A[k][c], 8 rows i x 16 columns c,
// 16 f16x8 accumulators. A = QT / P panels [k][16] (2 vector loads per k). B = packed panel:
//   mk 0 (LD1R, the ISA-ROOF winner: 0.770 TFMA/s L1 tile on 12 Oryon cores): B[k][8 rows],
//        each B[i][k] broadcast by LD1R, plain FMLA.8h;
//   mk 1 (by element): B[k/8][8 rows][8 k], one vector per row per 8 k, FMLA.8h by lane.
// Both packings put the 64 values of 8 k x 8 rows in one contiguous 128-byte block.
//
// Numerics (class of O-70 / fa2 / fa2x: f16 products, f32 combination; NOT the ggml FA-off order):
// every product accumulates in f16 over `fold` consecutive k (16 or 32; separate folds for QK and
// PV), then the f16 partial is added to an f32 accumulator (FMLAL by 1.0: the f16 -> f32 widening
// is exact, so this equals vcvt + vaddq bit for bit). Fold boundaries are absolute (head dims for
// QK, positions for PV: s0 is a multiple of ch and ch of fold), chunk boundaries are absolute, and
// every column's arithmetic only involves its own query, mask and running max: a column's output
// is the same bits in any unit, at any thread count.
// QK head-dim order: k = j * (hd / 32) + s holds head dim 32 s + j (j < 32), so the f16 chain
// of 8 consecutive k is exactly one lane chain of ggml's f16 dot (dims j, j + 32, .., the tiled
// kernel's order) and a fold of 16 / 32 joins 2 / 4 such chains: a block of consecutive outlier
// channels (massive activations; NeoX RoPE keeps the low-frequency dims of a rotary half together)
// is spread over the chains as in the tiled kernel instead of landing in one chain.
// The packing does it for free (the same 8x8 transposes, other store addresses).
//
// Profile (optional `prof`, 5 counters of CNTVCT ticks, accumulated): [0] packing (Q^T, K, V^T),
// [1] QK, [2] softmax, [3] PV, [4] normalise / transpose out. oxe_pfg_tick_hz() gives the rate.
#include "ggml-cpu/vec.h"

#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace {

constexpr int PG_MR = 8;   // rows per micro tile
constexpr int PG_NR = 16;  // columns per micro tile (2 x f16x8)
constexpr int PG_MAXCOL = 256;
constexpr int PG_MAXCH = 256;

static inline int64_t pg_al(int64_t b) { return (b + 63) & ~(int64_t) 63; }

static inline uint64_t pg_tick() {
    uint64_t v;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(v));
    return v;
}

// c + f32(a.lo) / c + f32(a.hi): exact widening then one rounding, equal to vaddq(c, vcvt(a)).
static inline float32x4_t pg_fold_lo(float32x4_t c, float16x8_t a) {
#if defined(__ARM_FEATURE_FP16_FML)
    return vfmlalq_low_f16(c, a, vdupq_n_f16((__fp16) 1.0f));
#else
    return vaddq_f32(c, vcvt_f32_f16(vget_low_f16(a)));
#endif
}
static inline float32x4_t pg_fold_hi(float32x4_t c, float16x8_t a) {
#if defined(__ARM_FEATURE_FP16_FML)
    return vfmlalq_high_f16(c, a, vdupq_n_f16((__fp16) 1.0f));
#else
    return vaddq_f32(c, vcvt_high_f32_f16(a));
#endif
}

// ---- mk 0: LD1R. b = 8 rows of one k ([k][8] packing), a = the A row of 16 columns.
template <bool FIRST>
static inline void pg_k1_dup(float16x8_t (&acc)[PG_MR][2], const __fp16 * b, const __fp16 * a) {
    const float16x8_t a0 = vld1q_f16(a);
    const float16x8_t a1 = vld1q_f16(a + 8);
    for (int i = 0; i < PG_MR; i++) {
        const float16x8_t bi = vld1q_dup_f16(b + i);
        if (FIRST) {
            acc[i][0] = vmulq_f16(a0, bi);
            acc[i][1] = vmulq_f16(a1, bi);
        } else {
            acc[i][0] = vfmaq_f16(acc[i][0], a0, bi);
            acc[i][1] = vfmaq_f16(acc[i][1], a1, bi);
        }
    }
}

// 8 consecutive k: bp = the 64-value block [kk][8 rows], a = A rows k .. k + 8 ([k][16])
template <bool FIRST>
static inline void pg_k8_dup(float16x8_t (&acc)[PG_MR][2], const __fp16 * bp, const __fp16 * a) {
    pg_k1_dup<FIRST>(acc, bp, a);
    pg_k1_dup<false>(acc, bp + 8, a + 16);
    pg_k1_dup<false>(acc, bp + 16, a + 32);
    pg_k1_dup<false>(acc, bp + 24, a + 48);
    pg_k1_dup<false>(acc, bp + 32, a + 64);
    pg_k1_dup<false>(acc, bp + 40, a + 80);
    pg_k1_dup<false>(acc, bp + 48, a + 96);
    pg_k1_dup<false>(acc, bp + 56, a + 112);
}

// ---- mk 1: by element. b[i] = row i's 8 consecutive k ([8 rows][8 k] packing).
template <int L, bool FIRST>
static inline void pg_step_elem(float16x8_t (&acc)[PG_MR][2], const float16x8_t (&b)[PG_MR], const __fp16 * a) {
    const float16x8_t a0 = vld1q_f16(a);
    const float16x8_t a1 = vld1q_f16(a + 8);
    for (int i = 0; i < PG_MR; i++) {
        if (FIRST) {
            acc[i][0] = vmulq_laneq_f16(a0, b[i], L);
            acc[i][1] = vmulq_laneq_f16(a1, b[i], L);
        } else {
            acc[i][0] = vfmaq_laneq_f16(acc[i][0], a0, b[i], L);
            acc[i][1] = vfmaq_laneq_f16(acc[i][1], a1, b[i], L);
        }
    }
}

template <bool FIRST>
static inline void pg_k8_elem(float16x8_t (&acc)[PG_MR][2], const __fp16 * bp, const __fp16 * a) {
    float16x8_t b[PG_MR];
    for (int i = 0; i < PG_MR; i++) {
        b[i] = vld1q_f16(bp + i * 8);
    }
    pg_step_elem<0, FIRST>(acc, b, a);
    pg_step_elem<1, false>(acc, b, a + 16);
    pg_step_elem<2, false>(acc, b, a + 32);
    pg_step_elem<3, false>(acc, b, a + 48);
    pg_step_elem<4, false>(acc, b, a + 64);
    pg_step_elem<5, false>(acc, b, a + 80);
    pg_step_elem<6, false>(acc, b, a + 96);
    pg_step_elem<7, false>(acc, b, a + 112);
}

template <int MK, bool FIRST>
static inline void pg_k8(float16x8_t (&acc)[PG_MR][2], const __fp16 * bp, const __fp16 * a) {
    if (MK == 0) {
        pg_k8_dup<FIRST>(acc, bp, a);
    } else {
        pg_k8_elem<FIRST>(acc, bp, a);
    }
}

// T[8][16] (f32, contiguous rows of 16) = sum over k < K (a multiple of 8, >= 8) of B[i][k] A[k][c]:
// f16 chains of FOLD consecutive k (absolute: k0 = 0, FOLD, ..), each folded into f32.
// Bp: packed B (64 values per 8 k), Ap: A rows [k][16].
template <int FOLD, int MK>
static void pg_mk(int K, const __fp16 * Bp, const __fp16 * Ap, float * T) {
    for (int k0 = 0; k0 < K; k0 += FOLD) {
        float16x8_t acc[PG_MR][2];
        pg_k8<MK, true>(acc, Bp + (int64_t) k0 * 8, Ap + (int64_t) k0 * PG_NR);
        if (K - k0 >= FOLD) {
            for (int j = 1; j < FOLD / 8; j++) {
                const int k = k0 + 8 * j;
                pg_k8<MK, false>(acc, Bp + (int64_t) k * 8, Ap + (int64_t) k * PG_NR);
            }
        } else {
            for (int k = k0 + 8; k < K; k += 8) {
                pg_k8<MK, false>(acc, Bp + (int64_t) k * 8, Ap + (int64_t) k * PG_NR);
            }
        }
        for (int i = 0; i < PG_MR; i++) {
            float * t = T + i * PG_NR;
            for (int h = 0; h < 2; h++) {
                if (k0 == 0) {
                    vst1q_f32(t + 8 * h, vcvt_f32_f16(vget_low_f16(acc[i][h])));
                    vst1q_f32(t + 8 * h + 4, vcvt_high_f32_f16(acc[i][h]));
                } else {
                    vst1q_f32(t + 8 * h, pg_fold_lo(vld1q_f32(t + 8 * h), acc[i][h]));
                    vst1q_f32(t + 8 * h + 4, pg_fold_hi(vld1q_f32(t + 8 * h + 4), acc[i][h]));
                }
            }
        }
    }
}

typedef void (*pg_mk_fn)(int, const __fp16 *, const __fp16 *, float *);

static pg_mk_fn pg_pick(int fold, int mk) {
    if (mk == 0) {
        return fold == 16 ? pg_mk<16, 0> : pg_mk<32, 0>;
    }
    return fold == 16 ? pg_mk<16, 1> : pg_mk<32, 1>;
}

// 8x8 transpose of f16 (as u16): r[i][j] -> r[j][i]
static inline void pg_tr8(uint16x8_t (&r)[8]) {
    const uint16x8_t t0 = vtrn1q_u16(r[0], r[1]), t1 = vtrn2q_u16(r[0], r[1]);
    const uint16x8_t t2 = vtrn1q_u16(r[2], r[3]), t3 = vtrn2q_u16(r[2], r[3]);
    const uint16x8_t t4 = vtrn1q_u16(r[4], r[5]), t5 = vtrn2q_u16(r[4], r[5]);
    const uint16x8_t t6 = vtrn1q_u16(r[6], r[7]), t7 = vtrn2q_u16(r[6], r[7]);
#define PG_U32(x) vreinterpretq_u32_u16(x)
    const uint32x4_t u0 = vtrn1q_u32(PG_U32(t0), PG_U32(t2)), u2 = vtrn2q_u32(PG_U32(t0), PG_U32(t2));
    const uint32x4_t u1 = vtrn1q_u32(PG_U32(t1), PG_U32(t3)), u3 = vtrn2q_u32(PG_U32(t1), PG_U32(t3));
    const uint32x4_t u4 = vtrn1q_u32(PG_U32(t4), PG_U32(t6)), u6 = vtrn2q_u32(PG_U32(t4), PG_U32(t6));
    const uint32x4_t u5 = vtrn1q_u32(PG_U32(t5), PG_U32(t7)), u7 = vtrn2q_u32(PG_U32(t5), PG_U32(t7));
#undef PG_U32
#define PG_U64(x) vreinterpretq_u64_u32(x)
#define PG_O(a, b, f) vreinterpretq_u16_u64(f(PG_U64(a), PG_U64(b)))
    r[0] = PG_O(u0, u4, vtrn1q_u64);
    r[4] = PG_O(u0, u4, vtrn2q_u64);
    r[1] = PG_O(u1, u5, vtrn1q_u64);
    r[5] = PG_O(u1, u5, vtrn2q_u64);
    r[2] = PG_O(u2, u6, vtrn1q_u64);
    r[6] = PG_O(u2, u6, vtrn2q_u64);
    r[3] = PG_O(u3, u7, vtrn1q_u64);
    r[7] = PG_O(u3, u7, vtrn2q_u64);
#undef PG_O
#undef PG_U64
}

// 8 f16 at p, lanes >= n (0..8) set to +0 (nothing at or beyond p + n is read)
static inline uint16x8_t pg_ld8(const uint16_t * p, int n) {
    if (n >= 8) {
        return vld1q_u16(p);
    }
    uint16_t t[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int j = 0; j < n; j++) {
        t[j] = p[j];
    }
    return vld1q_u16(t);
}

// Head dim of QK position k (see the header): k = j * T + s -> 32 s + j (j < 32), T = hd / 32.
static inline int pg_dim(int k, int T) { return 32 * (k % T) + k / T; }

// K rows [s0, s0 + klen) of the chunk (rows >= nr as +0) into KP: 8-row panel rp at
// KP + rp * hd * 8; inside, 64-value blocks per 8 consecutive k (head dims in pg_dim order):
// [k][8 rows] for LD1R, [8 rows][8 k] by element.
static void pg_pack_k(int mk, int hd, int nr, int klen, const uint16_t * K, uint16_t * KP) {
    const int T = hd / 32;
    const uint16x8_t z = vdupq_n_u16(0);
    for (int r0 = 0; r0 < klen; r0 += 8) {
        uint16_t * panel = KP + (int64_t) r0 * hd;
        const uint16_t * rows[8];
        bool ok[8];
        for (int i = 0; i < 8; i++) {
            ok[i] = r0 + i < nr;
            rows[i] = K + (int64_t) (r0 + i) * hd;
        }
        if (mk == 0) {
            // rows i x dims 32 s + j0 + jj -> transpose -> vector jj = the 8 rows of that dim,
            // which is k = (j0 + jj) * T + s
            for (int s = 0; s < T; s++) {
                for (int j0 = 0; j0 < 32; j0 += 8) {
                    uint16x8_t v[8];
                    for (int i = 0; i < 8; i++) {
                        v[i] = ok[i] ? vld1q_u16(rows[i] + 32 * s + j0) : z;
                    }
                    pg_tr8(v);
                    for (int jj = 0; jj < 8; jj++) {
                        vst1q_u16(panel + ((int64_t) (j0 + jj) * T + s) * 8, v[jj]);
                    }
                }
            }
        } else if (T == 8) {
            // per row: dims 32 s + j0 + jj for s < 8 -> transpose -> vector jj = k block of j0 + jj
            for (int i = 0; i < 8; i++) {
                for (int j0 = 0; j0 < 32; j0 += 8) {
                    uint16x8_t v[8];
                    for (int s = 0; s < 8; s++) {
                        v[s] = ok[i] ? vld1q_u16(rows[i] + 32 * s + j0) : z;
                    }
                    pg_tr8(v);
                    for (int jj = 0; jj < 8; jj++) {
                        vst1q_u16(panel + (int64_t) (j0 + jj) * 64 + 8 * i, v[jj]);
                    }
                }
            }
        } else {
            for (int k = 0; k < hd; k++) {
                const int d = pg_dim(k, T);
                for (int i = 0; i < 8; i++) {
                    panel[(int64_t) (k / 8) * 64 + i * 8 + k % 8] = ok[i] ? rows[i][d] : 0;
                }
            }
        }
    }
}

// V^T rows (hd dims, stride vs) at positions [s0, s0 + klen) (positions >= nr as +0) into VP:
// 8-dim panel dp at VP + dp * 8 * ch; inside, 64-value blocks per 8 positions ([pos][8 dims] for
// LD1R, [8 dims][8 pos] by element).
static void pg_pack_v(int mk, int hd, int ch, int nr, int klen, const uint16_t * VT, int64_t vs, uint16_t * VP) {
    for (int d0 = 0; d0 < hd; d0 += 8) {
        uint16_t * panel = VP + (int64_t) d0 * ch;
        for (int r0 = 0; r0 < klen; r0 += 8) {
            const int n = std::max(0, std::min(8, nr - r0));
            uint16x8_t v[8];
            for (int i = 0; i < 8; i++) {
                v[i] = pg_ld8(VT + (int64_t) (d0 + i) * vs + r0, n);
            }
            if (mk == 0) {
                pg_tr8(v);
            }
            uint16_t * dst = panel + (int64_t) r0 * 8;
            for (int j = 0; j < 8; j++) {
                vst1q_u16(dst + 8 * j, v[j]);
            }
        }
    }
}

struct PgLayout {
    int ncp;
    int64_t qt, kp, vp, s, p, o, m, l, al, vm, lim, t, total;
};

static PgLayout pg_layout(int hd, int ncol, int ch) {
    PgLayout L;
    L.ncp = (ncol + PG_NR - 1) / PG_NR * PG_NR;
    int64_t at = 0;
    L.qt = at;  at += pg_al((int64_t) hd * L.ncp * 2);  // QT [cb][hd][16] f16
    L.kp = at;  at += pg_al((int64_t) ch * hd * 2);     // KP [ch/8][hd][8] f16
    L.vp = at;  at += pg_al((int64_t) ch * hd * 2);     // VP [hd/8][ch][8] f16
    L.s = at;   at += pg_al((int64_t) ch * L.ncp * 4);  // S [cb][ch][16] f32
    L.p = at;   at += pg_al((int64_t) ch * L.ncp * 2);  // P [cb][ch][16] f16
    L.o = at;   at += pg_al((int64_t) hd * L.ncp * 4);  // O [cb][hd][16] f32
    L.m = at;   at += pg_al((int64_t) L.ncp * 4);
    L.l = at;   at += pg_al((int64_t) L.ncp * 4);
    L.al = at;  at += pg_al((int64_t) L.ncp * 4);
    L.vm = at;  at += pg_al((int64_t) L.ncp * 4);
    L.lim = at; at += pg_al((int64_t) L.ncp * 4);
    L.t = at;   at += pg_al((int64_t) PG_MR * PG_NR * 4);  // micro-kernel output tile
    L.total = at + 64;  // the caller's buffer is aligned up to 64 inside
    return L;
}

}  // namespace

extern "C" {

// Scratch bytes of oxe_pfg_unit for up to ncol columns of hd dims and chunks of up to ch positions.
int64_t oxe_pfg_scratch_bytes(int hd, int ncol, int ch) {
    return pg_layout(hd, std::max(ncol, 1), ch).total;
}

// Ticks per second of the counter the `prof` array of oxe_pfg_unit accumulates (CNTFRQ_EL0).
uint64_t oxe_pfg_tick_hz(void) {
    uint64_t f;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(f));
    return f;
}

// One pfg unit (see the header). Columns c = 0..ncol: token pos0 + c / group, query q16 + c * hd
// (f16); K rows k + p * hd and V^T rows vt + d * v_stride, read for positions p <= last only;
// out[c * hd + d] (normalised, ungated). ch: chunk positions (multiple of 32, <= 256); fold_qk /
// fold_pv: 16 or 32; mk: 0 LD1R, 1 by element; prof: NULL or 5 tick counters (accumulated).
// Requires hd % 32 == 0, n_ctx % 32 == 0, last < n_ctx, 1 <= ncol <= 256; returns 0, or -1 when
// a requirement fails (nothing written).
int oxe_pfg_unit(int hd, int ncol, int group, int pos0, float scale, const ggml_fp16_t * k, const ggml_fp16_t * vt,
                 int64_t v_stride, int n_ctx, const ggml_fp16_t * q16, int ch, int fold_qk, int fold_pv, int mk,
                 void * scratch, int64_t scratch_bytes, float * out, uint64_t * prof) {
    if (hd <= 0 || hd % 32 || ncol < 1 || ncol > PG_MAXCOL || group < 1 || pos0 < 0 || n_ctx <= 0 || n_ctx % 32 || ch < 32 ||
        ch % 32 || ch > PG_MAXCH || !(fold_qk == 16 || fold_qk == 32) || !(fold_pv == 16 || fold_pv == 32) ||
        !(mk == 0 || mk == 1)) {
        return -1;
    }
    const int last = pos0 + (ncol - 1) / group;
    if (last >= n_ctx) {
        return -1;
    }
    const PgLayout L = pg_layout(hd, ncol, ch);
    if (scratch_bytes < L.total) {
        return -1;
    }
    uint64_t tk = prof ? pg_tick() : 0;
    // phase boundary: add the ticks since the last boundary to prof[i]
    auto mark = [&](int i) {
        if (prof) {
            const uint64_t n = pg_tick();
            prof[i] += n - tk;
            tk = n;
        }
    };
    uint8_t * base = (uint8_t *) (((uintptr_t) scratch + 63) & ~(uintptr_t) 63);
    const int ncp = L.ncp;
    const int ncb = ncp / PG_NR;
    __fp16 * QT = (__fp16 *) (base + L.qt);
    uint16_t * KP = (uint16_t *) (base + L.kp);
    uint16_t * VP = (uint16_t *) (base + L.vp);
    float * S = (float *) (base + L.s);
    __fp16 * P = (__fp16 *) (base + L.p);
    float * O = (float *) (base + L.o);
    float * M = (float *) (base + L.m);
    float * Ls = (float *) (base + L.l);
    float * AL = (float *) (base + L.al);
    uint32_t * VM = (uint32_t *) (base + L.vm);
    int32_t * LIM = (int32_t *) (base + L.lim);
    float * T = (float *) (base + L.t);

    const pg_mk_fn mk_qk = pg_pick(fold_qk, mk);
    const pg_mk_fn mk_pv = pg_pick(fold_pv, mk);

    // Q^T panels in the QK head-dim order (pad columns 0), per-column position, running max /
    // sum, O = 0
    const __fp16 * q = (const __fp16 *) q16;
    const int nseg = hd / 32;
    for (int cb = 0; cb < ncb; cb++) {
        __fp16 * panel = QT + (int64_t) cb * hd * PG_NR;
        for (int j = 0; j < PG_NR; j++) {
            const int c = cb * PG_NR + j;
            if (c < ncol) {
                const __fp16 * qc = q + (int64_t) c * hd;
                for (int kk = 0; kk < hd; kk++) {
                    panel[(int64_t) kk * PG_NR + j] = qc[pg_dim(kk, nseg)];
                }
            } else {
                for (int d = 0; d < hd; d++) {
                    panel[(int64_t) d * PG_NR + j] = (__fp16) 0.0f;
                }
            }
        }
    }
    for (int c = 0; c < ncp; c++) {
        LIM[c] = c < ncol ? pos0 + c / group : -1;
        M[c] = -INFINITY;
        Ls[c] = 0.0f;
    }
    memset(O, 0, (size_t) hd * ncp * 4);
    mark(0);

    const uint16_t * K = (const uint16_t *) k;
    const uint16_t * VT = (const uint16_t *) vt;
    const int rows = last + 1;
    const float32x4_t vscale = vdupq_n_f32(scale);
    const float32x4_t ninf = vdupq_n_f32(-INFINITY);
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const float32x4_t one = vdupq_n_f32(1.0f);

    for (int s0 = 0; s0 < rows; s0 += ch) {
        const int nr = std::min(ch, rows - s0);  // valid rows of the chunk (positions <= last)
        const int klen = (nr + 7) / 8 * 8;       // rows computed (+0 beyond nr)
        // [pack] K and V^T of the chunk, once for all columns
        pg_pack_k(mk, hd, nr, klen, K + (int64_t) s0 * hd, KP);
        pg_pack_v(mk, hd, ch, nr, klen, VT + s0, v_stride, VP);
        if (s0 + ch < rows) {
            // next chunk's first K rows and V^T segments: start the DRAM stream early
            const char * pk = (const char *) (K + (int64_t) (s0 + ch) * hd);
            for (int b = 0; b < 8 * hd * 2; b += 64) {
                __builtin_prefetch(pk + b, 0, 3);
            }
            for (int d = 0; d < hd; d++) {
                __builtin_prefetch(VT + (int64_t) d * v_stride + s0 + ch, 0, 3);
            }
        }
        mark(0);
        // [QK] S tile (cb, row panel) = 8 rows x 16 columns, contiguous in S[cb][r][16]
        for (int cb = 0; cb < ncb; cb++) {
            const __fp16 * qa = QT + (int64_t) cb * hd * PG_NR;
            for (int r0 = 0; r0 < klen; r0 += PG_MR) {
                mk_qk(hd, (const __fp16 *) (KP + (int64_t) r0 * hd), qa, S + ((int64_t) cb * ch + r0) * PG_NR);
            }
        }
        mark(1);
        // [soft] online softmax, 4 columns per vector, vertical over the chunk's rows
        for (int cb = 0; cb < ncb; cb++) {
            float * sp = S + (int64_t) cb * ch * PG_NR;
            __fp16 * pp = P + (int64_t) cb * ch * PG_NR;
            for (int qv = 0; qv < 4; qv++) {
                const int c = cb * PG_NR + 4 * qv;
                const int32x4_t lim = vsubq_s32(vld1q_s32(LIM + c), vdupq_n_s32(s0));
                float32x4_t mx = ninf;
                for (int r = 0; r < nr; r++) {
                    const uint32x4_t ok = vcleq_s32(vdupq_n_s32(r), lim);
                    float * sr = sp + (int64_t) r * PG_NR + 4 * qv;
                    const float32x4_t x = vmulq_f32(vld1q_f32(sr), vscale);
                    vst1q_f32(sr, x);
                    mx = vmaxq_f32(mx, vbslq_f32(ok, x, ninf));
                }
                const float32x4_t mo = vld1q_f32(M + c);
                const float32x4_t mn = vmaxq_f32(mo, mx);
                // a column with no position yet keeps alpha = 1 (no -inf - -inf)
                const uint32x4_t fin = vcgtq_f32(mn, ninf);
                const float32x4_t al = vbslq_f32(fin, ggml_v_expf(vsubq_f32(mo, mn)), one);
                const float32x4_t mexp = vbslq_f32(fin, mn, zero);
                float32x4_t ls = zero;
                for (int r = 0; r < nr; r++) {
                    const uint32x4_t ok = vcleq_s32(vdupq_n_s32(r), lim);
                    const float32x4_t e = vbslq_f32(ok, ggml_v_expf(vsubq_f32(vld1q_f32(sp + (int64_t) r * PG_NR + 4 * qv), mexp)), zero);
                    const float16x4_t h = vcvt_f16_f32(e);
                    vst1_f16(pp + (int64_t) r * PG_NR + 4 * qv, h);
                    ls = vaddq_f32(ls, vcvt_f32_f16(h));
                }
                for (int r = nr; r < klen; r++) {
                    vst1_f16(pp + (int64_t) r * PG_NR + 4 * qv, vdup_n_f16((__fp16) 0.0f));
                }
                // valid in this chunk: the column's position reaches s0
                const uint32x4_t any = vcgeq_s32(lim, vdupq_n_s32(0));
                vst1q_u32(VM + c, any);
                vst1q_f32(AL + c, vbslq_f32(any, al, one));
                const float32x4_t lo = vld1q_f32(Ls + c);
                vst1q_f32(Ls + c, vbslq_f32(any, vaddq_f32(vmulq_f32(lo, al), ls), lo));
                vst1q_f32(M + c, vbslq_f32(any, mn, mo));
            }
        }
        mark(2);
        // [PV] O tile (8 dims, 16 columns) = valid ? O * alpha + V^T . P : O
        for (int d0 = 0; d0 < hd; d0 += PG_MR) {
            const __fp16 * vb = (const __fp16 *) (VP + (int64_t) d0 * ch);
            for (int cb = 0; cb < ncb; cb++) {
                mk_pv(klen, vb, P + (int64_t) cb * ch * PG_NR, T);
                float * ot = O + ((int64_t) cb * hd + d0) * PG_NR;
                for (int qv = 0; qv < 4; qv++) {
                    const int c = cb * PG_NR + 4 * qv;
                    const uint32x4_t vm = vld1q_u32(VM + c);
                    const float32x4_t va = vld1q_f32(AL + c);
                    for (int i = 0; i < PG_MR; i++) {
                        float * o = ot + i * PG_NR + 4 * qv;
                        const float32x4_t ov = vld1q_f32(o);
                        const float32x4_t n = vfmaq_f32(vld1q_f32(T + i * PG_NR + 4 * qv), ov, va);
                        vst1q_f32(o, vbslq_f32(vm, n, ov));
                    }
                }
            }
        }
        mark(3);
    }
    // normalise and transpose: out[c][d] = O[cb][d][j] * (1 / l)
    for (int c = 0; c < ncol; c++) {
        const float inv = 1.0f / Ls[c];
        const float * oc = O + (int64_t) (c / PG_NR) * hd * PG_NR + c % PG_NR;
        float * o = out + (int64_t) c * hd;
        for (int d = 0; d < hd; d++) {
            o[d] = oc[(int64_t) d * PG_NR] * inv;
        }
    }
    mark(4);
    return 0;
}

}  // extern "C"
