// PFA engine, native single-core version, compiled as one unit: this file picks the SIMD path at compile time from
// the target flags (see the Makefile) and includes the kernels,
//   spin.c                 spins: pfa_anneal_spin()
//   categorical.c          categorical variables: pfa_anneal_categorical()
//   swap.c                 variables that exchange values, a fixed count per value: pfa_anneal_swap()
//   sparse_spin.c          spins, sparse couplings: pfa_anneal_sparse_spin()
//   sparse_categorical.c   categorical variables, sparse couplings: pfa_anneal_sparse_categorical()
//   sparse_swap.c          exchanges, sparse couplings: pfa_anneal_sparse_swap()
//   tour.c                 tours through cities by coordinates: the tour structure and its edge lengths
//   tour_chain.c           the tour kernel, Lin and Kernighan's chain of sequential exchanges over the candidates at
//                          zero temperature, with don't-look bits: pfa_tour_descent(); the walk along the candidate
//                          edges that starts a restart: pfa_tour_walk()
//   alpha.c                the candidate edges of a tour problem by alpha-nearness: pfa_tour_alpha()
//   tour_fold.c            edge assembly between two tours, one fold for tours: pfa_tour_fold()
//   tour_ipt.c             partial transcription between two tours, the other: pfa_tour_ipt()
//   route.c                vehicle routes: the kernel of route moves at zero temperature, pfa_route_ls(); Split,
//                          pfa_route_split(); the walk over the customers, pfa_route_walk()
//   route_fold.c           edge assembly on the graph with the depot as one node, pfa_route_fold(); the loop with
//                          the pool, pfa_route_loop()
//   shop.c                 the job shop: heads and tails of the disjunctive graph, Giffler and Thompson's start,
//                          pfa_shop_gt(); the tabu search on the N7 neighbourhood at zero temperature, pfa_shop_tabu()
//   shop_loop.c            path relinking over a pool of schedules, the loop: pfa_shop_loop()
//   colour.c               graph colouring: TabuCol over the conflicting vertices at zero temperature with the lazy
//                          cache of best colours, pfa_colour_tabu(); the partition crossover, pfa_colour_gpx()
//   colour_loop.c          the duet of two parents and two elites under one budget of kernel iterations, the
//                          restart's length read from the children's returns: pfa_colour_run()
//   cover.c                minimum vertex cover: the two-stage exchange with edge weights at zero temperature under
//                          one budget of steps, the level dropping at every cover found, pfa_cover_search(); the
//                          greedy start, pfa_cover_greedy()
//   eliminate.c           categorical variables: pfa_eliminate(), before the anneal
//   fold.c                 pfa_fold_spin(), pfa_fold_sparse_spin(), pfa_fold_categorical(), pfa_fold_swap(): one
//                          state takes from another every part that it has lower, between the repetitions
// Each kernel holds its anneal loop. The first three include their pass over memory for the chosen path,
// spin_<path>.c, categorical_<path>.c or swap_<path>.c:
//   sve      vector-length agnostic, predicated   ARM with SVE (Graviton3 and 4, NVIDIA Grace, A64FX, ...)
//   neon     2 doubles / 4 floats per vector      every other 64-bit ARM (Apple Silicon, Graviton2, ...)
//   avx512   8 doubles per vector                 x86-64 with AVX-512
//   avx2     4 doubles per vector                 x86-64 with AVX2 (and FMA)
//   scalar                                        anything else
// The sparse three draw from the tree of sums of sums.c, the same code on every path.
// Build with -ffp-contract=off: each kernel rescores the winning chunk and must reproduce the pass's scores exactly,
// so the compiler may not fuse multiply-adds in one place and not the other.
//
// Test-only switches, for checking paths on hardware that lacks them:
//   -DTEST_STREAMING_SVE  runs the SVE path in SME streaming mode (e.g. Apple M4, -mcpu=apple-m4)
//   -DTEST_AVX512_SIMDE   runs the AVX-512 path through the SIMDe emulation headers (-I<simde include dir>)
//   -DTEST_AVX2_SIMDE     runs the AVX2 path through SIMDe
//   -DTEST_SCALAR         builds the scalar path on any target
#include <stdint.h>
#include <stdlib.h>

#if defined(__aarch64__) || defined(__FMA__)
#define FMAF(A, B, C) __builtin_fmaf(A, B, C)
#else
#define FMAF(A, B, C) ((A) * (B) + (C))   // no hardware FMA: a library fmaf would be far slower
#endif

#if defined(TEST_STREAMING_SVE)
#include <arm_sme.h>
#include <arm_sve.h>
#define SIMD_PATH "sve-streaming"
#define SVE_STREAMING __arm_streaming
#define ANNEAL_ATTR __arm_locally_streaming
#define SPIN_PASS "spin_sve.c"
#define CATEGORICAL_PASS "categorical_sve.c"
#define SWAP_PASS "swap_sve.c"
#elif defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#define SIMD_PATH "sve"
#define SVE_STREAMING
#define SPIN_PASS "spin_sve.c"
#define CATEGORICAL_PASS "categorical_sve.c"
#define SWAP_PASS "swap_sve.c"
#elif defined(TEST_AVX512_SIMDE)
#define SIMDE_ENABLE_NATIVE_ALIASES
#include <simde/x86/avx512.h>
#define SIMD_PATH "avx512-simde"
#define CVT512(D) _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(_mm512_castpd512_pd256(D))), \
                                       _mm256_cvtpd_ps(_mm512_extractf64x4_pd(D, 1)), 1)   // SIMDe lacks _mm512_cvtpd_ps
#define X86_PASSES
#define SPIN_PASS "spin_avx512.c"
#define CATEGORICAL_PASS "categorical_avx512.c"
#define SWAP_PASS "swap_avx512.c"
#elif defined(TEST_AVX2_SIMDE)
#define SIMDE_ENABLE_NATIVE_ALIASES
#include <simde/x86/avx2.h>
#include <simde/x86/fma.h>
#define SIMD_PATH "avx2-simde"
#define X86_PASSES
#define SPIN_PASS "spin_avx2.c"
#define CATEGORICAL_PASS "categorical_avx2.c"
#define SWAP_PASS "swap_avx2.c"
#elif defined(TEST_SCALAR)
#define SIMD_PATH "scalar-test"
#define SPIN_PASS "spin_scalar.c"
#define CATEGORICAL_PASS "categorical_scalar.c"
#define SWAP_PASS "swap_scalar.c"
#elif defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#define SIMD_PATH "neon"
#define SPIN_PASS "spin_neon.c"
#define CATEGORICAL_PASS "categorical_neon.c"
#define SWAP_PASS "swap_neon.c"
#elif defined(__AVX512F__)
#include <immintrin.h>
#define SIMD_PATH "avx512"
#define CVT512(D) _mm512_cvtpd_ps(D)
#define X86_PASSES
#define SPIN_PASS "spin_avx512.c"
#define CATEGORICAL_PASS "categorical_avx512.c"
#define SWAP_PASS "swap_avx512.c"
#elif defined(__AVX2__)
#include <immintrin.h>
#define SIMD_PATH "avx2"
#define X86_PASSES
#define SPIN_PASS "spin_avx2.c"
#define CATEGORICAL_PASS "categorical_avx2.c"
#define SWAP_PASS "swap_avx2.c"
#else
#define SIMD_PATH "scalar"
#define SPIN_PASS "spin_scalar.c"
#define CATEGORICAL_PASS "categorical_scalar.c"
#define SWAP_PASS "swap_scalar.c"
#endif

#ifndef ANNEAL_ATTR
#define ANNEAL_ATTR
#endif
#ifndef SVE_STREAMING
#define SVE_STREAMING
#endif

#ifdef X86_PASSES   // shared by the x86 passes: fused multiply-adds when built with FMA (any AVX2 CPU has it)
#if defined(__FMA__) || defined(TEST_AVX2_SIMDE)
#define FMA8(A, B, C) _mm256_fmadd_ps(A, B, C)
#define FMA4(A, B, C) _mm_fmadd_ps(A, B, C)
#else
#define FMA8(A, B, C) _mm256_add_ps(_mm256_mul_ps(A, B), C)
#define FMA4(A, B, C) _mm_add_ps(_mm_mul_ps(A, B), C)
#endif
static inline float hmax8(__m256 x) {
    __m128 m = _mm_max_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    return _mm_cvtss_f32(_mm_max_ss(m, _mm_shuffle_ps(m, m, 1)));
}
#endif

const char *pfa_simd_path(void) { return SIMD_PATH; }

#include "spin.c"
#include "categorical.c"
#include "swap.c"
#include "sums.c"
#include "sparse_spin.c"
#include "sparse_categorical.c"
#include "sparse_swap.c"
#include "tour.c"
#include "tour_chain.c"
#include "alpha.c"
#include "tour_fold.c"
#include "tour_ipt.c"
#include "route.c"
#include "route_fold.c"
#include "shop.c"
#include "shop_loop.c"
#include "colour.c"
#include "colour_loop.c"
#include "cover.c"
#include "eliminate.c"
#include "fold.c"
