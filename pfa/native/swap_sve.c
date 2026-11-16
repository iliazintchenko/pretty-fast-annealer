// SVE pass for exchanges (included by swap.c), vector-length agnostic: one chunk of 64 pairs of a row, a group is
// 2 float64 vectors of D = one float32 score vector. Applies the update to D and returns the largest score.
static inline float swap_chunk(double *D, const float *g, const double *au, const double *bu, const float *p1,
                               const float *p2, double at, double bt, float T) SVE_STREAMING {
    const int64_t VL = (int64_t)svcntd(), N = 64;
    const svbool_t all64 = svptrue_b64(), all32 = svptrue_b32();
    const svfloat64_t vat = svdup_f64(at), vbt = svdup_f64(bt);
    const svfloat32_t vT = svdup_f32(T);
    svfloat32_t B = svdup_f32(-__builtin_inff());
#define UPD(P, K) svsub_x(P, svld1(P, D + (K)), svmul_x(P, svsub_x(P, vat, svld1(P, au + (K))), svsub_x(P, vbt, svld1(P, bu + (K)))))
    for (int64_t k = 0; k < N; k += 2 * VL) {
        const svbool_t pa = svwhilelt_b64(k, N), pb = svwhilelt_b64(k + VL, N), pf = svwhilelt_b32(k, N);
        const svfloat64_t X0 = UPD(pa, k), X1 = UPD(pb, k + VL);
        svst1(pa, D + k, X0); svst1(pb, D + k + VL, X1);
        const svfloat32_t d32 = svuzp1(svcvt_f32_f64_x(all64, X0), svcvt_f32_f64_x(all64, X1));
        const svfloat32_t pen = svadd_x(pf, svld1(pf, p1 + k), svld1(pf, p2 + k));
        const svfloat32_t sc = svmla_x(pf, svsub_x(pf, svneg_x(pf, pen), d32), svld1(pf, g + k), vT);
        B = svmax_m(pf, B, sc);
    }
#undef UPD
    return svmaxv(all32, B);
}
