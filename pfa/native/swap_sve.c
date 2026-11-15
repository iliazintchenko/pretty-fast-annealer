// SVE pass for exchanges (included by swap.c), vector-length agnostic: one chunk of 64 pairs of a row, a group is
// 2 float64 vectors of D = one float32 score vector. Applies the update to D and returns the largest score.
static inline float swap_chunk(double *D, const float *g, const double *au, const double *bu, const float *pu,
                               double at, double bt, float c, float T) SVE_STREAMING {
    const int64_t VL = (int64_t)svcntd(), N = 64;
    const svbool_t all64 = svptrue_b64(), all32 = svptrue_b32();
    const svfloat64_t vat = svdup_f64(at), vbt = svdup_f64(bt);
    const svfloat32_t vc = svdup_f32(c), vT = svdup_f32(T);
    svfloat32_t B = svdup_f32(-__builtin_inff());
#define UPD(P, K) svsub_x(P, svld1(P, D + (K)), svmul_x(P, svsub_x(P, vat, svld1(P, au + (K))), svsub_x(P, vbt, svld1(P, bu + (K)))))
    for (int64_t k = 0; k < N; k += 2 * VL) {
        const svbool_t p0 = svwhilelt_b64(k, N), p1 = svwhilelt_b64(k + VL, N), pf = svwhilelt_b32(k, N);
        const svfloat64_t X0 = UPD(p0, k), X1 = UPD(p1, k + VL);
        svst1(p0, D + k, X0); svst1(p1, D + k + VL, X1);
        const svfloat32_t sc = svmla_x(pf, svsub_x(pf, svsub_x(pf, vc, svld1(pf, pu + k)),
                                                  svuzp1(svcvt_f32_f64_x(all64, X0), svcvt_f32_f64_x(all64, X1))),
                                       svld1(pf, g + k), vT);
        B = svmax_m(pf, B, sc);
    }
#undef UPD
    return svmaxv(all32, B);
}
