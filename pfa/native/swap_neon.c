// NEON pass for exchanges (included by swap.c): one chunk of 64 pairs of a row, a group of 16 pairs = 8 float64
// vectors of D -> 4 float32 score vectors. Applies the update to D and returns the largest score.
static inline float swap_chunk(double *D, const float *g, const double *au, const double *bu, const float *pu,
                               double at, double bt, float c, float T) {
    const float64x2_t vat = vdupq_n_f64(at), vbt = vdupq_n_f64(bt);
    const float32x4_t vc = vdupq_n_f32(c), vT = vdupq_n_f32(T);
    float32x4_t b0 = vdupq_n_f32(-__builtin_inff()), b1 = b0, b2 = b0, b3 = b0;
#define UPD(X, O) float64x2_t X = vsubq_f64(vld1q_f64(D + k + O), vmulq_f64(vsubq_f64(vat, vld1q_f64(au + k + O)), \
                                                                            vsubq_f64(vbt, vld1q_f64(bu + k + O)))); \
                  vst1q_f64(D + k + O, X);
#define SC(XA, XB, O) vfmaq_f32(vsubq_f32(vsubq_f32(vc, vld1q_f32(pu + k + O)), vcvt_high_f32_f64(vcvt_f32_f64(XA), XB)), \
                                vld1q_f32(g + k + O), vT)
    for (int k = 0; k < 64; k += 16) {
        UPD(X0, 0) UPD(X1, 2) UPD(X2, 4) UPD(X3, 6) UPD(X4, 8) UPD(X5, 10) UPD(X6, 12) UPD(X7, 14)
        b0 = vmaxq_f32(b0, SC(X0, X1, 0)); b1 = vmaxq_f32(b1, SC(X2, X3, 4));
        b2 = vmaxq_f32(b2, SC(X4, X5, 8)); b3 = vmaxq_f32(b3, SC(X6, X7, 12));
    }
#undef UPD
#undef SC
    return vmaxvq_f32(vmaxq_f32(vmaxq_f32(b0, b1), vmaxq_f32(b2, b3)));
}
