// Scalar pass for exchanges (included by swap.c), for targets without a SIMD path: one chunk of 64 pairs of a row.
// Applies the update to D and returns the largest score.
static inline float swap_chunk(double *D, const float *g, const double *au, const double *bu, const float *p1,
                               const float *p2, double at, double bt, float T) {
    float bv = -__builtin_inff();
    for (int k = 0; k < 64; k++) {
        D[k] = swap_upd(D[k], at, au[k], bt, bu[k]);
        const float v = swap_sc(D[k], p1[k], p2[k], g[k], T);
        if (v > bv) bv = v;
    }
    return bv;
}
