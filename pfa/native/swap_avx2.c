// AVX2 pass for exchanges (included by swap.c): one chunk of 64 pairs of a row, a group of 16 pairs = 4 float64
// vectors of D -> 2 float32 score vectors of 8. Applies the update to D and returns the largest score.
static inline float swap_chunk(double *D, const float *g, const double *au, const double *bu, const float *pu,
                               double at, double bt, float c, float T) {
    const __m256d vat = _mm256_set1_pd(at), vbt = _mm256_set1_pd(bt);
    const __m256 vc = _mm256_set1_ps(c), vT = _mm256_set1_ps(T);
    __m256 b0 = _mm256_set1_ps(-__builtin_inff()), b1 = b0;
#define UPD(X, O) __m256d X = _mm256_sub_pd(_mm256_loadu_pd(D + k + O), \
        _mm256_mul_pd(_mm256_sub_pd(vat, _mm256_loadu_pd(au + k + O)), _mm256_sub_pd(vbt, _mm256_loadu_pd(bu + k + O)))); \
    _mm256_storeu_pd(D + k + O, X);
#define CV8(XA, XB) _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(XA)), _mm256_cvtpd_ps(XB), 1)
#define SC(XA, XB, O) FMA8(_mm256_loadu_ps(g + k + O), vT, _mm256_sub_ps(_mm256_sub_ps(vc, _mm256_loadu_ps(pu + k + O)), CV8(XA, XB)))
    for (int k = 0; k < 64; k += 16) {
        UPD(X0, 0) UPD(X1, 4) UPD(X2, 8) UPD(X3, 12)
        b0 = _mm256_max_ps(b0, SC(X0, X1, 0)); b1 = _mm256_max_ps(b1, SC(X2, X3, 8));
    }
#undef UPD
#undef CV8
#undef SC
    return hmax8(_mm256_max_ps(b0, b1));
}
