// AVX-512 pass for exchanges (included by swap.c): one chunk of 64 pairs of a row, a group of 16 pairs = 2 float64
// vectors of D -> 2 float32 score vectors of 8. Applies the update to D and returns the largest score.
static inline float swap_chunk(double *D, const float *g, const double *au, const double *bu, const float *p1,
                               const float *p2, double at, double bt, float T) {
    const __m512d vat = _mm512_set1_pd(at), vbt = _mm512_set1_pd(bt);
    const __m256 vT = _mm256_set1_ps(T), sign = _mm256_set1_ps(-0.0f);
    __m256 b0 = _mm256_set1_ps(-__builtin_inff()), b1 = b0;
#define UPD(X, O) __m512d X = _mm512_sub_pd(_mm512_loadu_pd(D + k + O), \
        _mm512_mul_pd(_mm512_sub_pd(vat, _mm512_loadu_pd(au + k + O)), _mm512_sub_pd(vbt, _mm512_loadu_pd(bu + k + O)))); \
    _mm512_storeu_pd(D + k + O, X);
#define PEN(O) _mm256_add_ps(_mm256_loadu_ps(p1 + k + O), _mm256_loadu_ps(p2 + k + O))
#define SC(X, O) _mm256_fmadd_ps(_mm256_loadu_ps(g + k + O), vT, _mm256_sub_ps(_mm256_xor_ps(PEN(O), sign), CVT512(X)))
    for (int k = 0; k < 64; k += 16) {
        UPD(X0, 0) UPD(X1, 8)
        b0 = _mm256_max_ps(b0, SC(X0, 0)); b1 = _mm256_max_ps(b1, SC(X1, 8));
    }
#undef UPD
#undef SC
#undef PEN
    return hmax8(_mm256_max_ps(b0, b1));
}
