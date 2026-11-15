// PFA fold (included by pfa.c): state a takes from state b every part that b has lower.
// The variables on which a and b differ are put in clusters: two of them share a cluster when their choices interact,
// that is when taking both from b does not change the energy by the sum of taking each alone. For spins i, j the
// difference is 4 J_ij a_i a_j, nonzero whenever the coupling is; for variables r, s with values a_r, b_r and
// a_s, b_s it is
//     T(a_r, a_s) - T(a_r, b_s) - T(b_r, a_s) + T(b_r, b_s)
// with T the couplings between their values. Variables of different clusters do not interact, so the energy change
// of taking a cluster from b is the same whatever is taken elsewhere:
//     sum over the cluster of (the change of taking that variable alone) + sum over its pairs of the difference above
// and every cluster with a negative change is taken. The result is the lowest state that has each cluster from a or
// from b. An interaction counts when it is not exactly 0: one that is nonzero by rounding only joins two clusters,
// which loses a choice and nothing else. A change counts as negative when it is below 0 by more than the rounding of
// its own sum, 2^-40 of the magnitudes summed: two arrangements of a cluster that tie would otherwise be taken back
// and forth between the rounds, each time by a rounding error.
// Each function writes the result into a and returns the energy change, 0 or less.

static inline int fold_root(int *up, int p) {
    while (up[p] != p) { up[p] = up[up[p]]; p = up[p]; }
    return p;
}

static inline void fold_join(int *up, int p, int q) {
    p = fold_root(up, p); q = fold_root(up, q);
    if (p != q) up[q] = p;
}

// delta[p]: the change of taking differing variable p alone plus its interactions with the later ones, mag[p] the sum
// of the magnitudes of its terms; up: the clusters. Sets take[p] for the members of the clusters whose total is
// negative beyond rounding and returns the sum of those totals.
static double fold_clusters(int nd, int *up, const double *delta, const double *mag, double *sum, double *msum,
                            char *take) {
    double gain = 0.0;
    for (int p = 0; p < nd; p++) sum[p] = msum[p] = 0.0;
    for (int p = 0; p < nd; p++) { const int r = fold_root(up, p); sum[r] += delta[p]; msum[r] += mag[p]; }
    for (int p = 0; p < nd; p++) {
        const int r = fold_root(up, p);
        take[p] = sum[r] < -0x1p-40 * msum[r];
        if (up[p] == p && take[p]) gain += sum[p];
    }
    return gain;
}

#define FOLD_NEW(N) \
    int *d = malloc(sizeof(int) * (N)), *up = malloc(sizeof(int) * (N)), nd = 0; \
    double *delta = malloc(sizeof(double) * (N)), *mag = malloc(sizeof(double) * (N)); \
    double *sum = malloc(sizeof(double) * (N)), *msum = malloc(sizeof(double) * (N)); \
    char *take = malloc(N);
#define FOLD_FREE free(d); free(up); free(delta); free(mag); free(sum); free(msum); free(take);

// Spins, couplings as an n x n array: E = s J s / 2 + h s.
double pfa_fold_spin(const double *J, const double *h, int n, float *a, const float *b) {
    FOLD_NEW(n)
    for (int i = 0; i < n; i++) if (a[i] != b[i]) { up[nd] = nd; d[nd++] = i; }
    for (int p = 0; p < nd; p++) {
        const int i = d[p];
        const double *Ji = J + (long)i * n, ai = (double)a[i];
        double f = h[i], m = fabs(h[i]);
        for (int j = 0; j < n; j++) { f += Ji[j] * (double)a[j]; m += fabs(Ji[j]); }
        delta[p] = -2.0 * ai * f; mag[p] = 2.0 * m;   // flipping i alone
        for (int q = p + 1; q < nd; q++) {
            const int j = d[q];
            if (Ji[j] != 0.0) {
                delta[p] += 4.0 * Ji[j] * ai * (double)a[j]; mag[p] += 4.0 * fabs(Ji[j]);
                fold_join(up, p, q);
            }
        }
    }
    const double gain = fold_clusters(nd, up, delta, mag, sum, msum, take);
    for (int p = 0; p < nd; p++) if (take[p]) a[d[p]] = b[d[p]];
    FOLD_FREE
    return gain;
}

// Spins, couplings by row as in pfa_anneal_sparse_spin.
double pfa_fold_sparse_spin(const int64_t *ptr, const int *idx, const double *val, const double *h, int n, float *a,
                            const float *b) {
    FOLD_NEW(n)
    int *pos = malloc(sizeof(int) * n);   // place of a differing spin in d, else -1
    for (int i = 0; i < n; i++) {
        pos[i] = -1;
        if (a[i] != b[i]) { pos[i] = nd; up[nd] = nd; d[nd++] = i; }
    }
    for (int p = 0; p < nd; p++) {
        const int i = d[p];
        const double ai = (double)a[i];
        double f = h[i], m = fabs(h[i]), pairs = 0.0, mp = 0.0;
        for (int64_t e = ptr[i]; e < ptr[i + 1]; e++) {
            const int j = idx[e];
            f += val[e] * (double)a[j]; m += fabs(val[e]);
            if (pos[j] > p && val[e] != 0.0) {
                pairs += 4.0 * val[e] * ai * (double)a[j]; mp += 4.0 * fabs(val[e]);
                fold_join(up, p, pos[j]);
            }
        }
        delta[p] = -2.0 * ai * f + pairs; mag[p] = 2.0 * m + mp;
    }
    const double gain = fold_clusters(nd, up, delta, mag, sum, msum, take);
    for (int p = 0; p < nd; p++) if (take[p]) a[d[p]] = b[d[p]];
    free(pos);
    FOLD_FREE
    return gain;
}

// Categorical variables, with the slots and the couplings read as in eliminate.c (both layouts); a, b:
// the slot of each variable's value, -1 at value 0.
double pfa_fold_categorical(const int *first, int V, const int *vptr, const int *adj, const int64_t *seg,
                            const int64_t *rows, const int64_t *len, const double *vals, const double *fields, int *a,
                            const int *b) {
    FOLD_NEW(V)
    int *pos = malloc(sizeof(int) * V);   // place of a differing variable in d, else -1
    for (int r = 0; r < V; r++) {
        pos[r] = -1;
        if (a[r] != b[r]) { pos[r] = nd; up[nd] = nd; d[nd++] = r; }
    }
    for (int p = 0; p < nd; p++) {
        const int r = d[p], ar = a[r], br = b[r];
        const double *La = ar >= 0 ? vals + rows[r] + (int64_t)(ar - first[r]) * len[r] : 0;   // couplings of a_r
        const double *Lb = br >= 0 ? vals + rows[r] + (int64_t)(br - first[r]) * len[r] : 0;   // and of b_r
        double fa = ar >= 0 ? fields[ar] : 0.0, fb = br >= 0 ? fields[br] : 0.0, pairs = 0.0, m = fabs(fa) + fabs(fb);
        for (int e = vptr[r]; e < vptr[r + 1]; e++) {
            const int s = adj[e], as = a[s], bs = b[s];
            const int64_t at = seg[e] - first[s];
            const double Taa = La && as >= 0 ? La[at + as] : 0.0, Tba = Lb && as >= 0 ? Lb[at + as] : 0.0;
            fa += Taa; fb += Tba; m += fabs(Taa) + fabs(Tba);   // the fields of a_r and b_r in state a
            if (pos[s] > p) {
                const double Tab = La && bs >= 0 ? La[at + bs] : 0.0, Tbb = Lb && bs >= 0 ? Lb[at + bs] : 0.0;
                const double both = ((Taa - Tab) - Tba) + Tbb;
                if (both != 0.0) {
                    pairs += both; m += fabs(Taa) + fabs(Tab) + fabs(Tba) + fabs(Tbb);
                    fold_join(up, p, pos[s]);
                }
            }
        }
        delta[p] = (fb - fa) + pairs; mag[p] = m;
    }
    const double gain = fold_clusters(nd, up, delta, mag, sum, msum, take);
    for (int p = 0; p < nd; p++) if (take[p]) a[d[p]] = b[d[p]];
    free(pos);
    FOLD_FREE
    return gain;
}

// Variables that exchange values (swap.c): A n x n between the variables, B q x q between the values, f n x q; a, b
// the value of each variable, with the same count of variables at every value. Taking a cluster must keep the
// counts, so the clusters are joined further along the cycles of the exchanges: an arc a_r -> b_r for each differing
// r, and every value has as many arcs in as out, so the arcs split into cycles, each of which moves one variable in
// and one out of every value it passes. The differing variables of a cycle go into one cluster, and every cluster is
// then balanced. For a permutation the cycles are unique and the result is the lowest state that has each cluster
// from a or from b; otherwise which arcs form a cycle is one choice among several, which loses choices and nothing
// else.
double pfa_fold_swap(const double *A, const double *B, const double *f, int n, int q, int *a, const int *b) {
    FOLD_NEW(n)
    for (int r = 0; r < n; r++) if (a[r] != b[r]) { up[nd] = nd; d[nd++] = r; }
    for (int p = 0; p < nd; p++) {
        const int r = d[p], ar = a[r], br = b[r];
        const double *Ar = A + (long)r * n, *Ba = B + (long)ar * q, *Bb = B + (long)br * q;
        const double fa = f[(long)r * q + ar], fb = f[(long)r * q + br];
        double del = fb - fa, m = fabs(fa) + fabs(fb);
        for (int s = 0; s < n; s++) {   // r alone (A_rr = 0)
            del += Ar[s] * (Bb[a[s]] - Ba[a[s]]); m += fabs(Ar[s]) * (fabs(Bb[a[s]]) + fabs(Ba[a[s]]));
        }
        for (int o = p + 1; o < nd; o++) {
            const int s = d[o], as = a[s], bs = b[s];
            const double both = Ar[s] * (((Ba[as] - Ba[bs]) - Bb[as]) + Bb[bs]);
            if (both != 0.0) {
                del += both; m += fabs(Ar[s]) * (fabs(Ba[as]) + fabs(Ba[bs]) + fabs(Bb[as]) + fabs(Bb[bs]));
                fold_join(up, p, o);
            }
        }
        delta[p] = del; mag[p] = m;
    }
    // the arcs out of each value, as places in d: out[first[v] .. first[v + 1]); next[v]: the first unused one
    int *first = calloc(q + 1, sizeof(int)), *next = malloc(sizeof(int) * q), *out = malloc(sizeof(int) * (nd + 1));
    int *trail = malloc(sizeof(int) * (nd + 1)), *lastpos = malloc(sizeof(int) * q);
    for (int p = 0; p < nd; p++) first[a[d[p]] + 1]++;
    for (int v = 0; v < q; v++) { first[v + 1] += first[v]; next[v] = first[v]; lastpos[v] = -1; }
    for (int p = 0; p < nd; p++) out[next[a[d[p]]]++] = p;
    for (int v = 0; v < q; v++) next[v] = first[v];
    for (int v0 = 0; v0 < q; v0++) {
        while (next[v0] < first[v0 + 1]) {   // a walk from v0 along unused arcs: it can only end at v0
            int len = 0, v = v0;
            lastpos[v0] = 0;
            while (next[v] < first[v + 1]) {
                const int p = out[next[v]++], w = b[d[p]];
                trail[len++] = p;
                if (lastpos[w] >= 0) {   // the arcs since the walk was last at w form a cycle: one cluster
                    const int from = lastpos[w];
                    for (int i = from; i < len; i++) {
                        fold_join(up, trail[from], trail[i]);
                        lastpos[b[d[trail[i]]]] = -1;
                    }
                    len = from; lastpos[w] = from;
                } else lastpos[w] = len;
                v = w;
            }
            if (len) __builtin_trap();   // the counts of a and b differ
            lastpos[v0] = -1;
        }
    }
    const double gain = fold_clusters(nd, up, delta, mag, sum, msum, take);
    for (int p = 0; p < nd; p++) if (take[p]) a[d[p]] = b[d[p]];
    free(first); free(next); free(out); free(trail); free(lastpos);
    FOLD_FREE
    return gain;
}
#undef FOLD_NEW
#undef FOLD_FREE
