// The transcription fold for tours (included by pfa.c after tour_fold.c): iterative partial transcription (Mobius,
// Freisleben, Merz and Schreiber 1999), in the form of LKH's merge. Two tours are shrunk to the cities at which they
// differ, the common paths between them collapsed; a block is a run of cities that is a path in both tours with the
// same two ends; the shorter of its two paths replaces the longer, smallest blocks first, until no block is left; the
// shorter of the two tours so improved is expanded back and returned. Where edge assembly exchanges one alternating
// cycle, this exchanges every block whose two readings differ, independently.
static double ipt_sum(const double *c, int m, int from, int count) {
    double s = 0.0;
    for (int q = 0; q < count; q++) s += c[(from + q) % m];
    return s;
}

// a, b: the two tours (city at each position); out: the better tour after the exchanges. Returns its length.
double pfa_tour_ipt(int n, const double *x, const double *y, int round_mode, const int *a, const int *b, int *out) {
    int *an = malloc(sizeof(int) * 2 * n), *bn = malloc(sizeof(int) * 2 * n);
    unsigned char *V = malloc(n);
    for (int p = 0; p < n; p++) {
        int c = a[p]; an[2 * c] = a[p == 0 ? n - 1 : p - 1]; an[2 * c + 1] = a[p + 1 == n ? 0 : p + 1];
        c = b[p]; bn[2 * c] = b[p == 0 ? n - 1 : p - 1]; bn[2 * c + 1] = b[p + 1 == n ? 0 : p + 1];
    }
    int m = 0, first = -1;
    for (int c = 0; c < n; c++) {
        int same = (an[2 * c] == bn[2 * c] && an[2 * c + 1] == bn[2 * c + 1]) || (an[2 * c] == bn[2 * c + 1] && an[2 * c + 1] == bn[2 * c]);
        V[c] = !same; m += V[c];
        if (V[c] && first < 0) first = c;
    }
    double L = 0.0;
    if (m < 3) {
        memcpy(out, a, sizeof(int) * n);
        for (int p = 0; p < n; p++) L += fold_d(x, y, round_mode, a[p], a[p + 1 == n ? 0 : p + 1]);
        free(an); free(bn); free(V);
        return L;
    }
    // the shrunken tours: s1 the differing cities in a's order from first, s2 in b's order from first; r1, r2 their
    // positions; c1[p], c2[p] the length between position p and p + 1, zero for a collapsed common path
    int *s1 = malloc(sizeof(int) * m), *s2 = malloc(sizeof(int) * m), *r1 = malloc(sizeof(int) * n), *r2 = malloc(sizeof(int) * n);
    double *c1 = malloc(sizeof(double) * m), *c2 = malloc(sizeof(double) * m);
    int *pos = malloc(sizeof(int) * n);
    for (int p = 0; p < n; p++) pos[a[p]] = p;
    for (int q = 0, p = pos[first], k = 0; q < n; q++, p = p + 1 == n ? 0 : p + 1) if (V[a[p]]) { s1[k] = a[p]; r1[a[p]] = k; k++; }
    for (int p = 0; p < n; p++) pos[b[p]] = p;
    for (int q = 0, p = pos[first], k = 0; q < n; q++, p = p + 1 == n ? 0 : p + 1) if (V[b[p]]) { s2[k] = b[p]; r2[b[p]] = k; k++; }
    for (int p = 0; p < m; p++) {
        int u = s1[p], v = s1[(p + 1) % m];
        c1[p] = (an[2 * u] == v || an[2 * u + 1] == v) ? fold_d(x, y, round_mode, u, v) : 0.0;
        u = s2[p]; v = s2[(p + 1) % m];
        c2[p] = (bn[2 * u] == v || bn[2 * u + 1] == v) ? fold_d(x, y, round_mode, u, v) : 0.0;
    }
    double cost1 = ipt_sum(c1, m, 0, m), cost2 = ipt_sum(c2, m, 0, m);
    int best_min = 3;
    for (;;) {
        int min_size = m / 2, min_i = -1, min_count = 0, min_dir = 0;
        double min_gain = 0.0;
        for (int i0 = 0; i0 < m; i0++) {
            int N1 = s1[i0], suc1 = s1[(i0 + 1) % m], pred1 = s1[(i0 + m - 1) % m];
            if (suc1 == s2[(r2[N1] + 1) % m] || suc1 == s2[(r2[N1] + m - 1) % m]) continue;   // the first edge is common
            for (int dir = 1; dir >= -1; dir -= 2) {
                int j = r2[N1], count = 0, max1 = 0;
                double len2 = 0.0;
                for (;;) {
                    int jn = (j + dir + m) % m;
                    len2 += dir > 0 ? c2[j] : c2[jn];
                    j = jn; count++;
                    int N2 = s2[j];
                    if (count == 1 && (N2 == suc1 || N2 == pred1)) break;
                    if (count >= min_size) break;
                    int off = (r1[N2] - i0 + m) % m;
                    if (off >= min_size) break;
                    if (off > max1) {
                        if (off == count) {   // the walked cities are exactly the block s1[i0 .. i0 + count]
                            double gain = ipt_sum(c1, m, i0, count) - len2;
                            if (fabs(gain) > 1e-9) { min_size = off; min_i = i0; min_count = count; min_dir = dir; min_gain = gain; }
                            break;
                        }
                        max1 = off;
                    }
                }
            }
            if (min_i >= 0 && min_size == best_min) break;
        }
        if (min_i < 0) break;
        best_min = min_size;
        int N1 = s1[min_i], j0 = r2[N1];
        if (min_gain > 0.0) {   // the first tour takes the second's path through the block
            for (int q = 0; q <= min_count; q++) { int c = s2[(j0 + min_dir * q + m * 2) % m]; s1[(min_i + q) % m] = c; r1[c] = (min_i + q) % m; }
            for (int q = 0; q < min_count; q++) { int j = (j0 + min_dir * q + 2 * m) % m; c1[(min_i + q) % m] = min_dir > 0 ? c2[j] : c2[(j + m - 1) % m]; }
            cost1 -= min_gain;
        } else {   // the second takes the first's
            for (int q = 0; q <= min_count; q++) { int c = s1[(min_i + q) % m], j = (j0 + min_dir * q + 2 * m) % m; s2[j] = c; r2[c] = j; }
            for (int q = 0; q < min_count; q++) { int j = (j0 + min_dir * q + 2 * m) % m; if (min_dir > 0) c2[j] = c1[(min_i + q) % m]; else c2[(j + m - 1) % m] = c1[(min_i + q) % m]; }
            cost2 += min_gain;
        }
    }
    // expand the shorter shrunken tour: each city, then the common path to the next where the two are not adjacent
    const int *s = cost1 <= cost2 ? s1 : s2;
    int k = 0;
    int *path = malloc(sizeof(int) * n);
    for (int p = 0; p < m; p++) {
        int u = s[p], v = s[(p + 1) % m];
        out[k++] = u;
        if (an[2 * u] == v || an[2 * u + 1] == v || bn[2 * u] == v || bn[2 * u + 1] == v) continue;
        int done = 0;
        for (int side = 0; side < 2 && !done; side++) {
            int w = an[2 * u + side];
            if (V[w]) continue;
            int prev = u, cur = w, len = 0;
            while (!V[cur]) { path[len++] = cur; int nx = an[2 * cur] == prev ? an[2 * cur + 1] : an[2 * cur]; prev = cur; cur = nx; }
            if (cur == v) { memcpy(out + k, path, sizeof(int) * len); k += len; done = 1; }
        }
    }
    for (int p = 0; p < n; p++) L += fold_d(x, y, round_mode, out[p], out[p + 1 == n ? 0 : p + 1]);
    free(an); free(bn); free(V); free(s1); free(s2); free(r1); free(r2); free(c1); free(c2); free(pos); free(path);
    return L;
}
