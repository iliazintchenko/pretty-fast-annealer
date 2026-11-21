// Edge assembly between two tours (included by pfa.c): Nagata's EAX in its single-cycle form, the fold for tours.
// Two low tours do not differ in clusters a partition could exchange: their differing edges form a graph in which
// every city has as many edges of one tour as of the other, and it decomposes into AB-cycles, cycles alternating an
// edge of A and an edge of B. Removing a cycle's A-edges from A and adding its B-edges gives a 2-regular graph, a set
// of subtours, merged into one tour by the cheapest 2-exchange between the smallest subtour and any other (the
// exchange's first edge at a candidate pair), repeatedly. Each AB-cycle gives one child, both ways round (B's edges
// into A, A's into B); the shortest child is returned.

static inline double fold_d(const double *x, const double *y, int round_mode, int i, int j) {
    double dx = x[i] - x[j], dy = y[i] - y[j], r = sqrt(dx * dx + dy * dy);
    if (round_mode == 1) return floor(r + 0.5);
    if (round_mode == 2) return ceil(r);
    return r;
}

typedef struct {
    int n, K, round_mode;
    const double *x, *y;
    const int *nn;
    int *label, *size, *member, *stack;   // subtours: per city its subtour, per subtour its size and one member
    int *adj;                             // 2n, the child under construction
    double best;
    int *best_adj;
} Fold;

static inline void fold_replace(int *adj, int v, int from, int to) { if (adj[2 * v] == from) adj[2 * v] = to; else adj[2 * v + 1] = to; }

// the subtours of adj: returns their number
static int fold_subtours(Fold *F) {
    const int n = F->n;
    for (int v = 0; v < n; v++) F->label[v] = -1;
    int cnt = 0;
    for (int s = 0; s < n; s++) {
        if (F->label[s] >= 0) continue;
        int prev = -1, cur = s, sz = 0;
        F->member[cnt] = s;
        for (;;) {
            F->label[cur] = cnt; sz++;
            int a = F->adj[2 * cur], b = F->adj[2 * cur + 1], nxt = a != prev ? a : b;
            if (nxt == s || F->label[nxt] >= 0) break;
            prev = cur; cur = nxt;
        }
        F->size[cnt++] = sz;
    }
    return cnt;
}

// merges the subtours of adj into one tour
static void fold_merge(Fold *F) {
    const int n = F->n, K = F->K;
    const double *x = F->x, *y = F->y;
    const int rm = F->round_mode;
    int cnt = fold_subtours(F);
    while (cnt > 1) {
        int s = 0;
        for (int i = 1; i < cnt; i++) if (F->size[i] < F->size[s]) s = i;
        double best = INFINITY;
        int bu = -1, bv = -1, bp = -1, bq = -1;
        int prev = -1, u = F->member[s];
        for (int q = 0; q < F->size[s]; q++) {   // every edge (u, v) of the smallest subtour once, as it is walked
            int a = F->adj[2 * u], b = F->adj[2 * u + 1], v = a != prev ? a : b;
            for (int k = 0; k < K; k++) {
                int p = F->nn[u * K + k];
                if (F->label[p] == s) continue;
                for (int j = 0; j < 2; j++) {
                    int qq = F->adj[2 * p + j];
                    double base = -fold_d(x, y, rm, u, v) - fold_d(x, y, rm, p, qq);
                    double d1 = fold_d(x, y, rm, u, p) + fold_d(x, y, rm, v, qq) + base;
                    double d2 = fold_d(x, y, rm, u, qq) + fold_d(x, y, rm, v, p) + base;
                    if (d1 < best) { best = d1; bu = u; bv = v; bp = p; bq = qq; }
                    if (d2 < best) { best = d2; bu = u; bv = v; bp = qq; bq = p; }
                }
            }
            prev = u; u = v;
        }
        if (bu < 0) {   // no candidate reaches another subtour: any edge of another subtour
            int other = s == 0 ? 1 : 0;
            bu = F->member[s]; bv = F->adj[2 * bu]; bp = F->member[other]; bq = F->adj[2 * bp];
        }
        // remove (bu, bv) and (bp, bq), add (bu, bp) and (bv, bq)
        fold_replace(F->adj, bu, bv, bp); fold_replace(F->adj, bv, bu, bq);
        fold_replace(F->adj, bp, bq, bu); fold_replace(F->adj, bq, bp, bv);
        cnt = fold_subtours(F);
    }
}

// the children with A as the base: its differing edges removed and B's added along each AB-cycle
static void fold_children(Fold *F, const int *adjA, const int *adjB, uint64_t *rng) {
    const int n = F->n;
    int *dA = malloc(sizeof(int) * 2 * n), *dB = malloc(sizeof(int) * 2 * n), *cA = calloc(n, sizeof(int)), *cB = calloc(n, sizeof(int));
    for (int v = 0; v < n; v++)
        for (int j = 0; j < 2; j++) {
            int a = adjA[2 * v + j], b = adjB[2 * v + j];
            if (a != adjB[2 * v] && a != adjB[2 * v + 1]) dA[2 * v + cA[v]++] = a;
            if (b != adjA[2 * v] && b != adjA[2 * v + 1]) dB[2 * v + cB[v]++] = b;
        }
    int *order = malloc(sizeof(int) * n), no = 0;
    for (int v = 0; v < n; v++) if (cA[v]) order[no++] = v;
    for (int i = no - 1; i > 0; i--) { int j = (int)((tour_rng(rng) >> 32) % (uint64_t)(i + 1)), t = order[i]; order[i] = order[j]; order[j] = t; }
    int *path = malloc(sizeof(int) * (2 * n + 2)), *at = malloc(sizeof(int) * n);
    int (*cyc)[3] = malloc(sizeof(int[3]) * (2 * n + 2));
    for (int v = 0; v < n; v++) at[v] = -1;
    for (int oi = 0; oi < no; oi++) {
        int start = order[oi];
        while (cA[start] > 0) {
            int plen = 0, which = 0, v = start;
            path[plen] = start; at[start] = plen++;
            for (;;) {
                int *lst = which == 0 ? dA : dB, *cnt = which == 0 ? cA : cB;
                if (cnt[v] == 0) break;
                int r = (int)((tour_rng(rng) >> 32) % (uint64_t)cnt[v]), u = lst[2 * v + r];
                lst[2 * v + r] = lst[2 * v + cnt[v] - 1]; cnt[v]--;
                int *lu = lst + 2 * u;
                if (lu[0] == v) { lu[0] = lu[cnt[u] - 1]; } else { lu[1] = lu[cnt[u] - 1]; }
                cnt[u]--;
                which ^= 1;
                if (at[u] >= 0 && (plen - at[u]) % 2 == 0) {   // a closed alternating cycle from the earlier visit of u
                    int k = at[u], nc = 0, first = k % 2;
                    for (int i = k; i < plen; i++) { cyc[nc][0] = path[i]; cyc[nc][1] = i + 1 < plen ? path[i + 1] : u; cyc[nc][2] = (first + nc) % 2; nc++; }
                    // the child: A without the cycle's A-edges, with its B-edges
                    memcpy(F->adj, adjA, sizeof(int) * 2 * n);
                    for (int i = 0; i < nc; i++) if (cyc[i][2] == 0) { fold_replace(F->adj, cyc[i][0], cyc[i][1], -1); fold_replace(F->adj, cyc[i][1], cyc[i][0], -1); }
                    for (int i = 0; i < nc; i++) if (cyc[i][2] == 1) { fold_replace(F->adj, cyc[i][0], -1, cyc[i][1]); fold_replace(F->adj, cyc[i][1], -1, cyc[i][0]); }
                    fold_merge(F);
                    double L = 0.0;
                    for (int z = 0; z < n; z++) L += fold_d(F->x, F->y, F->round_mode, z, F->adj[2 * z]) + fold_d(F->x, F->y, F->round_mode, z, F->adj[2 * z + 1]);
                    L *= 0.5;
                    if (L < F->best) { F->best = L; memcpy(F->best_adj, F->adj, sizeof(int) * 2 * n); }
                    for (int i = 0; i < k; i++) {   // the edges before the cycle on the path go back to the lists
                        int a = path[i], b = path[i + 1], w = i % 2;
                        int *l = w == 0 ? dA : dB, *c = w == 0 ? cA : cB;
                        l[2 * a + c[a]++] = b; l[2 * b + c[b]++] = a;
                    }
                    for (int i = 0; i < plen; i++) at[path[i]] = -1;
                    break;
                }
                at[u] = plen; path[plen++] = u;
                v = u;
            }
            for (int i = 0; i < plen; i++) at[path[i]] = -1;
        }
    }
    free(dA); free(dB); free(cA); free(cB); free(order); free(path); free(at); free(cyc);
}

// a, b: two tours as the city at each position; out: the shortest child as the same, when one exists. Returns its
// length, or infinity when the tours share every edge.
double pfa_tour_fold(int n, const double *x, const double *y, int round_mode, int K, const int *nn, const int *a,
                     const int *b, int *out, uint64_t seed) {
    int *adjA = malloc(sizeof(int) * 2 * n), *adjB = malloc(sizeof(int) * 2 * n);
    for (int p = 0; p < n; p++) {
        int pa = a[p == 0 ? n - 1 : p - 1], na_ = a[p + 1 == n ? 0 : p + 1], pb = b[p == 0 ? n - 1 : p - 1], nb_ = b[p + 1 == n ? 0 : p + 1];
        adjA[2 * a[p]] = pa; adjA[2 * a[p] + 1] = na_;
        adjB[2 * b[p]] = pb; adjB[2 * b[p] + 1] = nb_;
    }
    Fold F = {n, K, round_mode, x, y, nn, malloc(sizeof(int) * n), malloc(sizeof(int) * n), malloc(sizeof(int) * n),
              malloc(sizeof(int) * n), malloc(sizeof(int) * 2 * n), INFINITY, malloc(sizeof(int) * 2 * n)};
    uint64_t rng = seed | 1;
    fold_children(&F, adjA, adjB, &rng);
    fold_children(&F, adjB, adjA, &rng);
    if (F.best < INFINITY) {
        int prev = -1, cur = 0;
        for (int p = 0; p < n; p++) {
            out[p] = cur;
            int x0 = F.best_adj[2 * cur], x1 = F.best_adj[2 * cur + 1], nxt = x0 != prev ? x0 : x1;
            prev = cur; cur = nxt;
        }
    }
    double best = F.best;
    free(adjA); free(adjB); free(F.label); free(F.size); free(F.member); free(F.stack); free(F.adj); free(F.best_adj);
    return best;
}
