// Candidate edges of a tour problem by alpha-nearness (included by pfa.c). The candidate list is the move set: an
// edge on no city's list never enters the tour, so the list is the problem's coupling graph as the kernel sees it.
// The K nearest cities miss the long edges an optimal tour has to use; the K alpha-nearest nearly never do.
//
// Held and Karp's relaxation: a 1-tree is a spanning tree over all cities but one plus that city's two cheapest
// edges; every tour is a 1-tree, so the minimum 1-tree is a lower bound on the tour, and the bound stays valid under
// a penalty pi_i added to every edge at city i (all tours change by the same 2 sum pi). Subgradient ascent on pi
// pushes every city's degree in the tree towards 2, where the tree is nearly a tour. The alpha of an edge is the
// increase of the minimum 1-tree's length when the edge is forced in: 0 for its edges, and otherwise the edge's
// penalised length minus the longest edge on the tree path between its ends (Helsgaun 2000). The K edges of
// smallest alpha at each city are its candidates.
//
// Costs O(n^2) per ascent iteration and in memory for the distances and the path maxima; the ascent runs at most a
// hundred iterations. nn_out: n x K candidate cities; the bound reached is returned.
#include <math.h>
#include <stdlib.h>
#include <string.h>

static inline float alpha_d(const double *x, const double *y, int round_mode, int i, int j) {
    double dx = x[i] - x[j], dy = y[i] - y[j], r = sqrt(dx * dx + dy * dy);
    if (round_mode == 1) return (float)floor(r + 0.5);
    if (round_mode == 2) return (float)ceil(r);
    return (float)r;
}

// Prim over the cities but s from (s + 1) % n, then s's two cheapest edges. order: the insertion order of the n - 1
// cities; parent[u] and key[u], its tree edge and that edge's penalised length; deg: the 1-tree degrees. Returns the
// 1-tree's penalised length.
static double alpha_tree(int n, const float *D, const double *pi, int s, int *parent, int *order, double *key,
                         int *deg, unsigned char *in) {
    const int r = (s + 1) % n;
    for (int j = 0; j < n; j++) { key[j] = INFINITY; in[j] = 0; deg[j] = 0; parent[j] = -1; }
    key[r] = 0.0;
    double L = 0.0;
    for (int it = 0; it < n - 1; it++) {
        int u = -1;
        for (int j = 0; j < n; j++) if (j != s && !in[j] && (u < 0 || key[j] < key[u])) u = j;
        in[u] = 1; order[it] = u;
        if (parent[u] >= 0) { deg[u]++; deg[parent[u]]++; L += key[u]; }
        const float *Du = D + (long)u * n;
        for (int j = 0; j < n; j++) {
            if (j == s || in[j]) continue;
            double w = Du[j] + pi[u] + pi[j];
            if (w < key[j]) { key[j] = w; parent[j] = u; }
        }
    }
    int j1 = -1, j2 = -1;
    for (int j = 0; j < n; j++) {
        if (j == s) continue;
        double w = D[(long)s * n + j] + pi[s] + pi[j];
        if (j1 < 0 || w < D[(long)s * n + j1] + pi[s] + pi[j1]) { j2 = j1; j1 = j; }
        else if (j2 < 0 || w < D[(long)s * n + j2] + pi[s] + pi[j2]) j2 = j;
    }
    deg[s] = 2; deg[j1]++; deg[j2]++;
    return L + D[(long)s * n + j1] + pi[s] + pi[j1] + D[(long)s * n + j2] + pi[s] + pi[j2];
}

double pfa_tour_alpha(int n, const double *x, const double *y, int round_mode, int K, int *nn_out) {
    float *D = malloc(sizeof(float) * (long)n * n);
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) D[(long)i * n + j] = alpha_d(x, y, round_mode, i, j);
    double *pi = calloc(n, sizeof(double)), *pi_best = calloc(n, sizeof(double)), *vprev = calloc(n, sizeof(double));
    double *key = malloc(sizeof(double) * n);
    int *parent = malloc(sizeof(int) * n), *order = malloc(sizeof(int) * n), *deg = malloc(sizeof(int) * n);
    unsigned char *in = malloc(n);
    // an upper bound for the step sizes: the nearest-neighbour tour from city 0
    double UB = 0.0;
    {
        memset(in, 0, n);
        int u = 0; in[0] = 1;
        for (int it = 1; it < n; it++) {
            int v = -1;
            for (int j = 0; j < n; j++) if (!in[j] && (v < 0 || D[(long)u * n + j] < D[(long)u * n + v])) v = j;
            UB += D[(long)u * n + v]; in[v] = 1; u = v;
        }
        UB += D[(long)u * n];
    }
    const int s = 0;
    double Wbest = -INFINITY, lambda = 2.0;
    int noimp = 0;
    for (int it = 0; it < 100; it++) {   // Polyak steps on the subgradient deg - 2, with Helsgaun's 0.7 / 0.3 momentum
        double L = alpha_tree(n, D, pi, s, parent, order, key, deg, in), sum = 0.0, norm = 0.0;
        for (int i = 0; i < n; i++) { sum += pi[i]; norm += (double)(deg[i] - 2) * (deg[i] - 2); }
        double W = L - 2.0 * sum;
        if (W > Wbest) { Wbest = W; memcpy(pi_best, pi, sizeof(double) * n); noimp = 0; }
        else if (++noimp >= 5) { lambda *= 0.5; noimp = 0; }
        if (norm == 0.0 || lambda < 1e-3 || UB <= W) break;
        double t = lambda * (UB - W) / norm;
        for (int i = 0; i < n; i++) { double v = deg[i] - 2; pi[i] += t * (0.7 * v + 0.3 * vprev[i]); vprev[i] = v; }
    }
    alpha_tree(n, D, pi_best, s, parent, order, key, deg, in);
    // the longest edge on the tree path between every two cities but s, in insertion order: a city's path to an
    // earlier city runs through its parent
    float *B = calloc((long)n * n, sizeof(float));
    for (int it = 1; it < n - 1; it++) {
        int u = order[it], p = parent[u];
        float wu = (float)key[u];
        for (int jj = 0; jj < it; jj++) {
            int j = order[jj];
            float b = B[(long)p * n + j] > wu ? B[(long)p * n + j] : wu;
            B[(long)u * n + j] = b; B[(long)j * n + u] = b;
        }
    }
    double d2s = INFINITY, d1s = INFINITY;   // s's two cheapest penalised edges: an edge at s costs its excess over the second
    for (int j = 1; j < n; j++) {
        double w = D[(long)s * n + j] + pi_best[s] + pi_best[j];
        if (w < d1s) { d2s = d1s; d1s = w; } else if (w < d2s) d2s = w;
    }
    double *ba = malloc(sizeof(double) * K), *bd = malloc(sizeof(double) * K);
    for (int i = 0; i < n; i++) {
        int cnt = 0, *bj = nn_out + (long)i * K;
        for (int j = 0; j < n; j++) {
            if (j == i) continue;
            double d = D[(long)i * n + j], w = d + pi_best[i] + pi_best[j];
            double a = (i == s || j == s) ? (w > d2s ? w - d2s : 0.0) : w - B[(long)i * n + j];
            if (cnt == K && (a > ba[K - 1] || (a == ba[K - 1] && d >= bd[K - 1]))) continue;
            int q = cnt < K ? cnt++ : K - 1;
            while (q > 0 && (a < ba[q - 1] || (a == ba[q - 1] && d < bd[q - 1]))) { ba[q] = ba[q - 1]; bd[q] = bd[q - 1]; bj[q] = bj[q - 1]; q--; }
            ba[q] = a; bd[q] = d; bj[q] = j;
        }
    }
    free(D); free(pi); free(pi_best); free(vprev); free(key); free(parent); free(order); free(deg); free(in); free(B);
    free(ba); free(bd);
    return Wbest;
}
