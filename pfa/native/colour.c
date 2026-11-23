// Graph colouring with k colours: every vertex one colour, the energy the number of edges whose ends share one. The
// state keeps, per vertex and colour, the number of neighbours in that colour (gamma), so a move's delta is one
// subtraction and its update one pass over the moved vertex's neighbours; the coupling is the graph once and the
// identity on colours, never a table per edge. The kernel is TabuCol in Galinier and Hao's form as HEAD runs it: one
// conflicting vertex recoloured per iteration, the best non-tabu move over the conflicting vertices (each vertex's
// best colours cached and maintained), ties broken uniformly, aspiration when a move beats the call's best, the colour
// left tabu for 0.6 |C| + rand(10) iterations, C the conflicting set; zero temperature. The fold is Galinier and Hao's
// partition crossover (GPX): the largest remaining colour class taken from the parents in turn, the rest coloured at
// random.
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int V, k;
    const int *off, *nbr;              // CSR adjacency, each edge in both lists
    int *col, *gamma, *tabu, *conf;    // colour; neighbours per colour (V*k); iteration until which v->c is tabu; conf[v] = gamma[v][col v]
    int *clist, *cpos, nc;             // the conflicting set as a list with positions
    int *bval, *bcnt, *blist;          // per vertex the least gamma over the other colours, how many colours attain it, and which (V*k)
    long total, best;                  // conflicting edges now and the call's best
    long it, best_it;
    double csum;                       // |C| summed over iterations
    uint64_t rng;
    const int *fixed;                  // vertices never moved (the pair test's fold arm), or NULL
} Col;

static inline uint64_t col_rng(uint64_t *s) { *s ^= *s >> 12; *s ^= *s << 25; *s ^= *s >> 27; return *s * 2685821657736338717ULL; }
static inline int col_rnd(uint64_t *s, int n) { return (int)(((col_rng(s) >> 32) * (uint64_t)n) >> 32); }
#define CG(S, v, c) (S)->gamma[(size_t)(v) * (S)->k + (c)]

static inline void col_cadd(Col *S, int v) { S->cpos[v] = S->nc; S->clist[S->nc++] = v; }
static inline void col_cdel(Col *S, int v) { int p = S->cpos[v], w = S->clist[--S->nc]; S->clist[p] = w; S->cpos[w] = p; S->cpos[v] = -1; }

static void col_best_of(Col *S, int v) {
    const int k = S->k, cv = S->col[v];
    int b = 1 << 30, n = 0, *l = S->blist + (size_t)v * k;
    for (int c = 0; c < k; c++) {
        if (c == cv) continue;
        int g = CG(S, v, c);
        if (g < b) { b = g; n = 0; }
        if (g == b) l[n++] = c;
    }
    S->bval[v] = b; S->bcnt[v] = n;
}

static void col_init(Col *S, int V, int k, const int *off, const int *nbr, int *col, uint64_t seed) {
    S->V = V; S->k = k; S->off = off; S->nbr = nbr; S->col = col;
    S->gamma = calloc((size_t)V * k, sizeof(int));
    S->tabu = calloc((size_t)V * k, sizeof(int));
    S->conf = malloc(V * sizeof(int));
    S->clist = malloc(V * sizeof(int));
    S->cpos = malloc(V * sizeof(int));
    S->bval = malloc(V * sizeof(int));
    S->bcnt = malloc(V * sizeof(int));
    S->blist = malloc((size_t)V * k * sizeof(int));
    S->rng = (seed << 1) | 1;
    S->fixed = NULL;
    for (int v = 0; v < V; v++)
        for (int i = off[v]; i < off[v + 1]; i++) CG(S, v, col[nbr[i]])++;
    S->nc = 0; S->total = 0;
    for (int v = 0; v < V; v++) {
        S->conf[v] = CG(S, v, col[v]);
        S->total += S->conf[v];
        S->cpos[v] = -1;
        if (S->conf[v]) col_cadd(S, v);
        col_best_of(S, v);
    }
    S->total /= 2;
    S->best = S->total;
    S->it = 0; S->best_it = 0; S->csum = 0;
}

static void col_free(Col *S) {
    free(S->gamma); free(S->tabu); free(S->conf); free(S->clist); free(S->cpos);
    free(S->bval); free(S->bcnt); free(S->blist);
}

// The move v: col[v] -> c, the gamma update over the neighbours and the maintenance of their cached best colours.
static void col_apply(Col *S, int v, int c) {
    const int k = S->k, cv = S->col[v];
    S->tabu[(size_t)v * k + cv] = (int)(S->it + col_rnd(&S->rng, 10) + 0.6 * S->nc);
    S->total += CG(S, v, c) - CG(S, v, cv);
    S->col[v] = c;
    for (int i = S->off[v]; i < S->off[v + 1]; i++) {
        int u = S->nbr[i], cu = S->col[u];
        int gcv = --CG(S, u, cv), gc = ++CG(S, u, c);
        if (cu == cv) { if (--S->conf[u] == 0) col_cdel(S, u); }
        else if (cu == c) { if (++S->conf[u] == 1) col_cadd(S, u); }
        if (cv != cu) {                                   // gamma[u][cv] fell by one
            if (gcv < S->bval[u]) { S->bval[u] = gcv; S->bcnt[u] = 1; S->blist[(size_t)u * k] = cv; }
            else if (gcv == S->bval[u]) S->blist[(size_t)u * k + S->bcnt[u]++] = cv;
        }
        if (c != cu && gc - 1 == S->bval[u]) {            // gamma[u][c] rose off the best value
            if (S->bcnt[u] > 1) {
                int *l = S->blist + (size_t)u * k, n = S->bcnt[u], p = 0;
                while (l[p] != c) p++;
                l[p] = l[n - 1]; S->bcnt[u] = n - 1;
            } else col_best_of(S, u);
        }
    }
    S->conf[v] = CG(S, v, c);
    if (S->conf[v] == 0 && S->cpos[v] >= 0) col_cdel(S, v);
    else if (S->conf[v] > 0 && S->cpos[v] < 0) col_cadd(S, v);
    col_best_of(S, v);
}

static inline int col_allowed(const Col *S, int v, int c, int d) {
    return S->tabu[(size_t)v * S->k + c] <= S->it || S->total + d < S->best;
}

// One TabuCol iteration: the best allowed move over the conflicting vertices, ties uniform. Returns 0 when C is empty.
static int col_step(Col *S) {
    if (S->nc == 0) return 0;
    const int k = S->k;
    int best = 1 << 30, nb = 0, bv = -1, bc = -1;
    for (int i = 0; i < S->nc; i++) {
        int v = S->clist[i], cv = S->col[v], gc = S->conf[v];
        int d = S->bval[v] - gc;
        if (d > best || (S->fixed && S->fixed[v])) continue;
        const int *l = S->blist + (size_t)v * k;
        int any = 0;
        for (int j = 0; j < S->bcnt[v]; j++) {
            int c = l[j];
            if (!col_allowed(S, v, c, d)) continue;
            any = 1;
            if (d < best) { best = d; nb = 1; bv = v; bc = c; }
            else if (col_rnd(&S->rng, ++nb) == 0) { bv = v; bc = c; }
        }
        if (!any && d < best) {                           // every best colour tabu: the whole row
            for (int c = 0; c < k; c++) {
                int dd = CG(S, v, c) - gc;
                if (dd > best || c == cv || !col_allowed(S, v, c, dd)) continue;
                if (dd < best) { best = dd; nb = 1; bv = v; bc = c; }
                else if (col_rnd(&S->rng, ++nb) == 0) { bv = v; bc = c; }
            }
        }
    }
    S->csum += S->nc;
    S->it++;
    if (bv >= 0) col_apply(S, bv, bc);
    return 1;
}

// TabuCol with the vertices marked in fixed never moved (the rest searched under them).
long pfa_colour_tabu_fixed(int V, int k, const int *off, const int *nbr, int *col_io, const int *fixed, long iters,
                           uint64_t seed, int *best_out, double *stats) {
    Col S;
    col_init(&S, V, k, off, nbr, col_io, seed);
    S.fixed = fixed;
    memcpy(best_out, col_io, V * sizeof(int));
    while (S.it < iters && col_step(&S)) {
        if (S.total <= S.best) { S.best = S.total; S.best_it = S.it; memcpy(best_out, col_io, V * sizeof(int)); }
    }
    if (stats) {
        stats[0] = (double)S.it; stats[1] = (double)S.best; stats[2] = (double)S.best_it;
        stats[3] = S.it ? S.csum / S.it : 0; stats[4] = (double)S.total;
    }
    long best = S.best;
    col_free(&S);
    return best;
}

// The temperature test's variant of the step: the allowed moves of the conflicting vertices bucketed by their integer
// delta, a bucket drawn with weight n_d e^(-beta (d - dmin)), a uniform member of it taken. beta = 0 keeps TabuCol
// (the best bucket only, which the draw gives as beta -> infinity).
static double col_beta = 0.0;
void pfa_colour_beta(double beta) { col_beta = beta; }

static int col_step_boltz(Col *S) {
    if (S->nc == 0) return 0;
    const int k = S->k;
    enum { L = 64 };
    long n[L];
    int dmin = 1 << 30, dmax = -(1 << 30);
    for (int i = 0; i < S->nc; i++) {                    // the levels present
        int v = S->clist[i], cv = S->col[v], gc = S->conf[v];
        for (int c = 0; c < k; c++) {
            if (c == cv) continue;
            int d = CG(S, v, c) - gc;
            if (!col_allowed(S, v, c, d)) continue;
            if (d < dmin) dmin = d;
            if (d > dmax) dmax = d;
        }
    }
    if (dmin > dmax) { S->csum += S->nc; S->it++; return 1; }   // everything tabu: no move, as HEAD's loop would skip
    if (dmax - dmin >= L) dmax = dmin + L - 1;
    for (int l = 0; l < L; l++) n[l] = 0;
    for (int i = 0; i < S->nc; i++) {
        int v = S->clist[i], cv = S->col[v], gc = S->conf[v];
        for (int c = 0; c < k; c++) {
            if (c == cv) continue;
            int d = CG(S, v, c) - gc;
            if (d > dmax || !col_allowed(S, v, c, d)) continue;
            n[d - dmin]++;
        }
    }
    double tot = 0, w[L];
    for (int l = 0; l <= dmax - dmin; l++) { w[l] = n[l] * exp(-col_beta * l); tot += w[l]; }
    double u = (col_rng(&S->rng) >> 11) * (1.0 / 9007199254740992.0) * tot;
    int lev = 0;
    while (lev < dmax - dmin && u >= w[lev]) { u -= w[lev]; lev++; }
    int target = dmin + lev, nb = 0, bv = -1, bc = -1;
    for (int i = 0; i < S->nc; i++) {                    // a uniform member of the level
        int v = S->clist[i], cv = S->col[v], gc = S->conf[v];
        for (int c = 0; c < k; c++) {
            if (c == cv) continue;
            int d = CG(S, v, c) - gc;
            if (d != target || !col_allowed(S, v, c, d)) continue;
            if (col_rnd(&S->rng, ++nb) == 0) { bv = v; bc = c; }
        }
    }
    S->csum += S->nc;
    S->it++;
    col_apply(S, bv, bc);
    return 1;
}

// Conflicting edges of a colouring.
long pfa_colour_conflicts(int V, const int *off, const int *nbr, const int *col) {
    long t = 0;
    for (int v = 0; v < V; v++)
        for (int i = off[v]; i < off[v + 1]; i++) t += col[nbr[i]] == col[v];
    return t / 2;
}

// TabuCol from col_io for at most iters iterations or until no edge conflicts; col_io ends as the last state, best_out
// as the best state (the most recent of the best when the search ends on one). stats: iterations run, best
// conflicts, iteration of the best, mean |C|, end conflicts. Returns the best conflicts.
long pfa_colour_tabu(int V, int k, const int *off, const int *nbr, int *col_io, long iters, uint64_t seed,
                     int *best_out, double *stats) {
    Col S;
    col_init(&S, V, k, off, nbr, col_io, seed);
    memcpy(best_out, col_io, V * sizeof(int));
    while (S.it < iters && (col_beta > 0 ? col_step_boltz(&S) : col_step(&S))) {
        if (S.total <= S.best) { S.best = S.total; S.best_it = S.it; memcpy(best_out, col_io, V * sizeof(int)); }   // the most recent of the best, as HEAD keeps it
    }
    if (stats) {
        stats[0] = (double)S.it; stats[1] = (double)S.best; stats[2] = (double)S.best_it;
        stats[3] = S.it ? S.csum / S.it : 0; stats[4] = (double)S.total;
    }
    long best = S.best;
    col_free(&S);
    return best;
}

// GPX: colour class i of the child is the largest remaining class of parent (start + i) % 2, its vertices removed
// from both parents' counts; the vertices left get a random colour.
void pfa_colour_gpx(int V, int k, const int *p0, const int *p1, int start, uint64_t seed, int *child) {
    uint64_t rng = (seed << 1) | 1;
    const int *par[2] = {p0, p1};
    int *sizes = calloc(2 * k, sizeof(int));
    for (int j = 0; j < V; j++) { sizes[p0[j]]++; sizes[k + p1[j]]++; child[j] = -1; }
    for (int i = 0; i < k; i++) {
        int idx = (start + i) % 2;
        const int *cp = par[idx];
        int *sz = sizes + idx * k, vmax = -1, cmax = -1, s0 = col_rnd(&rng, k);
        for (int j = 0; j < k; j++) { int c = (s0 + j) % k; if (sz[c] > vmax) { vmax = sz[c]; cmax = c; } }
        for (int j = 0; j < V; j++)
            if (cp[j] == cmax && child[j] < 0) { child[j] = i; sizes[p0[j]]--; sizes[k + p1[j]]--; }
    }
    for (int j = 0; j < V; j++) if (child[j] < 0) child[j] = col_rnd(&rng, k);
    free(sizes);
}

// A random colouring.
void pfa_colour_random(int V, int k, uint64_t seed, int *col) {
    uint64_t rng = (seed << 1) | 1;
    for (int v = 0; v < V; v++) col[v] = col_rnd(&rng, k);
}
