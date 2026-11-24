// Graph colouring with k colours: every vertex one colour, the energy the number of edges whose ends share one. The
// state keeps, per vertex and colour, the number of neighbours in that colour (gamma), so a move's delta is one
// subtraction and its update one pass over the moved vertex's neighbours; the coupling is the graph once and the
// identity on colours, never a table per edge. The kernel is TabuCol in Galinier and Hao's form as HEAD runs it: one
// conflicting vertex recoloured per iteration, the best non-tabu move over the conflicting vertices (each vertex's
// best colours cached and maintained), ties broken uniformly, aspiration when a move beats the call's best, the colour
// left tabu for 0.6 |C| + rand(10) iterations, C the conflicting set; zero temperature. The fold is Galinier and Hao's
// partition crossover (GPX): the largest remaining colour class taken from the parents in turn, the rest coloured at
// random.
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
        if (S->conf[v]) { col_cadd(S, v); col_best_of(S, v); }
        else { S->bval[v] = -1; S->bcnt[v] = 0; }
    }
    S->total /= 2;
    S->best = S->total;
    S->it = 0; S->best_it = 0; S->csum = 0;
}

static void col_free(Col *S) {
    free(S->gamma); free(S->tabu); free(S->conf); free(S->clist); free(S->cpos);
    free(S->bval); free(S->bcnt); free(S->blist);
}

// The move v: col[v] -> c, the gamma update over the neighbours and the maintenance of their cached best colours. The
// cache is exact for every vertex in C, the only ones the scan reads, and stale for the rest: a vertex leaving C is
// marked with bval = -1 and bcnt = 0, which no count is below or equal to, so the maintenance below passes it without a
// branch; a vertex entering C is rescanned once; the moved vertex is rescanned only if it stays in C. Near a legal
// colouring nearly every neighbour is outside C, and maintaining their true minima was half the iteration.
static void col_apply(Col *S, int v, int c) {
    const int k = S->k, cv = S->col[v];
    S->tabu[(size_t)v * k + cv] = (int)(S->it + col_rnd(&S->rng, 10) + 0.6 * S->nc);
    S->total += CG(S, v, c) - CG(S, v, cv);
    S->col[v] = c;
    for (int i = S->off[v]; i < S->off[v + 1]; i++) {
        int u = S->nbr[i], cu = S->col[u];
        int gcv = --CG(S, u, cv), gc = ++CG(S, u, c);
        if (cu == cv) { if (--S->conf[u] == 0) { col_cdel(S, u); S->bval[u] = -1; S->bcnt[u] = 0; continue; } }
        else if (cu == c) { if (++S->conf[u] == 1) { col_cadd(S, u); col_best_of(S, u); continue; } }
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
    if (S->conf[v] == 0) { if (S->cpos[v] >= 0) col_cdel(S, v); S->bval[v] = -1; S->bcnt[v] = 0; }
    else { if (S->cpos[v] < 0) col_cadd(S, v); col_best_of(S, v); }
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

// Conflicting edges of a colouring.
long pfa_colour_conflicts(int V, const int *off, const int *nbr, const int *col) {
    long t = 0;
    for (int v = 0; v < V; v++)
        for (int i = off[v]; i < off[v + 1]; i++) t += col[nbr[i]] == col[v];
    return t / 2;
}

// TabuCol from col_io for at most iters iterations or until no edge conflicts; col_io ends as the last state, best_out
// as the best state (the most recent of the best when the search ends on one). stats: iterations run, best
// conflicts, iteration of the best, mean |C|, end conflicts, the first iteration at or under the target (-1 if never),
// the start conflicts, the first iteration at the final best. Returns the best conflicts.
// The call log (the restart-length study): per call a row of 8: the target (the parents' level, set by the loop),
// the first iteration at or under it (-1 if never), the first and the last iteration at the call's best, the best,
// the iterations run, the start conflicts, the end conflicts.
static double *col_log = NULL;
static long col_log_n = 0, col_log_cap = 0, col_target = -1;
void pfa_colour_log(double *buf, long cap) { col_log = buf; col_log_cap = cap; col_log_n = 0; }
long pfa_colour_log_count(void) { return col_log_n; }

long pfa_colour_tabu(int V, int k, const int *off, const int *nbr, int *col_io, long iters, uint64_t seed,
                     int *best_out, double *stats) {
    Col S;
    col_init(&S, V, k, off, nbr, col_io, seed);
    memcpy(best_out, col_io, V * sizeof(int));
    long start = S.total, t_level = S.total <= col_target ? 0 : -1, first_best = 0;
    while (S.it < iters && col_step(&S)) {
        if (S.total < S.best) first_best = S.it;
        if (S.total <= S.best) { S.best = S.total; S.best_it = S.it; memcpy(best_out, col_io, V * sizeof(int)); }   // the most recent of the best, as HEAD keeps it
        if (t_level < 0 && S.total <= col_target) t_level = S.it;
    }
    if (stats) {
        stats[0] = (double)S.it; stats[1] = (double)S.best; stats[2] = (double)S.best_it;
        stats[3] = S.it ? S.csum / S.it : 0; stats[4] = (double)S.total;
        stats[5] = (double)t_level; stats[6] = (double)start; stats[7] = (double)first_best;
    }
    if (col_log && col_log_n < col_log_cap) {
        double *r = col_log + 8 * col_log_n++;
        r[0] = (double)col_target; r[1] = (double)t_level; r[2] = (double)first_best; r[3] = (double)S.best_it;
        r[4] = (double)S.best; r[5] = (double)S.it; r[6] = (double)start; r[7] = (double)S.total;
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
