// Minimum vertex cover: a set C of vertices, the energy the number of edges with neither end in C, zero a cover; the
// smallest C with energy zero is sought, so the search runs at a fixed |C|, drops one vertex when every edge is
// covered, and continues. The state keeps per vertex its score, the weighted count of edges whose coverage flipping
// the vertex would change (positive outside C: the uncovered edges it would cover; non-positive inside: minus the
// edges only it covers), the uncovered edges as a list with positions, and C as a list with positions, so a move's
// update is one pass over the moved vertex's neighbours. The kernel is NuMVC's two-stage exchange (Cai, Su, Luo and
// Sattar 2013) as its code runs it: remove the vertex of C with the greatest score (the least weighted loss), the
// vertex just added excepted, ties to the oldest; pick an uncovered edge at random and add the endpoint of greater
// score, an endpoint whose configuration has not changed since it left C never (configuration checking), ties to the
// oldest; then every uncovered edge's weight rises by one, and when the mean weight reaches half the vertex count every
// weight is scaled by 0.3 (forgetting). Zero temperature; the weights are the search's memory in place of a tabu.
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    int V, E;
    const int *eu, *ev;          // the edges' ends
    int *off, *nbr, *ned;        // CSR adjacency: neighbours and the index of the edge to each
    int *inc;                    // 1 if in C
    int *w;                      // edge weights (bounded: the mean is kept under V/2 by the forgetting)
    int *ds;                     // scores (bounded by a vertex's weighted degree)
    int *conf;                   // configuration changed since the vertex left C
    long *ts;                    // step of the vertex's last move
    int *ulist, *upos, nu;       // uncovered edges, a list with positions
    int *clist, *cpos, nc;       // C, a list with positions
    int *dsc; long *tsc;         // the scores (32-bit: a score is bounded by a vertex's weighted degree, far under 2^31) and time
                                 // stamps of C by position: the removal scan reads these two arrays and nothing else
    int tabu;                    // the vertex just added, or -1
    long ave, dtot, thr, forgets;
    long step;
    uint64_t rng;
} Cov;

static inline uint64_t cov_rng(uint64_t *s) { *s ^= *s >> 12; *s ^= *s << 25; *s ^= *s >> 27; return *s * 2685821657736338717ULL; }
static inline int cov_rnd(uint64_t *s, int n) { return (int)(((cov_rng(s) >> 32) * (uint64_t)n) >> 32); }
static double cov_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

static inline void cov_uncover(Cov *S, int e) { S->upos[e] = S->nu; S->ulist[S->nu++] = e; }
static inline void cov_cover(Cov *S, int e) { int p = S->upos[e], f = S->ulist[--S->nu]; S->ulist[p] = f; S->upos[f] = p; S->upos[e] = -1; }
static inline void cov_cadd(Cov *S, int v) { S->cpos[v] = S->nc; S->dsc[S->nc] = S->ds[v]; S->tsc[S->nc] = S->ts[v]; S->clist[S->nc++] = v; }
static inline void cov_cdel(Cov *S, int v) {
    int p = S->cpos[v], u = S->clist[--S->nc];
    S->clist[p] = u; S->cpos[u] = p; S->dsc[p] = S->dsc[S->nc]; S->tsc[p] = S->tsc[S->nc]; S->cpos[v] = -1;
}

static void cov_add(Cov *S, int v) {
    S->inc[v] = 1; S->ds[v] = -S->ds[v];
    for (int i = S->off[v]; i < S->off[v + 1]; i++) {
        int n = S->nbr[i], e = S->ned[i];
        if (!S->inc[n]) { S->ds[n] -= S->w[e]; S->conf[n] = 1; cov_cover(S, e); }
        else { S->ds[n] += S->w[e]; S->dsc[S->cpos[n]] += S->w[e]; }
    }
    cov_cadd(S, v);
}

static void cov_remove(Cov *S, int v) {
    S->inc[v] = 0; S->ds[v] = -S->ds[v]; S->conf[v] = 0;
    for (int i = S->off[v]; i < S->off[v + 1]; i++) {
        int n = S->nbr[i], e = S->ned[i];
        if (!S->inc[n]) { S->ds[n] += S->w[e]; S->conf[n] = 1; cov_uncover(S, e); }
        else { S->ds[n] -= S->w[e]; S->dsc[S->cpos[n]] -= S->w[e]; }
    }
    cov_cdel(S, v);
}

// Scores, the uncovered list and the C list from inc and w.
static void cov_rescore(Cov *S) {
    for (int v = 0; v < S->V; v++) S->ds[v] = 0;
    S->nu = 0; S->nc = 0;
    for (int e = 0; e < S->E; e++) {
        int a = S->eu[e], b = S->ev[e], c = S->inc[a] + S->inc[b];
        S->upos[e] = -1;
        if (c == 0) { S->ds[a] += S->w[e]; S->ds[b] += S->w[e]; cov_uncover(S, e); }
        else if (c == 1) S->ds[S->inc[a] ? a : b] -= S->w[e];
    }
    for (int v = 0; v < S->V; v++) { S->cpos[v] = -1; if (S->inc[v]) cov_cadd(S, v); }
}

static void cov_forget(Cov *S) {
    long total = 0;
    for (int e = 0; e < S->E; e++) { S->w[e] = (int)(S->w[e] * 0.3f); total += S->w[e]; }
    int nu = S->nu;                      // the uncovered list is unchanged by a rescale; rebuild only the scores
    for (int v = 0; v < S->V; v++) S->ds[v] = 0;
    for (int e = 0; e < S->E; e++) {
        int a = S->eu[e], b = S->ev[e], c = S->inc[a] + S->inc[b];
        if (c == 0) { S->ds[a] += S->w[e]; S->ds[b] += S->w[e]; }
        else if (c == 1) S->ds[S->inc[a] ? a : b] -= S->w[e];
    }
    (void)nu;
    for (int i = 0; i < S->nc; i++) S->dsc[i] = S->ds[S->clist[i]];
    S->ave = total / S->E;
    S->forgets++;
}

static void cov_weights(Cov *S) {
    for (int i = 0; i < S->nu; i++) { int e = S->ulist[i]; S->w[e]++; S->ds[S->eu[e]]++; S->ds[S->ev[e]]++; }   // both ends outside C: no mirror
    S->dtot += S->nu;
    if (S->dtot >= S->E) { S->ave++; S->dtot -= S->E; }
    if (S->ave >= S->thr) cov_forget(S);
}

static void cov_init(Cov *S, int V, int E, const int *eu, const int *ev, int *inc, int *w, uint64_t seed) {
    S->V = V; S->E = E; S->eu = eu; S->ev = ev; S->inc = inc; S->w = w;
    S->off = calloc(V + 1, sizeof(int)); S->nbr = malloc(2L * E * sizeof(int)); S->ned = malloc(2L * E * sizeof(int));
    for (int e = 0; e < E; e++) { S->off[eu[e] + 1]++; S->off[ev[e] + 1]++; }
    for (int v = 0; v < V; v++) S->off[v + 1] += S->off[v];
    int *fill = malloc(V * sizeof(int));
    memcpy(fill, S->off, V * sizeof(int));
    for (int e = 0; e < E; e++) {
        int a = eu[e], b = ev[e];
        S->nbr[fill[a]] = b; S->ned[fill[a]++] = e;
        S->nbr[fill[b]] = a; S->ned[fill[b]++] = e;
    }
    free(fill);
    S->ds = malloc(V * sizeof(int)); S->conf = malloc(V * sizeof(int)); S->ts = calloc(V, sizeof(long));
    S->ulist = malloc(E * sizeof(int)); S->upos = malloc(E * sizeof(int));
    S->clist = malloc(V * sizeof(int)); S->cpos = malloc(V * sizeof(int));
    S->dsc = malloc(V * sizeof(int)); S->tsc = malloc(V * sizeof(long));
    for (int v = 0; v < V; v++) S->conf[v] = 1;
    long total = 0;
    for (int e = 0; e < E; e++) total += w[e];
    S->tabu = -1; S->ave = total / E; S->dtot = 0; S->thr = V / 2; S->forgets = 0; S->step = 0;   // the mean weight carried with the weights
    S->rng = (seed << 1) | 1;
    cov_rescore(S);
}

static void cov_free(Cov *S) {
    free(S->off); free(S->nbr); free(S->ned); free(S->ds); free(S->conf); free(S->ts);
    free(S->ulist); free(S->upos); free(S->clist); free(S->cpos); free(S->dsc); free(S->tsc);
}

// The vertex of C with the greatest score, the one just added excepted, ties to the oldest (the first in C's order
// among equals). Two passes over the contiguous score array, the maximum then the oldest at it: the scan NuMVC makes
// through the vertex arrays, which was two thirds of the step, on the two arrays the scan reads and nothing else.
static int cov_best_remove(Cov *S) {
    int nc = S->nc, tp = S->tabu >= 0 ? S->cpos[S->tabu] : -1;
    int saved = 0, m = -(1 << 30);
    const int *d = S->dsc;
    if (tp >= 0) { saved = S->dsc[tp]; S->dsc[tp] = -(1 << 30); }
    for (int i = 0; i < nc; i++) m = d[i] > m ? d[i] : m;
    int b = -1; long bt = 0;
    for (int i = 0; i < nc; i++) if (d[i] == m && (b < 0 || S->tsc[i] < bt)) { b = i; bt = S->tsc[i]; }
    if (tp >= 0) S->dsc[tp] = saved;
    return S->clist[b < 0 ? 0 : b];
}

// The position in C of the greatest score (the first among equals), for the drop to the next level.
static int cov_argmax_c(Cov *S) {
    int m = -(1 << 30), b = 0;
    for (int i = 0; i < S->nc; i++) if (S->dsc[i] > m) { m = S->dsc[i]; b = i; }
    return S->clist[b];
}

long pfa_cover_uncovered(int E, const int *eu, const int *ev, const int *inc) {
    long n = 0;
    for (int e = 0; e < E; e++) n += !inc[eu[e]] && !inc[ev[e]];
    return n;
}

// NuMVC's greedy start: the vertex covering the most uncovered edges added until every edge is covered (ties to the
// lowest index). inc is written.
void pfa_cover_greedy(int V, int E, const int *eu, const int *ev, int *inc) {
    int *w = malloc(E * sizeof(int));
    for (int e = 0; e < E; e++) w[e] = 1;
    for (int v = 0; v < V; v++) inc[v] = 0;
    Cov S; cov_init(&S, V, E, eu, ev, inc, w, 1);
    while (S.nu > 0) {
        int b = -1;
        for (int v = 0; v < V; v++) if (!inc[v] && (b < 0 || S.ds[v] > S.ds[b])) b = v;
        cov_add(&S, b);
    }
    cov_free(&S); free(w);
}

// The search from the set inc (any set; its size is the level), with edge weights w (E ints, carried between calls;
// all ones for a fresh memory) and the vertex time stamps and configuration flags fresh. Runs until a cover of size
// target, max_steps steps, or seconds; writes the smallest cover found to best_out (inc itself if none is complete),
// returns its size (V + 1 if no cover was seen). inc is left at the search's end state, w at the end weights; low_out,
// if given, the state with the fewest uncovered edges seen at the end level. stats: steps, best size, step of the
// best, seconds, uncovered edges at the end, |C| at the end, forgettings, the fewest uncovered edges at the end level.
long pfa_cover_search(int V, int E, const int *eu, const int *ev, int *inc, int *w, int target, long max_steps,
                      double seconds, uint64_t seed, int *best_out, int *low_out, double *stats) {
    double t0 = cov_now();
    Cov S; cov_init(&S, V, E, eu, ev, inc, w, seed);
    long best = V + 1, best_step = 0, min_nu = 1L << 60;      // min_nu: the fewest uncovered edges seen at the final level
    memcpy(best_out, inc, V * sizeof(int));
    for (;;) {
        if (S.nu < min_nu) { min_nu = S.nu; if (low_out) memcpy(low_out, inc, V * sizeof(int)); }
        if (S.nu == 0) {
            if (S.nc < best) { best = S.nc; best_step = S.step; memcpy(best_out, inc, V * sizeof(int)); }
            if (S.nc <= target) break;
            cov_remove(&S, cov_argmax_c(&S));                     // one level down: drop the vertex of least loss
            min_nu = S.nu; if (low_out) memcpy(low_out, inc, V * sizeof(int));
            continue;
        }
        if (S.step >= max_steps) break;
        if (seconds > 0 && (S.step & 127) == 0 && cov_now() - t0 >= seconds) break;
        int rv = cov_best_remove(&S);
        cov_remove(&S, rv);
        int e = S.ulist[cov_rnd(&S.rng, S.nu)], a = S.eu[e], b = S.ev[e], av;
        if (!S.conf[a]) av = b;
        else if (!S.conf[b]) av = a;
        else av = (S.ds[a] > S.ds[b] || (S.ds[a] == S.ds[b] && S.ts[a] < S.ts[b])) ? a : b;
        cov_add(&S, av);
        S.ts[av] = S.ts[rv] = S.step; S.tsc[S.cpos[av]] = S.step;
        S.tabu = av;
        cov_weights(&S);
        S.step++;
    }
    if (stats) {
        stats[0] = (double)S.step; stats[1] = (double)best; stats[2] = (double)best_step; stats[3] = cov_now() - t0;
        stats[4] = (double)S.nu; stats[5] = (double)S.nc; stats[6] = (double)S.forgets; stats[7] = (double)min_nu;
    }
    cov_free(&S);
    return best;
}
