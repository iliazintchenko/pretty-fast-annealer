// The loop around the colouring kernel, HEAD's form (Moalic and Gondran): two parents, each generation two children by
// GPX in the two parent orders, each searched by TabuCol for iter_tc iterations, the children replacing the parents;
// two elite slots, the current one taking any child at least as good, and every swap_every generations, or when the
// two parents are 99% identical under a greedy matching of their colour classes, the other elite replaces a parent at
// random and the current slot resets. Stops at a legal colouring, at the iteration budget, or at the wall.
#include <time.h>

static int *col_dump = NULL;              // the pair test's record of parent pairs: 2 * max colourings of V
static int col_dump_every = 0, col_dump_max = 0, col_dump_from = 0;
void pfa_colour_dump(int *buf, int from, int every, int max) { col_dump = buf; col_dump_from = from; col_dump_every = every; col_dump_max = max; }

static double col_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

// Vertices matched between two colourings under a greedy matching of their classes (HEAD's Solution::proxi).
static int col_proxi(int V, int k, const int *a, const int *b, int *work) {
    int *M = work;                                   // k*k counts, k*k marks
    for (int i = 0; i < k * k; i++) M[i] = 0;
    for (int v = 0; v < V; v++) M[a[v] * k + b[v]]++;
    int prox = 0;
    for (int c = 0; c < k; c++) {
        int best = -1, bi = -1, bj = -1;
        for (int i = 0; i < k; i++)
            for (int j = 0; j < k; j++)
                if (M[i * k + j] > best) { best = M[i * k + j]; bi = i; bj = j; }
        prox += best;
        for (int i = 0; i < k; i++) { M[bi * k + i] = -1; M[i * k + bj] = -1; }
    }
    return prox;
}

static long col_tabu_run(int V, int k, const int *off, const int *nbr, int *col, long iters, uint64_t seed, int *best,
                         double *csum) {
    double st[8];
    long b = pfa_colour_tabu(V, k, off, nbr, col, iters, seed, best, st);
    *csum += st[0];
    return b;
}

// Returns the best conflicts; best_out the best colouring. stats: TabuCol iterations, generations, best conflicts,
// seconds, elite swaps, generation of the best, kicks. With kick, two parents still 99% identical after the elite
// swap do not end the run: one of them is replaced by a fresh random colouring searched for iter_tc iterations.
long pfa_colour_head(int V, int k, const int *off, const int *nbr, long iter_tc, int swap_every, long max_iters,
                     double seconds, uint64_t seed, int kick, int *best_out, double *stats) {
    double t0 = col_now();
    uint64_t rng = (seed << 1) | 1;
    int *pop[4], *child[2], *tmp = malloc(V * sizeof(int)), *work = malloc((size_t)k * k * sizeof(int));
    long fit[4], cfit[2], best = 1L << 60, best_gen = 0;
    for (int i = 0; i < 4; i++) { pop[i] = malloc(V * sizeof(int)); pfa_colour_random(V, k, col_rng(&rng), pop[i]); fit[i] = pfa_colour_conflicts(V, off, nbr, pop[i]); }
    for (int i = 0; i < 2; i++) child[i] = malloc(V * sizeof(int));
    memcpy(best_out, pop[0], V * sizeof(int)); best = fit[0];
    double iters = 0;
    long gen = 0, swaps = 0, kicks = 0, ndump = 0;
    int elite = 0, seuil = (int)(0.99 * V), prox = 0;
    while (best > 0 && (kick || prox < seuil) && iters < (double)max_iters && (seconds <= 0 || col_now() - t0 < seconds)) {
        gen++;
        if (col_dump && gen > col_dump_from && gen % col_dump_every == 0 && ndump < col_dump_max) {   // the parents of this generation, for the pair test
            memcpy(col_dump + 2 * ndump * V, pop[0], V * sizeof(int));
            memcpy(col_dump + (2 * ndump + 1) * V, pop[1], V * sizeof(int));
            ndump++;
        }
        col_target = fit[0] < fit[1] ? fit[0] : fit[1];   // the parents' level, for the call log
        for (int i = 0; i < 2; i++) {
            pfa_colour_gpx(V, k, pop[0], pop[1], i, col_rng(&rng), tmp);
            cfit[i] = col_tabu_run(V, k, off, nbr, tmp, iter_tc, col_rng(&rng), child[i], &iters);
        }
        for (int i = 0; i < 2; i++) {
            if (cfit[i] < best) { best = cfit[i]; best_gen = gen; memcpy(best_out, child[i], V * sizeof(int)); }
            memcpy(pop[i], child[i], V * sizeof(int)); fit[i] = cfit[i];
            if (cfit[i] <= fit[2 + elite]) { memcpy(pop[2 + elite], child[i], V * sizeof(int)); fit[2 + elite] = cfit[i]; }
        }
        prox = col_proxi(V, k, pop[0], pop[1], work);
        if (swap_every > 0 && (gen % swap_every == 0 || prox >= seuil)) {
            elite ^= 1;
            int r = col_rnd(&rng, 2);
            if (col_proxi(V, k, pop[r ^ 1], pop[2 + elite], work) >= seuil) r ^= 1;
            memcpy(pop[r], pop[2 + elite], V * sizeof(int)); fit[r] = fit[2 + elite];
            fit[2 + elite] = 1L << 60;
            prox = col_proxi(V, k, pop[0], pop[1], work);
            swaps++;
        }
        if (kick && prox >= seuil) {
            int r = col_rnd(&rng, 2);
            pfa_colour_random(V, k, col_rng(&rng), tmp);
            fit[r] = col_tabu_run(V, k, off, nbr, tmp, iter_tc, col_rng(&rng), pop[r], &iters);
            if (fit[r] < best) { best = fit[r]; best_gen = gen; memcpy(best_out, pop[r], V * sizeof(int)); }
            prox = col_proxi(V, k, pop[0], pop[1], work);
            kicks++;
        }
    }
    if (stats) { stats[0] = iters; stats[1] = (double)gen; stats[2] = (double)best; stats[3] = col_now() - t0; stats[4] = (double)swaps; stats[5] = (double)best_gen; stats[6] = (double)kicks; stats[7] = (double)ndump; }
    col_dump = NULL;
    for (int i = 0; i < 4; i++) free(pop[i]);
    free(child[0]); free(child[1]); free(tmp); free(work);
    return best;
}

// PFA's loop: HEAD's duet under one budget of kernel iterations, with the restart's length a quantity of the graph
// instead of a number tuned per instance. HEAD's tuned lengths on the random graphs, 3 to 70 iterations per vertex,
// are an eighth of the edges per vertex up to half density and flat past it, so the length is the smaller of E/8 and
// V^2/32: the iterations per vertex grow with the degree up to half density and then hold, within a factor of two of
// the tuned value on every random graph of HEAD's table. On the structured graphs HEAD tunes a hundred times longer
// per edge (le450_25c, R1000.5), and the loop does not follow it: six rules that read the length from the run were
// built and measured against the tuned values and none separated a call too short from a call whose descent fills it
// whatever its length, which is what the sparse random graphs do at the right length; the one that lengthened when
// the first best fell late in the call doubled the length on those graphs until they failed, and reached no success
// on the structured ones that the tuned length did not also reach. The loop never stops before the budget or a legal
// colouring: at convergence the elite swap comes first; parents and both elites all within 1%, one parent is replaced
// by a random colouring and searched. stats: kernel iterations, generations, best conflicts, seconds, elite swaps,
// generation of the best, kicks, the length, 0, the length again (the two fields the run-read rules reported).
long pfa_colour_run(int V, int k, const int *off, const int *nbr, long budget, double seconds, uint64_t seed,
                    int *best_out, double *stats) {
    double t0 = col_now();
    uint64_t rng = (seed << 1) | 1;
    int *pop[4], *child[2], *tmp = malloc(V * sizeof(int)), *work = malloc((size_t)k * k * sizeof(int));
    long fit[4], cfit[2], best, best_gen = 0;
    for (int i = 0; i < 4; i++) { pop[i] = malloc(V * sizeof(int)); pfa_colour_random(V, k, col_rng(&rng), pop[i]); fit[i] = pfa_colour_conflicts(V, off, nbr, pop[i]); }
    for (int i = 0; i < 2; i++) child[i] = malloc(V * sizeof(int));
    memcpy(best_out, pop[0], V * sizeof(int)); best = fit[0];
    const long floor_L = 1000, cap_L = 1000L * V;
    long L = off[V] / 16, doublings = 0;                  // an eighth of the edges (off[V] counts each twice) ...
    if (L > (long)V * V / 32) L = (long)V * V / 32;      // ... saturating at half density: a thirty-second of the vertex pairs

    double iters = 0, Lsum = 0, ncalls = 0;
    long gen = 0, swaps = 0, kicks = 0;
    int elite = 0, seuil = (int)(0.99 * V);
    if (L < floor_L) L = floor_L;
    if (L > cap_L) L = cap_L;
    while (best > 0 && iters < (double)budget && (seconds <= 0 || col_now() - t0 < seconds)) {
        gen++;
        col_target = fit[0] < fit[1] ? fit[0] : fit[1];
        for (int i = 0; i < 2; i++) {
            double st[8];
            pfa_colour_gpx(V, k, pop[0], pop[1], i, col_rng(&rng), tmp);
            cfit[i] = pfa_colour_tabu(V, k, off, nbr, tmp, L, col_rng(&rng), child[i], st);
            iters += st[0]; Lsum += L; ncalls++;
        }
        for (int i = 0; i < 2; i++) {
            if (cfit[i] < best) { best = cfit[i]; best_gen = gen; memcpy(best_out, child[i], V * sizeof(int)); }
            memcpy(pop[i], child[i], V * sizeof(int)); fit[i] = cfit[i];
            if (cfit[i] <= fit[2 + elite]) { memcpy(pop[2 + elite], child[i], V * sizeof(int)); fit[2 + elite] = cfit[i]; }
        }
        int prox = col_proxi(V, k, pop[0], pop[1], work);
        if (gen % 10 == 0 || prox >= seuil) {
            elite ^= 1;
            int r = col_rnd(&rng, 2);
            if (col_proxi(V, k, pop[r ^ 1], pop[2 + elite], work) >= seuil) r ^= 1;
            memcpy(pop[r], pop[2 + elite], V * sizeof(int)); fit[r] = fit[2 + elite];
            fit[2 + elite] = 1L << 60;
            prox = col_proxi(V, k, pop[0], pop[1], work);
            swaps++;
        }
        if (prox >= seuil) {                             // still converged after the swap
            int all = 1;
            for (int e = 2; e < 4 && all; e++) all = col_proxi(V, k, pop[0], pop[e], work) >= seuil;
            if (all) {
                int r = col_rnd(&rng, 2);
                double st[8];
                pfa_colour_random(V, k, col_rng(&rng), tmp);
                col_target = -1;
                fit[r] = pfa_colour_tabu(V, k, off, nbr, tmp, L, col_rng(&rng), pop[r], st);
                iters += st[0]; Lsum += L; ncalls++;
                if (fit[r] < best) { best = fit[r]; best_gen = gen; memcpy(best_out, pop[r], V * sizeof(int)); }
                kicks++;
            }
        }
    }
    if (stats) {
        stats[0] = iters; stats[1] = (double)gen; stats[2] = (double)best; stats[3] = col_now() - t0; stats[4] = (double)swaps;
        stats[5] = (double)best_gen; stats[6] = (double)kicks; stats[7] = (double)L; stats[8] = (double)doublings;
        stats[9] = ncalls ? Lsum / ncalls : 0;
    }
    for (int i = 0; i < 4; i++) free(pop[i]);
    free(child[0]); free(child[1]); free(tmp); free(work);
    return best;
}
