// The loop on the job shop: a pool of schedules, restarts from points on the path from one member toward another
// (adjacent inverted pairs bubbled toward the other's order, each swap passed through an O(1) acyclicity test), the
// tabu search of shop.c from the point with a cutoff in non-improving iterations, the result offered to the pool,
// which keeps the best distinct members by makespan and by arc distance to the others (the route class's rule). The
// budget is the number of restarts.
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// The arc distance of two machine orders: job pairs ordered differently, summed over machines. posB: work of n*m.
static long shop_inversions(int *v, int *tmp, int lo, int hi) {   // of v[lo..hi), v sorted in place
    if (hi - lo < 2) return 0;
    int mid = (lo + hi) / 2;
    long d = shop_inversions(v, tmp, lo, mid) + shop_inversions(v, tmp, mid, hi);
    int i = lo, j = mid, k = lo;
    while (i < mid && j < hi) { if (v[i] <= v[j]) tmp[k++] = v[i++]; else { d += mid - i; tmp[k++] = v[j++]; } }
    while (i < mid) tmp[k++] = v[i++];
    while (j < hi) tmp[k++] = v[j++];
    memcpy(v + lo, tmp + lo, sizeof(int) * (hi - lo));
    return d;
}
static long shop_arc_dist(int n, int m, const int *A, const int *B, int *work) {   // work: 3n ints
    int *posB = work, *v = work + n, *tmp = work + 2 * n;
    long d = 0;
    for (int k = 0; k < m; k++) {
        const int *a = A + k * n, *b = B + k * n;
        for (int p = 0; p < n; p++) posB[b[p]] = p;
        for (int p = 0; p < n; p++) v[p] = posB[a[p]];
        d += shop_inversions(v, tmp, 0, n);
    }
    return d;
}

// One relink step toward B: an adjacent pair of A's order inverted relative to B, drawn at random among those whose
// swap the O(1) test allows (u before v adjacent on the machine; a cycle would need a path JS(u) -> ... -> JP(v), which
// forces r_JP(v) >= r_JS(u) + p_JS(u)); heads are recomputed after. Returns 0 when no admissible inverted pair is left.
static int shop_relink_step(Shop *S, const int *B, int *posB, uint64_t *rng) {
    const int n = S->n, m = S->m;
    int cand_k = -1, cand_p = -1, seen = 0;
    for (int k = 0; k < m; k++) {
        const int *b = B + k * n, *a = S->seq + k * n;
        for (int p = 0; p < n; p++) posB[b[p]] = p;
        for (int p = 0; p + 1 < n; p++) {
            if (posB[a[p]] <= posB[a[p + 1]]) continue;
            int u = S->opof[k * n + a[p]], v = S->opof[k * n + a[p + 1]], js = shop_js(S, u), jp = shop_jp(S, v);
            if (!(js < 0 || jp < 0 || S->r[jp] < S->r[js] + S->dur[js])) continue;
            if ((int)((shop_rng(rng) >> 33) % (uint64_t)++seen) == 0) { cand_k = k; cand_p = p; }
        }
    }
    if (seen == 0) return 0;
    int *a = S->seq + cand_k * n, t = a[cand_p]; a[cand_p] = a[cand_p + 1]; a[cand_p + 1] = t;
    shop_reindex(S, cand_k, cand_p > 0 ? cand_p - 1 : 0, cand_p + 2 < n ? cand_p + 2 : n - 1);
    if (!shop_eval(S)) return -1;   // the test is sufficient: cannot happen
    return 1;
}

// The pool: P members of n*m ints, their makespans (INT_MAX where empty), the distance matrix dm (P*P longs). A
// newcomer replaces the member whose rank by makespan plus rank by closeness (mean distance to the 5 nearest) is
// worst, clones first, unless that is the newcomer (the route class's pool_offer with G = 0).
static void shop_pool_fitness(int P, const int *cost, const long *dm, double *fit) {
    double *close = malloc(sizeof(double) * P);
    for (int i = 0; i < P; i++) {
        long best[5] = {0}; int nb = 0;
        for (int j = 0; j < P; j++) {
            if (j == i) continue;
            long d = dm[i * P + j];
            int q = nb < 5 ? nb++ : 4;
            if (nb == 5 && q == 4 && d >= best[4]) continue;
            best[q] = d;
            while (q > 0 && best[q] < best[q - 1]) { long t = best[q]; best[q] = best[q - 1]; best[q - 1] = t; q--; }
        }
        double s = 0; for (int q = 0; q < nb; q++) s += best[q];
        close[i] = nb ? s / nb : 0;
    }
    for (int i = 0; i < P; i++) {
        int rc = 0, rd = 0;
        for (int j = 0; j < P; j++) { rc += cost[j] < cost[i]; rd += close[j] > close[i]; }
        fit[i] = rc + rd;
    }
    free(close);
}

static int shop_pool_offer(int P, int N, int n, int m, int *pool, int *pcost, long *dm, const int *t, int c, int *work) {
    int filled = 0;
    for (int i = 0; i < P; i++) filled += pcost[i] < 0x7fffffff;
    long *dt = malloc(sizeof(long) * P);
    for (int i = 0; i < P; i++) dt[i] = pcost[i] < 0x7fffffff ? shop_arc_dist(n, m, t, pool + (long)i * N, work) : 0;
    for (int i = 0; i < filled; i++) if (dt[i] == 0) { free(dt); return 0; }   // a clone of a member
    int slot = -1;
    if (filled < P) slot = filled;
    else {
        // the P + 1 candidates' fitness; the worst leaves (a clone among members first)
        int P1 = P + 1;
        int *cost = malloc(sizeof(int) * P1); long *dmx = malloc(sizeof(long) * P1 * P1); double *fit = malloc(sizeof(double) * P1);
        for (int i = 0; i < P; i++) { cost[i] = pcost[i]; for (int j = 0; j < P; j++) dmx[i * P1 + j] = dm[i * P + j]; dmx[i * P1 + P] = dmx[P * P1 + i] = dt[i]; }
        cost[P] = c; dmx[P * P1 + P] = 0;
        shop_pool_fitness(P1, cost, dmx, fit);
        int w = 0;
        for (int i = 1; i < P1; i++) if (fit[i] > fit[w]) w = i;
        free(cost); free(dmx); free(fit);
        if (w == P) { free(dt); return 0; }
        slot = w;
    }
    pcost[slot] = c; memcpy(pool + (long)slot * N, t, sizeof(int) * N);
    for (int i = 0; i < P; i++) dm[slot * P + i] = dm[i * P + slot] = i == slot ? 0 : dt[i];
    free(dt);
    return 1;
}

// `restarts` restarts of the loop. pool (P x n*m), pcost (P, 0x7fffffff where empty) and dm (P*P) persist across calls.
// A restart: two members by tournament of two (the better by makespan of two drawn), the path from A toward B walked
// to a random arc distance in [alpha, d - alpha] with alpha = max(1, d/5) (both directions over the restarts), the
// tabu search with `cutoff` non-improving iterations from there, the result offered; until the pool holds P members
// a restart is a Giffler-Thompson start. Returns the best makespan in the pool. stats: restarts, tabu iterations,
// relink steps, offers taken.
// mode 0: PFA's loop above (one path point per restart). mode 3: TS/PR's relink with PFA's pool (parents by tournament,
// biased replacement, the short search a 25th of the cutoff). mode 1: TS/PR's (Peng, Lu and Cheng): parents uniform, the path from A toward B walked
// with a point taken every beta = max(d/10, 2) steps from distance alpha = d/5 until within alpha of B, a tabu search
// of `si` non-improving iterations on each point, one of `cutoff` on the best point, the result replacing the worst
// member by makespan unless it is a clone.
static int shop_mode = 1; static long shop_si = 0;   // mode 1 ships; si 0: a 25th of the cutoff
void pfa_shop_loop_mode(int mode, long si) { shop_mode = mode; shop_si = si; }

static int shop_pool_offer_worst(int P, int N, int n, int m, int *pool, int *pcost, long *dm, const int *t, int c, int *work) {
    int filled = 0, worst = -1;
    for (int i = 0; i < P; i++) filled += pcost[i] < 0x7fffffff;
    for (int i = 0; i < filled; i++) if (shop_arc_dist(n, m, t, pool + (long)i * N, work) == 0) return 0;
    if (filled < P) worst = filled;
    else { worst = 0; for (int i = 1; i < P; i++) if (pcost[i] > pcost[worst]) worst = i; if (c >= pcost[worst]) return 0; }
    pcost[worst] = c; memcpy(pool + (long)worst * N, t, sizeof(int) * N);
    for (int i = 0; i < P; i++) dm[worst * P + i] = dm[i * P + worst] = i == worst ? 0 : shop_arc_dist(n, m, t, pool + (long)i * N, work);
    return 1;
}

int pfa_shop_loop(int n, int m, const int *mach, const int *dur, long restarts, long cutoff, uint64_t seed,
                  int P, int *pool, int *pcost, long *dm, long *stats) {
    const int N = n * m;
    int *t = malloc(sizeof(int) * N), *best = malloc(sizeof(int) * N), *work = malloc(sizeof(int) * (N > 3 * n ? N : 3 * n));
    long st[4] = {0}, tst[8];
    uint64_t rng = shop_seed(seed);
    for (long r = 0; r < restarts; r++) {
        int filled = 0;
        for (int i = 0; i < P; i++) filled += pcost[i] < 0x7fffffff;
        if (filled < (shop_mode == 1 ? P : 2)) {   // TS/PR fills its population first
            if (filled == 0) pfa_shop_start(n, m, mach, dur, shop_rng(&rng), SHOP_STARTS, t);   // the first member the best of the non-delay draws, the rest random for the pool's spread
            else pfa_shop_gt(n, m, mach, dur, shop_rng(&rng), 0, t);
        } else if (shop_mode == 1 || shop_mode == 3) {
            int ia, ib;
            if (shop_mode == 1) { ia = (int)((shop_rng(&rng) >> 33) % (uint64_t)filled); do { ib = (int)((shop_rng(&rng) >> 33) % (uint64_t)filled); } while (ib == ia); }
            else {
#define PICK2(out) { int a = (int)((shop_rng(&rng) >> 33) % (uint64_t)filled), b = (int)((shop_rng(&rng) >> 33) % (uint64_t)filled); out = pcost[a] <= pcost[b] ? a : b; }
                PICK2(ia); do { PICK2(ib); } while (ib == ia);
#undef PICK2
            }
            const long si = shop_si > 0 ? shop_si : (cutoff / 25 > 50 ? cutoff / 25 : 50);
            const int *A = pool + (long)ia * N, *B = pool + (long)ib * N;
            long d = shop_arc_dist(n, m, A, B, work), alpha = d / 5 > 1 ? d / 5 : 1, beta = d / 10 > 2 ? d / 10 : 2;
            Shop *S = shop_new(n, m, mach, dur, A);
            shop_eval(S);
            int bestc = 0x7fffffff; long walked = 0, tst2[8];
            for (long s2 = 0; s2 < alpha; s2++) { if (shop_relink_step(S, B, work, &rng) <= 0) break; walked++; }
            while (walked <= d - alpha) {   // a path point every beta steps, the short tabu search on each
                memcpy(t, S->seq, sizeof(int) * N);
                int c = pfa_shop_tabu(n, m, mach, dur, t, 0x7fffffffffffL, si, 0.0, shop_rng(&rng), best, tst2);
                st[1] += tst2[0];
                if (c < bestc) { bestc = c; memcpy(t, best, sizeof(int) * N); }
                int stop = 0;
                for (long s2 = 0; s2 < beta; s2++) { if (shop_relink_step(S, B, work, &rng) <= 0) { stop = 1; break; } walked++; }
                if (stop) break;
            }
            st[2] += walked;
            shop_free(S);
            if (bestc == 0x7fffffff) { memcpy(t, A, sizeof(int) * N); }
            int c = pfa_shop_tabu(n, m, mach, dur, t, 0x7fffffffffffL, cutoff, 0.0, shop_rng(&rng), best, tst);
            st[1] += tst[0];
            st[3] += shop_mode == 1 ? shop_pool_offer_worst(P, N, n, m, pool, pcost, dm, best, c, work) : shop_pool_offer(P, N, n, m, pool, pcost, dm, best, c, work);
            st[0]++;
            continue;
        } else {
#define PICK(out) { int a = (int)((shop_rng(&rng) >> 33) % (uint64_t)filled), b = (int)((shop_rng(&rng) >> 33) % (uint64_t)filled); out = pcost[a] <= pcost[b] ? a : b; }
            int ia, ib; PICK(ia); do { PICK(ib); } while (ib == ia && filled > 1);
#undef PICK
            const int *A = pool + (long)ia * N, *B = pool + (long)ib * N;
            long d = shop_arc_dist(n, m, A, B, work), alpha = d / 5 > 1 ? d / 5 : 1;
            long target = d > 2 * alpha ? alpha + (long)((shop_rng(&rng) >> 33) % (uint64_t)(d - 2 * alpha + 1)) : d / 2;
            Shop *S = shop_new(n, m, mach, dur, A);
            shop_eval(S);
            for (long s = 0; s < target; s++) { int ok = shop_relink_step(S, B, work, &rng); if (ok <= 0) break; st[2]++; }
            memcpy(t, S->seq, sizeof(int) * N);
            shop_free(S);
        }
        int c = pfa_shop_tabu(n, m, mach, dur, t, 0x7fffffffffffL, cutoff, 0.0, shop_rng(&rng), best, tst);
        st[1] += tst[0];
        st[3] += shop_mode ? shop_pool_offer_worst(P, N, n, m, pool, pcost, dm, best, c, work) : shop_pool_offer(P, N, n, m, pool, pcost, dm, best, c, work);
        st[0]++;
    }
    int bc = 0x7fffffff;
    for (int i = 0; i < P; i++) if (pcost[i] < bc) bc = pcost[i];
    if (stats) memcpy(stats, st, sizeof(st));
    free(t); free(best); free(work);
    return bc;
}
