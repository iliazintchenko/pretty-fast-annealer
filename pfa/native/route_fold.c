// The fold for routes (included by pfa.c after route.c): edge assembly on the graph with the depot as one node. Two
// route sets A and B are read as edge multisets on the customers plus the depot (every copy the one depot, an empty
// route a loop at it); the edges of one and not the other form alternating cycles; each cycle applied to A alone
// (A's edges of it out, B's in) gives a child when every resulting cycle passes through the depot, and the child of
// the least penalised cost is written out as a cycle with depot copies. Returns that cost, INFINITY when no cycle gives
// a child.
typedef struct { int a, b; } REdge;
static int redge_cmp(const void *p, const void *q) {
    const REdge *x = p, *y = q;
    return x->a != y->a ? x->a - y->a : x->b - y->b;
}

static void rfold_edges(int n, int nc, const int *t, REdge *E) {   // the cycle's edges on the depot graph, sorted
    for (int p = 0; p < n; p++) {
        int u = t[p], v = t[p + 1 == n ? 0 : p + 1];
        u = u < nc ? u + 1 : 0; v = v < nc ? v + 1 : 0;
        E[p].a = u < v ? u : v; E[p].b = u < v ? v : u;
    }
    qsort(E, n, sizeof(REdge), redge_cmp);
}

// The routes of a child given by the customers' neighbours nb (2 per customer) and the depot's edge counts dep (per
// node; dep[0] loops). Writes the customers route by route, 0 after each route, into rt; returns the penalised cost or
// INFINITY when a customer is missed or met twice (a cycle without the depot).
static double rfold_routes(const LS *S, const int *nb, int *dep, int nc, int m, int *rt, int *nroutes) {
    unsigned char *vis = calloc(nc + 1, 1);
    int k = 0, seen = 0, routes = 0;
    double cost = 0.0;
    for (int v = 1; v <= nc; v++) {
        while (dep[v] > 0) {
            dep[v]--;
            int prev = 0, c = v, load = 0;
            double len = ls_d(S, 0, c);
            while (c != 0) {
                if (vis[c]) { free(vis); return INFINITY; }
                vis[c] = 1; seen++; load += S->dem[c];
                int w = nb[2 * c] == prev ? nb[2 * c + 1] : nb[2 * c];
                if (w < 0) { free(vis); return INFINITY; }
                len += ls_d(S, c, w);
                rt[k++] = c;
                if (w == 0) dep[c]--;
                prev = c; c = w;
            }
            rt[k++] = 0;
            cost += len + ls_pen(S, load); routes++;
        }
    }
    free(vis);
    if (seen != nc || routes > m) return INFINITY;
    *nroutes = routes;
    return cost;
}

double pfa_route_fold(int nc, int m, const double *x, const double *y, int round_mode, const int *demand, int Q, double lambda,
                      const int *a, const int *b, int *out, uint64_t seed, const double *dist) {
    const int n = nc + m, N = nc + 1;
    LS S = {nc, m, 0, Q, round_mode, lambda};
    S.Dm = dist;
    double *xs = malloc(sizeof(double) * N), *ys = malloc(sizeof(double) * N);
    int *dem = malloc(sizeof(int) * N);
    xs[0] = x[nc]; ys[0] = y[nc]; dem[0] = 0;
    for (int c = 0; c < nc; c++) { xs[c + 1] = x[c]; ys[c + 1] = y[c]; dem[c + 1] = demand[c]; }
    S.x = xs; S.y = ys; S.dem = dem;
    REdge *EA = malloc(sizeof(REdge) * n), *EB = malloc(sizeof(REdge) * n), *GA = malloc(sizeof(REdge) * n), *GB = malloc(sizeof(REdge) * n);
    rfold_edges(n, nc, a, EA); rfold_edges(n, nc, b, EB);
    int na = 0, nb_ = 0;
    for (int i = 0, j = 0; i < n || j < n;) {   // the symmetric difference by a merge
        int c = i == n ? 1 : j == n ? -1 : redge_cmp(&EA[i], &EB[j]);
        if (c == 0) { i++; j++; } else if (c < 0) GA[na++] = EA[i++]; else GB[nb_++] = EB[j++];
    }
    double result = INFINITY;
    // A's adjacency: customers' two neighbours, the depot's edge counts per node
    int *nb = malloc(sizeof(int) * 2 * N), *cnt = calloc(N, sizeof(int)), *dep = calloc(N, sizeof(int));
    for (int e = 0; e < n; e++) {
        int u = EA[e].a, v = EA[e].b;
        if (u) nb[2 * u + cnt[u]++] = v;
        if (v && v != u) nb[2 * v + cnt[v]++] = u;
        if (u == 0) dep[v]++;
    }
    int *nb2 = malloc(sizeof(int) * 2 * N), *dep2 = malloc(sizeof(int) * N), *rt = malloc(sizeof(int) * (n + m + 1)), *best_rt = malloc(sizeof(int) * (n + m + 1));
    int best_k = 0;
    if (na > 0) {
        // incidence of the difference edges, per side
        int *stA = calloc(N + 1, sizeof(int)), *stB = calloc(N + 1, sizeof(int));
        for (int e = 0; e < na; e++) { stA[GA[e].a + 1]++; if (GA[e].b != GA[e].a) stA[GA[e].b + 1]++; }
        for (int e = 0; e < nb_; e++) { stB[GB[e].a + 1]++; if (GB[e].b != GB[e].a) stB[GB[e].b + 1]++; }
        for (int v = 0; v < N; v++) { stA[v + 1] += stA[v]; stB[v + 1] += stB[v]; }
        int *incA = malloc(sizeof(int) * stA[N]), *incB = malloc(sizeof(int) * stB[N]), *fa = calloc(N, sizeof(int)), *fb = calloc(N, sizeof(int));
        for (int e = 0; e < na; e++) { incA[stA[GA[e].a] + fa[GA[e].a]++] = e; if (GA[e].b != GA[e].a) incA[stA[GA[e].b] + fa[GA[e].b]++] = e; }
        for (int e = 0; e < nb_; e++) { incB[stB[GB[e].a] + fb[GB[e].a]++] = e; if (GB[e].b != GB[e].a) incB[stB[GB[e].b] + fb[GB[e].b]++] = e; }
        unsigned char *usedA = calloc(na, 1), *usedB = calloc(nb_ > 0 ? nb_ : 1, 1);
        int *path = malloc(sizeof(int) * (na + nb_ + 1)), *ptype = malloc(sizeof(int) * (na + nb_ + 1));
        int *posA = malloc(sizeof(int) * N), *posB = malloc(sizeof(int) * N), *perm = malloc(sizeof(int) * N);
        uint64_t rng = seed | 1;
        for (int v = 0; v < N; v++) perm[v] = v;
        for (int v = N - 1; v > 0; v--) { int r = (int)((tour_rng(&rng) >> 33) % (uint64_t)(v + 1)); int t = perm[v]; perm[v] = perm[r]; perm[r] = t; }
        for (int si = 0; si < N; si++) {
            const int s = perm[si];
            for (;;) {
                int have = 0;
                for (int q = stA[s]; q < stA[s + 1]; q++) if (!usedA[incA[q]]) { have = 1; break; }
                if (!have) break;
                for (int v = 0; v < N; v++) posA[v] = posB[v] = -1;
                int plen = 0, cur = s, typ = 0;
                posA[s] = 0;
                for (;;) {
                    int e = -1;
                    if (typ == 0) { for (int q = stA[cur]; q < stA[cur + 1]; q++) if (!usedA[incA[q]]) { e = incA[q]; break; } }
                    else { for (int q = stB[cur]; q < stB[cur + 1]; q++) if (!usedB[incB[q]]) { e = incB[q]; break; } }
                    if (e < 0) break;   // stuck on unbalanced degrees: the path is dropped
                    const REdge *E = typ == 0 ? &GA[e] : &GB[e];
                    if (typ == 0) usedA[e] = 1; else usedB[e] = 1;
                    int nxt = E->a == cur ? E->b : E->a;
                    path[plen] = e; ptype[plen] = typ; plen++;
                    typ ^= 1; cur = nxt;
                    int *pos = typ == 0 ? posA : posB;
                    if (pos[cur] < 0) { pos[cur] = plen; continue; }
                    // a cycle: path[from .. plen). Evaluate it on A.
                    const int from = pos[cur];
                    memcpy(nb2, nb, sizeof(int) * 2 * N); memcpy(dep2, dep, sizeof(int) * N);
                    for (int q = from; q < plen; q++) if (ptype[q] == 0) {   // A's edges out
                        int u = GA[path[q]].a, v = GA[path[q]].b;
                        if (u == 0) dep2[v]--;
                        else { if (nb2[2 * u] == v) nb2[2 * u] = -1; else nb2[2 * u + 1] = -1; }
                        if (v && v != u) { if (nb2[2 * v] == u) nb2[2 * v] = -1; else nb2[2 * v + 1] = -1; }
                    }
                    for (int q = from; q < plen; q++) if (ptype[q] == 1) {   // B's edges in
                        int u = GB[path[q]].a, v = GB[path[q]].b;
                        if (u == 0) dep2[v]++;
                        else { if (nb2[2 * u] < 0) nb2[2 * u] = v; else nb2[2 * u + 1] = v; }
                        if (v && v != u) { if (nb2[2 * v] < 0) nb2[2 * v] = u; else nb2[2 * v + 1] = u; }
                    }
                    int nroutes = 0, k = 0;
                    double cost = rfold_routes(&S, nb2, dep2, nc, m, rt, &nroutes);
                    if (cost < result) { result = cost; best_k = nc + nroutes; memcpy(best_rt, rt, sizeof(int) * best_k); }
                    (void)k;
                    // back to the cut: replay the path up to from
                    plen = from;
                    for (int v = 0; v < N; v++) posA[v] = posB[v] = -1;
                    cur = s; typ = 0; posA[s] = 0;
                    for (int q = 0; q < plen; q++) {
                        const REdge *E2 = ptype[q] == 0 ? &GA[path[q]] : &GB[path[q]];
                        cur = E2->a == cur ? E2->b : E2->a; typ ^= 1;
                        (typ == 0 ? posA : posB)[cur] = q + 1;
                    }
                    if (plen == 0) break;
                }
            }
        }
        free(stA); free(stB); free(incA); free(incB); free(fa); free(fb); free(usedA); free(usedB); free(path); free(ptype); free(posA); free(posB); free(perm);
    }
    if (result < INFINITY) {   // the child as a cycle with copies: a copy before each route, the unused copies at the end
        int k = 0, r = 0;
        out[k++] = nc + r++;
        for (int q = 0; q < best_k; q++) {
            if (best_rt[q] == 0) { if (q + 1 < best_k) out[k++] = nc + r++; }
            else out[k++] = best_rt[q] - 1;
        }
        while (r < m) out[k++] = nc + r++;
    }
    free(xs); free(ys); free(dem); free(EA); free(EB); free(GA); free(GB); free(nb); free(cnt); free(dep); free(nb2); free(dep2); free(rt); free(best_rt);
    return result;
}

// PFA's loop on routes, `restarts` of them: a walk over the customers biased to the best's route edges (share), Split,
// the descent repairing the departures, a repair descent at 10 lambda when infeasible, the edge-assembly fold of the
// result into the best and the child descended; the best feasible kept. best_io holds the best cycle in and out
// (*best_len INFINITY on the first call: the first restart's result is taken). nn: nc x K candidates (0-based).
// Returns the restarts whose own result or fold improved the best.
// PFA's loop on routes. A pool of P solutions, kept the best distinct ones by cost and by distance to the others: the
// pool takes a generation of G newcomers (or none) and is culled to P by Vidal's biased fitness, the rank by cost plus
// the rank by closeness (the mean distance to the five nearest), clones first. A restart is a child of two members,
// each the better by cost of two drawn at random, by Vidal's ordered crossover (a random stretch of one's customer
// order, the rest in the other's order) through Split; every fourth restart is instead a walk along the candidate
// edges biased to a member's route edges, which keeps the pool apart. The result is descended from its departures,
// repaired at ten times the penalty when infeasible, folded by edge assembly into the best and into the member it
// came from, each child descended from the parent it leaves; every feasible result and child is offered to the pool.
// The pool's pairwise distances (route edges of one not in the other) are kept in dm ((P + G)^2).
static int pool_dist(int nc, int m, const int *a, const int *b, int *bn) {
    const int n = nc + m;
    for (int p = 0; p < n; p++) {
        int c = b[p];
        if (c >= nc) continue;
        int u = b[p == 0 ? n - 1 : p - 1], v = b[p + 1 == n ? 0 : p + 1];
        bn[2 * c] = u < nc ? u : -1; bn[2 * c + 1] = v < nc ? v : -1;
    }
    int d = 0;
    for (int p = 0; p < n; p++) {
        int c = a[p];
        if (c >= nc) continue;
        int v = a[p + 1 == n ? 0 : p + 1]; v = v < nc ? v : -1;
        if (bn[2 * c] != v && bn[2 * c + 1] != v) d++;
    }
    return d;
}

// The biased fitness of the filled members among the PT slots: the cost rank plus the closeness rank, closeness the
// mean distance to the five nearest others (Vidal's rule, the elite weight 4).
static void pool_fitness(int PT, const double *plen, const int *dm, double *fit) {
    double *close = malloc(sizeof(double) * PT);
    int cnt = 0;
    for (int i = 0; i < PT; i++) if (plen[i] < INFINITY) cnt++;
    for (int i = 0; i < PT; i++) {
        if (plen[i] == INFINITY) { fit[i] = -1.0; close[i] = 0.0; continue; }
        int best[5], nb = 0;
        for (int j = 0; j < PT; j++) {
            if (j == i || plen[j] == INFINITY) continue;
            int d = dm[i * PT + j];
            if (nb == 5 && d >= best[nb - 1]) continue;
            int q = nb < 5 ? nb++ : nb - 1;
            while (q > 0 && d < best[q - 1]) { best[q] = best[q - 1]; q--; }
            best[q] = d;
        }
        double sum = 0.0;
        for (int q = 0; q < nb; q++) sum += best[q];
        close[i] = nb ? sum / nb : 0.0;
    }
    for (int i = 0; i < PT; i++) {
        if (plen[i] == INFINITY) continue;
        int rc = 0, rd = 0;
        for (int j = 0; j < PT; j++) { if (plen[j] == INFINITY) continue; if (plen[j] < plen[i]) rc++; if (close[j] > close[i]) rd++; }
        fit[i] = (double)rc / cnt + (1.0 - 4.0 / cnt) * (double)rd / cnt;
    }
    free(close);
}

// A newcomer joins a free slot of the P + G; when the G children of a generation are in, the pool is culled to P, a
// clone first (a member at distance zero from another, the costlier), then the worst biased fitness. With G = 0 the
// worst among the P + 1 leaves at once (measured better on the large instances, the generation on the small ones).
static void pool_offer(int P, int G, int nc, int m, int *pool, double *plen, const int *t, double len, int *bn, int *dm) {
    const int n = nc + m, PT = P + G;
    for (int i = 0; i < PT; i++) if (fabs(plen[i] - len) < 1e-9) return;   // the same length: taken as the same solution
    int slot = -1;
    for (int i = 0; i < PT; i++) if (plen[i] == INFINITY) { slot = i; break; }
    if (slot < 0) {   // full with no generation to wait for: the worst by biased fitness among the P + 1 leaves at once
        const int P1 = PT + 1;
        double *L = malloc(sizeof(double) * P1), *fit = malloc(sizeof(double) * P1);
        int *dmx = malloc(sizeof(int) * P1 * P1);
        for (int i = 0; i < PT; i++) {
            L[i] = plen[i];
            for (int j = 0; j < PT; j++) dmx[i * P1 + j] = dm[i * PT + j];
            dmx[i * P1 + PT] = dmx[PT * P1 + i] = pool_dist(nc, m, t, pool + (long)i * n, bn);
        }
        L[PT] = len; dmx[PT * P1 + PT] = 0;
        pool_fitness(P1, L, dmx, fit);
        int w = 0;
        for (int i = 1; i < P1; i++) if (fit[i] > fit[w]) w = i;
        free(L); free(fit); free(dmx);
        if (w == PT) return;
        slot = w;
    }
    plen[slot] = len; memcpy(pool + (long)slot * n, t, sizeof(int) * n);
    for (int i = 0; i < PT; i++) dm[slot * PT + i] = dm[i * PT + slot] = i == slot || plen[i] == INFINITY ? 0 : pool_dist(nc, m, t, pool + (long)i * n, bn);
    int filled = 0;
    for (int i = 0; i < PT; i++) filled += plen[i] < INFINITY;
    if (filled < PT) return;
    double *fit = malloc(sizeof(double) * PT);
    while (filled > P) {
        int w = -1;
        for (int i = 0; i < PT && w < 0; i++) if (plen[i] < INFINITY) for (int j = 0; j < PT; j++) if (j != i && plen[j] < INFINITY && dm[i * PT + j] == 0) { w = plen[i] > plen[j] ? i : j; break; }
        if (w < 0) { pool_fitness(PT, plen, dm, fit); for (int i = 0; i < PT; i++) if (plen[i] < INFINITY && (w < 0 || fit[i] > fit[w])) w = i; }
        plen[w] = INFINITY; filled--;
    }
    free(fit);
}

static int pool_best(int P, const double *plen) { int b = 0; for (int i = 1; i < P; i++) if (plen[i] < plen[b]) b = i; return b; }

// Vidal's ordered crossover on the customer orders of two cycles: a random stretch of a, the rest in b's order
static void pool_ox(int nc, int m, const int *a, const int *b, int *order, uint64_t *rng, unsigned char *taken) {
    const int n = nc + m;
    int k = 0;
    for (int p = 0; p < n; p++) if (a[p] < nc) order[k++] = a[p];
    int i = (int)((tour_rng(rng) >> 33) % (uint64_t)nc), j = (int)((tour_rng(rng) >> 33) % (uint64_t)nc);
    if (i > j) { int t = i; i = j; j = t; }
    memset(taken, 0, nc);
    for (int q = i; q <= j; q++) taken[order[q]] = 1;
    int pos = (j + 1) % nc;
    for (int q = 0; q < n; q++) {
        int c = b[q];
        if (c >= nc || taken[c]) continue;
        while (pos >= i && pos <= j) pos = (pos + 1) % nc;
        order[pos] = c; pos = (pos + 1) % nc;
    }
}

// the route neighbours of each customer in the cycle t, -1 at the depot, into bn (2 per customer)
static void pool_edges(int nc, int m, const int *t, int *bn) {
    const int n = nc + m;
    for (int p = 0; p < n; p++) {
        int c = t[p];
        if (c >= nc) continue;
        int a = t[p == 0 ? n - 1 : p - 1], d = t[p + 1 == n ? 0 : p + 1];
        bn[2 * c] = a < nc ? a : -1; bn[2 * c + 1] = d < nc ? d : -1;
    }
}

// `restarts` restarts of the loop. nn: nc x K candidate customers (0-based); dist: the (nc + 1)^2 distance matrix with
// the depot first, or NULL. The pool has P members and room for a generation of G more: pool ((P + G) x n), plen
// (INFINITY where empty) and dm ((P + G)^2) persist across calls. Returns the restarts after which the best changed.
int pfa_route_loop(int nc, int m, const double *x, const double *y, int round_mode, const int *demand, int Q, double lambda,
                   int K, const int *nn, int restarts, uint64_t seed, double share, int P, int G, int *pool, double *plen, int *dm, const double *dist) {
    const int n = nc + m, PT = P + G;
    int *order = malloc(sizeof(int) * nc), *t = malloc(sizeof(int) * n), *child = malloc(sizeof(int) * n), *bn = malloc(sizeof(int) * 2 * nc);
    unsigned char *taken = malloc(nc);
    double *fit = malloc(sizeof(double) * PT);
    uint64_t rng = seed | 1;
    int improved = 0;
    double len, ex, lc, exc;
    long mv;
    for (int r = 0; r < restarts; r++) {
        int filled = 0;
        for (int i = 0; i < PT; i++) filled += plen[i] < INFINITY;
        const int b = pool_best(PT, plen);
        const double best0 = plen[b];
        int j = -1, j2 = -1;
#define PICK(out) { int pick = (int)((tour_rng(&rng) >> 33) % (uint64_t)filled); for (int i = 0; i < PT; i++) if (plen[i] < INFINITY && pick-- == 0) { out = i; break; } }
        if (filled) {   // two parents, each the better by cost of two drawn
            PICK(j); { int alt = j; PICK(alt); if (plen[alt] < plen[j]) j = alt; }
            if (filled >= 2) { do { PICK(j2); int alt = j2; PICK(alt); if (plen[alt] < plen[j2]) j2 = alt; } while (j2 == j); }
            pool_edges(nc, m, pool + (long)j * n, bn);
        }
#undef PICK
        const int crossed = j2 >= 0 && r % 4 != 3;
        if (crossed) pool_ox(nc, m, pool + (long)j * n, pool + (long)j2 * n, order, &rng, taken);
        else pfa_route_walk(nc, x, y, K, nn, j >= 0 ? bn : NULL, share, tour_rng(&rng), order);
        pfa_route_split(nc, m, x, y, round_mode, demand, Q, lambda, order, t);
        pfa_route_ls(nc, m, x, y, round_mode, demand, Q, lambda, K, nn, t, &len, &ex, &mv, j >= 0 && !crossed ? bn : NULL, dist);
        if (ex > 0) pfa_route_ls(nc, m, x, y, round_mode, demand, Q, 10 * lambda, K, nn, t, &len, &ex, &mv, NULL, dist);
        if (ex == 0) {
            if (filled) {   // the fold into the best and into the member
                int partners[2] = {b, j}, np = j == b ? 1 : 2;
                for (int q = 0; q < np; q++) {
                    const int *par = pool + (long)partners[q] * n;
                    double c = pfa_route_fold(nc, m, x, y, round_mode, demand, Q, lambda, par, t, child, tour_rng(&rng), dist);
                    if (c == INFINITY) continue;
                    pool_edges(nc, m, par, bn);
                    pfa_route_ls(nc, m, x, y, round_mode, demand, Q, lambda, K, nn, child, &lc, &exc, &mv, bn, dist);
                    if (exc > 0) pfa_route_ls(nc, m, x, y, round_mode, demand, Q, 10 * lambda, K, nn, child, &lc, &exc, &mv, NULL, dist);
                    if (exc == 0) pool_offer(P, G, nc, m, pool, plen, child, lc, bn, dm);
                }
            }
            pool_offer(P, G, nc, m, pool, plen, t, len, bn, dm);
        }
        if (plen[pool_best(PT, plen)] < best0 - 1e-9) improved++;
    }
    free(order); free(t); free(child); free(bn); free(taken);
    return improved;
}
