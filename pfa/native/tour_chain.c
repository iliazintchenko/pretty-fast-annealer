// The tour kernel (included by pfa.c after tour.c): Lin and Kernighan's chain of sequential exchanges over the
// candidates, at zero temperature. The move is the Boltzmann draw's limit on this landscape: at any finite
// temperature the distribution over tours is exp(-beta L) whatever the moves, so a richer move only mixes faster,
// while a tour's quality is the local optimum of the chained neighbourhood, which is reached cold.
//
// A chain starts at a tour edge (t1, t2). Its state is the open path from t1 to a tip f, the tour without the edge
// (f, t1). From the tip, every sequential exchange of at most depth + 1 edges over the candidates is enumerated:
// add (f, t3) for a candidate t3 of f, remove either tour edge (t3, t4) at t3, including the one that leaves a cycle
// and a path until the next exchange repairs it, and so on, as long as the open path stays shorter than the start
// tour (the gain criterion); each one that closes to a tour is an option with the exact change of the tour's length.
// The first option met that leaves the tour shorter than at the chain's start is applied and the chain ends.
// Otherwise the option of the full depth whose open path is shortest is applied, if its gain pays for the cheapest
// candidate edge at its tip (else the next step could not pass the criterion), and the chain goes on from its tip;
// else the chain's exchanges are undone. The chain is kept from undoing itself by Lin and Kernighan's rule in LKH's
// form: the last edge an open option removes must be an edge of the tour the chain started from that no earlier
// option removed; every other edge removed may be any edge of the tour, the chain's own added ones included.
// depth 4 gives the 5-opt basis. The gains are read on the penalised distances of the ascent (every tour changes by
// the same constant, so the choices change and the lengths do not), as in LKH.
//
// A descent runs chains from both tour edges of every active city, with don't-look bits. Given a reference tour (the
// best tour, after a restart), no chain starts by removing one of its edges: the descent repairs the restart's
// departures from it and leaves what they share to the fold.
//
// Non-sequential moves. A close that would shorten the tour but leaves two cycles instead is bridged: an edge of the
// smaller cycle and an edge of the other are exchanged for the two edges that join the cycles, the first at a
// candidate pair, and the whole is an option when it shortens the tour. That is LKH's Gain23 in its 2-opt-bridge
// form: a double bridge when the first exchange was a 2-opt, a non-sequential 5-opt when it was a 3-opt.
//
// An exchange is checked and applied generically from its removed and added edge sets: the removed edges cut the tour
// into segments, the added edges link the segments' ends, and the exchange is a tour iff following those links visits
// every segment once. The tour array is rewritten from the longest segment on, the others copied in the order and
// direction the links give, so a 2-opt costs its shorter side. Interior cities of a copied segment keep their
// neighbours; those of a reversed one flip o; the ends get new neighbours and their table entries recomputed.
#define KOPT_MAXK 7
#define CHAIN_MAX 64

typedef struct { int k; int rem[KOPT_MAXK][2], add[KOPT_MAXK][2]; } Move;   // k removed and k added edges
typedef struct { int na, nr; int add[CHAIN_MAX * KOPT_MAXK][2], rem[CHAIN_MAX * KOPT_MAXK][2]; } Chain;

static inline int edge_in(const int (*edges)[2], int cnt, int u, int v) {
    for (int i = 0; i < cnt; i++) if ((edges[i][0] == u && edges[i][1] == v) || (edges[i][0] == v && edges[i][1] == u)) return 1;
    return 0;
}

static inline int tour_edge_pos(const Tour *T, int u, int v) {   // the position p of tour edge (u, v): tour[p], tour[p + 1]
    int p = T->pos[u], s = p + 1 == T->n ? 0 : p + 1;
    return T->tour[s] == v ? p : T->pos[v];
}

// the sequential exchange t[0 .. 2k): removed (t[2i], t[2i+1]), added (t[2i+1], t[2i+2 mod 2k])
static void move_from_t(Move *M, const int *t, int k) {
    M->k = k;
    for (int i = 0; i < k; i++) {
        M->rem[i][0] = t[2 * i]; M->rem[i][1] = t[2 * i + 1];
        M->add[i][0] = t[2 * i + 1]; M->add[i][1] = t[(2 * i + 2) % (2 * k)];
    }
}

// The segments and links of an exchange: e the removed edges' positions sorted (segment i: positions e[i] + 1 ..
// e[i+1]), order the segments in the order the links lead from segment 0 and rev whether each is run backwards, cnt
// how many the walk from segment 0 visits before it closes. Returns 1 iff the result is a tour (cnt == k); 0 with
// cnt < k when the result is several cycles, -1 when the exchange is malformed.
static int kopt_check(const Tour *T, const Move *M, int *e, int *order, int *rev, int *cnt) {
    const int n = T->n, k = M->k;
    for (int i = 0; i < k; i++) {
        int v = tour_edge_pos(T, M->rem[i][0], M->rem[i][1]), j = i;
        while (j > 0 && e[j - 1] > v) { e[j] = e[j - 1]; j--; }
        e[j] = v;
    }
    for (int i = 1; i < k; i++) if (e[i] == e[i - 1]) return -1;
    int sc[2 * KOPT_MAXK], link[2 * KOPT_MAXK];   // the city at each segment end: 2i its start, 2i + 1 its end
    for (int i = 0; i < k; i++) {
        sc[2 * i] = T->tour[(e[i] + 1) % n];
        sc[2 * i + 1] = T->tour[e[(i + 1) % k]];
        link[2 * i] = link[2 * i + 1] = -1;
    }
    for (int i = 0; i < k; i++) {
        int u = M->add[i][0], v = M->add[i][1], su = -1, sv = -1;
        if (u == v) return -1;
        for (int s = 0; s < 2 * k && su < 0; s++) if (link[s] < 0 && sc[s] == u) su = s;
        for (int s = 0; s < 2 * k && sv < 0; s++) if (link[s] < 0 && s != su && sc[s] == v) sv = s;
        if (su < 0 || sv < 0) return -1;
        link[su] = sv; link[sv] = su;
    }
    int c = 0, s = 0;
    for (;;) {
        order[c] = s >> 1; rev[c] = s & 1; c++;
        s = link[s ^ 1];
        if (s == 0 || c >= k) break;
    }
    *cnt = c;
    return c == k && s == 0;
}

// Applies a checked exchange; buf is scratch of n ints. The ends get their neighbours and entries recomputed.
static void kopt_apply(Tour *T, const Move *M, const int *e, const int *order, const int *rev, int *buf) {
    const int n = T->n, k = M->k;
    int len[KOPT_MAXK], m = 0, om = 0, seq[KOPT_MAXK], rv[KOPT_MAXK], off[KOPT_MAXK], total = 0;
    for (int i = 0; i < k; i++) { len[i] = (e[(i + 1) % k] - e[i] + n) % n; if (len[i] > len[m]) m = i; }
    while (order[om] != m) om++;
    for (int c = 0; c < k; c++) {   // the cycle read from the longest segment forwards
        int q = rev[om] ? (om - c + k) % k : (om + c) % k;
        seq[c] = order[q]; rv[c] = rev[om] ? !rev[q] : rev[q];
    }
    for (int c = 1; c < k; c++) {
        int j = seq[c], p = (e[j] + 1) % n;
        off[j] = total;
        for (int q = 0; q < len[j]; q++) { buf[total++] = T->tour[p]; p = p + 1 == n ? 0 : p + 1; }
    }
    int p = (e[(m + 1) % k] + 1) % n;
    for (int c = 1; c < k; c++) {
        int j = seq[c];
        for (int q = 0; q < len[j]; q++) {
            int city = buf[off[j] + (rv[c] ? len[j] - 1 - q : q)];
            T->tour[p] = city; T->pos[city] = p; p = p + 1 == n ? 0 : p + 1;
            if (rv[c]) T->o[city] ^= 1;
        }
    }
    for (int i = 0; i < k; i++) { tour_set_nb(T, M->rem[i][0]); tour_set_nb(T, M->rem[i][1]); }
}

// The best options at a tip: the close that leaves the tour shortest, when one leaves it shorter than at the chain's
// start, and otherwise the exchange whose open path is shortest. dL is the chain's change of the tour so far.
typedef struct { double dL, best_close, best_open; int has_close, has_open; Move mc, mo; } Best;

// a city's neighbours after the exchange M: its tour neighbours with M's removed edges taken out and added ones in
static inline void kopt_nbs(const Tour *T, const Move *M, int u, int *out) {
    int cnt = 0;
    for (int j = 0; j < 2; j++) { int v = T->nb[2 * u + j]; if (!edge_in(M->rem, M->k, u, v)) out[cnt++] = v; }
    for (int i = 0; i < M->k && cnt < 2; i++) {
        if (M->add[i][0] == u) out[cnt++] = M->add[i][1];
        else if (M->add[i][1] == u) out[cnt++] = M->add[i][0];
    }
    while (cnt < 2) out[cnt++] = -1;
}

// The bridge: M leaves two cycles (the walk from segment 0 visited cnt of its k segments); an edge (u, u') of the
// smaller cycle and (v, v') of the other, v a candidate of u, are exchanged for (u, v) and (u', v'), which joins the
// cycles either way round. The best such, when the whole is shorter than best_close, becomes the close.
static void kopt_bridge(const Tour *T, const Move *M, double total, const int *e, const int *order, int cnt, Best *B) {
    const int n = T->n, k = M->k;
    unsigned char first[KOPT_MAXK] = {0};
    for (int i = 0; i < cnt; i++) first[order[i]] = 1;
    long size_first = 0, size_all = 0;
    for (int i = 0; i < k; i++) { long len = (e[(i + 1) % k] - e[i] + n) % n; size_all += len; if (first[i]) size_first += len; }
    const int small = 2 * size_first <= size_all ? 1 : 0;   // the cycle to walk: the smaller
    double best = B->best_close - total - 1e-9;   // the bridge must beat this
    int bu = -1, bu2 = 0, bv = 0, bv2 = 0;
    for (int i = 0; i < k; i++) {
        if (first[i] != small) continue;
        long len = (e[(i + 1) % k] - e[i] + n) % n;
        int p = (e[i] + 1) % n;
        for (long q = 0; q < len; q++, p = p + 1 == n ? 0 : p + 1) {
            const int u = T->tour[p];
            int nu[2]; kopt_nbs(T, M, u, nu);
            for (int kk = 0; kk < T->K; kk++) {
                const int v = T->nn[u * T->K + kk];
                // v's segment: the position's place among the sorted cut positions
                int pv = T->pos[v], sv = 0;
                while (sv + 1 < k && e[sv + 1] < pv) sv++;
                if (e[sv] >= pv) sv = k - 1;   // before the first cut: the wrapping segment
                if (first[sv] == small) continue;   // v must lie on the other cycle
                int nv[2]; kopt_nbs(T, M, v, nv);
                for (int a = 0; a < 2; a++) {
                    const int u2 = nu[a];
                    if (u2 < 0 || edge_in(M->add, k, u, u2)) continue;
                    for (int b = 0; b < 2; b++) {
                        const int v2 = nv[b];
                        if (v2 < 0 || edge_in(M->add, k, v, v2) || u2 == v2 || u == v2 || v == u2) continue;
                        double d = tour_c(T, u, v) + tour_c(T, u2, v2) - tour_c(T, u, u2) - tour_c(T, v, v2);
                        if (d < best) { best = d; bu = u; bu2 = u2; bv = v; bv2 = v2; }
                    }
                }
            }
        }
    }
    if (bu < 0 || k + 2 > KOPT_MAXK) return;
    Move M2 = *M;
    M2.rem[k][0] = bu; M2.rem[k][1] = bu2; M2.rem[k + 1][0] = bv; M2.rem[k + 1][1] = bv2;
    M2.add[k][0] = bu; M2.add[k][1] = bv; M2.add[k + 1][0] = bu2; M2.add[k + 1][1] = bv2;
    M2.k = k + 2;
    int e2[KOPT_MAXK], order2[KOPT_MAXK], rev2[KOPT_MAXK], c2;
    if (kopt_check(T, &M2, e2, order2, rev2, &c2) == 1) { B->best_close = total + best; B->has_close = 1; B->mc = M2; }
}

static inline int kopt_removed(const int *t, int k, int u, int v) {
    for (int i = 0; i < k; i++) if ((t[2 * i] == u && t[2 * i + 1] == v) || (t[2 * i] == v && t[2 * i + 1] == u)) return 1;
    return 0;
}
static inline int kopt_added(const int *t, int k, int u, int v) {   // among the k - 1 added edges so far
    for (int i = 0; i + 1 < k; i++) if ((t[2 * i + 1] == u && t[2 * i + 2] == v) || (t[2 * i + 1] == v && t[2 * i + 2] == u)) return 1;
    return 0;
}

// The options from the exchange t[0 .. 2k) with k removed edges, g its removed minus added length so far, C the chain.
// Only an option that would be chosen is checked for being a tour; the enumeration ends at the first improving close.
static void kopt_extend(const Tour *T, int *t, int k, double g, int depth, Best *B, const Chain *C) {
    const int f = t[2 * k - 1];
    if (k >= 2 && !tour_adjacent(T, f, t[0])) {   // close: (f, t1) is neither present nor just removed
        double total = B->dL + tour_c(T, f, t[0]) - g, G = g - B->dL;
        int improving = total < B->best_close - 1e-9;
        int better_open = !B->has_close && k == depth + 1 && G > B->best_open + 1e-9
                          && !edge_in(C->add, C->na, t[2 * k - 2], f) && !edge_in(C->rem, C->nr, t[2 * k - 2], f);   // the last removed edge: of the start tour, not removed before
        if (better_open) {   // the open gain must pay for the cheapest candidate edge at the tip
            double cheapest = INFINITY;
            for (int kk = 0; kk < T->K; kk++) if (!tour_adjacent(T, f, T->nn[f * T->K + kk]) && T->cc[f * T->K + kk] < cheapest) cheapest = T->cc[f * T->K + kk];
            better_open = G > cheapest;
        }
        if (improving || better_open) {
            Move M;
            move_from_t(&M, t, k);
            int e[KOPT_MAXK], order[KOPT_MAXK], rev[KOPT_MAXK], cnt;
            int ok = kopt_check(T, &M, e, order, rev, &cnt);
            if (ok == 1) {
                if (improving) { B->best_close = total; B->has_close = 1; B->mc = M; }
                else { B->best_open = G; B->has_open = 1; B->mo = M; }
            } else if (ok == 0 && improving) {   // two cycles (or more, which the bridge's own check rejects)
                kopt_bridge(T, &M, total, e, order, cnt, B);
            }
        }
    }
    if (k > depth) return;
    for (int kk = 0; kk < T->K && !B->has_close; kk++) {
        int c = T->nn[f * T->K + kk];
        double g1 = g - T->cc[f * T->K + kk];
        if (g1 - B->dL <= 0.0) continue;   // the gain criterion
        if (tour_adjacent(T, f, c) || kopt_added(t, k, f, c)) continue;
        for (int j = 0; j < 2 && !B->has_close; j++) {
            int c2 = T->nb[2 * c + j];
            if (kopt_removed(t, k, c, c2)) continue;
            t[2 * k] = c; t[2 * k + 1] = c2;
            kopt_extend(T, t, k + 1, g1 + T->ec[2 * c + j], depth, B, C);
        }
    }
}

// A chain from the edge (t1, t2). Returns the change of the tour's length (negative, or zero when undone); the cities
// of the applied exchanges are appended to touched (up to 2 * KOPT_MAXK * CHAIN_MAX).
static double kopt_chain(Tour *T, int t1, int t2, int depth, Chain *C, int *buf, int *touched, int *ntouched) {
    static Move moves[CHAIN_MAX];
    int nm = 0, f = t2;
    double dL = 0.0;
    C->na = C->nr = 0;
    *ntouched = 0;
    for (int s = 0; s < CHAIN_MAX; s++) {
        int t[2 * KOPT_MAXK] = {t1, f};
        Best B = {dL, 0.0, 0.0, 0, 0};
        kopt_extend(T, t, 1, tour_c(T, t1, f), depth, &B, C);
        if (!B.has_close && !B.has_open) break;
        const Move *M = B.has_close ? &B.mc : &B.mo;
        int e[KOPT_MAXK], order[KOPT_MAXK], rev[KOPT_MAXK], cnt;
        kopt_check(T, M, e, order, rev, &cnt);
        kopt_apply(T, M, e, order, rev, buf);
        moves[nm++] = *M;
        for (int i = 0; i < M->k; i++) { C->rem[C->nr][0] = M->rem[i][0]; C->rem[C->nr][1] = M->rem[i][1]; C->nr++; }
        if (B.has_close) {
            dL = B.best_close;
            for (int m = 0; m < nm; m++) for (int i = 0; i < moves[m].k; i++) { touched[(*ntouched)++] = moves[m].rem[i][0]; touched[(*ntouched)++] = moves[m].rem[i][1]; }
            T->L += dL;
            return dL;
        }
        f = M->add[M->k - 1][0];   // the sequential exchange's tip: the closing edge (f, t1) is add[k - 1]
        dL = tour_c(T, f, t1) - B.best_open;   // the closed change: the open gain less the closing edge
        for (int i = 0; i + 1 < M->k; i++) { C->add[C->na][0] = M->add[i][0]; C->add[C->na][1] = M->add[i][1]; C->na++; }
    }
    for (int m = nm - 1; m >= 0; m--) {   // undo: the inverse exchange removes what was added and adds what was removed
        Move inv = {moves[m].k};
        for (int i = 0; i < inv.k; i++) { inv.rem[i][0] = moves[m].add[i][0]; inv.rem[i][1] = moves[m].add[i][1]; inv.add[i][0] = moves[m].rem[i][0]; inv.add[i][1] = moves[m].rem[i][1]; }
        int e[KOPT_MAXK], order[KOPT_MAXK], rev[KOPT_MAXK], cnt;
        kopt_check(T, &inv, e, order, rev, &cnt);
        kopt_apply(T, &inv, e, order, rev, buf);
    }
    return 0.0;
}

// The descent with don't-look bits: chains from both edges of every active city, a city made inactive when no chain
// from it shortens the tour and active again when a chain touches it. Ends at a local optimum of the chained
// neighbourhood. x, y: coordinates; nn: n x K candidates; pi: the ascent's penalties; tour_io: the city at each
// position, in and out; chains: the number of chains run, added to; ref: a tour (the city at each position) none of
// whose edges a chain starts by removing, or NULL. Returns the length.
double pfa_tour_descent(int n, const double *x, const double *y, int round_mode, int K, const int *nn, const double *pi,
                        int *tour_io, int depth, long *chains, const int *ref) {
    Tour T = tour_new(n, K, round_mode, x, y, nn, tour_io, pi);
    int *queue = malloc(sizeof(int) * n), *buf = malloc(sizeof(int) * n);
    int *touched = malloc(sizeof(int) * 2 * KOPT_MAXK * CHAIN_MAX), ntouched;
    unsigned char *active = malloc(n);
    Chain *C = malloc(sizeof(Chain));
    int *rn = NULL;   // ref's two neighbours of each city
    if (ref) {
        rn = malloc(sizeof(int) * 2 * n);
        for (int p = 0; p < n; p++) { int c = ref[p]; rn[2 * c] = ref[p == 0 ? n - 1 : p - 1]; rn[2 * c + 1] = ref[p + 1 == n ? 0 : p + 1]; }
    }
    int head = 0, count = n;
    for (int a = 0; a < n; a++) { queue[a] = a; active[a] = 1; }
    while (count > 0) {
        const int t1 = queue[head]; head = head + 1 == n ? 0 : head + 1; count--; active[t1] = 0;
        for (int i = 0; i < 2 && !active[t1]; i++) {
            if (rn && (rn[2 * t1] == T.nb[2 * t1 + i] || rn[2 * t1 + 1] == T.nb[2 * t1 + i])) continue;
            (*chains)++;
            if (kopt_chain(&T, t1, T.nb[2 * t1 + i], depth, C, buf, touched, &ntouched) < 0.0) {
                touched[ntouched++] = t1;
                for (int q = 0; q < ntouched; q++) {
                    const int z = touched[q];
                    if (!active[z]) { active[z] = 1; queue[(head + count) % n] = z; count++; }
                }
            }
        }
    }
    double L = T.L;
    tour_free(&T); free(queue); free(buf); free(touched); free(active); free(C); free(rn);
    return L;
}

static inline uint64_t tour_rng(uint64_t *s) { *s ^= *s >> 12; *s ^= *s << 25; *s ^= *s >> 27; return *s * 2685821657736338717ULL; }

// A tour by a random walk along the candidate edges: from a random city, an unvisited candidate on the given tour's
// edge (best: the city at each position, or NULL) with probability share when there is one, else a random unvisited
// candidate, else the nearest unvisited city. tour_out: the city at each position.
void pfa_tour_walk(int n, const double *x, const double *y, int round_mode, int K, const int *nn, const int *best,
                   double share, uint64_t seed, int *tour_out) {
    int *bn = malloc(sizeof(int) * 2 * n), *left = malloc(sizeof(int) * n), *at = malloc(sizeof(int) * n);
    unsigned char *seen = calloc(n, 1);
    if (best) for (int p = 0; p < n; p++) { int c = best[p]; bn[2 * c] = best[p == 0 ? n - 1 : p - 1]; bn[2 * c + 1] = best[p + 1 == n ? 0 : p + 1]; }
    for (int c = 0; c < n; c++) { left[c] = c; at[c] = c; }   // the unvisited cities, swap-removed
    int nleft = n;
    uint64_t rng = seed | 1;
    int cur = (int)((tour_rng(&rng) >> 32) % (uint64_t)n);
    for (int p = 0; p < n; p++) {
        tour_out[p] = cur; seen[cur] = 1;
        int l = at[cur]; nleft--; left[l] = left[nleft]; at[left[l]] = l;
        if (p == n - 1) break;
        int free_[16], kept[16], nf = 0, nk = 0;
        for (int k = 0; k < K; k++) {
            int c = nn[cur * K + k];
            if (seen[c]) continue;
            free_[nf++] = c;
            if (best && (bn[2 * cur] == c || bn[2 * cur + 1] == c)) kept[nk++] = c;
        }
        if (nk && (double)(tour_rng(&rng) >> 11) * 0x1p-53 < share) cur = kept[(tour_rng(&rng) >> 32) % (uint64_t)nk];
        else if (nf) cur = free_[(tour_rng(&rng) >> 32) % (uint64_t)nf];
        else {
            int bestc = left[0];
            double bd = INFINITY;
            for (int i = 0; i < nleft; i++) {
                double dx = x[cur] - x[left[i]], dy = y[cur] - y[left[i]], d = dx * dx + dy * dy;
                if (d < bd) { bd = d; bestc = left[i]; }
            }
            cur = bestc;
        }
    }
    free(bn); free(left); free(at); free(seen);
}
