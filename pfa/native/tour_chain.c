// The tour kernel (included by pfa.c after tour.c): Lin and Kernighan's chain of sequential exchanges over the
// candidates, at zero temperature. The move is the Boltzmann draw's limit on this landscape: at any finite
// temperature the distribution over tours is exp(-beta L) whatever the moves, so a richer move only mixes faster,
// while a tour's quality is the local optimum of the chained neighbourhood, which is reached cold.
//
// A chain starts at a tour edge (t1, t2). Its state is the open path from t1 to a tip f, the tour without the edge
// (f, t1). From the tip, every sequential exchange of at most depth + 1 edges over the candidates is enumerated:
// add (f, t3) for a candidate t3 of f, remove either tour edge (t3, t4) at t3, including the one that leaves a cycle
// and a path until the next exchange repairs it, and so on; each one that closes to a tour is an option with the
// exact change of the tour's length. An option that leaves the tour shorter than at the chain's start is applied and
// the chain ends. Otherwise the option whose open path is shortest is applied if that path is shorter than the start
// tour (the positive gain criterion) and the chain goes on from its tip; else the chain's exchanges are undone. An
// edge the chain added may not be removed and one it removed not added again, which keeps a chain from undoing
// itself. depth 2 gives the 3-opt basis, about 2K + 4K^2 options at a tip.
//
// An exchange is checked and applied generically: its k removed edges cut the tour into k segments, the added edges
// link the segments' ends, and the exchange is a tour iff following those links visits every segment once. The tour
// array is rewritten from the longest segment on, the others copied in the order and direction the links give, so a
// 2-opt costs its shorter side. Interior cities of a copied segment keep their neighbours; those of a reversed one
// flip o; the 2k ends get new neighbours and their table entries recomputed.
#define KOPT_MAXK 5
#define KOPT_MAXOPT 1200
#define CHAIN_MAX 64

typedef struct { int cnt; int k[KOPT_MAXOPT]; int t[KOPT_MAXOPT][2 * KOPT_MAXK]; double dl[KOPT_MAXOPT]; } Opts;
typedef struct { int na, nr; int add[CHAIN_MAX * KOPT_MAXK][2], rem[CHAIN_MAX * KOPT_MAXK][2]; } Chain;

static inline int chain_has(const int (*edges)[2], int cnt, int u, int v) {
    for (int i = 0; i < cnt; i++) if ((edges[i][0] == u && edges[i][1] == v) || (edges[i][0] == v && edges[i][1] == u)) return 1;
    return 0;
}

static inline int tour_edge_pos(const Tour *T, int u, int v) {   // the position p of tour edge (u, v): tour[p], tour[p + 1]
    int p = T->pos[u], s = p + 1 == T->n ? 0 : p + 1;
    return T->tour[s] == v ? p : T->pos[v];
}

// The exchange t[0 .. 2k): removed (t[2i], t[2i+1]), added (t[2i+1], t[2i+2 mod 2k]). Returns 1 if the result is a
// tour, with e the removed edges' positions sorted, order the segments (segment i: positions e[i] + 1 .. e[i+1]) in
// the order the added edges link them from segment 0, and rev whether each is run backwards.
static int kopt_check(const Tour *T, const int *t, int k, int *e, int *order, int *rev) {
    const int n = T->n;
    for (int i = 0; i < k; i++) {
        int v = tour_edge_pos(T, t[2 * i], t[2 * i + 1]), j = i;
        while (j > 0 && e[j - 1] > v) { e[j] = e[j - 1]; j--; }
        e[j] = v;
    }
    for (int i = 1; i < k; i++) if (e[i] == e[i - 1]) return 0;
    int sc[2 * KOPT_MAXK], link[2 * KOPT_MAXK];   // the city at each segment end: 2i its start, 2i + 1 its end
    for (int i = 0; i < k; i++) {
        sc[2 * i] = T->tour[(e[i] + 1) % n];
        sc[2 * i + 1] = T->tour[e[(i + 1) % k]];
        link[2 * i] = link[2 * i + 1] = -1;
    }
    for (int i = 0; i < k; i++) {
        int u = t[2 * i + 1], v = t[(2 * i + 2) % (2 * k)], su = -1, sv = -1;
        if (u == v) return 0;
        for (int s = 0; s < 2 * k && su < 0; s++) if (link[s] < 0 && sc[s] == u) su = s;
        for (int s = 0; s < 2 * k && sv < 0; s++) if (link[s] < 0 && s != su && sc[s] == v) sv = s;
        if (su < 0 || sv < 0) return 0;
        link[su] = sv; link[sv] = su;
    }
    int cnt = 0, s = 0;
    for (;;) {
        order[cnt] = s >> 1; rev[cnt] = s & 1; cnt++;
        s = link[s ^ 1];
        if (s == 0) return cnt == k;
        if (cnt >= k) return 0;
    }
}

// Applies a checked exchange; buf is scratch of n ints. The 2k ends get their neighbours and entries recomputed.
static void kopt_apply(Tour *T, const int *t, int k, const int *e, const int *order, const int *rev, int *buf) {
    const int n = T->n;
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
    for (int i = 0; i < 2 * k; i++) tour_set_nb(T, t[i]);
    for (int i = 0; i < 2 * k; i++) tour_refresh(T, t[i]);
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
static void kopt_extend(const Tour *T, int *t, int k, double g, int depth, Opts *O, const Chain *C) {
    const int f = t[2 * k - 1];
    if (k >= 2 && !tour_adjacent(T, f, t[0]) && O->cnt < KOPT_MAXOPT) {   // close: (f, t1) is neither present nor just removed
        int e[KOPT_MAXK], order[KOPT_MAXK], rev[KOPT_MAXK];
        if (kopt_check(T, t, k, e, order, rev)) {
            O->k[O->cnt] = k;
            memcpy(O->t[O->cnt], t, sizeof(int) * 2 * k);
            O->dl[O->cnt] = tour_d(T, f, t[0]) - g;
            O->cnt++;
        }
    }
    if (k > depth) return;
    for (int kk = 0; kk < T->K; kk++) {
        int c = T->nn[f * T->K + kk];
        if (tour_adjacent(T, f, c) || kopt_added(t, k, f, c) || chain_has(C->rem, C->nr, f, c)) continue;
        double g1 = g - tour_d(T, f, c);
        for (int j = 0; j < 2; j++) {
            int c2 = T->nb[2 * c + j];
            if (kopt_removed(t, k, c, c2) || chain_has(C->add, C->na, c, c2)) continue;
            t[2 * k] = c; t[2 * k + 1] = c2;
            kopt_extend(T, t, k + 1, g1 + tour_d(T, c, c2), depth, O, C);
        }
    }
}

// A chain from the edge (t1, t2). Returns the change of the tour's length (negative, or zero when undone); the cities
// of the applied exchanges are appended to touched (up to 2 * KOPT_MAXK * CHAIN_MAX).
static double kopt_chain(Tour *T, int t1, int t2, int depth, Opts *O, Chain *C, int *buf, int *touched, int *ntouched) {
    static int moves[CHAIN_MAX][2 * KOPT_MAXK], mk[CHAIN_MAX];
    int nm = 0, f = t2;
    double dL = 0.0;
    C->na = C->nr = 0;
    *ntouched = 0;
    for (int s = 0; s < CHAIN_MAX; s++) {
        int t[2 * KOPT_MAXK] = {t1, f};
        O->cnt = 0;
        kopt_extend(T, t, 1, tour_d(T, t1, f), depth, O, C);
        int bc = -1, bo = -1;
        double best_close = 0.0, best_open = 0.0;
        for (int o = 0; o < O->cnt; o++) {
            double total = dL + O->dl[o], G = tour_d(T, O->t[o][2 * O->k[o] - 1], t1) - total;
            if (total < -1e-9 && total < best_close) { best_close = total; bc = o; }
            if (G > 1e-9 && G > best_open) { best_open = G; bo = o; }
        }
        int ch = bc >= 0 ? bc : bo;
        if (ch < 0) break;
        const int k = O->k[ch], *tc = O->t[ch];
        int e[KOPT_MAXK], order[KOPT_MAXK], rev[KOPT_MAXK];
        kopt_check(T, tc, k, e, order, rev);
        kopt_apply(T, tc, k, e, order, rev, buf);
        dL += O->dl[ch];
        memcpy(moves[nm], tc, sizeof(int) * 2 * k); mk[nm++] = k;
        for (int i = 0; i < k; i++) { C->rem[C->nr][0] = tc[2 * i]; C->rem[C->nr][1] = tc[2 * i + 1]; C->nr++; }
        for (int i = 0; i + 1 < k; i++) { C->add[C->na][0] = tc[2 * i + 1]; C->add[C->na][1] = tc[2 * i + 2]; C->na++; }
        f = tc[2 * k - 1];
        if (bc >= 0) {
            for (int m = 0; m < nm; m++) for (int i = 0; i < 2 * mk[m]; i++) touched[(*ntouched)++] = moves[m][i];
            T->L += dL;
            return dL;
        }
    }
    for (int m = nm - 1; m >= 0; m--) {   // undo: the inverse exchange removes what was added and adds what was removed
        int k = mk[m], inv[2 * KOPT_MAXK], e[KOPT_MAXK], order[KOPT_MAXK], rev[KOPT_MAXK];
        for (int i = 0; i < 2 * k; i++) inv[i] = moves[m][(i + 1) % (2 * k)];
        kopt_check(T, inv, k, e, order, rev);
        kopt_apply(T, inv, k, e, order, rev, buf);
    }
    return 0.0;
}

// The descent with don't-look bits: chains from both edges of every active city, a city made inactive when no chain
// from it shortens the tour and active again when a chain touches it. Ends at a local optimum of the chained
// neighbourhood. x, y: coordinates; nn: n x K candidates; tour_io: the city at each position, in and out; chains: the
// number of chains run, added to. Returns the length.
double pfa_tour_descent(int n, const double *x, const double *y, int round_mode, int K, const int *nn, int *tour_io,
                        int depth, long *chains) {
    Tour T = tour_new(n, K, round_mode, x, y, nn, tour_io);
    int *queue = malloc(sizeof(int) * n), *buf = malloc(sizeof(int) * n);
    int *touched = malloc(sizeof(int) * 2 * KOPT_MAXK * CHAIN_MAX), ntouched;
    unsigned char *active = malloc(n);
    Opts *O = malloc(sizeof(Opts));
    Chain *C = malloc(sizeof(Chain));
    int head = 0, count = n;
    for (int a = 0; a < n; a++) { queue[a] = a; active[a] = 1; }
    while (count > 0) {
        const int t1 = queue[head]; head = head + 1 == n ? 0 : head + 1; count--; active[t1] = 0;
        for (int i = 0; i < 2 && !active[t1]; i++) {
            (*chains)++;
            if (kopt_chain(&T, t1, T.nb[2 * t1 + i], depth, O, C, buf, touched, &ntouched) < 0.0) {
                touched[ntouched++] = t1;
                for (int q = 0; q < ntouched; q++) {
                    const int z = touched[q];
                    if (!active[z]) { active[z] = 1; queue[(head + count) % n] = z; count++; }
                }
            }
        }
    }
    double L = T.L;
    tour_free(&T); free(queue); free(buf); free(touched); free(active); free(O); free(C);
    return L;
}
