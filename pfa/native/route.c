// The route kernel (included by pfa.c after tour_chain.c): the local search over a set of vehicle routes with the
// moves native to routes, each evaluated in O(1) with the loads first, at zero temperature. Customers are 1 .. nc,
// the depot 0; a route is an array of customers, read as depot, customers, depot. The objective is length plus
// lambda per unit of a route's load over Q.
//
// The moves, for a customer u and a candidate v (one of u's K nearest, or the start of a route): relocate u after v;
// relocate the pair (u, next u) after v, as is or reversed; swap u and v; swap (u, next u) with v; swap (u, next u)
// with (v, next v); within a route, reverse the path after u up to v (2-opt); across routes, exchange the tails after
// u and v, as they are or reversed (2-opt*). A customer is tested against its candidates until no move improves; a
// route's change re-activates the tests of the customers whose candidates it holds (time stamps, as HGS).
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int *seq, *cum, len, load, stamp; double dist; } Route;   // cum[i]: the load of seq[0 .. i - 1]

typedef struct {
    int nc, m, K, Q, round_mode;
    double lambda;
    const double *x, *y;   // nc + 1, the depot at 0
    const int *dem, *nn;   // dem: nc + 1; nn: nc x K candidate customers (1-based) per customer, row u - 1
    const double *Dm;      // the distance matrix (nc + 1)^2, or NULL to compute from the coordinates
    Route *R;
    int *rof, *pof;        // each customer's route and index
    int *tested, clock, pass;
    long moves;
    int *buf;              // scratch, 2 * (nc + 1)
} LS;

static inline double ls_d(const LS *S, int i, int j) {
    if (S->Dm) return S->Dm[(long)i * (S->nc + 1) + j];
    double dx = S->x[i] - S->x[j], dy = S->y[i] - S->y[j], r = sqrt(dx * dx + dy * dy);
    if (S->round_mode == 1) return floor(r + 0.5);
    if (S->round_mode == 2) return ceil(r);
    return r;
}
static inline double ls_pen(const LS *S, int load) { return load > S->Q ? S->lambda * (load - S->Q) : 0.0; }
static inline int ls_city(const Route *R, int i) { return i < 0 || i >= R->len ? 0 : R->seq[i]; }

static void ls_update(LS *S, int r) {
    Route *R = &S->R[r];
    int load = 0;
    double dist = 0.0;
    R->cum[0] = 0;
    for (int i = 0; i < R->len; i++) {
        int c = R->seq[i];
        S->rof[c] = r; S->pof[c] = i;
        load += S->dem[c]; R->cum[i + 1] = load;
        dist += ls_d(S, i == 0 ? 0 : R->seq[i - 1], c);
    }
    dist += R->len ? ls_d(S, R->seq[R->len - 1], 0) : 0.0;
    R->load = load; R->dist = dist; R->stamp = ++S->clock;
}

// Removes the block seq[i .. i + cnt) from route r, inserts block (cnt cities) into route r2 before index j2 (indices
// of the state after the removal when r2 == r is handled by the caller).
static void ls_remove(Route *R, int i, int cnt) { memmove(R->seq + i, R->seq + i + cnt, sizeof(int) * (R->len - i - cnt)); R->len -= cnt; }
static void ls_insert(Route *R, int j, const int *block, int cnt) { memmove(R->seq + j + cnt, R->seq + j, sizeof(int) * (R->len - j)); memcpy(R->seq + j, block, sizeof(int) * cnt); R->len += cnt; }

// Moves the block seq[i .. i + cnt) of route ru to sit after index j of route rv (j = -1: the route's start), the
// block reversed if asked.
static void ls_relocate(LS *S, int ru, int i, int cnt, int rv, int j, int reversed) {
    int block[2];
    for (int q = 0; q < cnt; q++) block[q] = S->R[ru].seq[reversed ? i + cnt - 1 - q : i + q];
    ls_remove(&S->R[ru], i, cnt);
    if (ru == rv && j > i) j -= cnt;
    ls_insert(&S->R[rv], j + 1, block, cnt);
    ls_update(S, ru);
    if (rv != ru) ls_update(S, rv);
    S->moves++;
}

// Swaps the block seq[i .. i + a) of route ru with seq[j .. j + b) of route rv (disjoint blocks).
static void ls_swap_blocks(LS *S, int ru, int i, int a, int rv, int j, int b) {
    int A[2], B[2];
    for (int q = 0; q < a; q++) A[q] = S->R[ru].seq[i + q];
    for (int q = 0; q < b; q++) B[q] = S->R[rv].seq[j + q];
    if (ru != rv) {
        ls_remove(&S->R[ru], i, a); ls_insert(&S->R[ru], i, B, b);
        ls_remove(&S->R[rv], j, b); ls_insert(&S->R[rv], j, A, a);
        ls_update(S, ru); ls_update(S, rv);
    } else {   // the later block first, so the earlier one's index stands
        if (i < j) { ls_remove(&S->R[ru], j, b); ls_insert(&S->R[ru], j, A, a); ls_remove(&S->R[ru], i, a); ls_insert(&S->R[ru], i, B, b); }
        else { ls_remove(&S->R[ru], i, a); ls_insert(&S->R[ru], i, B, b); ls_remove(&S->R[ru], j, b); ls_insert(&S->R[ru], j, A, a); }
        ls_update(S, ru);
    }
    S->moves++;
}

static inline double ls_dpen2(const LS *S, int la, int lb, int d) { return ls_pen(S, la - d) - ls_pen(S, la) + ls_pen(S, lb + d) - ls_pen(S, lb); }

// u (route ru, index i) against the place after index j of route rv (j = -1 the start; v the city there, 0 at the
// start). Returns 1 when a move was applied.
static int ls_try(LS *S, int u, int rv, int j) {
    const int ru = S->rof[u], i = S->pof[u];
    Route *RU = &S->R[ru], *RV = &S->R[rv];
    const int inter = ru != rv;
    const int P = ls_city(RU, i - 1), N = ls_city(RU, i + 1), N2 = ls_city(RU, i + 2);
    const int v = ls_city(RV, j), Nv = ls_city(RV, j + 1), Pv = ls_city(RV, j - 1);
    if (v == u) return 0;   // a customer against itself (a candidate list may hold it when coordinates coincide)
    const int du = S->dem[u], dv = j >= 0 ? S->dem[v] : 0;
    const int x = i + 1 < RU->len ? RU->seq[i + 1] : 0, dx = x ? S->dem[x] : 0;
    double g;
    // relocate u after v
    if (!(ru == rv && (j == i || j == i - 1))) {
        g = ls_d(S, P, N) - ls_d(S, P, u) - ls_d(S, u, N) + ls_d(S, v, u) + ls_d(S, u, Nv) - ls_d(S, v, Nv);
        if (inter) g += ls_dpen2(S, RU->load, RV->load, du);
        if (g < -1e-9) { ls_relocate(S, ru, i, 1, rv, j, 0); return 1; }
    }
    // relocate (u, x) after v, as is or reversed
    if (x && !(ru == rv && (j == i || j == i + 1 || j == i - 1))) {
        double base = ls_d(S, P, N2) - ls_d(S, P, u) - ls_d(S, x, N2) - ls_d(S, v, Nv);
        if (inter) base += ls_dpen2(S, RU->load, RV->load, du + dx);
        double g1 = base + ls_d(S, v, u) + ls_d(S, x, Nv), g2 = base + ls_d(S, v, x) + ls_d(S, u, Nv);
        if (g1 < -1e-9 || g2 < -1e-9) { ls_relocate(S, ru, i, 2, rv, j, g2 < g1); return 1; }
    }
    if (j < 0) {   // v is the depot: the 2-opt* with the route's whole tail
        if (!inter) return 0;
        int tailU = RU->load - RU->cum[i + 1];
        g = ls_d(S, u, Nv) + ls_d(S, 0, N) - ls_d(S, u, N) - ls_d(S, 0, Nv)
            + ls_pen(S, RU->cum[i + 1] + RV->load) + ls_pen(S, tailU) - ls_pen(S, RU->load) - ls_pen(S, RV->load);
        if (g < -1e-9) {   // u's head takes v's route; v's route becomes u's tail
            int *b = S->buf, lenT = RU->len - i - 1;
            memcpy(b, RU->seq + i + 1, sizeof(int) * lenT);
            memcpy(RU->seq + i + 1, RV->seq, sizeof(int) * RV->len); RU->len = i + 1 + RV->len;
            memcpy(RV->seq, b, sizeof(int) * lenT); RV->len = lenT;
            ls_update(S, ru); ls_update(S, rv); S->moves++; return 1;
        }
        g = ls_d(S, u, 0) + ls_d(S, N, Nv) - ls_d(S, u, N) - ls_d(S, 0, Nv)   // the other 2-opt*: u's route ends at u, u's tail reversed goes before v's route
            + ls_pen(S, RU->cum[i + 1]) + ls_pen(S, tailU + RV->load) - ls_pen(S, RU->load) - ls_pen(S, RV->load);
        if (g < -1e-9) {
            int *b = S->buf, lenT = RU->len - i - 1;
            for (int q = 0; q < lenT; q++) b[q] = RU->seq[RU->len - 1 - q];
            memmove(RV->seq + lenT, RV->seq, sizeof(int) * RV->len); memcpy(RV->seq, b, sizeof(int) * lenT); RV->len += lenT;
            RU->len = i + 1;
            ls_update(S, ru); ls_update(S, rv); S->moves++; return 1;
        }
        return 0;
    }
    // swap u and v
    if (!(ru == rv && (j == i - 1 || j == i + 1 || j == i))) {
        g = ls_d(S, P, v) + ls_d(S, v, N) - ls_d(S, P, u) - ls_d(S, u, N) + ls_d(S, Pv, u) + ls_d(S, u, Nv) - ls_d(S, Pv, v) - ls_d(S, v, Nv);
        if (inter) g += ls_dpen2(S, RU->load, RV->load, du - dv);
        if (g < -1e-9) { ls_swap_blocks(S, ru, i, 1, rv, j, 1); return 1; }
    }
    // swap (u, x) with v
    if (x && !(ru == rv && (j >= i - 1 && j <= i + 2))) {
        g = ls_d(S, P, v) + ls_d(S, v, N2) - ls_d(S, P, u) - ls_d(S, x, N2) + ls_d(S, Pv, u) + ls_d(S, x, Nv) - ls_d(S, Pv, v) - ls_d(S, v, Nv);
        if (inter) g += ls_dpen2(S, RU->load, RV->load, du + dx - dv);
        if (g < -1e-9) { ls_swap_blocks(S, ru, i, 2, rv, j, 1); return 1; }
    }
    // swap (u, x) with (v, y)
    const int y = j + 1 < RV->len ? RV->seq[j + 1] : 0, dy = y ? S->dem[y] : 0, Ny = ls_city(RV, j + 2);
    if (x && y && !(ru == rv && (j >= i - 2 && j <= i + 2))) {
        g = ls_d(S, P, v) + ls_d(S, y, N2) - ls_d(S, P, u) - ls_d(S, x, N2) + ls_d(S, Pv, u) + ls_d(S, x, Ny) - ls_d(S, Pv, v) - ls_d(S, y, Ny);
        if (inter) g += ls_dpen2(S, RU->load, RV->load, du + dx - dv - dy);
        if (g < -1e-9) { ls_swap_blocks(S, ru, i, 2, rv, j, 2); return 1; }
    }
    if (!inter) {   // 2-opt: reverse the path strictly between u and v's successor edges
        int a = i < j ? i : j, b = i < j ? j : i;
        if (b == a + 1) return 0;
        int ca = RU->seq[a], cb = RU->seq[b], na = RU->seq[a + 1], nb = ls_city(RU, b + 1);
        g = ls_d(S, ca, cb) + ls_d(S, na, nb) - ls_d(S, ca, na) - ls_d(S, cb, nb);
        if (g < -1e-9) {
            for (int lo = a + 1, hi = b; lo < hi; lo++, hi--) { int t = RU->seq[lo]; RU->seq[lo] = RU->seq[hi]; RU->seq[hi] = t; }
            ls_update(S, ru); S->moves++; return 1;
        }
        return 0;
    }
    // 2-opt* across routes
    {
        int headU = RU->cum[i + 1], headV = RV->cum[j + 1], tailU = RU->load - headU, tailV = RV->load - headV;
        g = ls_d(S, u, Nv) + ls_d(S, v, N) - ls_d(S, u, N) - ls_d(S, v, Nv) + ls_pen(S, headU + tailV) + ls_pen(S, headV + tailU) - ls_pen(S, RU->load) - ls_pen(S, RV->load);
        if (g < -1e-9) {   // the tails exchanged
            int *b = S->buf, lenTU = RU->len - i - 1, lenTV = RV->len - j - 1;
            memcpy(b, RU->seq + i + 1, sizeof(int) * lenTU);
            memcpy(RU->seq + i + 1, RV->seq + j + 1, sizeof(int) * lenTV); RU->len = i + 1 + lenTV;
            memcpy(RV->seq + j + 1, b, sizeof(int) * lenTU); RV->len = j + 1 + lenTU;
            ls_update(S, ru); ls_update(S, rv); S->moves++; return 1;
        }
        g = ls_d(S, u, v) + ls_d(S, N, Nv) - ls_d(S, u, N) - ls_d(S, v, Nv) + ls_pen(S, headU + headV) + ls_pen(S, tailU + tailV) - ls_pen(S, RU->load) - ls_pen(S, RV->load);
        if (g < -1e-9) {   // u's head joined to v's head reversed; u's tail reversed joined to v's tail
            int *b = S->buf, *b2 = S->buf + S->nc + 1, lenTU = RU->len - i - 1, lenHV = j + 1;
            for (int q = 0; q < lenTU; q++) b[q] = RU->seq[RU->len - 1 - q];   // u's tail reversed
            for (int q = 0; q < lenHV; q++) b2[q] = RV->seq[j - q];             // v's head reversed
            memcpy(RU->seq + i + 1, b2, sizeof(int) * lenHV); RU->len = i + 1 + lenHV;
            memmove(RV->seq + lenTU, RV->seq + j + 1, sizeof(int) * (RV->len - j - 1)); memcpy(RV->seq, b, sizeof(int) * lenTU); RV->len = lenTU + (RV->len - j - 1);
            ls_update(S, ru); ls_update(S, rv); S->moves++; return 1;
        }
    }
    return 0;
}

// SWAP* (Vidal): for two routes, a customer of each exchanged, each inserted at its best place in the other route
// (its three best insertion points, or the place the other leaves), or one customer moved to its best place in the
// other route; the best such move of the pair applied when it improves. ins: per customer of r1 its three best insertion points in r2 (position after which, and cost).
typedef struct { int p[3]; double c[3]; } Ins3;
static void ls_ins3(const LS *S, int u, const Route *R, Ins3 *I) {
    for (int q = 0; q < 3; q++) { I->p[q] = -2; I->c[q] = INFINITY; }
    for (int p = -1; p < R->len; p++) {
        int a = ls_city(R, p), b = ls_city(R, p + 1);
        double c = ls_d(S, a, u) + ls_d(S, u, b) - ls_d(S, a, b);
        if (c < I->c[2]) {
            int q = 2;
            while (q > 0 && c < I->c[q - 1]) { I->p[q] = I->p[q - 1]; I->c[q] = I->c[q - 1]; q--; }
            I->p[q] = p; I->c[q] = c;
        }
    }
}
// the best insertion of u into R from its three points, avoiding the points touching index j (where v sits), or in v's place
static inline double ls_best_ins(const LS *S, int u, const Route *R, const Ins3 *I, int j, int *pos) {
    double best = INFINITY;
    for (int q = 0; q < 3; q++) if (I->p[q] != -2 && I->p[q] != j && I->p[q] != j - 1 && I->c[q] < best) { best = I->c[q]; *pos = I->p[q]; }
    int Pv = ls_city(R, j - 1), Nv = ls_city(R, j + 1);
    double inplace = ls_d(S, Pv, u) + ls_d(S, u, Nv) - ls_d(S, Pv, Nv);
    if (inplace < best) { best = inplace; *pos = j - 1; }   // after Pv, which is index j - 1 once v is gone
    return best;
}

static int ls_swapstar_pair(LS *S, int r1, int r2, Ins3 *I1, Ins3 *I2) {
    Route *A = &S->R[r1], *B = &S->R[r2];
    if (A->len == 0 || B->len == 0) return 0;
    for (int i = 0; i < A->len; i++) ls_ins3(S, A->seq[i], B, &I1[i]);
    for (int j = 0; j < B->len; j++) ls_ins3(S, B->seq[j], A, &I2[j]);
    double best = -1e-9;
    int bi = -1, bj = -1, bpu = 0, bpv = 0;
    for (int i = 0; i < A->len; i++) {
        int u = A->seq[i], Pu = ls_city(A, i - 1), Nu = ls_city(A, i + 1);
        double rem_u = ls_d(S, Pu, u) + ls_d(S, u, Nu) - ls_d(S, Pu, Nu);
        for (int j = 0; j < B->len; j++) {
            int v = B->seq[j], Pv = ls_city(B, j - 1), Nv = ls_city(B, j + 1);
            double rem_v = ls_d(S, Pv, v) + ls_d(S, v, Nv) - ls_d(S, Pv, Nv);
            double pen = ls_pen(S, A->load - S->dem[u] + S->dem[v]) + ls_pen(S, B->load - S->dem[v] + S->dem[u]) - ls_pen(S, A->load) - ls_pen(S, B->load);
            // the cheapest conceivable: the best insertion with the other still present, or in the other's place, which can be cheaper
            double inpl_u = ls_d(S, Pv, u) + ls_d(S, u, Nv) - ls_d(S, Pv, Nv), inpl_v = ls_d(S, Pu, v) + ls_d(S, v, Nu) - ls_d(S, Pu, Nu);
            double bound = -rem_u - rem_v + pen + (inpl_u < I1[i].c[0] ? inpl_u : I1[i].c[0]) + (inpl_v < I2[j].c[0] ? inpl_v : I2[j].c[0]);
            if (bound >= best) continue;
            int pu, pv;
            double g = -rem_u - rem_v + pen + ls_best_ins(S, u, B, &I1[i], j, &pu) + ls_best_ins(S, v, A, &I2[j], i, &pv);
            if (g < best) { best = g; bi = i; bj = j; bpu = pu; bpv = pv; }
        }
    }
    // the one-sided relocations at the best insertion point, which Vidal's SWAP* includes: the point need not touch a
    // candidate of the customer, so the move reaches what the relocate over the candidates does not
    int reloc = 0;   // 1: u from A into B after bpu; 2: v from B into A after bpv
    for (int i = 0; i < A->len; i++) {
        int u = A->seq[i], Pu = ls_city(A, i - 1), Nu = ls_city(A, i + 1);
        double g = ls_d(S, Pu, Nu) - ls_d(S, Pu, u) - ls_d(S, u, Nu) + I1[i].c[0]
                   + ls_pen(S, A->load - S->dem[u]) + ls_pen(S, B->load + S->dem[u]) - ls_pen(S, A->load) - ls_pen(S, B->load);
        if (g < best) { best = g; bi = i; bj = -1; bpu = I1[i].p[0]; reloc = 1; }
    }
    for (int j = 0; j < B->len; j++) {
        int v = B->seq[j], Pv = ls_city(B, j - 1), Nv = ls_city(B, j + 1);
        double g = ls_d(S, Pv, Nv) - ls_d(S, Pv, v) - ls_d(S, v, Nv) + I2[j].c[0]
                   + ls_pen(S, B->load - S->dem[v]) + ls_pen(S, A->load + S->dem[v]) - ls_pen(S, A->load) - ls_pen(S, B->load);
        if (g < best) { best = g; bj = j; bi = -1; bpv = I2[j].p[0]; reloc = 2; }
    }
    if (bi < 0 && bj < 0) return 0;
    if (reloc == 1) { int u = A->seq[bi]; ls_remove(A, bi, 1); ls_insert(B, bpu + 1, &u, 1); ls_update(S, r1); ls_update(S, r2); S->moves++; return 1; }
    if (reloc == 2) { int v = B->seq[bj]; ls_remove(B, bj, 1); ls_insert(A, bpv + 1, &v, 1); ls_update(S, r1); ls_update(S, r2); S->moves++; return 1; }
    int u = A->seq[bi], v = B->seq[bj];
    ls_remove(A, bi, 1); ls_remove(B, bj, 1);
    if (bpv >= bi) bpv--;   // insertion points on the arrays without u / v
    if (bpu >= bj) bpu--;
    ls_insert(A, bpv + 1, &v, 1); ls_insert(B, bpu + 1, &u, 1);
    ls_update(S, r1); ls_update(S, r2); S->moves++;
    return 1;
}

static int ls_swapstar(LS *S, int *pair_tested, Ins3 *I1, Ins3 *I2) {
    int any = 0;
    for (int r1 = 0; r1 < S->m; r1++) {
        if (S->R[r1].len == 0) continue;
        for (int r2 = r1 + 1; r2 < S->m; r2++) {
            if (S->R[r2].len == 0) continue;
            int *t = &pair_tested[r1 * S->m + r2];
            if (S->R[r1].stamp <= *t && S->R[r2].stamp <= *t) continue;
            // the routes must come near each other: a customer of one has a candidate in the other
            int near = 0;
            for (int i = 0; i < S->R[r1].len && !near; i++) {
                int u = S->R[r1].seq[i];
                for (int k = 0; k < S->K; k++) if (S->rof[S->nn[(u - 1) * S->K + k]] == r2) { near = 1; break; }
            }
            *t = S->clock;   // stamped before the test: an exchange applied bumps the routes' stamps past it, so the pair is tried again
            if (!near) continue;
            if (ls_swapstar_pair(S, r1, r2, I1, I2)) any = 1;
        }
    }
    return any;
}

// The descent: the moves over the candidates until none improves, then SWAP* over the route pairs, again while SWAP*
// improves. Returns the moves applied.
static long ls_run(LS *S) {
    const int nc = S->nc, K = S->K;
    long moves0 = S->moves;
    int *pair_tested = calloc((size_t)S->m * S->m, sizeof(int));
    Ins3 *I1 = malloc(sizeof(Ins3) * (nc + 1)), *I2 = malloc(sizeof(Ins3) * (nc + 1));
    S->pass = 0;
    for (;;) {
        int improved = 1;
        while (improved) {
            improved = 0;
            for (int u = 1; u <= nc; u++) {
                int moved = 0, last = S->tested[u];
                for (int k = 0; k < K && !moved; k++) {
                    int v = S->nn[(u - 1) * K + k], rv = S->rof[v];
                    if (S->R[S->rof[u]].stamp <= last && S->R[rv].stamp <= last) continue;
                    if (ls_try(S, u, rv, S->pof[v])) moved = 1;
                    else if (ls_try(S, v, S->rof[u], S->pof[u])) moved = 1;
                    else if (rv != S->rof[u] && ls_try(S, u, rv, -1)) moved = 1;
                }
                if (!moved) moved = ls_try(S, u, S->rof[u], -1);   // u, or u and x, to the start of its own route
                if (!moved && S->pass) {   // one empty route, as the depot candidate
                    for (int r = 0; r < S->m; r++) if (S->R[r].len == 0) { if (ls_try(S, u, r, -1)) moved = 1; break; }
                }
                if (moved) improved = 1;
                else S->tested[u] = S->clock;
            }
            S->pass = 1;
        }
        if (!ls_swapstar(S, pair_tested, I1, I2)) break;
    }
    free(pair_tested); free(I1); free(I2);
    return S->moves - moves0;
}

// From the cycle with depot copies (customers 0 .. nc - 1, copies nc .. nc + m - 1) to routes, the descent, and back.
// x, y and demand are per city of the cycle (the copies at the depot). nn: nc x K candidate customers (0-based).
double pfa_route_ls(int nc, int m, const double *x, const double *y, int round_mode, const int *demand, int Q, double lambda,
                    int K, const int *nn, int *tour_io, double *length, double *excess, long *moves, const int *bn, const double *dist) {
    const int n = nc + m;
    LS S = {nc, m, K, Q, round_mode, lambda};
    S.Dm = dist;
    double *xs = malloc(sizeof(double) * (nc + 1)), *ys = malloc(sizeof(double) * (nc + 1));
    int *dem = malloc(sizeof(int) * (nc + 1)), *cnn = malloc(sizeof(int) * nc * K);
    xs[0] = x[nc]; ys[0] = y[nc]; dem[0] = 0;
    for (int c = 0; c < nc; c++) { xs[c + 1] = x[c]; ys[c + 1] = y[c]; dem[c + 1] = demand[c]; }
    for (int i = 0; i < nc * K; i++) cnn[i] = nn[i] + 1;
    S.x = xs; S.y = ys; S.dem = dem; S.nn = cnn;
    S.R = malloc(sizeof(Route) * m);
    int *pool = malloc(sizeof(int) * 2 * m * (nc + 1));
    for (int r = 0; r < m; r++) { S.R[r].seq = pool + (long)r * 2 * (nc + 1); S.R[r].cum = S.R[r].seq + nc + 1; S.R[r].len = 0; }
    S.rof = malloc(sizeof(int) * (nc + 1)); S.pof = malloc(sizeof(int) * (nc + 1));
    S.tested = calloc(nc + 1, sizeof(int)); S.clock = 0; S.moves = 0;
    S.buf = malloc(sizeof(int) * 2 * (nc + 1));
    int p0 = 0;
    while (tour_io[p0] < nc) p0++;
    for (int q = 0, r = -1; q < n; q++) {
        int c = tour_io[(p0 + q) % n];
        if (c >= nc) r++;
        else S.R[r].seq[S.R[r].len++] = c + 1;
    }
    for (int r = 0; r < m; r++) ls_update(&S, r);
    S.clock = 1;
    for (int r = 0; r < m; r++) S.R[r].stamp = 1;
    if (bn) {   // the restart's repair: a customer whose two route neighbours are the reference's starts as tested
        for (int r = 0; r < m; r++)
            for (int i = 0; i < S.R[r].len; i++) {
                int c = S.R[r].seq[i], a = ls_city(&S.R[r], i - 1) - 1, b = ls_city(&S.R[r], i + 1) - 1;   // -1 at the depot
                int ra = bn[2 * (c - 1)], rb = bn[2 * (c - 1) + 1];
                if ((a == ra && b == rb) || (a == rb && b == ra)) S.tested[c] = 1;
            }
    }
    ls_run(&S);
    int k = 0;
    double L = 0.0, X = 0.0;
    for (int r = 0; r < m; r++) {
        tour_io[k++] = nc + r;
        for (int i = 0; i < S.R[r].len; i++) tour_io[k++] = S.R[r].seq[i] - 1;
        L += S.R[r].dist; X += S.R[r].load > Q ? S.R[r].load - Q : 0;
    }
    *length = L; *excess = X; *moves = S.moves;
    free(xs); free(ys); free(dem); free(cnn); free(S.R); free(pool); free(S.rof); free(S.pof); free(S.tested); free(S.buf);
    return L + lambda * X;
}

// Split (Prins): the customer order (nc cities) cut into routes at the least penalised cost, the fleet free but for
// m copies; a route may carry up to 2Q (penalised). Writes the cycle with the copies: a copy before each route, the
// unused copies adjacent at the end. Returns the number of routes.
int pfa_route_split(int nc, int m, const double *x, const double *y, int round_mode, const int *demand, int Q, double lambda,
                    const int *order, int *tour_out) {
    Tour T = {nc + m, 0, round_mode, x, y};   // for tour_d only
    const int depot = nc;
    double *best = malloc(sizeof(double) * (nc + 1));
    int *prev = malloc(sizeof(int) * (nc + 1)), *cnt = malloc(sizeof(int) * (nc + 1));
    best[0] = 0.0; cnt[0] = 0;
    for (int i = 1; i <= nc; i++) best[i] = INFINITY;
    for (int i = 0; i < nc; i++) {   // a route from order[i] .. order[j - 1]
        if (best[i] == INFINITY) continue;
        int load = 0;
        double len = 0.0;
        for (int j = i + 1; j <= nc; j++) {
            int c = order[j - 1];
            load += demand[c];
            if (load > 2 * Q) break;
            len += j == i + 1 ? tour_d(&T, depot, c) : tour_d(&T, order[j - 2], c);
            double cost = best[i] + len + tour_d(&T, c, depot) + lambda * (load > Q ? load - Q : 0);
            if (cost < best[j]) { best[j] = cost; prev[j] = i; cnt[j] = cnt[i] + 1; }
        }
    }
    int R = cnt[nc];
    if (R > m) {   // too many routes for the copies: a limited-fleet pass, exactly m routes at most
        double *b2 = malloc(sizeof(double) * (nc + 1) * (m + 1));
        int *p2 = malloc(sizeof(int) * (nc + 1) * (m + 1));
        for (int v = 0; v <= m; v++) for (int i = 0; i <= nc; i++) b2[v * (nc + 1) + i] = INFINITY;
        b2[0] = 0.0;
        for (int v = 0; v < m; v++)
            for (int i = 0; i < nc; i++) {
                double bi = b2[v * (nc + 1) + i];
                if (bi == INFINITY) continue;
                int load = 0;
                double len = 0.0;
                for (int j = i + 1; j <= nc; j++) {
                    int c = order[j - 1];
                    load += demand[c];
                    if (load > 2 * Q) break;
                    len += j == i + 1 ? tour_d(&T, depot, c) : tour_d(&T, order[j - 2], c);
                    double cost = bi + len + tour_d(&T, c, depot) + lambda * (load > Q ? load - Q : 0);
                    if (cost < b2[(v + 1) * (nc + 1) + j]) { b2[(v + 1) * (nc + 1) + j] = cost; p2[(v + 1) * (nc + 1) + j] = i; }
                }
            }
        int vb = 1;
        for (int v = 1; v <= m; v++) if (b2[v * (nc + 1) + nc] < b2[vb * (nc + 1) + nc]) vb = v;
        R = vb;
        for (int v = vb, j = nc; v >= 1; v--) { int i = p2[v * (nc + 1) + j]; prev[j] = i; j = i; }
        free(b2); free(p2);
    }
    int *starts = malloc(sizeof(int) * (R + 1)), r = R;
    for (int j = nc; j > 0; j = prev[j]) starts[--r] = prev[j];
    int k = 0;
    for (int q = 0; q < R; q++) {
        tour_out[k++] = depot + q;
        int from = starts[q], to = q + 1 < R ? starts[q + 1] : nc;
        for (int i = from; i < to; i++) tour_out[k++] = order[i];
    }
    for (int q = R; q < m; q++) tour_out[k++] = depot + q;
    free(best); free(prev); free(cnt); free(starts);
    return R;
}

// A walk over the customers along the candidate edges (K nearest customers per customer): from a random customer, an
// unvisited candidate that is a neighbour in bn (2 per customer, -1 for none: the best cycle's route edges) with
// probability share when there is one, else a random unvisited candidate, else the nearest unvisited customer.
void pfa_route_walk(int nc, const double *x, const double *y, int K, const int *nn, const int *bn, double share, uint64_t seed, int *order) {
    int *left = malloc(sizeof(int) * nc), *at = malloc(sizeof(int) * nc), *free_ = malloc(sizeof(int) * K), *kept = malloc(sizeof(int) * K);
    unsigned char *seen = calloc(nc, 1);
    for (int c = 0; c < nc; c++) { left[c] = c; at[c] = c; }
    int nleft = nc;
    uint64_t rng = seed | 1;
    int cur = (int)((tour_rng(&rng) >> 32) % (uint64_t)nc);
    for (int p = 0; p < nc; p++) {
        order[p] = cur; seen[cur] = 1;
        int l = at[cur]; nleft--; left[l] = left[nleft]; at[left[l]] = l;
        if (p == nc - 1) break;
        int nf = 0, nk = 0;
        for (int k = 0; k < K; k++) {
            int c = nn[cur * K + k];
            if (c >= nc || seen[c]) continue;
            free_[nf++] = c;
            if (bn && (bn[2 * cur] == c || bn[2 * cur + 1] == c)) kept[nk++] = c;
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
    free(free_); free(kept); free(left); free(at); free(seen);
}
