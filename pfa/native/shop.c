// The job shop: n jobs, m machines, job j the operations o = j*m + i in order, each on machine mach[o] for dur[o].
// The state is the order of the jobs on every machine (seq[k*n + p] the job at position p of machine k); its makespan
// is the longest path of the disjunctive graph, every operation's start the later of its job predecessor's end and
// its machine predecessor's end. The kernel is Zhang et al.'s tabu search on the N7 neighbourhood at zero temperature:
// the critical path is cut into blocks (maximal runs of one machine), every inner operation of a block is moved to
// the block's front or back and the block's ends are moved inside, each move generated only when Balas and
// Vazacopoulos's O(1) condition guarantees it leaves the graph acyclic, evaluated by the head/tail recurrence over
// the re-sequenced stretch (exact for the adjacent swap when the estimate is at least the makespan, an estimate
// otherwise), the best non-tabu move taken uphill included, the tabu on the arcs a move reverses, aspiration when the
// estimate beats the best makespan, heads and tails recomputed exactly after every applied move.
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    int n, m, N;                 // jobs, machines, operations
    const int *mach, *dur;       // per operation
    int *opof;                   // opof[k*n + j]: the operation of job j on machine k
    int *seq, *pos;              // seq[k*n + p]: job; pos[o]: position of o on its machine
    int *r, *q, *topo, *indeg;   // heads, tails, a topological order, work
    int *mark, *set;             // the incremental recompute's stamps and the affected set
    int stamp, incr;             // incr: recompute incrementally after a move (large instances)
    long affected;               // operations whose head or tail was recomputed, summed
    int cmax;
    uint64_t rng;
} Shop;

static int shop_incr_threshold = 50000;   // operations from which a move's recompute is incremental
static int shop_tabu_mode = 0;             // 0: the arc tabu (reversed arcs, tenure L..1.5L); 1: the hash tabu (visited solutions)
static long shop_kick = 1000;              // a random move after this many iterations without a new best (Xie et al.'s rule has 200; 1000 measured better; 0: never)
void pfa_shop_knob(int incr_threshold) { shop_incr_threshold = incr_threshold; }
void pfa_shop_kick(long kick) { shop_kick = kick; }
void pfa_shop_tabu_mode(int mode) { shop_tabu_mode = mode; }

static inline uint64_t shop_rng(uint64_t *s) { *s ^= *s >> 12; *s ^= *s << 25; *s ^= *s >> 27; return *s * 2685821657736338717ULL; }

static inline int shop_jp(const Shop *S, int o) { return o % S->m ? o - 1 : -1; }
static inline int shop_js(const Shop *S, int o) { return (o + 1) % S->m ? o + 1 : -1; }
static inline int shop_mp(const Shop *S, int o) { int k = S->mach[o], p = S->pos[o]; return p ? S->opof[k * S->n + S->seq[k * S->n + p - 1]] : -1; }
static inline int shop_ms(const Shop *S, int o) { int k = S->mach[o], p = S->pos[o]; return p + 1 < S->n ? S->opof[k * S->n + S->seq[k * S->n + p + 1]] : -1; }
static inline int shop_end(const Shop *S, int o) { return o < 0 ? 0 : S->r[o] + S->dur[o]; }          // r + p, 0 for none
static inline int shop_tail(const Shop *S, int o) { return o < 0 ? 0 : S->q[o] + S->dur[o]; }         // q + p, 0 for none

// Heads, tails and the makespan by one topological pass; returns 0 when the orders are cyclic.
static int shop_eval(Shop *S) {
    const int N = S->N, m = S->m;
    int *indeg = S->indeg, *topo = S->topo, head = 0, tail = 0;
    for (int o = 0; o < N; o++) { indeg[o] = (o % m ? 1 : 0) + (S->pos[o] ? 1 : 0); if (!indeg[o]) topo[tail++] = o; }
    while (head < tail) {
        int o = topo[head++], jp = shop_jp(S, o), mp = shop_mp(S, o);
        int t = shop_end(S, jp), t2 = shop_end(S, mp);
        S->r[o] = t > t2 ? t : t2;
        int js = shop_js(S, o), ms = shop_ms(S, o);
        if (js >= 0 && --indeg[js] == 0) topo[tail++] = js;
        if (ms >= 0 && --indeg[ms] == 0) topo[tail++] = ms;
    }
    if (tail < N) return 0;
    int cmax = 0;
    for (int i = N - 1; i >= 0; i--) {
        int o = topo[i], js = shop_js(S, o), ms = shop_ms(S, o);
        int t = shop_tail(S, js), t2 = shop_tail(S, ms);
        S->q[o] = t > t2 ? t : t2;
        int c = S->r[o] + S->dur[o] + S->q[o];
        if (c > cmax) cmax = c;
    }
    S->cmax = cmax;
    return 1;
}

// After a move of the operations X (nx of them): heads recomputed over the operations reachable from X, tails over
// those reaching X, each set collected by a search and ordered by Kahn's pass within the set (predecessors outside
// it hold their values); the makespan from the jobs' last operations. Returns 0 on a cycle.
static int shop_eval_incr(Shop *S, const int *X, int nx) {
    const int m = S->m;
    int *set = S->set, *mark = S->mark, *indeg = S->indeg;
    for (int pass = 0; pass < 2; pass++) {   // 0: heads forward, 1: tails backward
        int stamp = ++S->stamp, ns = 0;
        for (int i = 0; i < nx; i++) if (mark[X[i]] != stamp) { mark[X[i]] = stamp; set[ns++] = X[i]; }
        for (int i = 0; i < ns; i++) {   // the reachable set
            int o = set[i], a = pass ? shop_jp(S, o) : shop_js(S, o), b = pass ? shop_mp(S, o) : shop_ms(S, o);
            if (a >= 0 && mark[a] != stamp) { mark[a] = stamp; set[ns++] = a; }
            if (b >= 0 && mark[b] != stamp) { mark[b] = stamp; set[ns++] = b; }
        }
        S->affected += ns;
        for (int i = 0; i < ns; i++) {   // in-set degrees
            int o = set[i], a = pass ? shop_js(S, o) : shop_jp(S, o), b = pass ? shop_ms(S, o) : shop_mp(S, o);
            indeg[o] = (a >= 0 && mark[a] == stamp) + (b >= 0 && mark[b] == stamp);
        }
        int head = 0, tail = 0, *topo = S->topo;
        for (int i = 0; i < ns; i++) if (indeg[set[i]] == 0) topo[tail++] = set[i];
        while (head < tail) {
            int o = topo[head++];
            if (!pass) { int t = shop_end(S, shop_jp(S, o)), t2 = shop_end(S, shop_mp(S, o)); S->r[o] = t > t2 ? t : t2; }
            else { int t = shop_tail(S, shop_js(S, o)), t2 = shop_tail(S, shop_ms(S, o)); S->q[o] = t > t2 ? t : t2; }
            int a = pass ? shop_jp(S, o) : shop_js(S, o), b = pass ? shop_mp(S, o) : shop_ms(S, o);
            if (a >= 0 && mark[a] == stamp && --indeg[a] == 0) topo[tail++] = a;
            if (b >= 0 && mark[b] == stamp && --indeg[b] == 0) topo[tail++] = b;
        }
        if (tail < ns) return 0;
    }
    int cmax = 0;
    for (int j = 0; j < S->n; j++) { int o = j * m + m - 1, c = S->r[o] + S->dur[o]; if (c > cmax) cmax = c; }
    S->cmax = cmax;
    return 1;
}

static void shop_index(Shop *S) {
    for (int k = 0; k < S->m; k++) for (int p = 0; p < S->n; p++) S->pos[S->opof[k * S->n + S->seq[k * S->n + p]]] = p;
}

static Shop *shop_new(int n, int m, const int *mach, const int *dur, const int *seq) {
    Shop *S = calloc(1, sizeof(Shop));
    S->n = n; S->m = m; S->N = n * m; S->mach = mach; S->dur = dur;
    S->opof = malloc(sizeof(int) * n * m); S->seq = malloc(sizeof(int) * n * m); S->pos = malloc(sizeof(int) * n * m);
    S->r = malloc(sizeof(int) * n * m); S->q = malloc(sizeof(int) * n * m); S->topo = malloc(sizeof(int) * n * m); S->indeg = malloc(sizeof(int) * n * m);
    S->mark = calloc(n * m, sizeof(int)); S->set = malloc(sizeof(int) * n * m); S->stamp = 0; S->incr = n * m >= shop_incr_threshold; S->affected = 0;
    for (int o = 0; o < n * m; o++) S->opof[mach[o] * n + o / m] = o;
    if (seq) { memcpy(S->seq, seq, sizeof(int) * n * m); shop_index(S); }
    return S;
}

static void shop_free(Shop *S) { free(S->opof); free(S->seq); free(S->pos); free(S->r); free(S->q); free(S->topo); free(S->indeg); free(S->mark); free(S->set); free(S); }

// The makespan of the machine orders, or -1 when cyclic. r_out, q_out (n*m each) may be NULL.
int pfa_shop_makespan(int n, int m, const int *mach, const int *dur, const int *seq, int *r_out, int *q_out) {
    Shop *S = shop_new(n, m, mach, dur, seq);
    int ok = shop_eval(S), c = ok ? S->cmax : -1;
    if (ok && r_out) memcpy(r_out, S->r, sizeof(int) * n * m);
    if (ok && q_out) memcpy(q_out, S->q, sizeof(int) * n * m);
    shop_free(S);
    return c;
}

// Giffler and Thompson's active schedules with a random choice in the conflict set: the schedulable operation with
// the earliest completion fixes the machine and the time c*; among that machine's schedulable operations that could
// start before c*, one is drawn at random and scheduled. Writes the machine orders.
void pfa_shop_gt(int n, int m, const int *mach, const int *dur, uint64_t seed, int *seq_out) {
    int *next = calloc(n, sizeof(int)), *jfree = calloc(n, sizeof(int)), *mfree = calloc(m, sizeof(int)), *fill = calloc(m, sizeof(int));
    int *cand = malloc(sizeof(int) * n);
    uint64_t rng = seed | 1;
    for (int step = 0; step < n * m; step++) {
        int best = -1, bc = 0;
        for (int j = 0; j < n; j++) {
            if (next[j] == m) continue;
            int o = j * m + next[j], s = jfree[j] > mfree[mach[o]] ? jfree[j] : mfree[mach[o]], c = s + dur[o];
            if (best < 0 || c < bc) { best = j; bc = c; }
        }
        int k = mach[best * m + next[best]], nc = 0;
        for (int j = 0; j < n; j++) {
            if (next[j] == m) continue;
            int o = j * m + next[j];
            if (mach[o] != k) continue;
            int s = jfree[j] > mfree[k] ? jfree[j] : mfree[k];
            if (s < bc) cand[nc++] = j;
        }
        int j = cand[(int)((shop_rng(&rng) >> 33) % (uint64_t)nc)], o = j * m + next[j];
        int s = jfree[j] > mfree[k] ? jfree[j] : mfree[k];
        jfree[j] = mfree[k] = s + dur[o];
        seq_out[k * n + fill[k]++] = j;
        next[j]++;
    }
    free(next); free(jfree); free(mfree); free(fill); free(cand);
}

// The critical path from the sink back along the predecessors attaining each head (the machine predecessor preferred,
// for longer blocks); path_out in order from the source, returns its length.
static int shop_critical(const Shop *S, int *path_out) {
    int x = -1;
    for (int j = 0; j < S->n && x < 0; j++) { int o = j * S->m + S->m - 1; if (S->r[o] + S->dur[o] == S->cmax) x = o; }
    int len = 0;
    while (x >= 0) {
        path_out[len++] = x;
        int mp = shop_mp(S, x), jp = shop_jp(S, x);
        if (mp >= 0 && shop_end(S, mp) == S->r[x]) x = mp;
        else if (jp >= 0 && shop_end(S, jp) == S->r[x]) x = jp;
        else x = -1;
    }
    for (int i = 0; i < len / 2; i++) { int t = path_out[i]; path_out[i] = path_out[len - 1 - i]; path_out[len - 1 - i] = t; }
    return len;
}

// A move: the operation at block position `from` (0-based within the block) moved to block position `to` (the others
// shifting), on one machine's block of positions pb .. pb + k - 1.
typedef struct { int k, pb, len, from, to, est; } ShopMove;

// The estimate of the makespan after the move: heads forward and tails backward along the block's new order from
// the stored values of everything outside it. order[] is work of length len.
static int shop_estimate(const Shop *S, const int *blk, int len, int from, int to, int *order) {
    int w = 0;
    for (int i = 0; i < len; i++) if (i != from) { if (w == to) order[w++] = blk[from]; order[w++] = blk[i]; }
    if (w == to) order[w++] = blk[from];
    int rr[64], qq[64];   // blocks longer than 64 are evaluated on the heap
    int *R = len <= 64 ? rr : malloc(sizeof(int) * len), *Q = len <= 64 ? qq : malloc(sizeof(int) * len);
    int prev = shop_end(S, shop_mp(S, blk[0]));
    for (int i = 0; i < len; i++) {
        int o = order[i], t = shop_end(S, shop_jp(S, o));
        R[i] = prev > t ? prev : t;
        prev = R[i] + S->dur[o];
    }
    int next = shop_tail(S, shop_ms(S, blk[len - 1]));
    for (int i = len - 1; i >= 0; i--) {
        int o = order[i], t = shop_tail(S, shop_js(S, o));
        Q[i] = next > t ? next : t;
        next = Q[i] + S->dur[o];
    }
    int est = 0;
    for (int i = 0; i < len; i++) { int c = R[i] + S->dur[order[i]] + Q[i]; if (c > est) est = c; }
    if (len > 64) { free(R); free(Q); }
    return est;
}

// Balas and Vazacopoulos: u moved right after v (u before v on the machine) leaves the graph acyclic when
// p_v + q_v >= p_JS(u) + q_JS(u); v moved right before u when r_u + p_u >= r_JP(v) + p_JP(v).
static inline int shop_ok_after(const Shop *S, int u, int v) { return shop_tail(S, v) >= shop_tail(S, shop_js(S, u)); }
static inline int shop_ok_before(const Shop *S, int u, int v) { return shop_end(S, u) >= shop_end(S, shop_jp(S, v)); }

// The tabu: arcs (a before b on machine k) a move reversed, forbidden to be recreated until an iteration.
typedef struct { int k, a, b; long until, added; } ShopArc;
#define SHOP_TABU_CAP 8192
typedef struct { ShopArc arcs[SHOP_TABU_CAP]; int head, count; long max_tenure; } ShopTabu;

static int tabu_has(const ShopTabu *T, int k, int a, int b, long it) {
    for (int i = 0; i < T->count; i++) {
        const ShopArc *A = &T->arcs[(T->head - 1 - i + SHOP_TABU_CAP) % SHOP_TABU_CAP];
        if (it - A->added > T->max_tenure) break;   // everything older has expired
        if (A->until > it && A->k == k && A->a == a && A->b == b) return 1;
    }
    return 0;
}
static void tabu_add(ShopTabu *T, int k, int a, int b, long until, long it) {
    T->arcs[T->head] = (ShopArc){k, a, b, until, it};
    T->head = (T->head + 1) % SHOP_TABU_CAP;
    if (T->count < SHOP_TABU_CAP) T->count++;
}

// The hash tabu (Li, Hao and Wu): a solution's hash is the sum over machine adjacencies (k, a before b) of a random
// weight, three weight tables and three bit vectors; a visited solution has its three bits set and is tabu for the
// run. The hash of a move's result follows from the adjacencies it removes and adds, O(block).
#define SHOP_HASH_BITS 27
typedef struct { uint64_t *w[3]; uint8_t *bits[3]; int n, m; } ShopHash;
static inline uint64_t shop_hw(const ShopHash *H, int t, int k, int a, int b) {   // the weight of adjacency a->b on machine k
    uint64_t x = ((uint64_t)k * H->n + a) * H->n + b + (uint64_t)t * 0x9E3779B97F4A7C15ull;
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 27; x *= 0x94D049BB133111EBull; x ^= x >> 31;
    return x;
}
static int shop_hash_seen(const ShopHash *H, const uint64_t *h) {
    for (int t = 0; t < 3; t++) { uint64_t i = h[t] & ((1ull << SHOP_HASH_BITS) - 1); if (!(H->bits[t][i >> 3] >> (i & 7) & 1)) return 0; }
    return 1;
}
static void shop_hash_mark(ShopHash *H, const uint64_t *h) {
    for (int t = 0; t < 3; t++) { uint64_t i = h[t] & ((1ull << SHOP_HASH_BITS) - 1); H->bits[t][i >> 3] |= 1 << (i & 7); }
}
static void shop_hash_all(const Shop *S, const ShopHash *H, uint64_t *h) {
    for (int t = 0; t < 3; t++) { h[t] = 0; for (int k = 0; k < S->m; k++) for (int p = 0; p + 1 < S->n; p++) h[t] += shop_hw(H, t, k, S->seq[k * S->n + p], S->seq[k * S->n + p + 1]); }
}
// the hash after the move: the block's adjacencies (with its two outer neighbours) removed, the new order's added
static void shop_hash_move(const Shop *S, const ShopHash *H, const ShopMove *M, const uint64_t *h, uint64_t *out, int *order) {
    const int k = M->k, n = S->n, *sq = S->seq + k * n;
    int lo = M->pb - 1, hi = M->pb + M->len;   // outer neighbours, -1 / n when absent
    int w = 0;
    for (int i = 0; i < M->len; i++) if (i != M->from) { if (w == M->to) order[w++] = sq[M->pb + M->from]; order[w++] = sq[M->pb + i]; }
    if (w == M->to) order[w++] = sq[M->pb + M->from];
    for (int t = 0; t < 3; t++) {
        uint64_t x = h[t];
        for (int p = (lo < 0 ? 0 : lo); p < (hi >= n ? n - 1 : hi); p++) x -= shop_hw(H, t, k, sq[p], sq[p + 1]);
        int prev = lo < 0 ? -1 : sq[lo];
        for (int i = 0; i < M->len; i++) { if (prev >= 0) x += shop_hw(H, t, k, prev, order[i]); prev = order[i]; }
        if (hi < n) x += shop_hw(H, t, k, prev, sq[hi]);
        out[t] = x;
    }
}

// Is the move tabu: does it create an arc that is forbidden? Moving the job at `from` to `to` creates, for every
// position strictly between, the arc (that job, moved job) when to > from and (moved job, that job) when to < from.
static int shop_move_tabu(const Shop *S, const ShopTabu *T, const ShopMove *M, long it) {
    const int k = M->k, *s = S->seq + k * S->n + M->pb, x = s[M->from];
    if (shop_tabu_mode == 2) {   // only the arc between x and the operation it lands next to on the side it came from
        return M->to > M->from ? tabu_has(T, k, s[M->to], x, it) : tabu_has(T, k, x, s[M->to], it);
    }
    if (M->to > M->from) { for (int i = M->from + 1; i <= M->to; i++) if (tabu_has(T, k, s[i], x, it)) return 1; }
    else { for (int i = M->to; i < M->from; i++) if (tabu_has(T, k, x, s[i], it)) return 1; }
    return 0;
}

// Apply the move: the reversed arcs join the tabu (forbidden to be recreated), the machine order shifts.
static void shop_apply(Shop *S, ShopTabu *T, const ShopMove *M, long until, long it) {
    const int k = M->k, n = S->n;
    int *s = S->seq + k * n + M->pb, x = s[M->from];
    if (M->to > M->from) {
        for (int i = M->from + 1; i <= M->to; i++) tabu_add(T, k, x, s[i], until, it);   // x was before s[i]: that arc is forbidden to return
        memmove(s + M->from, s + M->from + 1, sizeof(int) * (M->to - M->from));
    } else {
        for (int i = M->to; i < M->from; i++) tabu_add(T, k, s[i], x, until, it);
        memmove(s + M->to + 1, s + M->to, sizeof(int) * (M->from - M->to));
    }
    s[M->to] = x;
    int lo = M->from < M->to ? M->from : M->to, hi = M->from < M->to ? M->to : M->from;
    for (int p = M->pb + lo; p <= M->pb + hi; p++) S->pos[S->opof[k * n + S->seq[k * n + p]]] = p;
}

// The N7 moves of the critical path's blocks, evaluated; returns the count. moves has room for 4 * path length.
static int shop_moves(const Shop *S, const int *path, int plen, ShopMove *moves, int *work) {
    int nm = 0, i = 0;
    while (i < plen) {
        int j = i, k = S->mach[path[i]];
        while (j + 1 < plen && S->mach[path[j + 1]] == k && S->pos[path[j + 1]] == S->pos[path[j]] + 1) j++;
        int len = j - i + 1;
        if (len >= 2) {
            const int *blk = path + i, pb = S->pos[blk[0]];
            // Nowicki and Smutnicki: a change at the front of the path's first block or at the back of its last block
            // cannot shorten the path, so those are left out (both allowed when the path is one block)
            const int front = i > 0 || j == plen - 1, back = j < plen - 1 || i == 0;
            for (int t = 1; t < len - 1; t++) {   // inner operations to the front and to the back
                if (front && shop_ok_before(S, blk[0], blk[t]))
                    moves[nm++] = (ShopMove){k, pb, len, t, 0, shop_estimate(S, blk, len, t, 0, work)};
                if (back && shop_ok_after(S, blk[t], blk[len - 1]))
                    moves[nm++] = (ShopMove){k, pb, len, t, len - 1, shop_estimate(S, blk, len, t, len - 1, work)};
            }
            for (int t = 1; t < len; t++) if (front && shop_ok_after(S, blk[0], blk[t]))   // the first operation after an inner one or the last
                moves[nm++] = (ShopMove){k, pb, len, 0, t, shop_estimate(S, blk, len, 0, t, work)};
            for (int t = 0; t < len - 1; t++)   // the last operation before an inner one or the first (for a block of two, the one swap once)
                if (back && !(len == 2 && front) && shop_ok_before(S, blk[t], blk[len - 1]))
                    moves[nm++] = (ShopMove){k, pb, len, len - 1, t, shop_estimate(S, blk, len, len - 1, t, work)};
        }
        i = j + 1;
    }
    return nm;
}

// Measurement B: `steps` tabu iterations from seq; out[0] steps, out[1] steps whose best-estimated move is not among the
// exactly best, out[2] applied moves estimated to improve on the current makespan that did not, out[3] moves evaluated.
void pfa_shop_estimate_check(int n, int m, const int *mach, const int *dur, const int *seq, long steps, uint64_t seed, long *out) {
    Shop *S = shop_new(n, m, mach, dur, seq), *C = shop_new(n, m, mach, dur, seq);
    shop_eval(S);
    ShopTabu *T = calloc(1, sizeof(ShopTabu)); T->max_tenure = 1;
    int *path = malloc(sizeof(int) * S->N), *work = malloc(sizeof(int) * S->N);
    ShopMove *moves = malloc(sizeof(ShopMove) * 4 * S->N);
    long o[4] = {0};
    S->rng = seed | 1;
    for (long it = 0; it < steps; it++) {
        int plen = shop_critical(S, path), nm = shop_moves(S, path, plen, moves, work);
        if (!nm) break;
        int be = 0, exact_of_be, exact_min = 0x7fffffff;
        for (int i = 0; i < nm; i++) if (moves[i].est < moves[be].est) be = i;
        for (int i = 0; i < nm; i++) {
            memcpy(C->seq, S->seq, sizeof(int) * S->N); memcpy(C->pos, S->pos, sizeof(int) * S->N);
            ShopTabu dummy = {0}; shop_apply(C, &dummy, &moves[i], 0, 0); shop_eval(C);
            if (i == be) exact_of_be = C->cmax;
            if (C->cmax < exact_min) exact_min = C->cmax;
        }
        o[3] += nm;
        if (exact_of_be != exact_min) o[1]++;
        if (moves[be].est < S->cmax && exact_of_be >= S->cmax) o[2]++;
        ShopTabu dummy = {0}; shop_apply(S, &dummy, &moves[be], 0, 0); shop_eval(S);
        o[0]++;
    }
    memcpy(out, o, sizeof(o));
    shop_free(S); shop_free(C); free(T); free(path); free(work); free(moves);
}

// The tabu search from the given orders: at most `iters` iterations or `seconds` of CPU time (0: no limit), stopping
// after `cutoff` iterations without a new best. Tenure drawn per iteration from [L, 1.5 L], L = 10 + n / m (Zhang et al.). seq_io ends as the last
// state; best_seq the best found, its makespan returned. stats (8 longs): iterations, moves evaluated, iterations
// with every move tabu, improvements of the best, applied moves whose estimate missed the exact makespan, moves that
// were adjacent swaps, the sum of path lengths, the operations recomputed incrementally.
int pfa_shop_tabu(int n, int m, const int *mach, const int *dur, int *seq_io, long iters, long cutoff, double seconds, uint64_t seed,
                  int *best_seq, long *stats) {
    clock_t t_end = seconds > 0 ? clock() + (clock_t)(seconds * CLOCKS_PER_SEC) : 0;
    Shop *S = shop_new(n, m, mach, dur, seq_io);
    if (!shop_eval(S)) { shop_free(S); return -1; }
    ShopTabu *T = calloc(1, sizeof(ShopTabu));
    const int L = 10 + n / m;
    T->max_tenure = L + L / 2 + 1;
    ShopHash H = {{0}, {0}, n, m}; uint64_t hcur[3], hnew[3];
    if (shop_tabu_mode == 1) { for (int t = 0; t < 3; t++) H.bits[t] = calloc(1u << (SHOP_HASH_BITS - 3), 1); }
    int *path = malloc(sizeof(int) * S->N), *work = malloc(sizeof(int) * S->N), *work2 = malloc(sizeof(int) * S->N);
    if (shop_tabu_mode == 1) { shop_hash_all(S, &H, hcur); shop_hash_mark(&H, hcur); }
    ShopMove *moves = malloc(sizeof(ShopMove) * 4 * S->N);
    int best = S->cmax;
    memcpy(best_seq, S->seq, sizeof(int) * S->N);
    long since = 0, st[8] = {0};
    S->rng = seed | 1;
    for (long it = 0; it < iters && since < cutoff; it++) {
        if (t_end && (it & 1023) == 0 && clock() >= t_end) break;   // the wall-clock limit, checked every 1024 iterations
        int plen = shop_critical(S, path);
        int nm = shop_moves(S, path, plen, moves, work);
        st[1] += nm; st[6] += plen;
        if (nm == 0) break;   // one block of length one everywhere: no move (cannot happen with n, m > 1)
        int pick = -1, bestest = 0, ties = 0, admissible = 0;
        for (int i = 0; i < nm; i++) {
            int tabu;
            if (shop_tabu_mode == 1) { shop_hash_move(S, &H, &moves[i], hcur, hnew, work2); tabu = shop_hash_seen(&H, hnew); }
            else tabu = shop_move_tabu(S, T, &moves[i], it);
            if (tabu && !(shop_tabu_mode == 0 && moves[i].est < best)) continue;   // aspiration (arc tabu): a tabu move that beats the best is allowed
            admissible++;
            if (pick < 0 || moves[i].est < bestest) { pick = i; bestest = moves[i].est; ties = 1; }
            else if (moves[i].est == bestest && (int)((shop_rng(&S->rng) >> 33) % (uint64_t)++ties) == 0) pick = i;
        }
        if (!admissible || (shop_kick && since > 0 && since % shop_kick == 0)) {   // every move tabu, or the kick: a random move
            st[2] += !admissible; pick = (int)((shop_rng(&S->rng) >> 33) % (uint64_t)nm); bestest = moves[pick].est;
        }
        const ShopMove *M = &moves[pick];
        if (M->from + 1 == M->to || M->to + 1 == M->from) st[5]++;
        long tenure = L + (long)((shop_rng(&S->rng) >> 33) % (uint64_t)(L / 2 + 1));
        int nx = 0;
        if (S->incr) { for (int p = M->pb; p < M->pb + M->len; p++) work[nx++] = S->opof[M->k * n + S->seq[M->k * n + p]]; }
        if (shop_tabu_mode == 1) { shop_hash_move(S, &H, M, hcur, hnew, work2); memcpy(hcur, hnew, sizeof(hcur)); shop_hash_mark(&H, hcur); }
        shop_apply(S, T, M, it + tenure, it);
                if (!(S->incr ? shop_eval_incr(S, work, nx) : shop_eval(S))) { shop_free(S); free(T); free(path); free(work); free(work2); free(moves); return -2; }   // the condition failed: a bug
        if (S->cmax != bestest) st[4]++;
        if (S->cmax < best) { best = S->cmax; memcpy(best_seq, S->seq, sizeof(int) * S->N); since = 0; st[3]++; }
        else since++;
        st[0]++;
    }
    st[7] = S->affected;
    memcpy(seq_io, S->seq, sizeof(int) * S->N);
    if (stats) memcpy(stats, st, sizeof(st));
    shop_free(S); free(T); free(path); free(work); free(work2); free(moves);
    for (int t = 0; t < 3; t++) free(H.bits[t]);
    return best;
}
