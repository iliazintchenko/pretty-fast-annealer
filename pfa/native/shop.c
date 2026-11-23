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
    int *mpred, *msucc;          // the machine predecessor and successor of every operation (-1 at the ends), kept with seq
    int *r, *q, *topo, *indeg;   // heads, tails, a topological order, work
    int *mark, *set;             // the incremental recompute's stamps and the affected set
    int stamp, incr;             // incr: recompute incrementally after a move (large instances)
    long affected;               // operations whose head or tail was recomputed, summed
    int cmax;
    uint64_t rng;
} Shop;

static int shop_incr_threshold = 0x7fffffff;   // operations from which a move's recompute is incremental: off by default, since a critical move's
                                                // reach measured 94-97% of the operations at 10,000-100,000 of them and the full pass is twice as fast
static int shop_tabu_mode = 0;             // 0: the arc tabu (reversed arcs, tenure L..1.5L); 1: the hash tabu (visited solutions)
static int shop_all_ends = 0;              // 1: no front/back exclusion of N7 moves (the path is a truncated window segment)
#define SHOP_STARTS 8                       // non-delay draws the first schedule is the best of
static long shop_kick = 1000;              // a random move after this many iterations without a new best (Xie et al.'s rule has 200; 1000 measured better; 0: never)
void pfa_shop_knob(int incr_threshold) { shop_incr_threshold = incr_threshold; }
void pfa_shop_kick(long kick) { shop_kick = kick; }
void pfa_shop_tabu_mode(int mode) { shop_tabu_mode = mode; }

static inline uint64_t shop_seed(uint64_t seed) { return (seed << 1) | 1; }   // an odd state per seed, distinct seeds distinct (or-ing in the low bit made seeds 2k and 2k+1 the same)
static inline uint64_t shop_rng(uint64_t *s) { *s ^= *s >> 12; *s ^= *s << 25; *s ^= *s >> 27; return *s * 2685821657736338717ULL; }

static inline int shop_jp(const Shop *S, int o) { return o % S->m ? o - 1 : -1; }
static inline int shop_js(const Shop *S, int o) { return (o + 1) % S->m ? o + 1 : -1; }
static inline int shop_mp(const Shop *S, int o) { return S->mpred[o]; }
static inline int shop_ms(const Shop *S, int o) { return S->msucc[o]; }
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

// Positions and machine neighbours of the operations at positions lo..hi of machine k refreshed from seq.
static void shop_reindex(Shop *S, int k, int lo, int hi) {
    const int n = S->n, *sq = S->seq + k * n, *op = S->opof + k * n;
    for (int p = lo; p <= hi; p++) {
        int o = op[sq[p]];
        S->pos[o] = p;
        S->mpred[o] = p ? op[sq[p - 1]] : -1;
        S->msucc[o] = p + 1 < n ? op[sq[p + 1]] : -1;
    }
}
static void shop_index(Shop *S) { for (int k = 0; k < S->m; k++) shop_reindex(S, k, 0, S->n - 1); }

static Shop *shop_new(int n, int m, const int *mach, const int *dur, const int *seq) {
    Shop *S = calloc(1, sizeof(Shop));
    S->n = n; S->m = m; S->N = n * m; S->mach = mach; S->dur = dur;
    S->opof = malloc(sizeof(int) * n * m); S->seq = malloc(sizeof(int) * n * m); S->pos = malloc(sizeof(int) * n * m);
    S->mpred = malloc(sizeof(int) * n * m); S->msucc = malloc(sizeof(int) * n * m);
    S->r = malloc(sizeof(int) * n * m); S->q = malloc(sizeof(int) * n * m); S->topo = malloc(sizeof(int) * n * m); S->indeg = malloc(sizeof(int) * n * m);
    S->mark = calloc(n * m, sizeof(int)); S->set = malloc(sizeof(int) * n * m); S->stamp = 0; S->incr = n * m >= shop_incr_threshold; S->affected = 0;
    for (int o = 0; o < n * m; o++) S->opof[mach[o] * n + o / m] = o;
    if (seq) { memcpy(S->seq, seq, sizeof(int) * n * m); shop_index(S); }
    return S;
}

static void shop_free(Shop *S) { free(S->opof); free(S->seq); free(S->pos); free(S->mpred); free(S->msucc); free(S->r); free(S->q); free(S->topo); free(S->indeg); free(S->mark); free(S->set); free(S); }

// The makespan of the machine orders, or -1 when cyclic. r_out, q_out (n*m each) may be NULL.
int pfa_shop_makespan(int n, int m, const int *mach, const int *dur, const int *seq, int *r_out, int *q_out) {
    Shop *S = shop_new(n, m, mach, dur, seq);
    int ok = shop_eval(S), c = ok ? S->cmax : -1;
    if (ok && r_out) memcpy(r_out, S->r, sizeof(int) * n * m);
    if (ok && q_out) memcpy(q_out, S->q, sizeof(int) * n * m);
    shop_free(S);
    return c;
}

// Giffler and Thompson's active schedules: the schedulable operation with the earliest completion fixes the machine
// and the time c*; among that machine's schedulable operations that could start before c* (the conflict set) one is
// scheduled: `rule` 0 draws it at random (the pool's diversity), 1 takes the one that can start earliest, ties at
// random (the non-delay schedule; a third of the gap less than the random choice on the million-operation instance,
// 0.13 s). A heap over the jobs keyed by their next operation's earliest completion, re-keyed (lazily, by version)
// only for the jobs waiting on the machine just used. Writes the machine orders.
typedef struct { long key; int job, ver; } GtItem;
static void gt_push(GtItem *h, int *hn, GtItem it) {
    int i = (*hn)++; h[i] = it;
    while (i > 0) { int par = (i - 1) / 2; if (h[par].key <= h[i].key) break; GtItem t = h[par]; h[par] = h[i]; h[i] = t; i = par; }
}
static GtItem gt_pop(GtItem *h, int *hn) {
    GtItem top = h[0]; h[0] = h[--(*hn)];
    int i = 0;
    for (;;) { int l = 2 * i + 1, r = l + 1, sm = i; if (l < *hn && h[l].key < h[sm].key) sm = l; if (r < *hn && h[r].key < h[sm].key) sm = r; if (sm == i) break; GtItem t = h[sm]; h[sm] = h[i]; h[i] = t; i = sm; }
    return top;
}
void pfa_shop_gt(int n, int m, const int *mach, const int *dur, uint64_t seed, int rule, int *seq_out) {
    int *next = calloc(n, sizeof(int)), *jfree = calloc(n, sizeof(int)), *mfree = calloc(m, sizeof(int)), *fill = calloc(m, sizeof(int));
    int *ver = calloc(n, sizeof(int)), *wait = malloc(sizeof(int) * n), *wnext = malloc(sizeof(int) * n), *whead = malloc(sizeof(int) * m), *wprev = malloc(sizeof(int) * n);
    GtItem *heap = malloc(sizeof(GtItem) * (size_t)(n * m + n)); int hn = 0;
    int *cand = malloc(sizeof(int) * n);
    uint64_t rng = shop_seed(seed);
    for (int k = 0; k < m; k++) whead[k] = -1;
#define WAIT_ADD(j, k) { wnext[j] = whead[k]; wprev[j] = -1; if (whead[k] >= 0) wprev[whead[k]] = j; whead[k] = j; wait[j] = k; }
#define WAIT_DEL(j) { int k_ = wait[j]; if (wprev[j] >= 0) wnext[wprev[j]] = wnext[j]; else whead[k_] = wnext[j]; if (wnext[j] >= 0) wprev[wnext[j]] = wprev[j]; }
#define KEY(j) ((long)(jfree[j] > mfree[mach[(j) * m + next[j]]] ? jfree[j] : mfree[mach[(j) * m + next[j]]]) + dur[(j) * m + next[j]])
    for (int j = 0; j < n; j++) { WAIT_ADD(j, mach[j * m]); gt_push(heap, &hn, (GtItem){KEY(j), j, 0}); }
    for (int step = 0; step < n * m; step++) {
        GtItem it;
        do { it = gt_pop(heap, &hn); } while (it.ver != ver[it.job]);   // stale keys are skipped
        int best = it.job, k = mach[best * m + next[best]]; long bc = it.key;
        int nc = 0;
        for (int j = whead[k]; j >= 0; j = wnext[j]) { int s0 = jfree[j] > mfree[k] ? jfree[j] : mfree[k]; if (s0 < bc) cand[nc++] = j; }
        if (rule == 1) {   // the earliest starters only, then the draw among them
            int smin = 0x7fffffff, w = 0;
            for (int c = 0; c < nc; c++) { int a = cand[c], sa = jfree[a] > mfree[k] ? jfree[a] : mfree[k]; if (sa < smin) smin = sa; }
            for (int c = 0; c < nc; c++) { int a = cand[c], sa = jfree[a] > mfree[k] ? jfree[a] : mfree[k]; if (sa == smin) cand[w++] = a; }
            nc = w;
        }
        int j = cand[(int)((shop_rng(&rng) >> 33) % (uint64_t)nc)];
        int o = j * m + next[j];
        int s0 = jfree[j] > mfree[k] ? jfree[j] : mfree[k];
        jfree[j] = mfree[k] = s0 + dur[o];
        seq_out[k * n + fill[k]++] = j;
        WAIT_DEL(j);
        next[j]++;
        if (next[j] < m) { WAIT_ADD(j, mach[j * m + next[j]]); ver[j]++; gt_push(heap, &hn, (GtItem){KEY(j), j, ver[j]}); }
        else ver[j] = -1;
        for (int j2 = whead[k]; j2 >= 0; j2 = wnext[j2]) { ver[j2]++; gt_push(heap, &hn, (GtItem){KEY(j2), j2, ver[j2]}); }   // the machine's time moved
    }
#undef WAIT_ADD
#undef WAIT_DEL
#undef KEY
    free(next); free(jfree); free(mfree); free(fill); free(ver); free(wait); free(wnext); free(whead); free(wprev); free(heap); free(cand);
}

// The first schedule: the best of `draws` non-delay starts (their makespans spread about two percent at a million
// operations, 0.15 s each there; the shipped count is SHOP_STARTS). Returns its makespan.
int pfa_shop_start(int n, int m, const int *mach, const int *dur, uint64_t seed, int draws, int *seq_out) {
    uint64_t rng = shop_seed(seed);
    int *t = malloc(sizeof(int) * n * m), best = 0x7fffffff;
    for (int d = 0; d < draws; d++) {
        pfa_shop_gt(n, m, mach, dur, shop_rng(&rng), 1, t);
        int c = pfa_shop_makespan(n, m, mach, dur, t, NULL, NULL);
        if (c < best) { best = c; memcpy(seq_out, t, sizeof(int) * n * m); }
    }
    free(t);
    return best;
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
    lo = M->pb + lo - 1; hi = M->pb + hi + 1;   // the neighbours just outside the stretch get a new machine neighbour
    shop_reindex(S, k, lo < 0 ? 0 : lo, hi >= n ? n - 1 : hi);
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
            const int front = shop_all_ends || i > 0 || j == plen - 1, back = shop_all_ends || j < plen - 1 || i == 0;
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

// Heads and tails recomputed over a given set of operations W (marked with `stamp`), everything outside it holding
// its values: Kahn's pass within W counting in-W predecessors only. Returns max over W of r + p + q, or -1 on a cycle.
static int shop_eval_set(Shop *S, const int *set, int ns, int stamp) {
    int *mark = S->mark, *indeg = S->indeg, *topo = S->topo;
    for (int pass = 0; pass < 2; pass++) {
        for (int i = 0; i < ns; i++) {
            int o = set[i], a = pass ? shop_js(S, o) : shop_jp(S, o), b = pass ? shop_ms(S, o) : shop_mp(S, o);
            indeg[o] = (a >= 0 && mark[a] == stamp) + (b >= 0 && mark[b] == stamp);
        }
        int head = 0, tail = 0;
        for (int i = 0; i < ns; i++) if (indeg[set[i]] == 0) topo[tail++] = set[i];
        while (head < tail) {
            int o = topo[head++];
            if (!pass) { int t = shop_end(S, shop_jp(S, o)), t2 = shop_end(S, shop_mp(S, o)); S->r[o] = t > t2 ? t : t2; }
            else { int t = shop_tail(S, shop_js(S, o)), t2 = shop_tail(S, shop_ms(S, o)); S->q[o] = t > t2 ? t : t2; }
            int a = pass ? shop_jp(S, o) : shop_js(S, o), b = pass ? shop_mp(S, o) : shop_ms(S, o);
            if (a >= 0 && mark[a] == stamp && --indeg[a] == 0) topo[tail++] = a;
            if (b >= 0 && mark[b] == stamp && --indeg[b] == 0) topo[tail++] = b;
        }
        if (tail < ns) return -1;
    }
    int cm = 0;
    for (int i = 0; i < ns; i++) { int o = set[i], c = S->r[o] + S->dur[o] + S->q[o]; if (c > cm) cm = c; }
    return cm;
}

// The critical segment inside W through the operation x (the one with the largest r + p + q in W): back along
// head-attaining predecessors and forward along tail-attaining successors while they stay in W. path_out in order.
static int shop_segment_in(const Shop *S, int x, int stamp, int *path_out) {
    int len = 0, y = x;
    while (y >= 0) {   // backwards
        path_out[len++] = y;
        int mp = shop_mp(S, y), jp = shop_jp(S, y), nx = -1;
        if (mp >= 0 && S->mark[mp] == stamp && shop_end(S, mp) == S->r[y]) nx = mp;
        else if (jp >= 0 && S->mark[jp] == stamp && shop_end(S, jp) == S->r[y]) nx = jp;
        y = nx;
    }
    for (int i = 0; i < len / 2; i++) { int t = path_out[i]; path_out[i] = path_out[len - 1 - i]; path_out[len - 1 - i] = t; }
    y = x;
    for (;;) {   // forwards
        int ms = shop_ms(S, y), js = shop_js(S, y), nx = -1;
        if (ms >= 0 && S->mark[ms] == stamp && shop_tail(S, ms) == S->q[y]) nx = ms;
        else if (js >= 0 && S->mark[js] == stamp && shop_tail(S, js) == S->q[y]) nx = js;
        if (nx < 0) break;
        path_out[len++] = nx; y = nx;
    }
    return len;
}

// The window search's machinery: a worklist keyed by the stored head or tail (a topological key, durations being
// positive, so repeats are rare; the empty worklist is consistency with every neighbour and hence exactness), and a
// segment tree over W's local indices holding r + p + q with the maximum and its operation at the root.
// The worklist as a bucket queue over the key (a head or a tail, at most the makespan): O(1) push and pop, the next
// non-empty bucket found in a bitmap word scan; an operation is queued at most once (re-queued after it is popped),
// a key below the sweep pointer is clamped to it (order only keeps repeats rare, the empty queue is exactness).
typedef struct { int *bucket, *next; char *in; uint64_t *bits; int cap, ptr, hi, n; } WQueue;
static void wq_init(WQueue *Q, int cap, int nloc) {
    Q->cap = cap; Q->bucket = malloc(sizeof(int) * cap); for (int k = 0; k < cap; k++) Q->bucket[k] = -1;
    Q->bits = calloc((cap + 63) / 64, sizeof(uint64_t)); Q->next = malloc(sizeof(int) * nloc); Q->in = calloc(nloc, 1); Q->ptr = 0; Q->hi = -1; Q->n = 0;
}
static void wq_free(WQueue *Q) { free(Q->bucket); free(Q->bits); free(Q->next); free(Q->in); }
static inline void wq_push(WQueue *Q, int key, int i) {
    if (Q->in[i]) return;
    if (key >= Q->cap) key = Q->cap - 1;   // the search can go above the start's makespan; the order only keeps repeats rare
    if (Q->n == 0) Q->ptr = key; else if (key < Q->ptr) key = Q->ptr;
    Q->in[i] = 1; Q->next[i] = Q->bucket[key]; Q->bucket[key] = i; Q->bits[key >> 6] |= 1ull << (key & 63);
    if (key > Q->hi) Q->hi = key;
    Q->n++;
}
static inline int wq_pop(WQueue *Q) {
    int w = Q->ptr >> 6;
    uint64_t word = Q->bits[w] & (~0ull << (Q->ptr & 63));
    while (!word) word = Q->bits[++w];
    int key = (w << 6) + __builtin_ctzll(word), i = Q->bucket[key];
    Q->bucket[key] = Q->next[i];
    if (Q->bucket[key] < 0) Q->bits[w] &= ~(1ull << (key & 63));
    Q->in[i] = 0; Q->ptr = key; Q->n--;
    return i;
}
typedef struct { int size, *val, *op; } WTree;
static inline void wt_up(WTree *T, int x) {
    for (x >>= 1; x >= 1; x >>= 1) { int l = 2 * x, r = l + 1; if (T->val[l] >= T->val[r]) { T->val[x] = T->val[l]; T->op[x] = T->op[l]; } else { T->val[x] = T->val[r]; T->op[x] = T->op[r]; } }
}

typedef struct {
    int *set, ns, stamp;   // W, in machine order
    int *loc;              // a W operation's index in set
    int *chg, nchg, *cst, cver;   // the local indices changed by the last move
    // the compact window: W's values and links by local index, so the recompute streams from cache (seat 4's design;
    // the global arrays at a million operations put every neighbour lookup in another cache line, 90 ns a value)
    int *lr, *lq, *lp;             // heads, tails, durations
    int *ljp, *ljs, *lmp, *lms;    // neighbours as local indices, -1 when outside W or absent
    int *bjp, *bjs, *bmp, *bms;    // the fixed value of a neighbour outside W: a predecessor's end, a successor's tail + p (0 when absent)
    WQueue q; WTree tree;
    long pops;             // worklist pops, summed (stats[15])
} Win;

static inline void win_link_mach(const Shop *S, Win *W, int i, int o) {   // the machine links of local i (operation o) from the global ones
    int mp = S->mpred[o], ms = S->msucc[o], stamp = W->stamp;
    W->lmp[i] = mp >= 0 && S->mark[mp] == stamp ? W->loc[mp] : -1; W->bmp[i] = mp >= 0 && S->mark[mp] != stamp ? S->r[mp] + S->dur[mp] : 0;
    W->lms[i] = ms >= 0 && S->mark[ms] == stamp ? W->loc[ms] : -1; W->bms[i] = ms >= 0 && S->mark[ms] != stamp ? S->q[ms] + S->dur[ms] : 0;
}

// The window's local arrays from the global state (the job links, the machine links, the values).
static void win_build(const Shop *S, Win *W) {
    const int stamp = W->stamp;
    for (int i = 0; i < W->ns; i++) {
        int o = W->set[i], jp = shop_jp(S, o), js = shop_js(S, o);
        W->lr[i] = S->r[o]; W->lq[i] = S->q[o]; W->lp[i] = S->dur[o];
        W->ljp[i] = jp >= 0 && S->mark[jp] == stamp ? W->loc[jp] : -1; W->bjp[i] = jp >= 0 && S->mark[jp] != stamp ? S->r[jp] + S->dur[jp] : 0;
        W->ljs[i] = js >= 0 && S->mark[js] == stamp ? W->loc[js] : -1; W->bjs[i] = js >= 0 && S->mark[js] != stamp ? S->q[js] + S->dur[js] : 0;
        win_link_mach(S, W, i, o);
    }
}

static inline int win_head(const Win *W, int i) { int a = W->ljp[i] >= 0 ? W->lr[W->ljp[i]] + W->lp[W->ljp[i]] : W->bjp[i], b = W->lmp[i] >= 0 ? W->lr[W->lmp[i]] + W->lp[W->lmp[i]] : W->bmp[i]; return a > b ? a : b; }
static inline int win_tail(const Win *W, int i) { int a = W->ljs[i] >= 0 ? W->lq[W->ljs[i]] + W->lp[W->ljs[i]] : W->bjs[i], b = W->lms[i] >= 0 ? W->lq[W->lms[i]] + W->lp[W->lms[i]] : W->bms[i]; return a > b ? a : b; }

// Heads and tails over the whole window by Kahn's passes on the local arrays, written back to the global ones; the
// tree built from the result. Returns max of r + p + q, or -1 on a cycle.
static int win_eval(Shop *S, Win *W, int *indeg, int *topo) {
    const int ns = W->ns;
    for (int pass = 0; pass < 2; pass++) {
        int head = 0, tail = 0;
        for (int i = 0; i < ns; i++) { indeg[i] = pass ? (W->ljs[i] >= 0) + (W->lms[i] >= 0) : (W->ljp[i] >= 0) + (W->lmp[i] >= 0); if (!indeg[i]) topo[tail++] = i; }
        while (head < tail) {
            int i = topo[head++];
            if (!pass) { W->lr[i] = win_head(W, i); if (W->ljs[i] >= 0 && --indeg[W->ljs[i]] == 0) topo[tail++] = W->ljs[i]; if (W->lms[i] >= 0 && --indeg[W->lms[i]] == 0) topo[tail++] = W->lms[i]; }
            else { W->lq[i] = win_tail(W, i); if (W->ljp[i] >= 0 && --indeg[W->ljp[i]] == 0) topo[tail++] = W->ljp[i]; if (W->lmp[i] >= 0 && --indeg[W->lmp[i]] == 0) topo[tail++] = W->lmp[i]; }
        }
        if (tail < ns) return -1;
    }
    int size = 1; while (size < ns) size <<= 1;
    W->tree.size = size;
    int cm = 0;
    for (int i = 0; i < ns; i++) { int o = W->set[i]; S->r[o] = W->lr[i]; S->q[o] = W->lq[i]; int c = W->lr[i] + W->lp[i] + W->lq[i]; if (c > cm) cm = c; W->tree.val[size + i] = c; W->tree.op[size + i] = o; }
    for (int i = ns; i < size; i++) { W->tree.val[size + i] = -1; W->tree.op[size + i] = -1; }
    for (int x = size - 1; x >= 1; x--) { int l = 2 * x, r = l + 1; if (W->tree.val[l] >= W->tree.val[r]) { W->tree.val[x] = W->tree.val[l]; W->tree.op[x] = W->tree.op[l]; } else { W->tree.val[x] = W->tree.val[r]; W->tree.op[x] = W->tree.op[r]; } }
    return cm;
}

// After a move on machine k over the block at positions pb .. pb + len - 1: the machine links of the stretch and its
// two outer neighbours refreshed, then the heads and tails that change, by the worklist from the block outward on the
// local arrays, written back to the global heads and tails as they change; the tree's leaves updated.
static void win_recompute(Shop *S, Win *W, int k, int pb, int len, long *nheads, long *ntails) {
    const int n = S->n, stamp = W->stamp, *sq = S->seq + k * n, *op = S->opof + k * n;
    int lo = pb > 0 ? pb - 1 : 0, hi = pb + len < n ? pb + len : n - 1;
    for (int p = lo; p <= hi; p++) { int o = op[sq[p]]; if (S->mark[o] == stamp) win_link_mach(S, W, W->loc[o], o); }
    W->nchg = 0; W->cver++;
    for (int p = pb; p <= hi; p++) { int o = op[sq[p]]; if (S->mark[o] == stamp) { int i = W->loc[o]; wq_push(&W->q, W->lr[i], i); } }
    while (W->q.n) {   // a successor is queued only when its head will change (its new value read at once, two loads)
        int i = wq_pop(&W->q), h = win_head(W, i);
        W->pops++;
        if (h == W->lr[i]) continue;
        W->lr[i] = h; S->r[W->set[i]] = h; (*nheads)++;
        if (W->cst[i] != W->cver) { W->cst[i] = W->cver; W->chg[W->nchg++] = i; }
        int a = W->ljs[i], b = W->lms[i];
        if (a >= 0 && win_head(W, a) != W->lr[a]) wq_push(&W->q, W->lr[a], a);
        if (b >= 0 && win_head(W, b) != W->lr[b]) wq_push(&W->q, W->lr[b], b);
    }
    for (int p = lo; p < pb + len; p++) { int o = op[sq[p]]; if (S->mark[o] == stamp) { int i = W->loc[o]; wq_push(&W->q, W->lq[i], i); } }
    while (W->q.n) {
        int i = wq_pop(&W->q), h = win_tail(W, i);
        W->pops++;
        if (h == W->lq[i]) continue;
        W->lq[i] = h; S->q[W->set[i]] = h; (*ntails)++;
        if (W->cst[i] != W->cver) { W->cst[i] = W->cver; W->chg[W->nchg++] = i; }
        int a = W->ljp[i], b = W->lmp[i];
        if (a >= 0 && win_tail(W, a) != W->lq[a]) wq_push(&W->q, W->lq[a], a);
        if (b >= 0 && win_tail(W, b) != W->lq[b]) wq_push(&W->q, W->lq[b], b);
    }
    for (int c = 0; c < W->nchg; c++) { int i = W->chg[c], x = W->tree.size + i; W->tree.val[x] = W->lr[i] + W->lp[i] + W->lq[i]; wt_up(&W->tree, x); }
}

static int shop_check = 0;   // 1: the change-driven recompute asserted against the full window pass on every step (slow; the tests)
void pfa_shop_check(int on) { shop_check = on; }

static inline long shop_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1000000000L + t.tv_nsec; }

// The tabu search in time windows (for large instances). An interval [T0, T1) of width `width` on the current heads is
// a window; P = operations ending by T0, F = those starting at or after T1, W = the rest. P is predecessor-closed and F
// successor-closed, so a move among W's arcs changes no head on P and no tail on F, and the makespan after it is
// max(K, max_W r' + p + q') with r', q' recomputed over W alone and K the longest path that jumps W (an arc from P to
// F, at most one per machine and per job, or a path ending in P), a constant of the window: the window's search is
// exact. After a move only the heads and tails that change are recomputed (`win_recompute`), the window's maximum
// kept in a tree. The windows are swept backward in time, from the makespan to 0 in steps of width / 2: sweeping
// backward the new P is a subset of the old (heads exact, never written), the new F is the old F plus the old W's
// operations with exact head at or above the new T1 (successor-closed; F's tails are exact since moves in W change
// tails upstream only), and membership by the stored heads is exact since both predicates are monotone along every
// machine and job, so W is a position range per machine found by two binary searches; only the tails of the new W's
// old-P members are stale, which one pass over W at entry fixes. (The forward sweep cannot drop its global pass: an
// uphill move raises a head above the next T1 while its old-F successor's stale head stays below, and F is not
// successor-closed at the seam.) The full pass runs once per sweep and asserts the tracked makespan (stats[3] counts
// mismatches; zero or the run is wrong). Each window is searched for at most `win_iters` iterations and left after
// `win_idle` without a window best; the tabu list, the best and the kick carry across windows. Stops after `seconds`
// of CPU time or `iters` iterations. seq_io: the trajectory (in and out); best_seq: the best; t0_io: the sweep
// position carried across calls. stats (16): iterations, moves evaluated, windows searched, mismatches at the sweep
// assertion, sweeps, new bests, |W| summed, the start's makespan, heads changed, tails changed, entry ns, recompute ns,
// windows left by the iteration cap, segment+moves+selection ns, windows skipped (the path jumps them), worklist pops.
int pfa_shop_tabu_win(int n, int m, const int *mach, const int *dur, int *seq_io, double seconds, long iters, uint64_t seed,
                      int *best_seq, double width, long win_iters, long win_idle, double *t0_io, long *stats) {
    Shop *S = shop_new(n, m, mach, dur, seq_io);
    if (!shop_eval(S)) { shop_free(S); return -1; }
    const int N = S->N, L = 10 + n / m;
    ShopTabu *T = calloc(1, sizeof(ShopTabu)); T->max_tenure = L + L / 2 + 1;
    int *path = malloc(sizeof(int) * N), *work = malloc(sizeof(int) * N);
    ShopMove *moves = malloc(sizeof(ShopMove) * 4 * N);
    Win W = {0};
    W.set = malloc(sizeof(int) * N); W.loc = malloc(sizeof(int) * N); W.cst = calloc(N, sizeof(int)); W.chg = malloc(sizeof(int) * N);
    int *lbuf = malloc(sizeof(int) * 11 * (size_t)N);   // the compact window's eleven arrays
    W.lr = lbuf; W.lq = lbuf + N; W.lp = lbuf + 2 * (size_t)N; W.ljp = lbuf + 3 * (size_t)N; W.ljs = lbuf + 4 * (size_t)N; W.lmp = lbuf + 5 * (size_t)N; W.lms = lbuf + 6 * (size_t)N; W.bjp = lbuf + 7 * (size_t)N; W.bjs = lbuf + 8 * (size_t)N; W.bmp = lbuf + 9 * (size_t)N; W.bms = lbuf + 10 * (size_t)N;
    wq_init(&W.q, S->cmax + 1, N);   // keys are heads and tails, below the start's makespan
    { int size = 1; while (size < N) size <<= 1; W.tree.val = malloc(sizeof(int) * 2 * size); W.tree.op = malloc(sizeof(int) * 2 * size); }
    int best = S->cmax;
    long st[16] = {0}; st[7] = S->cmax;
    S->rng = shop_seed(seed);
    clock_t t_end = seconds > 0 ? clock() + (clock_t)(seconds * CLOCKS_PER_SEC) : 0;   // a clock limit, an iteration limit, or both
#define WIN_OUT ((t_end && clock() >= t_end) || it >= iters)
    long since = 0, it = 0;
    typedef struct { int k, from, to; } Undo;   // a move as applied, absolute positions on the machine
    long ulen = 0, ucap = 1 << 16; Undo *ulog = malloc(sizeof(Undo) * ucap);
    shop_all_ends = 1;
    double top = S->cmax - width > 0 ? S->cmax - width : 0;
    double T0 = (t0_io && *t0_io > 0 && *t0_io < S->cmax) ? *t0_io : top;   // the sweep continues where the previous call left it
    for (;;) {
        if (WIN_OUT) break;
        long t_entry = shop_ns();
        const double T1 = T0 + width;
        const int stamp = ++S->stamp;
        int ns = 0;
        long K = 0;
        for (int k = 0; k < m; k++) {   // W's position range on machine k: ends > T0 from lo, heads >= T1 from hi (both monotone)
            const int *sq = S->seq + k * n, *op = S->opof + k * n;
            int lo = 0, hi = n;
            for (int a = 0, b = n; a < b;) { int c = (a + b) / 2; if (shop_end(S, op[sq[c]]) > T0) b = c; else a = c + 1; lo = a; }
            for (int a = lo, b = n; a < b;) { int c = (a + b) / 2; if (S->r[op[sq[c]]] >= T1) b = c; else a = c + 1; hi = a; }
            for (int p = lo; p < hi; p++) { int o = op[sq[p]]; S->mark[o] = stamp; W.loc[o] = ns; W.set[ns++] = o; }
            if (lo > 0) {   // the last P operation: its end (a path ending in P) and, with no W operation on the machine, its arc into F
                int a = op[sq[lo - 1]]; long e = shop_end(S, a);
                if (e > K) K = e;
                if (lo == hi && lo < n) { int b = op[sq[lo]]; long c = e + S->dur[b] + S->q[b]; if (c > K) K = c; }
            }
        }
        for (int j = 0; j < n; j++) {   // the one job arc from P to F, if any
            const int o0 = j * m;
            int i0 = 0;
            for (int a = 0, b = m; a < b;) { int c = (a + b) / 2; if (shop_end(S, o0 + c) > T0) b = c; else a = c + 1; i0 = a; }
            if (i0 > 0) {
                long e = shop_end(S, o0 + i0 - 1);
                if (e > K) K = e;
                if (i0 < m && S->r[o0 + i0] >= T1) { int b = o0 + i0; long c = e + S->dur[b] + S->q[b]; if (c > K) K = c; }
            }
        }
        W.ns = ns; W.stamp = stamp;
        int cmW = 0;
        if (ns > 0) { win_build(S, &W); cmW = win_eval(S, &W, S->indeg, S->topo); }
        if (cmW < 0) { shop_all_ends = 0; shop_free(S); free(T); free(path); free(work); free(moves); free(W.set); free(W.loc); free(W.cst); free(W.chg); free(lbuf); wq_free(&W.q); free(W.tree.val); free(W.tree.op); free(ulog); return -2; }
        st[10] += shop_ns() - t_entry;
        if (ns > 0 && cmW > K) {
            st[2]++; st[6] += ns;
            int wbest = cmW; long widle = 0, wi = 0;
            for (; wi < win_iters && widle < win_idle; wi++, it++) {
                if (it >= iters || ((it & 255) == 0 && t_end && clock() >= t_end)) break;
                long t_seg = shop_ns();
                int plen = shop_segment_in(S, W.tree.op[1], stamp, path);
                int nm = shop_moves(S, path, plen, moves, work);
                st[1] += nm;
                if (nm == 0) break;
                int pick = -1, bestest = 0, ties = 0, admissible = 0;
                for (int i = 0; i < nm; i++) {
                    int tabu = shop_move_tabu(S, T, &moves[i], it);
                    long est_true = moves[i].est > K ? moves[i].est : K;
                    if (tabu && !(est_true < best)) continue;
                    admissible++;
                    if (pick < 0 || moves[i].est < bestest) { pick = i; bestest = moves[i].est; ties = 1; }
                    else if (moves[i].est == bestest && (int)((shop_rng(&S->rng) >> 33) % (uint64_t)++ties) == 0) pick = i;
                }
                if (!admissible || (shop_kick && since > 0 && since % shop_kick == 0)) { pick = (int)((shop_rng(&S->rng) >> 33) % (uint64_t)nm); }
                const ShopMove *M = &moves[pick];
                long tenure = L + (long)((shop_rng(&S->rng) >> 33) % (uint64_t)(L / 2 + 1));
                shop_apply(S, T, M, it + tenure, it);
                if (ulen == ucap) { ucap *= 2; ulog = realloc(ulog, sizeof(Undo) * ucap); }
                ulog[ulen++] = (Undo){M->k, M->pb + M->from, M->pb + M->to};
                long t_rec = shop_ns();
                st[13] += t_rec - t_seg;
                win_recompute(S, &W, M->k, M->pb, M->len, &st[8], &st[9]);
                st[11] += shop_ns() - t_rec;
                int cmW2 = W.tree.val[1];
                if (shop_check) {   // the full window pass over the global arrays must agree on every head and tail
                    int c2 = shop_eval_set(S, W.set, ns, stamp);
                    if (c2 != cmW2) st[3]++;
                    for (int i = 0; i < ns; i++) { int o = W.set[i]; if (W.lr[i] != S->r[o] || W.lq[i] != S->q[o] || W.tree.val[W.tree.size + i] != S->r[o] + S->dur[o] + S->q[o]) { st[3]++; break; } }
                    cmW2 = win_eval(S, &W, S->indeg, S->topo);
                }
                long cm = cmW2 > K ? cmW2 : K;
                S->cmax = (int)cm;
                if (cm < best) { best = (int)cm; ulen = 0; since = 0; st[5]++; } else since++;   // the log restarts at a new best
                if (cmW2 < wbest) { wbest = cmW2; widle = 0; } else widle++;
                st[0]++;
            }
            if (wi >= win_iters) st[12]++;
        } else if (ns > 0) st[14]++;
        if (T0 <= 0) {   // the sweep wraps: everything exact again, the tracked makespan asserted
            int tracked = S->cmax;
            if (!shop_eval(S)) break;
            if (S->cmax != tracked) st[3]++;
            if (S->cmax < best) best = S->cmax;
            st[4]++;
            top = S->cmax - width > 0 ? S->cmax - width : 0;
            T0 = top;
        } else { T0 -= width / 2; if (T0 < 0) T0 = 0; }
    }
    if (t0_io) *t0_io = T0;
#undef WIN_OUT
    shop_all_ends = 0;
    memcpy(seq_io, S->seq, sizeof(int) * N);
    memcpy(best_seq, S->seq, sizeof(int) * N);
    for (long u = ulen - 1; u >= 0; u--) {   // the best: the moves since it undone in reverse (the job at `to` back to `from`)
        int *sq = best_seq + ulog[u].k * n, from = ulog[u].from, to = ulog[u].to, x = sq[to];
        if (to > from) { memmove(sq + from + 1, sq + from, sizeof(int) * (to - from)); sq[from] = x; }
        else { memmove(sq + to, sq + to + 1, sizeof(int) * (from - to)); sq[from] = x; }
    }
    st[15] = W.pops;
    if (stats) memcpy(stats, st, sizeof(st));
    shop_free(S); free(T); free(path); free(work); free(moves); free(W.set); free(W.loc); free(W.cst); free(W.chg); free(lbuf); wq_free(&W.q); free(W.tree.val); free(W.tree.op); free(ulog);
    return best;
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
    S->rng = shop_seed(seed);
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
    S->rng = shop_seed(seed);
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
