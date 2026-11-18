// PFA sparse swap kernel (included by pfa.c): the exchanges of swap.c for problems where each variable is coupled to
// few others, with the same schedule, moves, tabu and bookkeeping. A is stored by rows of nonzero entries (ptr, idx,
// val; symmetric, columns ascending), and each coupled pair r < s has a number, pid[e] at both of its entries.
//   - F_r(k) is kept for every variable and value and changes on the two exchanged variables and their neighbours.
//   - The move is drawn with probability proportional to exp(beta x), x = -dE - tab_r(b) - tab_s(a) with tab the
//     tabu penalties by variable and value, in two parts. For a pair with A_rs = 0, x = x_r(b) + x_s(a) with
//     x_r(b) = -(F_r(b) - F_r(a_r)) - tab_r(b) the single move of r to b, so
//     the weights factor: the variables at value a form the class a -> b with the sum Z(a -> b) of their weights, and
//     the uncoupled pairs of the block of values (a, b) weigh Z(a -> b) Z(b -> a) - S(a, b), S the same products over
//     the coupled pairs of the block, kept as a running sum. A draw picks a block by that weight, then one variable
//     from each class by its weight, again while the two are coupled, and after 16 such draws by enumerating the two
//     classes (the pairs' weights are then as drawn, whatever the share of the coupled ones). The coupled pairs at
//     different values are the leaves of a second tree with their full x, and are drawn from it with its share of the
//     total. Together that is the distribution above.
//   - The classes are the subtrees of one tree of sums (sums.c): the segment of value b holds the classes a -> b, each
//     of P = 8^l leaves, P the smallest power of 8 at or above the largest count, so a class is one node at level l,
//     its entry the class sum, and a draw within it walks down from that node. Variables sit in positions sorted by
//     value as in swap.c; the leaf of a variable in a class is its position's offset in its value's block.
//   - Weights are at the beta of the last rebuild, beta0, as exp(beta0 (x - x0)) with x0 the largest x of the class,
//     a block's weight times exp(beta0 (x0(a -> b) + x0(b -> a) - X0)) and a coupled pair's exp(beta0 (x - X0)), X0
//     the largest x of any pair then: every weight is then exp(beta0 (x - X0)) and none overflows. The draw is
//     corrected for the rise of beta since by rejection as in sums.c, against the largest x of any pair now, from
//     trees of maxima kept next to the sums. All weights are rebuilt by the rule of sums.c, and the running sums
//     S with them. A block whose uncoupled weight is below the rounding of Z Z, 2^-40 of it, gets 0.
//   - A step gives new x to the leaves of the exchanged variables and of their neighbours (one exponential each), to
//     the coupled pairs of those variables, and to the blocks of the classes touched.

typedef struct { Sums s; double *mx[SUMS_LEVELS]; long cnt[SUMS_LEVELS]; } Tree;   // sums, and maxima of the x's

static inline double max8(const double *m) {
    double v = m[0];
    for (int i = 1; i < 8; i++) if (m[i] > v) v = m[i];
    return v;
}

static void tree_new(Tree *t, long n) {
    sums_new(&t->s, (int)n);
    long cnt = (n + 7) & ~7L;
    for (int l = 0; l <= t->s.top; l++) {
        t->cnt[l] = cnt;
        t->mx[l] = malloc(sizeof(double) * cnt);
        for (long i = 0; i < cnt; i++) t->mx[l][i] = -__builtin_inf();
        cnt = ((cnt >> 3) + 7) & ~7L;
    }
}

static void tree_free(Tree *t) {
    sums_free(&t->s);
    for (int l = 0; l <= t->s.top; l++) free(t->mx[l]);
}

// Leaf k gets weight w and value x, and the entries above it follow.
static inline void tree_set(Tree *t, long k, double w, double x) {
    t->s.w[0][k] = w; t->mx[0][k] = x;
    for (int l = 1; l <= t->s.top; l++) {
        const long c = k >> (3 * l);
        t->s.w[l][c] = sums_8(t->s.w[l - 1] + 8 * c);
        t->mx[l][c] = max8(t->mx[l - 1] + 8 * c);
    }
}

// All entries above the leaves, after the leaves were set directly.
static void tree_levels(Tree *t) {
    for (int l = 1; l <= t->s.top; l++)
        for (long c = 0; c < t->cnt[l - 1] / 8; c++) {
            t->s.w[l][c] = sums_8(t->s.w[l - 1] + 8 * c);
            t->mx[l][c] = max8(t->mx[l - 1] + 8 * c);
        }
}

static inline double tree_total(const Tree *t) { return sums_8(t->s.w[t->s.top]); }
static inline double tree_max(const Tree *t) { return max8(t->mx[t->s.top]); }

// The leaf under entry c of level l (level top + 1, entry 0: the whole tree) that r, in [0, its sum), falls in.
static inline long tree_walk(const Tree *t, int l, long c, double r) {
    for (; l > 0; l--) {
        const double *w = t->s.w[l - 1] + 8 * c;
        int k = 0, last = 0;
        for (; k < 8; k++) {
            if (w[k] > 0.0) last = k;
            if (r < w[k]) break;
            r -= w[k];
        }
        if (k == 8) { k = last; r = 0.0; }   // rounding took r past the end
        c = 8 * c + k;
    }
    return c;
}

// The weight of x relative to x0 at beta: 0 for a move that does not exist (x = -inf), at any beta.
static inline double ss_w(double beta, double x, double x0) {
    return x > -__builtin_inf() ? exp(beta * (x - x0)) : 0.0;
}

typedef struct {
    int n, q, P, lP;                        // variables, values, class size 8^lP
    const int64_t *ptr; const int *idx; const double *val; const int *pid;   // A, and the pair of each entry
    const int *pr, *ps; int np;             // the coupled pairs, r < s
    const double *B; double *kappa;
    int *start, *count;                     // positions of each value: start[v] .. start[v] + count[v]
    int *pos, *at, *x;                      // variable at each position; position and value of each variable
    double *F, *Fcur, *tab;                 // F by variable and value; F of the current value; tabu penalties by
                                            // variable and value
    Tree L, K, C;                           // the classes, the blocks, the coupled pairs
    int *ba, *bb;                           // the values of each block
    double *x0c, *fac, X0, beta0;           // see above
    double *S, *fterm; int *fblk;           // per block, the factorized weight of its coupled pairs; per pair, its
                                            // term in S and the block it is in
    long leaves, rejected;                  // leaves of all trees; draws rejected since the rebuild
    int *mark, tick;                        // mark[v] == tick: v is in list
    int *list, nlist;                       // the variables whose x changed this step
    int *bmark, *blist, nblist;             // the same for blocks
    int *pmark;                             // the same for pairs
    int *nmark, ntick; double *scratch;     // for the enumeration of a block
} SS;

static inline int ss_cls(const SS *S, int a, int b) { return b * (S->q - 1) + (a < b ? a : a - 1); }
static inline int ss_blk(const SS *S, int a, int b) {
    return a < b ? a * S->q - a * (a + 1) / 2 + (b - a - 1) : ss_blk(S, b, a);
}
static inline long ss_leaf(const SS *S, int w, int b) {
    return (long)ss_cls(S, S->x[w], b) * S->P + (S->at[w] - S->start[S->x[w]]);
}
static inline double ss_xleaf(const SS *S, int w, int b) {   // x of the single move of w to b
    return S->x[w] == b ? -__builtin_inf() : (S->Fcur[w] - S->F[(long)w * S->q + b]) - S->tab[(long)w * S->q + b];
}
static inline double ss_dE(const SS *S, int r, int s, double Ars) {   // dE of the exchange of r and s
    const int a = S->x[r], b = S->x[s], q = S->q;
    return (S->F[(long)r * q + b] - S->Fcur[r]) + (S->F[(long)s * q + a] - S->Fcur[s]) + Ars * S->kappa[a * q + b];
}
static inline double ss_xpair(const SS *S, int r, int s, double Ars) {   // x of the coupled pair r, s
    return S->x[r] == S->x[s] ? -__builtin_inf()
                              : -ss_dE(S, r, s, Ars) - S->tab[(long)r * S->q + S->x[s]] - S->tab[(long)s * S->q + S->x[r]];
}
static inline double ss_Ars(const SS *S, int r, int s) {   // A_rs, by bisection in row r
    int64_t lo = S->ptr[r], hi = S->ptr[r + 1];
    while (lo < hi) {
        const int64_t m = (lo + hi) >> 1;
        if (S->idx[m] < s) lo = m + 1; else hi = m;
    }
    return lo < S->ptr[r + 1] && S->idx[lo] == s ? S->val[lo] : 0.0;
}
static inline double ss_fterm(const SS *S, int r, int s) {   // the factorized weight of the coupled pair r, s
    return S->x[r] == S->x[s] ? 0.0 : S->L.s.w[0][ss_leaf(S, r, S->x[s])] * S->L.s.w[0][ss_leaf(S, s, S->x[r])];
}

static inline void ss_touch(SS *S, int w) {
    if (S->mark[w] != S->tick) { S->mark[w] = S->tick; S->list[S->nlist++] = w; }
}
static inline void ss_touch_block(SS *S, int blk) {
    if (S->bmark[blk] != S->tick) { S->bmark[blk] = S->tick; S->blist[S->nblist++] = blk; }
}

// New x for the leaves of variable w, and its blocks marked.
static void ss_var(SS *S, int w) {
    const int a = S->x[w];
    for (int b = 0; b < S->q; b++) {
        if (b == a) continue;
        const int c = ss_cls(S, a, b);
        const long k = (long)c * S->P + (S->at[w] - S->start[a]);
        const double x = ss_xleaf(S, w, b);
        S->L.s.x[k] = x;
        tree_set(&S->L, k, ss_w(S->beta0, x, S->x0c[c]), x);
        ss_touch_block(S, ss_blk(S, a, b));
    }
}

// New x and factorized weight for the coupled pairs of variable w, after the leaves of every changed variable.
static void ss_pairs(SS *S, int w) {
    for (int64_t e = S->ptr[w]; e < S->ptr[w + 1]; e++) {
        const int id = S->pid[e], z = S->idx[e];
        if (S->pmark[id] == S->tick) continue;
        S->pmark[id] = S->tick;
        const double x = ss_xpair(S, w, z, S->val[e]);
        S->C.s.x[id] = x;
        tree_set(&S->C, id, ss_w(S->beta0, x, S->X0), x);
        S->S[S->fblk[id]] -= S->fterm[id]; ss_touch_block(S, S->fblk[id]);
        if (S->x[w] != S->x[z]) {
            S->fblk[id] = ss_blk(S, S->x[w], S->x[z]); S->fterm[id] = ss_fterm(S, w, z);
            S->S[S->fblk[id]] += S->fterm[id]; ss_touch_block(S, S->fblk[id]);
        } else S->fterm[id] = 0.0;
    }
}

static inline void ss_block(SS *S, int blk) {
    const int ab = ss_cls(S, S->ba[blk], S->bb[blk]), ba = ss_cls(S, S->bb[blk], S->ba[blk]);
    const double ZZ = S->L.s.w[S->lP][ab] * S->L.s.w[S->lP][ba];
    double rest = ZZ - S->S[blk];
    if (rest <= 0x1p-40 * ZZ) rest = 0.0;
    tree_set(&S->K, blk, rest * S->fac[blk], S->L.mx[S->lP][ab] + S->L.mx[S->lP][ba]);
}

// All weights at beta, from the x's.
static void ss_rebuild(SS *S, double beta) {
    const int q = S->q, P = S->P;
    S->beta0 = beta; S->rejected = 0;
    for (int c = 0; c < q * (q - 1); c++) {
        double m = -__builtin_inf();
        for (long k = (long)c * P; k < (long)(c + 1) * P; k++) if (S->L.s.x[k] > m) m = S->L.s.x[k];
        S->x0c[c] = m > -__builtin_inf() ? m : 0.0;
    }
    double X0 = -__builtin_inf();
    for (int blk = 0; blk < S->K.s.n; blk++) {
        const double m = S->x0c[ss_cls(S, S->ba[blk], S->bb[blk])] + S->x0c[ss_cls(S, S->bb[blk], S->ba[blk])];
        if (m > X0) X0 = m;
    }
    for (int id = 0; id < S->np; id++) if (S->C.s.x[id] > X0) X0 = S->C.s.x[id];
    S->X0 = X0;
    for (int c = 0; c < q * (q - 1); c++)
        for (long k = (long)c * P; k < (long)(c + 1) * P; k++) {
            S->L.s.w[0][k] = ss_w(beta, S->L.s.x[k], S->x0c[c]); S->L.mx[0][k] = S->L.s.x[k];
        }
    tree_levels(&S->L);
    for (int id = 0; id < S->np; id++) {
        S->C.s.w[0][id] = ss_w(beta, S->C.s.x[id], X0); S->C.mx[0][id] = S->C.s.x[id];
    }
    tree_levels(&S->C);
    for (int blk = 0; blk < S->K.s.n; blk++) S->S[blk] = 0.0;
    for (int id = 0; id < S->np; id++) {
        const int r = S->pr[id], s = S->ps[id];
        S->fterm[id] = ss_fterm(S, r, s);
        S->fblk[id] = S->x[r] != S->x[s] ? ss_blk(S, S->x[r], S->x[s]) : 0;
        S->S[S->fblk[id]] += S->fterm[id];
    }
    for (int blk = 0; blk < S->K.s.n; blk++) {
        const int ab = ss_cls(S, S->ba[blk], S->bb[blk]), ba = ss_cls(S, S->bb[blk], S->ba[blk]);
        S->fac[blk] = exp(beta * (S->x0c[ab] + S->x0c[ba] - X0));
        const double ZZ = S->L.s.w[S->lP][ab] * S->L.s.w[S->lP][ba];
        double rest = ZZ - S->S[blk];
        if (rest <= 0x1p-40 * ZZ) rest = 0.0;
        S->K.s.w[0][blk] = rest * S->fac[blk];
        S->K.mx[0][blk] = S->L.mx[S->lP][ab] + S->L.mx[S->lP][ba];
    }
    tree_levels(&S->K);
}

// An uncoupled pair of block blk, drawn by the weights of its two variables; returns 0 when the block has none.
static int ss_block_pair(SS *S, int blk, uint64_t *rng, int *r_out, int *s_out, double *xp) {
    const int a = S->ba[blk], b = S->bb[blk], ab = ss_cls(S, a, b), ba = ss_cls(S, b, a);
    const double Zab = S->L.s.w[S->lP][ab], Zba = S->L.s.w[S->lP][ba];
    for (int tries = 0; tries < 16; tries++) {
        const long kr = tree_walk(&S->L, S->lP, ab, sums_uniform(rng) * Zab);
        const long ks = tree_walk(&S->L, S->lP, ba, sums_uniform(rng) * Zba);
        const int r = S->pos[S->start[a] + (int)(kr - (long)ab * S->P)];
        const int s = S->pos[S->start[b] + (int)(ks - (long)ba * S->P)];
        if (ss_Ars(S, r, s) != 0.0) continue;
        *r_out = r; *s_out = s; *xp = S->L.s.x[kr] + S->L.s.x[ks];
        return 1;
    }
    // enumeration: r by w_r(b) times the weight of the variables at b it is not coupled to, then s among those
    double tot = 0.0;
    for (int t = 0; t < S->count[a]; t++) {
        const int r = S->pos[S->start[a] + t];
        double coupled = 0.0;
        for (int64_t e = S->ptr[r]; e < S->ptr[r + 1]; e++)
            if (S->x[S->idx[e]] == b) coupled += S->L.s.w[0][ss_leaf(S, S->idx[e], a)];
        double rest = Zba - coupled;
        if (rest < 0.0) rest = 0.0;
        tot += S->scratch[t] = S->L.s.w[0][ss_leaf(S, r, b)] * rest;
    }
    if (!(tot > 0.0)) return 0;
    double u = sums_uniform(rng) * tot;
    int tr = S->count[a] - 1;
    for (int t = 0; t < S->count[a]; t++) { if (u < S->scratch[t]) { tr = t; break; } u -= S->scratch[t]; }
    const int r = S->pos[S->start[a] + tr];
    S->ntick++;
    for (int64_t e = S->ptr[r]; e < S->ptr[r + 1]; e++) S->nmark[S->idx[e]] = S->ntick;
    tot = 0.0;
    for (int t = 0; t < S->count[b]; t++) {
        const int s = S->pos[S->start[b] + t];
        tot += S->scratch[t] = S->nmark[s] == S->ntick ? 0.0 : S->L.s.w[0][ss_leaf(S, s, a)];
    }
    if (!(tot > 0.0)) return 0;
    u = sums_uniform(rng) * tot;
    int ts = S->count[b] - 1;
    for (int t = 0; t < S->count[b]; t++) { if (u < S->scratch[t]) { ts = t; break; } u -= S->scratch[t]; }
    const int s = S->pos[S->start[b] + ts];
    *r_out = r; *s_out = s; *xp = S->L.s.x[ss_leaf(S, r, b)] + S->L.s.x[ss_leaf(S, s, a)];
    return 1;
}

// A pair drawn with probability proportional to exp(beta x): the variables *r_out, *s_out.
static void ss_draw(SS *S, double beta, uint64_t *rng, int *r_out, int *s_out) {
    for (;;) {
        double WK = tree_total(&S->K), WC = tree_total(&S->C), W = WK + WC;
        if (S->rejected * (S->L.s.top + 2) >= S->leaves || !(W > 0.0 && W < __builtin_inf())) {
            ss_rebuild(S, beta);
            WK = tree_total(&S->K); WC = tree_total(&S->C); W = WK + WC;
        }
        const double u = sums_uniform(rng) * W;
        int r, s; double xp;
        if (u < WK) {
            const int blk = (int)tree_walk(&S->K, S->K.s.top + 1, 0, u);
            if (!ss_block_pair(S, blk, rng, &r, &s, &xp)) { S->rejected++; continue; }
        } else {
            const int id = (int)tree_walk(&S->C, S->C.s.top + 1, 0, u - WK);
            r = S->pr[id]; s = S->ps[id]; xp = S->C.s.x[id];
        }
        const double X = tree_max(&S->K) > tree_max(&S->C) ? tree_max(&S->K) : tree_max(&S->C);
        if (beta == S->beta0 || sums_uniform(rng) < exp((beta - S->beta0) * (xp - X))) {
            *r_out = r; *s_out = s;
            return;
        }
        S->rejected++;
    }
}

static void sparse_swap_loop(SS *S, Tabu *tb, long *out, int *moved, long nsteps, double beta_start, double beta_max,
                             double *E_io, double *best_io, int *best_pos, double *late_io, int *late_pos,
                             double *marks, uint64_t seed, double lambda) {
    const int n = S->n, q = S->q;
    double E = *E_io, best = *best_io, beta = beta_start, dbeta = (beta_max - beta_start) / (double)(nsteps - 1);
    uint64_t rng = seed | 1;
    int at_best = 0;     // the current state is the best one of this half and best_pos does not hold it yet
    long nmoved = 0;     // positions moved since best_pos was written; more than n: copy the state
#define KEEP_BEST { \
        if (nmoved > n) __builtin_memcpy(best_pos, S->pos, sizeof(int) * n); \
        else for (long i = 0; i < nmoved; i++) best_pos[moved[i]] = S->pos[moved[i]]; \
        nmoved = 0; at_best = 0; }
    for (long step = 0; step < nsteps; step++) {
        if (step == 3 * nsteps / 4) marks[1] = best;   // the half's lowest at three quarters
        if (step == nsteps / 2) {   // the second half starts: its lowest state so far is the current one
            marks[0] = E;   // the energy at the half
            if (at_best) KEEP_BEST
            *best_io = best; best = E; best_pos = late_pos;
            __builtin_memcpy(best_pos, S->pos, sizeof(int) * n); nmoved = 0;
        }
        beta += dbeta;
        int r, s;
        ss_draw(S, beta, &rng, &r, &s);
        const int a = S->x[r], b = S->x[s], t = S->at[r], u = S->at[s];
        const double dE = ss_dE(S, r, s, ss_Ars(S, r, s));
        if (at_best && dE >= 0.0) KEEP_BEST   // leaving it
        E += dE;
        S->pos[t] = s; S->pos[u] = r; S->at[r] = u; S->at[s] = t; S->x[r] = b; S->x[s] = a;
        if (nmoved < n) { moved[nmoved] = t; moved[nmoved + 1] = u; }
        nmoved += 2;
        // F of the neighbours: F_w(k) += A_wr (B(k, b) - B(k, a)) + A_ws (B(k, a) - B(k, b))
        S->tick++; S->nlist = 0; S->nblist = 0;
        ss_touch(S, r); ss_touch(S, s);
        const double *Ba = S->B + (long)a * q, *Bb = S->B + (long)b * q;
        for (int64_t e = S->ptr[r]; e < S->ptr[r + 1]; e++) {
            const int w = S->idx[e]; double *Fw = S->F + (long)w * q;
            for (int k = 0; k < q; k++) Fw[k] += S->val[e] * (Bb[k] - Ba[k]);
            ss_touch(S, w);
        }
        for (int64_t e = S->ptr[s]; e < S->ptr[s + 1]; e++) {
            const int w = S->idx[e]; double *Fw = S->F + (long)w * q;
            for (int k = 0; k < q; k++) Fw[k] += S->val[e] * (Ba[k] - Bb[k]);
            ss_touch(S, w);
        }
        const int cnt = tabu_expire(tb, step, out);
        for (int i = 0; i < cnt; i++) { S->tab[out[i]] = 0.0; ss_touch(S, (int)(out[i] / q)); }
        if (tabu_push(tb, r, a, step)) S->tab[(long)r * q + a] = lambda / beta;   // r left a, s left b
        if (tabu_push(tb, s, b, step)) S->tab[(long)s * q + b] = lambda / beta;
        for (int i = 0; i < S->nlist; i++) S->Fcur[S->list[i]] = S->F[(long)S->list[i] * q + S->x[S->list[i]]];
        for (int i = 0; i < S->nlist; i++) ss_var(S, S->list[i]);
        for (int i = 0; i < S->nlist; i++) ss_pairs(S, S->list[i]);
        for (int i = 0; i < S->nblist; i++) ss_block(S, S->blist[i]);
        if (E < best) { best = E; at_best = 1; }
    }
    if (at_best) KEEP_BEST
#undef KEEP_BEST
    *E_io = E; *late_io = best;
}

// ptr, idx, val: A by rows; pid: the pair of each entry; pr, ps: the np coupled pairs, r < s; B: q x q; val_pos: the
// value of each position; pos: the variable at each position (updated in place); F: n x q (updated in place); seed:
// for the draws. No noise buffer; everything else as pfa_anneal_swap.
void pfa_anneal_sparse_swap(const int64_t *ptr, const int *idx, const double *val, const int *pid, int np,
                            const int *pr, const int *ps, const double *B, int n, int q, const int *val_pos, int *pos,
                            double *F, long nsteps, double beta_start, double beta_max, double *E_io, double *best_io,
                            int *best_pos, double *late_io, int *late_pos, double *marks, uint64_t seed,
                            double lambda, const int64_t *tenure) {
    SS S = {0};
    S.n = n; S.q = q; S.ptr = ptr; S.idx = idx; S.val = val; S.pid = pid; S.pr = pr; S.ps = ps; S.np = np; S.B = B;
    S.pos = pos; S.F = F;
    S.start = calloc(q + 1, sizeof(int)); S.count = calloc(q, sizeof(int));
    S.at = malloc(sizeof(int) * n); S.x = malloc(sizeof(int) * n);
    S.Fcur = malloc(sizeof(double) * n); S.tab = calloc((long)n * q, sizeof(double));
    S.kappa = malloc(sizeof(double) * (long)q * q);
    S.mark = calloc(n, sizeof(int)); S.list = malloc(sizeof(int) * n);
    S.nmark = calloc(n, sizeof(int)); S.scratch = malloc(sizeof(double) * n);
    for (int t = 0; t < n; t++) { S.count[val_pos[t]]++; S.at[pos[t]] = t; S.x[pos[t]] = val_pos[t]; }
    for (int t = n - 1; t >= 0; t--) S.start[val_pos[t]] = t;
    int cmax = 0;
    for (int v = 0; v < q; v++) if (S.count[v] > cmax) cmax = S.count[v];
    for (S.P = 1, S.lP = 0; S.P < cmax; S.P *= 8) S.lP++;
    for (int a = 0; a < q; a++)
        for (int b = 0; b < q; b++)
            S.kappa[(long)a * q + b] = 2.0 * B[(long)a * q + b] - B[(long)a * q + a] - B[(long)b * q + b];
    for (int w = 0; w < n; w++) S.Fcur[w] = F[(long)w * q + S.x[w]];
    const int nb = q * (q - 1) / 2;
    S.ba = malloc(sizeof(int) * nb); S.bb = malloc(sizeof(int) * nb); S.fac = malloc(sizeof(double) * nb);
    S.S = malloc(sizeof(double) * nb);
    S.bmark = calloc(nb, sizeof(int)); S.blist = malloc(sizeof(int) * (nb + 1));
    for (int a = 0; a < q; a++)
        for (int b = a + 1; b < q; b++) { S.ba[ss_blk(&S, a, b)] = a; S.bb[ss_blk(&S, a, b)] = b; }
    S.x0c = malloc(sizeof(double) * q * (q - 1));
    S.fterm = malloc(sizeof(double) * (np + 1)); S.fblk = malloc(sizeof(int) * (np + 1));
    S.pmark = calloc(np + 1, sizeof(int));
    tree_new(&S.L, (long)q * (q - 1) * S.P); tree_new(&S.K, nb); tree_new(&S.C, np > 0 ? np : 1);
    S.leaves = (long)q * (q - 1) * S.P + nb + np;
    for (long k = 0; k < S.L.s.n; k++) S.L.s.x[k] = -__builtin_inf();
    for (int w = 0; w < n; w++)
        for (int b = 0; b < q; b++) if (b != S.x[w]) S.L.s.x[ss_leaf(&S, w, b)] = ss_xleaf(&S, w, b);
    for (int id = 0; id < np; id++) S.C.s.x[id] = ss_xpair(&S, pr[id], ps[id], ss_Ars(&S, pr[id], ps[id]));
    if (np == 0) { S.C.s.x[0] = -__builtin_inf(); S.C.s.n = 0; }
    ss_rebuild(&S, beta_start);
    Tabu tb;
    tabu_new(&tb, n, q, tenure);
    long *out = malloc(sizeof(long) * (q + 1)); int *moved = malloc(sizeof(int) * (n + 2));
    sparse_swap_loop(&S, &tb, out, moved, nsteps, beta_start, beta_max, E_io, best_io, best_pos, late_io, late_pos,
                     marks, seed, lambda);
    tabu_free(&tb);
    tree_free(&S.L); tree_free(&S.K); tree_free(&S.C);
    free(S.start); free(S.count); free(S.at); free(S.x); free(S.Fcur); free(S.tab); free(S.kappa); free(S.mark);
    free(S.list); free(S.nmark); free(S.scratch); free(S.ba); free(S.bb); free(S.fac); free(S.S); free(S.bmark);
    free(S.blist); free(S.x0c); free(S.fterm); free(S.fblk); free(S.pmark); free(out); free(moved);
}
