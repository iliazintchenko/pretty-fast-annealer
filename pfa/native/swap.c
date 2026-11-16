// PFA swap kernel (included by pfa.c): n variables that exchange values, with a fixed count of variables at each of
// q values. Variable r at value a_r costs f_r(a_r), and a pair of variables A_rs B(a_r, a_s): A (n x n, symmetric,
// zero diagonal) couples the variables, B (q x q, symmetric) the values. With
//     F_r(k) = f_r(k) + sum over s of A_rs B(k, a_s)
// the exchange of the values a, b of r and s costs
//     dE = (F_r(b) - F_r(a)) + (F_s(a) - F_s(b)) + A_rs kappa(a, b),    kappa(a, b) = 2 B(a, b) - B(a, a) - B(b, b).
// The variables sit in positions sorted by value: val[t] is the value of position t and never changes, pos[t] the
// variable there. A move exchanges the variables of two positions of different values, and D holds the dE of every
// such pair: row t covers the positions e[t] .. n - 1, those of the later values, padded to a multiple of 64 pairs
// with +inf (rs[t]: where the row starts). After the exchange of positions t and u (variables r and s, values a and
// b), with alpha_w = A_wr - A_ws for every variable and beta_k = B(k, b) - B(k, a) for every value,
//     F_w(k) += alpha_w beta_k                           for every w (A_rr = A_ss = 0 makes r and s fit),
//     D_xy -= (alpha_x - alpha_y) (beta_x - beta_y)      for x, y other than t and u (alpha, beta of their variables
//                                                        and values),
// and the pairs of t and of u are recomputed from F. The update of D is applied in the next step's pass, which
// scores every pair -dE - pen_x - pen_y + noise / beta, in float32 with the same operations on every SIMD path, and
// picks the first pair at the maximum; the recomputed pairs are stored with the update that pass will subtract
// added back. The tabu raises the return of a variable to a value it has just left by lambda / beta, for a tenure
// per value left: tabu_tenure times the mean number of steps between two moves of a variable at that value. The pair
// (t, u) moves pos[t] to val[u] and pos[u] to val[t], so the pass streams two penalty vectors over the columns u of a
// row t: R[pos[t]][u], the penalty of the row's variable at the column's value, and C[val[t]][u], the penalty of the
// column's variable at the row's value. Noise, schedule, lowest states and the energy are as in the other array
// kernels.

// One pair: the update of D, and the score, in float32. Every path does these operations in this order, so all paths
// give bit-identical results.
static inline double swap_upd(double d, double at, double au, double bt, double bu) {
    return d - (at - au) * (bt - bu);
}
static inline float swap_sc(double d, float p1, float p2, float g, float T) {
    return FMAF(g, T, (-(p1 + p2)) - (float)d);
}

#include SWAP_PASS   // swap_chunk(): 64 pairs of one row

typedef struct {
    int n, q;
    const double *A, *B;
    const int *val, *e;        // per position: its value, and the first position of its row of pairs
    const int64_t *rs;         // where each position's row starts in D, n + 1
    int *pos, *at;             // variable at each position; position of each variable
    double *F, *D, *Fcur;      // F: n x q by variable; D: the pairs; Fcur[t] = F[pos[t]][val[t]]
    double *kappa;             // q x q
    double *alpha, *beta;      // the pending update by variable and by value,
    double *ap, *bp;           // and by position (n + 64, zero beyond n)
    double *Fa, *Fb;           // F[w][a], F[w][b] of the last exchange by variable
    const int *start;          // first position of each value (0 for a value no variable has)
    int W;                     // n + 64: the width of R and C
    float *R, *C;              // tabu penalties: R[r][u] of variable r at val[u]; C[k][u] of pos[u] at value k
} Swap;

// The tabu queues, shared with sparse_swap.c: per value c, the variables that left c, with the step, in a ring of
// tau[c] + 1 entries at off[c] (at most one variable leaves a value per step). last[v q + c]: the step at which v
// last left c.
typedef struct { int q; const int64_t *tau; int64_t *off; int *var; long *when, *head, *tail, *last; } Tabu;

static void tabu_new(Tabu *tb, int n, int q, const int64_t *tenure) {
    tb->q = q; tb->tau = tenure;
    tb->off = malloc(sizeof(int64_t) * (q + 1));
    tb->off[0] = 0;
    for (int c = 0; c < q; c++) tb->off[c + 1] = tb->off[c] + tenure[c] + 1;
    tb->var = malloc(sizeof(int) * tb->off[q]); tb->when = malloc(sizeof(long) * tb->off[q]);
    tb->head = calloc(q, sizeof(long)); tb->tail = calloc(q, sizeof(long));
    tb->last = malloc(sizeof(long) * (long)n * q);
    for (long i = 0; i < (long)n * q; i++) tb->last[i] = -1;
}

static void tabu_free(Tabu *tb) {
    free(tb->off); free(tb->var); free(tb->when); free(tb->head); free(tb->tail); free(tb->last);
}

// The (variable, value) whose tenure is over at this step and that have not been renewed since, into out as
// variable * q + value; returns how many.
static int tabu_expire(Tabu *tb, long step, long *out) {
    int cnt = 0;
    for (int c = 0; c < tb->q; c++) {
        const long cap = tb->tau[c] + 1;
        while (tb->head[c] < tb->tail[c]) {
            const long i = tb->off[c] + tb->head[c] % cap;
            if (tb->when[i] > step - tb->tau[c]) break;
            tb->head[c]++;
            const long vc = (long)tb->var[i] * tb->q + c;
            if (tb->last[vc] == tb->when[i]) out[cnt++] = vc;
        }
    }
    return cnt;
}

// Variable v left value c at this step; returns whether its return to c is tabu now (value c has a tenure).
static int tabu_push(Tabu *tb, int v, int c, long step) {
    tb->last[(long)v * tb->q + c] = step;
    if (tb->tau[c] == 0) return 0;
    const long j = tb->off[c] + tb->tail[c] % (tb->tau[c] + 1);
    tb->var[j] = v; tb->when[j] = step; tb->tail[c]++;
    return 1;
}

// One pass: applies the pending update to D, scores every pair and returns the first pair at the maximum as its
// row *t_out and column *u_out.
static inline void swap_pass(const Swap *S, const float *g, float T, int *t_out, int *u_out) SVE_STREAMING {
    float bv = -__builtin_inff(); int64_t bf = 0; int bt = 0;
    for (int t = 0; t < S->n; t++) {
        const int64_t lo = S->rs[t], hi = S->rs[t + 1];
        const double at = S->ap[t], btv = S->bp[t];
        const float *p1 = S->R + (long)S->pos[t] * S->W, *p2 = S->C + (long)S->val[t] * S->W;
        for (int64_t j = lo; j < hi; j += 64) {
            const int64_t u0 = S->e[t] + (j - lo);
            const float v = swap_chunk(S->D + j, g + j, S->ap + u0, S->bp + u0, p1 + u0, p2 + u0, at, btv, T);
            if (v > bv) { bv = v; bf = j; bt = t; }
        }
    }
    const int64_t u0 = S->e[bt] + (bf - S->rs[bt]);
    const float *p1 = S->R + (long)S->pos[bt] * S->W, *p2 = S->C + (long)S->val[bt] * S->W;
    for (int i = 0; i < 64; i++) {
        if (swap_sc(S->D[bf + i], p1[u0 + i], p2[u0 + i], g[bf + i], T) != bv) continue;
        *t_out = bt; *u_out = (int)(u0 + i);
        return;
    }
    __builtin_trap();
}

// D of the current state, with no update pending.
static void swap_fill(Swap *S) {
    const int n = S->n, q = S->q;
    double *FT = malloc(sizeof(double) * (long)n * q);   // F by value, then variable
    for (int w = 0; w < n; w++) for (int k = 0; k < q; k++) FT[(long)k * n + w] = S->F[(long)w * q + k];
    for (int t = 0; t < n; t++) S->Fcur[t] = S->F[(long)S->pos[t] * q + S->val[t]];
    for (int t = 0; t < n; t++) {
        const int r = S->pos[t], a = S->val[t], e = S->e[t];
        const double *Ar = S->A + (long)r * n, *Fr = S->F + (long)r * q, *Fta = FT + (long)a * n;
        const double *ka = S->kappa + (long)a * q;
        double *row = S->D + S->rs[t];
        for (int u = e; u < n; u++) {
            const int s = S->pos[u], b = S->val[u];
            row[u - e] = (Fr[b] - S->Fcur[t]) + (Fta[s] - S->Fcur[u]) + Ar[s] * ka[b];
        }
        for (int64_t j = n - e; j < S->rs[t + 1] - S->rs[t]; j++) row[j] = __builtin_inf();
    }
    free(FT);
}

// Applies the pending update to D.
static void swap_settle(Swap *S) {
    for (int t = 0; t < S->n; t++) {
        const int64_t lo = S->rs[t], hi = S->rs[t + 1];
        for (int64_t j = lo; j < hi; j++) {
            const int64_t u = S->e[t] + (j - lo);
            S->D[j] = swap_upd(S->D[j], S->ap[t], S->ap[u], S->bp[t], S->bp[u]);
        }
    }
    for (int x = 0; x < S->n; x++) S->ap[x] = S->bp[x] = 0.0;
}

// Stores the dE of the pair of positions x, y with the pending update added back.
static inline void swap_store(Swap *S, int x, int y, double de) {
    if (x > y) { const int w = x; x = y; y = w; }
    S->D[S->rs[x] + (y - S->e[x])] = de + (S->ap[x] - S->ap[y]) * (S->bp[x] - S->bp[y]);
}

// The exchange of positions t and u: positions, the pending update, F, and the pairs of t and u.
static void swap_step(Swap *S, int t, int u) {
    const int n = S->n, q = S->q, r = S->pos[t], s = S->pos[u], a = S->val[t], b = S->val[u];
    const double *Ar = S->A + (long)r * n, *As = S->A + (long)s * n, *Ba = S->B + (long)a * q, *Bb = S->B + (long)b * q;
    const double *ka = S->kappa + (long)a * q, *kb = S->kappa + (long)b * q;
    S->pos[t] = s; S->pos[u] = r; S->at[s] = t; S->at[r] = u;
    for (int w = 0; w < n; w++) S->alpha[w] = Ar[w] - As[w];
    for (int k = 0; k < q; k++) S->beta[k] = Bb[k] - Ba[k];
    for (int x = 0; x < n; x++) { S->ap[x] = S->alpha[S->pos[x]]; S->bp[x] = S->beta[S->val[x]]; }
    for (int w = 0; w < n; w++) {
        double *Fw = S->F + (long)w * q; const double aw = S->alpha[w];
        for (int k = 0; k < q; k++) Fw[k] += aw * S->beta[k];
        S->Fa[w] = Fw[a]; S->Fb[w] = Fw[b];
    }
    for (int x = 0; x < n; x++) S->Fcur[x] += S->ap[x] * S->bp[x];
    const double *Fs = S->F + (long)s * q, *Fr = S->F + (long)r * q;
    S->Fcur[t] = Fs[a]; S->Fcur[u] = Fr[b];
    for (int y = 0; y < n; y++) {   // the pairs of t (now s at a), the pair (t, u) included
        const int d = S->val[y], z = S->pos[y];
        if (y != t && d != a) swap_store(S, t, y, (Fs[d] - S->Fcur[t]) + (S->Fa[z] - S->Fcur[y]) + As[z] * ka[d]);
    }
    for (int y = 0; y < n; y++) {   // the pairs of u (now r at b)
        const int d = S->val[y], z = S->pos[y];
        if (y != u && y != t && d != b)
            swap_store(S, u, y, (Fr[d] - S->Fcur[u]) + (S->Fb[z] - S->Fcur[y]) + Ar[z] * kb[d]);
    }
}

// Sets the penalty of variable v at value k: in R over the positions of k, in C at v's position.
static inline void swap_pen(Swap *S, int v, int k, float p) {
    float *Rv = S->R + (long)v * S->W;
    for (int x = S->start[k]; x < S->e[S->start[k]]; x++) Rv[x] = p;
    S->C[(long)k * S->W + S->at[v]] = p;
}

// The tabu after the exchange of positions t and u: the columns t and u of C follow their new variables, the
// (variable, value) whose tenure is over leave, and the two variables that moved become tabu at the values they left.
static void swap_tabu(Swap *S, Tabu *tb, long *out, int t, int u, long step, double beta, double lambda) {
    const int q = S->q, r = S->pos[u], s = S->pos[t];   // r now at u, s at t
    for (int k = 0; k < q; k++) {
        S->C[(long)k * S->W + t] = S->R[(long)s * S->W + S->start[k]];
        S->C[(long)k * S->W + u] = S->R[(long)r * S->W + S->start[k]];
    }
    const int cnt = tabu_expire(tb, step, out);
    for (int i = 0; i < cnt; i++) swap_pen(S, (int)(out[i] / q), (int)(out[i] % q), 0.0f);
    const int left[2] = {r, s}, from[2] = {S->val[t], S->val[u]};
    for (int i = 0; i < 2; i++)
        if (tabu_push(tb, left[i], from[i], step)) swap_pen(S, left[i], from[i], (float)(lambda / beta));
}

ANNEAL_ATTR static void swap_loop(Swap *S, Tabu *tb, long *out, const float *noise, long nsteps, double beta_start,
                                  double beta_max, double *E_io, double *best_io, int *best_pos, double *late_io,
                                  int *late_pos, uint64_t seed, double lambda) {
    const int n = S->n;
    const uint64_t P = (uint64_t)S->rs[n];
    double E = *E_io, best = *best_io, beta = beta_start, dbeta = (beta_max - beta_start) / (double)(nsteps - 1);
    uint64_t x = seed | 1;
    int at_best = 0;   // the current state is the best one of this half, not yet in best_pos
    for (long step = 0; step < nsteps; step++) {
        if (step == nsteps / 2) {   // the second half starts: its lowest state so far is the current one
            if (at_best) __builtin_memcpy(best_pos, S->pos, sizeof(int) * n);
            *best_io = best; best = E; best_pos = late_pos; at_best = 1;
        }
        beta += dbeta;
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        const float *g = noise + (int64_t)(((x * 2685821657736338717ULL) >> 32) * P >> 32);
        int t, u;
        swap_pass(S, g, (float)(1.0 / beta), &t, &u);
        const double dE = S->D[S->rs[t] + (u - S->e[t])];
        if (at_best && dE >= 0.0) { __builtin_memcpy(best_pos, S->pos, sizeof(int) * n); at_best = 0; }   // leaving it
        E += dE;
        swap_step(S, t, u);
        swap_tabu(S, tb, out, t, u, step, beta, lambda);
        if (E < best) { best = E; at_best = 1; }
    }
    if (at_best) __builtin_memcpy(best_pos, S->pos, sizeof(int) * n);
    swap_settle(S);
    *E_io = E; *late_io = best;
}

// A: n x n couplings between the variables; B: q x q table between the values; val, e: per position its value and
// the first position of its row of pairs; rs: n + 1 row starts in D (rows padded to multiples of 64); pos: the
// variable at each position (updated in place); F: n x q, the fields of every variable at every value given the
// others (updated in place); D: rs[n] pairs (out: the dE of every pair of the final state); noise: rs[n] noise values
// stored twice; best_pos, late_pos: n positions each; tenure: per value, in steps (0: no tabu at that value). As
// pfa_anneal_categorical otherwise.
void pfa_anneal_swap(const double *A, const double *B, int n, int q, const int *val, const int *e, const int64_t *rs,
                     int *pos, double *F, double *D, const float *noise, long nsteps, double beta_start,
                     double beta_max, double *E_io, double *best_io, int *best_pos, double *late_io, int *late_pos,
                     uint64_t seed, double lambda, const int64_t *tenure) {
    int *start = calloc(q, sizeof(int));   // first position of each value present
    for (int t = n - 1; t >= 0; t--) start[val[t]] = t;
    Swap S = {n, q, A, B, val, e, rs, pos, malloc(sizeof(int) * n), F, D, malloc(sizeof(double) * n),
              malloc(sizeof(double) * (long)q * q), malloc(sizeof(double) * n), malloc(sizeof(double) * q),
              calloc(n + 64, sizeof(double)), calloc(n + 64, sizeof(double)), malloc(sizeof(double) * n),
              malloc(sizeof(double) * n), start, n + 64, calloc((long)n * (n + 64), sizeof(float)),
              calloc((long)q * (n + 64), sizeof(float))};
    for (int t = 0; t < n; t++) S.at[pos[t]] = t;
    for (int a = 0; a < q; a++)
        for (int b = 0; b < q; b++)
            S.kappa[(long)a * q + b] = 2.0 * B[(long)a * q + b] - B[(long)a * q + a] - B[(long)b * q + b];
    swap_fill(&S);
    Tabu tb;
    tabu_new(&tb, n, q, tenure);
    long *out = malloc(sizeof(long) * (q + 1));   // the (variable, value) leaving the tabu at a step: one per value
    swap_loop(&S, &tb, out, noise, nsteps, beta_start, beta_max, E_io, best_io, best_pos, late_io, late_pos, seed,
              lambda);
    free(S.at); free(S.Fcur); free(S.kappa); free(S.alpha); free(S.beta); free(S.ap); free(S.bp); free(S.Fa);
    free(S.Fb); free(start); free(S.R); free(S.C); free(out);
    tabu_free(&tb);
}
