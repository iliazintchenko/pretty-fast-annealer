// (Weighted partial) Max-SAT: an assignment of n Boolean variables; the hard clauses must hold, the energy is the
// weight of the soft clauses falsified. The state keeps per clause the count of its true literals and the one variable
// satisfying it when that count is one, per variable its score (the weighted change in falsified clauses if it
// flips, positive when flipping helps), the variables of positive score as a list with positions, and the falsified
// hard and soft clauses as two lists with positions, so a flip's update is one pass over the variable's clauses. The
// kernel is noSAT-MaxSATv3's (Lübke, MSE 2024), the NuWLS-c-2023 local search written without a SAT solver, as its
// code runs it: at zero temperature the variable of greatest score, sampled from the positive-score list when that
// list is longer than the sample size (best from multiple selections), ties to the first drawn; at a local optimum
// every falsified hard clause's weight rises by h_inc and, when every hard clause holds, every soft clause's by its
// increment (its weight over the mean weight in the weighted case), and a falsified clause is drawn at random, hard
// first, and its least-flipped variable is flipped (frequency fitness). A try is max_flips flips from a start by
// decimation (unit propagation with random choices under a best-of-sample rule on the variables longest unassigned
// in the previous decimation) of the best assignment, or of a random one after three tries without improvement; a
// try that improved continues without a restart. The weights are the search's memory in place of a tabu.
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MS_NONE UINT32_MAX

typedef struct {
    uint32_t n, m, mall, nh;       // variables, clauses in use (hard first), clauses in all, hard clauses
    const int32_t *off, *lits;     // clauses as literal ranges: lits[off[c]..off[c+1]), a literal 2v + negated
    const int64_t *w;              // original weights, 0 for hard
    int32_t *pcl;                  // the clause of each literal position
    int32_t *voff, *vlit;          // per variable the literal positions it occurs at (CSR), in the order given
    uint8_t *val;                  // the assignment
    uint64_t *cw, *tuned;          // clause weights (fixed point), the soft increments
    uint32_t *nsat, *satvar;       // true literals per clause, the one satisfying variable when nsat == 1
    int64_t *score;
    uint32_t *dec, *decpos, ndec;                   // positive-score variables
    uint32_t *fh, *fhpos, nfh;                      // falsified hard clauses
    uint32_t *fs, *fspos, nfs;                      // falsified soft clauses
    uint32_t *hw, *hwpos, nhw;                      // soft clauses of raised weight (non-partial only)
    uint64_t *flips;                                // flip counts
    uint64_t cost;                                  // weight of the soft clauses falsified now
    // decimation
    uint32_t *un, *unpos, nun;                      // unassigned variables
    int32_t *hu, *huposc; uint32_t nhu;             // hard unit clauses as literal positions, the position per clause
    int32_t *su, *suposc; uint32_t nsu;             // soft unit clauses likewise
    uint32_t *prevdec;                              // the step a variable was assigned at in the previous decimation
    uint8_t *crem; uint32_t *cnl;                   // clause removed, literals left
    // parameters (nsms_params)
    int partial, weighted;
    uint64_t bms, hinc, sinc, sprob, scoef, soff, max_tries_wo;
    unsigned shift;
    // the result
    uint8_t *best; uint64_t best_cost; int status;  // 0 optimum, 1 feasible, 3 infeasible
    uint64_t total_flips, best_flip, budget, tries; double best_time;
    int32_t *trace; long ntrace, ktrace;
    uint64_t rng;
    double t0, seconds;
} MS;

static double ms_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

#ifdef PFA_MAXSAT_LIBC_RAND   // the certificate build: the C library's generator in the reference's call order
static inline uint32_t ms_rnd(MS *S, uint32_t n) { (void)S; return (uint32_t)(rand() % n); }
#else
static inline uint32_t ms_rnd(MS *S, uint32_t n) {
    S->rng ^= S->rng >> 12; S->rng ^= S->rng << 25; S->rng ^= S->rng >> 27;
    return (uint32_t)((((S->rng * 2685821657736338717ULL) >> 32) * (uint64_t)n) >> 32);
}
#endif
static inline int ms_coin(MS *S, uint64_t bias) { return ms_rnd(S, 100000000u) < bias; }   // bias in 1e-8

static inline uint32_t ms_var(int32_t lit) { return (uint32_t)lit >> 1; }
static inline int ms_true(const MS *S, int32_t lit) { return S->val[lit >> 1] != (lit & 1); }
static inline int ms_hard(const MS *S, uint32_t c) { return c < S->nh; }
static inline uint32_t ms_clause_of(const MS *S, int32_t pos) { return (uint32_t)S->pcl[pos]; }

static inline void ms_ladd(uint32_t x, uint32_t *list, uint32_t *pos, uint32_t *n) { list[*n] = x; pos[x] = (*n)++; }
static inline void ms_ldel(uint32_t x, uint32_t *list, uint32_t *pos, uint32_t *n) {
    uint32_t p = pos[x]; pos[x] = MS_NONE; (*n)--;
    if (p < *n) { uint32_t y = list[*n]; list[p] = y; pos[y] = p; }
}
static inline void ms_dec_add(MS *S, uint32_t v) { if (S->decpos[v] == MS_NONE && S->score[v] > 0) ms_ladd(v, S->dec, S->decpos, &S->ndec); }
static inline void ms_dec_del(MS *S, uint32_t v) { if (S->decpos[v] != MS_NONE && S->score[v] <= 0) ms_ldel(v, S->dec, S->decpos, &S->ndec); }

static int ms_feasible(const MS *S) {
    for (uint32_t c = 0; c < S->nh; c++) {
        int sat = 0;
        for (int32_t p = S->off[c]; p < S->off[c + 1] && !sat; p++) sat = ms_true(S, S->lits[p]);
        if (!sat) return 0;
    }
    return 1;
}

static int ms_done(const MS *S) { return S->status == 0 || (S->nfh == 0 && S->nfs == 0); }

static int ms_update_result(MS *S, int infeasible) {
    if (!infeasible && S->cost < S->best_cost) {
        memcpy(S->best, S->val, S->n);
        S->best_cost = S->cost; S->best_flip = S->total_flips; S->best_time = ms_now() - S->t0;
        S->status = S->cost == 0 ? 0 : 1;
        return 1;
    }
    return 0;
}

static void ms_init_vars(MS *S, int force_random) {
    if (!force_random && S->status != 3) memcpy(S->val, S->best, S->n);
    else for (uint32_t v = 0; v < S->n; v++) S->val[v] = (uint8_t)ms_coin(S, 50000000u);
}

// ---- decimation (nsms_decimation) ----
static void ms_unit_add(MS *S, int32_t pos) {
    uint32_t c = ms_clause_of(S, pos);
    if (ms_hard(S, c)) { S->hu[S->nhu] = pos; S->huposc[c] = (int32_t)S->nhu++; }
    else { S->su[S->nsu] = pos; S->suposc[c] = (int32_t)S->nsu++; }
}
static void ms_unit_del(MS *S, uint32_t c) {
    int32_t *list, *posc; uint32_t *n;
    if (ms_hard(S, c)) { list = S->hu; posc = S->huposc; n = &S->nhu; } else { list = S->su; posc = S->suposc; n = &S->nsu; }
    int32_t p = posc[c]; posc[c] = -1; (*n)--;
    if ((uint32_t)p < *n) { int32_t mv = list[*n]; list[p] = mv; posc[ms_clause_of(S, mv)] = p; }
}

static void ms_decimation(MS *S, int reuse) {
    S->nun = S->n;
    for (uint32_t v = 0; v < S->n; v++) { S->un[v] = v; S->unpos[v] = v; if (!reuse) S->prevdec[v] = MS_NONE; }
    S->nhu = S->nsu = 0;
    for (uint32_t c = 0; c < S->m; c++) {
        S->huposc[c] = S->suposc[c] = -1; S->crem[c] = 0; S->cnl[c] = (uint32_t)(S->off[c + 1] - S->off[c]);
        if (S->cnl[c] == 1) ms_unit_add(S, S->off[c]);
    }
    for (uint32_t it = 0; it < S->n; it++) {
        uint32_t v;
        if (S->nhu > 0 || S->nsu > 0) {
            int32_t *list = S->nhu > 0 ? S->hu : S->su; uint32_t nl = S->nhu > 0 ? S->nhu : S->nsu;
            int32_t lit = S->lits[list[ms_rnd(S, nl)]]; uint32_t pd = S->prevdec[ms_var(lit)];
            for (uint64_t i = 1; i < S->bms; i++) {
                int32_t cl = S->lits[list[ms_rnd(S, nl)]]; uint32_t cd = S->prevdec[ms_var(cl)];
                if (cd > pd) { pd = cd; lit = cl; }
            }
            v = ms_var(lit); S->val[v] = (uint8_t)!(lit & 1);
        } else {
            v = S->un[ms_rnd(S, S->nun)]; uint32_t pd = S->prevdec[v];
            for (uint64_t i = 1; i < S->bms; i++) {
                uint32_t cv = S->un[ms_rnd(S, S->nun)], cd = S->prevdec[cv];
                if (cd > pd) { pd = cd; v = cv; }
            }
        }
        ms_ldel(v, S->un, S->unpos, &S->nun);
        S->prevdec[v] = it;
        for (int32_t i = S->voff[v]; i < S->voff[v + 1]; i++) {        // simplify
            int32_t pos = S->vlit[i]; uint32_t c = ms_clause_of(S, pos);
            if (c >= S->m || S->crem[c]) continue;
            if (ms_true(S, S->lits[pos])) {
                S->crem[c] = 1;
                if (S->cnl[c] == 1) ms_unit_del(S, c);
            } else {
                S->cnl[c]--;
                if (S->cnl[c] == 1) {
                    int32_t unit = -1;
                    for (int32_t p = S->off[c]; p < S->off[c + 1]; p++) if (S->unpos[ms_var(S->lits[p])] != MS_NONE) { unit = p; break; }
                    if (unit >= 0) ms_unit_add(S, unit); else S->crem[c] = 1;
                } else if (S->cnl[c] == 0) { S->crem[c] = 1; ms_unit_del(S, c); }
            }
        }
    }
}

// ---- weights ----
static inline uint64_t ms_udiv(uint64_t a, uint64_t b, unsigned shift) { return a / (b >> shift); }
static inline uint64_t ms_umul(uint64_t a, uint64_t b, unsigned shift) { if (a >= b) a >>= shift; else b >>= shift; return a * b; }

static void ms_hw_reset(MS *S) { S->nhw = 0; for (uint32_t c = 0; c < S->m; c++) S->hwpos[c] = MS_NONE; }

static void ms_init_weights(MS *S, uint64_t avg) {
    uint64_t one = 1ULL << S->shift, sinc = S->sinc << S->shift, coef = S->scoef << S->shift;
    if (S->partial) {
        for (uint32_t c = 0; c < S->nh; c++) S->cw[c] = one;
        for (uint32_t c = S->nh; c < S->m; c++) {
            if (S->weighted) { S->tuned[c] = ms_udiv((uint64_t)S->w[c] << S->shift, avg, S->shift); S->cw[c] = 0; }
            else S->cw[c] = one;
        }
    } else {
        ms_hw_reset(S);
        for (uint32_t c = 0; c < S->m; c++) {
            S->tuned[c] = S->weighted ? ms_umul(coef, ms_udiv((uint64_t)S->w[c] << S->shift, avg, S->shift), S->shift) : coef;
            S->cw[c] = S->tuned[c];
            if (S->cw[c] > sinc) ms_ladd(c, S->hw, S->hwpos, &S->nhw);
        }
    }
}

static void ms_update_scores(MS *S, uint32_t c, int64_t inc) {
    if (S->fspos[c] != MS_NONE) {
        for (int32_t p = S->off[c]; p < S->off[c + 1]; p++) { uint32_t v = ms_var(S->lits[p]); S->score[v] += inc; ms_dec_add(S, v); }
    } else if (S->nsat[c] == 1) { uint32_t v = S->satvar[c]; S->score[v] -= inc; ms_dec_del(S, v); }
}

static void ms_update_weights(MS *S) {
    if (S->partial) {
        uint64_t inc = S->hinc << S->shift;
        for (uint32_t i = 0; i < S->nfh; i++) {
            uint32_t c = S->fh[i]; S->cw[c] += inc;
            for (int32_t p = S->off[c]; p < S->off[c + 1]; p++) { uint32_t v = ms_var(S->lits[p]); S->score[v] += (int64_t)inc; ms_dec_add(S, v); }
        }
        if (S->nfh == 0) {
            uint64_t sinc = S->sinc << S->shift;
            for (uint32_t c = S->nh; c < S->m; c++) {
                uint64_t d = S->weighted ? S->tuned[c] : sinc;
                S->cw[c] += d; ms_update_scores(S, c, (int64_t)d);
            }
        }
    } else if (ms_coin(S, S->sprob)) {               // smooth
        uint64_t dec = S->sinc << S->shift;
        for (long i = 0; i < (long)S->nhw; i++) {
            uint32_t c = S->hw[i];
            if (S->fspos[c] == MS_NONE) {
                S->cw[c] -= dec;
                if (S->cw[c] < dec) { ms_ldel(c, S->hw, S->hwpos, &S->nhw); i--; }
                if (S->nsat[c] == 1) { uint32_t v = S->satvar[c]; S->score[v] += (int64_t)dec; ms_dec_add(S, v); }
            }
        }
    } else {                                           // increase the falsified soft clauses under their bounds
        uint64_t inc = S->sinc << S->shift, offb = S->soff << S->shift;
        for (uint32_t i = 0; i < S->nfs; i++) {
            uint32_t c = S->fs[i];
            if (S->cw[c] >= S->tuned[c] + offb) continue;
            S->cw[c] += inc;
            if (S->hwpos[c] == MS_NONE) ms_ladd(c, S->hw, S->hwpos, &S->nhw);
            ms_update_scores(S, c, (int64_t)inc);
        }
    }
}

// ---- the state from the assignment and the weights ----
static void ms_init_algo(MS *S) {
    S->cost = 0; S->ndec = 0; S->nfh = 0; S->nfs = 0;
    for (uint32_t v = 0; v < S->n; v++) { S->score[v] = 0; S->decpos[v] = MS_NONE; }
    for (uint32_t c = 0; c < S->m; c++) {
        S->fhpos[c] = S->fspos[c] = MS_NONE;
        uint32_t k = 0;
        for (int32_t p = S->off[c]; p < S->off[c + 1]; p++) if (ms_true(S, S->lits[p])) { if (++k == 1) S->satvar[c] = ms_var(S->lits[p]); }
        S->nsat[c] = k;
        int64_t cs = (int64_t)S->cw[c];
        if (k == 1) S->score[S->satvar[c]] -= cs;
        else if (k == 0) {
            for (int32_t p = S->off[c]; p < S->off[c + 1]; p++) S->score[ms_var(S->lits[p])] += cs;
            if (ms_hard(S, c)) ms_ladd(c, S->fh, S->fhpos, &S->nfh);
            else { ms_ladd(c, S->fs, S->fspos, &S->nfs); S->cost += (uint64_t)S->w[c]; }
        }
    }
    for (uint32_t v = 0; v < S->n; v++) ms_dec_add(S, v);
}

static uint32_t ms_select(MS *S) {
    if (S->ndec > S->bms) {
        uint32_t b = S->dec[ms_rnd(S, S->ndec)]; int64_t bs = S->score[b];
        for (uint64_t i = 1; i < S->bms; i++) { uint32_t v = S->dec[ms_rnd(S, S->ndec)]; if (S->score[v] > bs) { b = v; bs = S->score[v]; } }
        return b;
    }
    if (S->ndec > 0) {
        uint32_t b = S->dec[0]; int64_t bs = S->score[b];
        for (uint32_t i = 1; i < S->ndec; i++) { uint32_t v = S->dec[i]; if (S->score[v] > bs) { b = v; bs = S->score[v]; } }
        return b;
    }
    ms_update_weights(S);
    uint32_t c = S->nfh > 0 ? S->fh[ms_rnd(S, S->nfh)] : S->fs[ms_rnd(S, S->nfs)];
    uint32_t b = ms_var(S->lits[S->off[c]]);
    for (int32_t p = S->off[c] + 1; p < S->off[c + 1]; p++) { uint32_t v = ms_var(S->lits[p]); if (S->flips[v] < S->flips[b]) b = v; }
    return b;
}

static void ms_flip(MS *S, uint32_t v) {
    S->val[v] = !S->val[v]; S->flips[v]++; S->total_flips++;
    if (S->trace && S->ktrace < S->ntrace) S->trace[S->ktrace++] = (int32_t)v;
    for (int32_t i = S->voff[v]; i < S->voff[v + 1]; i++) {
        int32_t pos = S->vlit[i]; uint32_t c = ms_clause_of(S, pos);
        if (c >= S->m) continue;
        int64_t cs = (int64_t)S->cw[c];
        if (ms_true(S, S->lits[pos])) {
            if (++S->nsat[c] == 1) {
                S->score[v] -= cs;
                for (int32_t p = S->off[c]; p < S->off[c + 1]; p++) { uint32_t u = ms_var(S->lits[p]); S->score[u] -= cs; ms_dec_del(S, u); }
                S->satvar[c] = v;
                if (ms_hard(S, c)) ms_ldel(c, S->fh, S->fhpos, &S->nfh);
                else { S->cost -= (uint64_t)S->w[c]; ms_ldel(c, S->fs, S->fspos, &S->nfs); }
            } else if (S->nsat[c] == 2) { uint32_t u = S->satvar[c]; S->score[u] += cs; ms_dec_add(S, u); }
        } else {
            if (--S->nsat[c] == 1) {
                for (int32_t p = S->off[c]; p < S->off[c + 1]; p++) if (ms_true(S, S->lits[p])) {
                    uint32_t u = ms_var(S->lits[p]); S->score[u] -= cs; ms_dec_del(S, u); S->satvar[c] = u; break;
                }
            } else if (S->nsat[c] == 0) {
                S->score[S->satvar[c]] += cs;
                for (int32_t p = S->off[c]; p < S->off[c + 1]; p++) { uint32_t u = ms_var(S->lits[p]); S->score[u] += cs; ms_dec_add(S, u); }
                S->satvar[c] = MS_NONE;
                if (ms_hard(S, c)) ms_ladd(c, S->fh, S->fhpos, &S->nfh);
                else { S->cost += (uint64_t)S->w[c]; ms_ladd(c, S->fs, S->fspos, &S->nfs); }
            }
        }
    }
}

static inline int ms_stop(const MS *S) {
    return S->total_flips >= S->budget || (S->seconds > 0 && (S->total_flips & 1023) == 0 && ms_now() - S->t0 >= S->seconds);
}

// nsms_params: the reference's settings by the instance's kind
static void ms_params(MS *S) {
    S->max_tries_wo = S->partial ? 3 : 2;
    if (!S->weighted) {
        S->sinc = 1; S->hinc = 1;
        if (S->nh == 0) { S->bms = 94; S->sprob = 200000; S->scoef = 397; S->soff = 550; }
        else { S->bms = 50; S->sprob = 0; S->scoef = 0; S->soff = 0; }
    } else {
        if (S->nh == 0) { S->bms = 22; S->hinc = 0; S->sinc = 3; S->sprob = 100000; S->scoef = 1000; S->soff = 10; }
        else { S->bms = 50; S->hinc = 5; S->sinc = 0; S->sprob = 0; S->scoef = 0; S->soff = 0; }
    }
}

// The hard clauses alone until they hold (p_initVars → solveHard): the clauses in use are the hard ones, the
// parameters the unweighted partial ones, random starts, each try max_flips flips.
static void ms_solve_hard(MS *S, uint64_t max_flips, int force_random, int *reuse) {
    uint32_t m = S->m; S->m = S->nh;
    MS saved = *S;                                    // parameters only; the arrays are shared
    S->weighted = 0; S->partial = 1; ms_params(S);
    S->nfh = MS_NONE;                                 // ms_done false on the first check
    while (!ms_done(S) && !ms_stop(S)) {
        ms_init_vars(S, force_random);
        ms_decimation(S, *reuse); *reuse = 1;
        for (uint32_t c = 0; c < S->m; c++) S->cw[c] = 1ULL << S->shift;
        ms_init_algo(S);
        for (uint64_t f = 0; f < max_flips && !ms_done(S) && !ms_stop(S); f++) ms_flip(S, ms_select(S));
        S->tries++;
    }
    S->weighted = saved.weighted; S->partial = saved.partial; S->bms = saved.bms; S->hinc = saved.hinc; S->sinc = saved.sinc;
    S->sprob = saved.sprob; S->scoef = saved.scoef; S->soff = saved.soff; S->max_tries_wo = saved.max_tries_wo;
    S->m = m;
}

// n variables, m clauses with hard ones first (nh of them), as literal ranges off[m + 1] into lits (a literal is
// 2 v + negated); w[m] the weights, 0 for hard; occ_order, if given, the order the clauses' literals are appended to
// the variables' occurrence lists (the reference appends in file order; NULL for 0..m-1); no clause is empty, has a
// repeated variable, and every soft weight is positive. The search runs max_flips flips per try until budget flips
// in all, seconds, or a feasible assignment of cost at most target. Writes the best assignment to best_out (0/1 per
// variable) and the first ntrace flipped variables to trace, returns the best cost (-1 if no feasible assignment).
// stats: flips, tries, the flip of the best, seconds, status, seconds at the best.
int64_t pfa_maxsat_run(int n, int m, int nh, const int32_t *off, const int32_t *lits, const int64_t *w, const int32_t *occ_order,
                       long max_flips, long budget, double seconds, int64_t target, uint64_t seed,
                       uint8_t *best_out, int32_t *trace, long ntrace, double *stats) {
    MS S; memset(&S, 0, sizeof S);
    S.n = (uint32_t)n; S.m = S.mall = (uint32_t)m; S.nh = (uint32_t)nh; S.off = off; S.lits = lits; S.w = w;
    S.budget = (uint64_t)budget; S.seconds = seconds; S.t0 = ms_now(); S.trace = trace; S.ntrace = ntrace;
    S.rng = (seed << 1) | 1;
#ifdef PFA_MAXSAT_LIBC_RAND
    srand((unsigned)seed);
#endif
    int32_t L = off[m];
    S.pcl = malloc(L * sizeof(int32_t));
    for (int c = 0; c < m; c++) for (int32_t p = off[c]; p < off[c + 1]; p++) S.pcl[p] = c;
    S.voff = calloc(n + 1, sizeof(int32_t)); S.vlit = malloc(L * sizeof(int32_t));
    for (int32_t p = 0; p < L; p++) S.voff[ms_var(lits[p]) + 1]++;
    for (int v = 0; v < n; v++) S.voff[v + 1] += S.voff[v];
    int32_t *fill = malloc(n * sizeof(int32_t)); memcpy(fill, S.voff, n * sizeof(int32_t));
    for (int i = 0; i < m; i++) {
        int c = occ_order ? occ_order[i] : i;
        for (int32_t p = off[c]; p < off[c + 1]; p++) S.vlit[fill[ms_var(lits[p])]++] = p;
    }
    free(fill);
    S.val = calloc(n, 1); S.best = best_out;
    S.cw = malloc(m * sizeof(uint64_t)); S.tuned = malloc(m * sizeof(uint64_t));
    S.nsat = malloc(m * sizeof(uint32_t)); S.satvar = malloc(m * sizeof(uint32_t));
    S.score = malloc(n * sizeof(int64_t));
    S.dec = malloc(n * sizeof(uint32_t)); S.decpos = malloc(n * sizeof(uint32_t));
    S.fh = malloc(m * sizeof(uint32_t)); S.fhpos = malloc(m * sizeof(uint32_t));
    S.fs = malloc(m * sizeof(uint32_t)); S.fspos = malloc(m * sizeof(uint32_t));
    S.hw = malloc(m * sizeof(uint32_t)); S.hwpos = malloc(m * sizeof(uint32_t));
    S.flips = calloc(n, sizeof(uint64_t));
    S.un = malloc(n * sizeof(uint32_t)); S.unpos = malloc(n * sizeof(uint32_t));
    S.hu = malloc(m * sizeof(int32_t)); S.huposc = malloc(m * sizeof(int32_t));
    S.su = malloc(m * sizeof(int32_t)); S.suposc = malloc(m * sizeof(int32_t));
    S.prevdec = malloc(n * sizeof(uint32_t)); S.crem = malloc(m); S.cnl = malloc(m * sizeof(uint32_t));

    S.partial = nh > 0; S.weighted = 0;
    uint64_t total = 0;
    for (int c = nh; c < m; c++) { total += (uint64_t)w[c]; if (w[c] != 1) S.weighted = 1; }
    ms_params(&S);
    S.shift = 0;
    if (S.weighted) { unsigned msb = 32; uint64_t t = total; while (t >>= 1) msb++; int r = 63 - (int)msb; S.shift = r > 0 ? (unsigned)r : 0; }
    uint32_t nsoft = (uint32_t)(m - nh);
    uint64_t avg = nsoft > 0 ? ms_udiv(total << S.shift, (uint64_t)nsoft << S.shift, S.shift) : 0;

    S.status = 3; S.best_cost = UINT64_MAX;
    int reuse = 0, skip = 0; uint64_t wo = 0;
    S.nfh = MS_NONE;
    while (!ms_done(&S) && !ms_stop(&S)) {
        if (!skip) {
            int force = wo >= S.max_tries_wo;
            if (S.partial) { if (!ms_feasible(&S)) ms_solve_hard(&S, (uint64_t)max_flips, force, &reuse); }   // p_initVars: the current assignment stands when feasible
            else ms_init_vars(&S, force);
            ms_decimation(&S, reuse); reuse = 1;
            ms_init_weights(&S, avg);
            ms_init_algo(&S);
        }
        skip = 0;
        for (uint64_t f = 0; f < (uint64_t)max_flips && !ms_done(&S) && !ms_stop(&S); f++) {
            skip |= ms_update_result(&S, S.nfh > 0);
            if (S.status != 3 && (int64_t)S.best_cost <= target) break;
            ms_flip(&S, ms_select(&S));
        }
        skip |= ms_update_result(&S, S.nfh > 0);
        if (S.status != 3 && (int64_t)S.best_cost <= target) break;
        wo = skip ? 0 : wo + 1;
        S.tries++;
    }
    if (stats) { stats[0] = (double)S.total_flips; stats[1] = (double)S.tries; stats[2] = (double)S.best_flip; stats[3] = ms_now() - S.t0; stats[4] = S.status; stats[5] = S.best_time; }
    free(S.pcl); free(S.voff); free(S.vlit); free(S.val); free(S.cw); free(S.tuned); free(S.nsat); free(S.satvar); free(S.score);
    free(S.dec); free(S.decpos); free(S.fh); free(S.fhpos); free(S.fs); free(S.fspos); free(S.hw); free(S.hwpos); free(S.flips);
    free(S.un); free(S.unpos); free(S.hu); free(S.huposc); free(S.su); free(S.suposc); free(S.prevdec); free(S.crem); free(S.cnl);
    return S.status == 3 ? -1 : (int64_t)S.best_cost;
}
