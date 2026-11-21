// Tours: a cycle through n cities given by coordinates (included by pfa.c). The structure the tour kernel works on:
// the city at each position and the position of each city, each city's two tour neighbours, and for each (city a,
// candidate c) pair the length changes of the four ways to cut one edge at a and one at c and reconnect, kept up to
// date for the cities whose edges change. Distances come from the coordinates, with TSPLIB's rounding on request
// (round_mode 0 none, 1 nearest integer, 2 ceiling).
//
// The table is orientation-free (Helsgaun's trick in a small form): nb[2a], nb[2a + 1] are a's neighbours in a fixed
// order and o[a] says whether nb[2a] is a's successor, so a reversal of a path flips o for its cities and leaves their
// entries alone; exactly two of a pair's four entries are tours, (i, j) in {(0, 0), (1, 1)} when a and c face the
// same way along the tour and {(0, 1), (1, 0)} otherwise.
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int n, K, round_mode;
    const double *x, *y;
    const int *nn;        // n x K candidates
    int *tour, *pos;      // city at each position; position of each city
    int *nb;              // 2n: the two tour neighbours of each city, in the table's order
    unsigned char *o;     // per city: nb[2z] is its successor (0) or its predecessor (1)
    int *inv_start, *inv; // the (a, k) pairs naming each city as the candidate c, CSR by c
    float *q;             // 4 per (a, k): dL of the four cuts
    double L;
    const double *pi;     // the ascent's city penalties, or NULL: the chain's gains are read on the penalised distances
} Tour;

static inline double tour_d(const Tour *T, int i, int j) {
    double dx = T->x[i] - T->x[j], dy = T->y[i] - T->y[j], r = sqrt(dx * dx + dy * dy);
    if (T->round_mode == 1) return floor(r + 0.5);
    if (T->round_mode == 2) return ceil(r);
    return r;
}
static inline double tour_c(const Tour *T, int i, int j) {   // the penalised distance: every tour changes by the same 2 sum pi
    return T->pi ? tour_d(T, i, j) + T->pi[i] + T->pi[j] : tour_d(T, i, j);
}
static inline int tour_succ(const Tour *T, int a) { int p = T->pos[a] + 1; return T->tour[p == T->n ? 0 : p]; }
static inline int tour_pred(const Tour *T, int a) { int p = T->pos[a] - 1; return T->tour[p < 0 ? T->n - 1 : p]; }
static inline int tour_adjacent(const Tour *T, int u, int v) { return T->nb[2 * u] == v || T->nb[2 * u + 1] == v; }

static void tour_set_nb(Tour *T, int a) {
    int s = tour_succ(T, a), p = tour_pred(T, a);
    if (T->nb[2 * a] == s || T->nb[2 * a + 1] == p) { T->nb[2 * a] = s; T->nb[2 * a + 1] = p; T->o[a] = 0; }
    else { T->nb[2 * a] = p; T->nb[2 * a + 1] = s; T->o[a] = 1; }
}

static void tour_pair(Tour *T, int a, int k) {
    int c = T->nn[a * T->K + k];
    float *Q = T->q + 4 * (a * T->K + k);
    double dac = tour_d(T, a, c);
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++) {
            int xx = T->nb[2 * a + i], yy = T->nb[2 * c + j];
            int bad = (xx == c || yy == a || xx == yy);
            Q[2 * i + j] = bad ? INFINITY : (float)(dac + tour_d(T, xx, yy) - tour_d(T, a, xx) - tour_d(T, c, yy));
        }
}

static void tour_refresh(Tour *T, int z) {
    for (int k = 0; k < T->K; k++) tour_pair(T, z, k);
    for (int r = T->inv_start[z]; r < T->inv_start[z + 1]; r++) tour_pair(T, T->inv[r] / T->K, T->inv[r] % T->K);
}

static Tour tour_new(int n, int K, int round_mode, const double *x, const double *y, const int *nn, int *tour_io) {
    Tour T = {n, K, round_mode, x, y, nn, tour_io, malloc(sizeof(int) * n), malloc(sizeof(int) * 2 * n), malloc(n),
              calloc(n + 1, sizeof(int)), malloc(sizeof(int) * n * K), malloc(sizeof(float) * 4 * n * K), 0.0, NULL};
    for (int k = 0; k < n; k++) T.pos[T.tour[k]] = k;
    for (int a = 0; a < n; a++) { T.nb[2 * a] = tour_succ(&T, a); T.nb[2 * a + 1] = tour_pred(&T, a); T.o[a] = 0; }
    for (int m = 0; m < n * K; m++) T.inv_start[nn[m] + 1]++;
    for (int a = 0; a < n; a++) T.inv_start[a + 1] += T.inv_start[a];
    int *fill = calloc(n, sizeof(int));
    for (int m = 0; m < n * K; m++) { int c = nn[m]; T.inv[T.inv_start[c] + fill[c]++] = m; }
    free(fill);
    for (int a = 0; a < n; a++) for (int k = 0; k < K; k++) tour_pair(&T, a, k);
    for (int a = 0; a < n; a++) T.L += tour_d(&T, a, tour_succ(&T, a));
    return T;
}

static void tour_free(Tour *T) { free(T->pos); free(T->nb); free(T->o); free(T->inv_start); free(T->inv); free(T->q); }
