// Tours: a cycle through n cities given by coordinates (included by pfa.c). The structure the tour kernel works on:
// the city at each position and the position of each city, each city's two tour neighbours with the lengths of those
// two edges, and the length of every candidate edge, so that the chain's enumeration reads lengths and computes one
// distance per option, the closing edge's. Distances come from the coordinates, with TSPLIB's rounding on request
// (round_mode 0 none, 1 nearest integer, 2 ceiling).
//
// The neighbours are orientation-free (Helsgaun's trick in a small form): nb[2a], nb[2a + 1] are a's neighbours in a
// fixed order and o[a] says whether nb[2a] is a's successor, so a reversal of a path flips o for its cities and leaves
// their entries alone; only the ends of an exchange's segments get new neighbours.
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
    double *cc, *ec;      // the penalised length of each candidate edge (n x K) and of each city's two tour edges (2n, nb's order)
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
    T->ec[2 * a] = tour_c(T, a, T->nb[2 * a]); T->ec[2 * a + 1] = tour_c(T, a, T->nb[2 * a + 1]);
}

static Tour tour_new(int n, int K, int round_mode, const double *x, const double *y, const int *nn, int *tour_io, const double *pi) {
    Tour T = {n, K, round_mode, x, y, nn, tour_io, malloc(sizeof(int) * n), malloc(sizeof(int) * 2 * n), malloc(n),
              malloc(sizeof(double) * n * K), malloc(sizeof(double) * 2 * n), 0.0, pi};
    for (int k = 0; k < n; k++) T.pos[T.tour[k]] = k;
    for (int a = 0; a < n; a++) { T.nb[2 * a] = tour_succ(&T, a); T.nb[2 * a + 1] = tour_pred(&T, a); tour_set_nb(&T, a); }
    for (int a = 0; a < n; a++) for (int k = 0; k < K; k++) T.cc[a * K + k] = tour_c(&T, a, nn[a * K + k]);
    for (int a = 0; a < n; a++) T.L += tour_d(&T, a, tour_succ(&T, a));
    return T;
}

static void tour_free(Tour *T) { free(T->pos); free(T->nb); free(T->o); free(T->cc); free(T->ec); }
