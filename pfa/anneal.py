import ctypes
import os

import numpy as np

from . import eax
from .utils import validate

# built by running `make` in the repository root
_LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "native", "libpfa.so")
_lib = ctypes.CDLL(_LIB_PATH)

_f64 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
_f32 = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
_f64_2d = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS")
_i32 = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
_i64 = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS")
_lib.pfa_anneal_spin.argtypes = [
    _f64_2d,  # couplings, read in place
    _f64,  # local fields / 2: (fields + couplings @ spins) / 2 (updated in place)
    _f32,  # spins as +-1 (updated in place)
    _f32,  # noise buffer, the noise vector stored twice
    ctypes.c_int,  # number of spins
    ctypes.c_long,  # number of flips
    ctypes.c_double,  # beta at the start of the ramp
    ctypes.c_double,  # beta_max
    ctypes.c_int,  # index of the first spin to flip
    ctypes.POINTER(ctypes.c_double),  # energy (in: initial, out: final)
    ctypes.POINTER(ctypes.c_double),  # lowest energy of the first half of the flips (in: initial)
    _f32,  # spins of that state (in: initial)
    ctypes.POINTER(ctypes.c_double),  # lowest energy of the second half (out)
    _f32,  # spins of that state (out)
    _f64,  # the energy at the half and the second half's lowest at three quarters (out)
    ctypes.c_uint64,  # seed for the per-step noise offsets
    ctypes.c_double,  # tabu penalty lambda, in temperatures
    ctypes.c_int,  # tabu tenure, in flips
]
_lib.pfa_anneal_spin.restype = None
_lib.pfa_anneal_categorical.argtypes = [
    _f64_2d,  # couplings between the values other than 0
    _f64,  # fields of those values, plus their couplings to the current values (updated in place)
    ctypes.c_int,  # number of values other than 0
    ctypes.c_int,  # number of variables with more than one value
    _i32,  # variable of each value other than 0, 0 .. V - 1, grouped by variable
    _i32,  # current value of each variable as its row, -1 at value 0 (updated in place)
    _f32,  # noise buffer, the noise vector stored twice
    ctypes.c_long,  # number of moves
    ctypes.c_double,  # beta at the start of the ramp
    ctypes.c_double,  # beta_max
    ctypes.POINTER(ctypes.c_double),  # energy (in: initial, out: final)
    ctypes.POINTER(ctypes.c_double),  # lowest energy of the first half of the moves (in: initial)
    _i32,  # rows of that state (in: initial)
    ctypes.POINTER(ctypes.c_double),  # lowest energy of the second half (out)
    _i32,  # rows of that state (out)
    _f64,  # the energy at the half and the second half's lowest at three quarters (out)
    ctypes.c_uint64,  # seed for the per-step noise offsets
    ctypes.c_double,  # tabu penalty lambda, in temperatures
    ctypes.c_int,  # tabu tenure, in moves
]
_lib.pfa_anneal_categorical.restype = None
# the sparse kernels: the couplings argument is replaced by the stored couplings, and they take no noise buffer
# (and no first spin): the seed is for their draws
_spin_args, _cat_args = _lib.pfa_anneal_spin.argtypes, _lib.pfa_anneal_categorical.argtypes
_lib.pfa_anneal_sparse_spin.argtypes = [
    _i64,  # where each row of the couplings starts
    _i32,  # column of each stored coupling
    _f64,  # value of each stored coupling
] + _spin_args[1:3] + _spin_args[4:8] + _spin_args[9:]
_lib.pfa_anneal_sparse_spin.restype = None
_lib.pfa_anneal_sparse_categorical.argtypes = [
    _i32,  # first row of each variable, and the number of rows
    _i32,  # where each variable's neighbours start
    _i32,  # the neighbours: the variables it is coupled to
    _i64,  # where each variable's couplings start
    _f64,  # per variable, a table: one line per value other than 0, its couplings to the neighbours' values
] + _cat_args[1:4] + _cat_args[5:6] + _cat_args[7:]
_lib.pfa_anneal_sparse_categorical.restype = None
_lib.pfa_eliminate.argtypes = [
    _i32,  # first row of each variable, and the number of rows
    ctypes.c_int,  # number of variables
    _i32,  # where each variable's neighbours start
    _i32,  # the neighbours: the variables it is coupled to
    _i64,  # per neighbour, where its rows start in a line of the variable's couplings
    _i64,  # where each variable's lines start
    _i64,  # the length of a line of each variable
    _f64,  # the lines: the tables of the sparse kernel, or the array of the array kernel
    _f64,  # fields
    np.ctypeslib.ndpointer(dtype=np.int8, ndim=1, flags="C_CONTIGUOUS"),  # per value, value 0 included: allowed (out)
]
_lib.pfa_eliminate.restype = ctypes.c_long
# the folds: the first state takes from the second every cluster of variables that the second has lower (in place),
# and the energy change is returned
_lib.pfa_fold_spin.argtypes = [_f64_2d, _f64, ctypes.c_int, _f32, _f32]   # couplings, fields, spins, two states
_lib.pfa_fold_spin.restype = ctypes.c_double
_lib.pfa_fold_sparse_spin.argtypes = _lib.pfa_anneal_sparse_spin.argtypes[:3] + [_f64, ctypes.c_int, _f32, _f32]
_lib.pfa_fold_sparse_spin.restype = ctypes.c_double
_lib.pfa_fold_categorical.argtypes = _lib.pfa_eliminate.argtypes[:9] + [_i32, _i32]   # two states, as rows
_lib.pfa_fold_categorical.restype = ctypes.c_double
_lib.pfa_anneal_swap.argtypes = [
    _f64_2d,  # couplings between the variables, n x n
    _f64_2d,  # table between the values, q x q
    ctypes.c_int,  # number of variables
    ctypes.c_int,  # number of values
    _i32,  # value of each position
    _i32,  # first position of each position's row of pairs: the first position of the next value
    _i64,  # where each position's row starts in the pairs, n + 1
    _i32,  # variable at each position (updated in place)
    _f64_2d,  # F: the field of every variable at every value given the others, n x q (updated in place)
    _f64,  # dE of every pair, row by row (out)
    _f32,  # noise buffer, the noise vector stored twice
    ctypes.c_long,  # number of steps
    ctypes.c_double,  # beta at the start of the ramp
    ctypes.c_double,  # beta_max
    ctypes.POINTER(ctypes.c_double),  # energy (in: initial, out: final)
    ctypes.POINTER(ctypes.c_double),  # lowest energy of the first half of the steps (in: initial)
    _i32,  # positions of that state (in: initial)
    ctypes.POINTER(ctypes.c_double),  # lowest energy of the second half (out)
    _i32,  # positions of that state (out)
    _f64,  # the energy at the half and the second half's lowest at three quarters (out)
    ctypes.c_uint64,  # seed for the per-step noise offsets
    ctypes.c_double,  # tabu penalty lambda, in temperatures
    _i64,  # tabu tenure of each value, in steps
]
_lib.pfa_anneal_swap.restype = None
_swap_args = _lib.pfa_anneal_swap.argtypes
_lib.pfa_anneal_sparse_swap.argtypes = [
    _i64,  # where each row of the couplings starts
    _i32,  # column of each stored coupling
    _f64,  # value of each stored coupling
    _i32,  # the coupled pair of each stored coupling
    ctypes.c_int,  # number of coupled pairs
    _i32,  # first variable of each coupled pair
    _i32,  # second variable of each coupled pair, the larger
    _f64_2d,  # table between the values, q x q
    ctypes.c_int,  # number of variables
    ctypes.c_int,  # number of values
    _i32,  # value of each position
    _i32,  # variable at each position (updated in place)
    _f64_2d,  # F (updated in place)
] + _swap_args[11:12] + _swap_args[12:]
_lib.pfa_anneal_sparse_swap.restype = None
# A, B, fields, n, q, two states as the value of each variable
_lib.pfa_fold_swap.argtypes = [_f64_2d, _f64_2d, _f64_2d, ctypes.c_int, ctypes.c_int, _i32, _i32]
_lib.pfa_fold_swap.restype = ctypes.c_double
# n, x, y, rounding (0 none, 1 nearest integer, 2 ceiling), K, the candidates out (n x K); returns the Held-Karp bound
_lib.pfa_tour_alpha.argtypes = [ctypes.c_int, _f64, _f64, ctypes.c_int, ctypes.c_int, _i32]
_lib.pfa_tour_alpha.restype = ctypes.c_double
# n, x, y, rounding, K, the candidates (n x K), the tour (city at each position, in and out), the exchanges' depth,
# the number of chains run (added to); returns the length
_lib.pfa_tour_descent.argtypes = [ctypes.c_int, _f64, _f64, ctypes.c_int, ctypes.c_int, _i32, _i32, ctypes.c_int,
                                  ctypes.POINTER(ctypes.c_long)]
_lib.pfa_tour_descent.restype = ctypes.c_double
_lib.pfa_from_one_hot.argtypes = [
    _f64_2d,  # one-hot couplings over all slots
    _f64,  # one-hot fields
    ctypes.c_int,  # number of slots
    ctypes.c_int,  # number of variables
    _i32,  # reference slot of each variable
    ctypes.c_int,  # number of other slots
    _i32,  # the other slots, grouped by variable
    _i32,  # variable of each other slot
    _f64_2d,  # couplings between the other slots, relative to the reference slots (out)
    _f64,  # fields of the other slots, relative to the reference slots (out)
]
_lib.pfa_from_one_hot.restype = ctypes.c_double
_lib.pfa_simd_path.argtypes = []
_lib.pfa_simd_path.restype = ctypes.c_char_p


def simd_path() -> str:
    """Name of the SIMD code path the native library was compiled with."""
    return _lib.pfa_simd_path().decode()


def run(
    couplings: np.ndarray,
    fields: np.ndarray | None = None,
    sizes: np.ndarray | None = None,
    counts: np.ndarray | None = None,
    table: np.ndarray | None = None,
    num_steps: int = 1000,
    seed: int = 42,
    tabu_lambda: float = 3.0,
    tabu_tenure: float = 0.25,
) -> np.ndarray:
    """
    Minimizes 0.5 * s @ couplings @ s + fields @ s over spins s = +-1, or
    with sizes the energy of a categorical problem (below), with PFA, a
    rejection-free annealer (see the README): every step flips one spin,
    chosen with probability proportional to exp(-beta * dE) by Gumbel-max
    sampling, while beta rises. For the tabu_tenure * n flips after a spin
    flips, its dE is raised by tabu_lambda temperatures at the time of its
    flip (a tabu). The budget is num_steps, the total number of steps, and
    the only number: everything else is read from the problem. A scan from a
    random start cools in slices of eight sweeps, doubling beta every two
    slices, and records at each doubling the weight mobility, the share of
    the coupling weight (sum of |J| and |h| per spin) carried by the spins
    that still move between slices; it ends two doublings after that share
    has fallen under a tenth. Each restart then melts the best state so far
    (one slice from beta = 0) and anneals again over the same range, its
    slices shared among the doublings in proportion to the weight that froze
    in each, so the time goes where the energy is decided and none to the
    tail of light spins; the restart costs what the scan cost, and the budget
    buys their number. After each anneal the best state takes from the
    anneal's lowest states (one per half of its slices) every cluster of
    spins that they have lower: the spins on which two states differ fall
    into clusters not coupled to each other, and each cluster is taken from
    the state where it costs less; the spins that still differ are annealed
    again as a problem of their own (see the README). Compiled to native
    code on one core.

    The couplings are an array or a scipy.sparse matrix; the format does not
    change the run. run() picks the kernel from the number of nonzero
    couplings. The array kernels update the local fields and select the next
    spin in one SIMD pass over memory, and read a C-contiguous float64 array
    in place. When a step changes few enough local fields, the sparse kernels
    are faster: they store only the nonzero couplings, update the fields of
    the neighbours only, and draw the next spin with the same probability
    from a tree of sums of the weights exp(-beta * dE). A seed gives a
    different run on the sparse kernels than on the array kernels. If the
    array of the array kernels would take more than half of the machine's
    memory, the kernel whose couplings take less memory runs instead.

    With sizes, the problem is categorical: variable r takes one integer value
    x_r in 0 .. sizes[r] - 1. Each value other than 0 has a row and column of
    the couplings and an entry of the fields, variable by variable (variable
    0's values 1 .. sizes[0] - 1 first), and the energy sums the fields of the
    values taken plus half the couplings between every two of them; value 0
    adds nothing. Couplings between two values of the same variable must be 0.
    Every step changes one variable's value, chosen among all variables and
    values by the same rule. The tabu acts per variable, for tabu_tenure times
    the number of variables steps. Before the anneal, values that another
    value of the same variable is at least as good as, whatever the other
    variables do, are removed (dead-end elimination): a state of the lowest
    energy always remains. from_one_hot() converts the one-hot
    form.

    With counts, the variables exchange values: variable r takes a value x_r
    in 0 .. q - 1, counts[v] variables at value v throughout, and the energy is
        sum over r of fields[r, x_r] + sum over r < s of couplings[r, s] * table[x_r, x_s]
    with couplings n x n between the variables, table q x q between the values
    and fields n x q. Every step exchanges the values of two variables, chosen
    among all pairs at different values by the same rule, so the counts never
    change: k of n (counts (n - k, k)), balanced partitions (equal counts),
    permutations (counts all 1: quadratic assignment, tours by position).
    The tabu holds a variable's return to the value it left, for tabu_tenure
    times the mean number of steps between two moves of a variable at that
    value.

    Parameters:
        couplings (np.ndarray or scipy.sparse matrix): The couplings matrix for the system, such that couplings[i,j] is the coupling between spin_i and spin_j.
        fields (np.ndarray, optional): External field terms. Defaults to None.
        sizes (np.ndarray, optional): For categorical problems, the number of values of each variable. Defaults to None.
        counts (np.ndarray, optional): For variables that exchange values, the number of variables at each value. Defaults to None.
        table (np.ndarray, optional): With counts, the q x q symmetric table between the values. Defaults to None.
        num_steps (int): The budget: steps in all, each flipping a spin or changing one variable's value, at least
            sixteen per variable (one doubling of the scan). The scan takes what it needs and the rest buys restarts of
            the scan's length; the re-anneals of the differing variables come on top, in proportion to their share.
            Defaults to 1000.
        seed (int): Seed for the random number generator
        tabu_lambda (float): Tabu penalty on a just-flipped spin, in temperatures. Defaults to 3.0;
            rarely worth changing.
        tabu_tenure (float): How long a spin stays tabu, as a fraction of n flips; 0 turns the tabu
            off. Defaults to 0.25; rarely worth changing.
    Returns:
        np.ndarray: The lowest-energy state found: +-1 spins, or with sizes or counts the value of each variable.
    Raises:
        ValueError: If input parameters are invalid.
    """

    if counts is not None:
        if sizes is not None:
            raise ValueError("sizes and counts cannot both be given")
        if table is None:
            raise ValueError("counts needs table, the couplings between the values")
        validate(couplings)
        counts = np.asarray(counts, dtype=np.int64)
        table = np.ascontiguousarray(table, dtype=np.float64)
        n, q = couplings.shape[0], len(counts)
        if np.any(counts < 0) or int(counts.sum()) != n:
            raise ValueError("counts must be non-negative and sum to the number of variables")
        if table.shape != (q, q) or not np.array_equal(table, table.T):
            raise ValueError("table must be a symmetric q x q matrix, q the length of counts")
        if fields is not None and np.shape(fields) != (n, q):
            raise ValueError("fields must be n x q with counts")
    else:
        validate(couplings, fields)
    sparse = hasattr(couplings, "tocsr")
    if sparse:
        couplings = couplings.tocsr().astype(np.float64)
        couplings.sum_duplicates()
        couplings.eliminate_zeros()
    if sizes is not None:
        sizes = np.asarray(sizes, dtype=np.int64)
        if np.any(sizes < 1) or int(np.sum(sizes - 1)) != couplings.shape[0]:
            raise ValueError("sizes must be positive, with one couplings row per value other than 0: sum(sizes - 1)")
        ends = np.cumsum(sizes - 1)
        if sparse:
            var = np.repeat(np.arange(len(sizes)), sizes - 1)   # variable of each row
            row, col = couplings.nonzero()
            within = bool(np.any(var[row] == var[col]))
        else:
            within = any(np.any(couplings[e - q + 1 : e, e - q + 1 : e]) for q, e in zip(sizes, ends) if q > 2)
        if within:
            raise ValueError("Couplings between values of the same variable must be zero")

    variables = len(sizes) if sizes is not None else couplings.shape[0]
    if num_steps < K * W * variables:
        raise ValueError(f"num_steps must be at least {K * W} per variable, one doubling of the scan")
    if tabu_lambda < 0.0:
        raise ValueError("tabu_lambda must be non-negative")
    if tabu_tenure < 0.0:
        raise ValueError("tabu_tenure must be non-negative")
    if counts is not None:
        fields = np.zeros((n, q)) if fields is None else np.ascontiguousarray(fields, dtype=np.float64)
        return _run_swap(couplings, fields, counts, table, num_steps, seed, tabu_lambda, tabu_tenure)
    fields = np.zeros(couplings.shape[0]) if fields is None else np.ascontiguousarray(fields, dtype=np.float64)
    if sizes is not None:
        return _run_categorical(couplings, fields, sizes, num_steps, seed, tabu_lambda, tabu_tenure)
    form = _spin_form(couplings, len(fields))
    return _run_spins(form, fields, num_steps, seed, tabu_lambda, tabu_tenure)


def _sparse_is_faster(changed, size, spins):
    """Which kernel takes less time per step. A step of a sparse kernel costs a fixed part plus one exponential per
    weight it changes; a step of an array kernel is one pass over all size candidates. changed: the weights a step
    changes on average, the neighbours of a spin or the values of a variable's neighbours. The constants are
    fitted to timings of both kernels (on an Apple M4)."""
    return changed < (size / 30 - 7 if spins else size / 9 - 20)


def _memory():
    """Bytes of memory of this machine."""
    return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")


def _use_sparse(changed, size, spins, stored):
    """Whether the sparse kernel runs the problem: where its step is faster. The array kernel needs the couplings as
    a size x size float64 array; if that takes more than half of the machine's memory, memory decides and not speed:
    the sparse kernel runs where its couplings take less, stored bytes."""
    array = 8 * size * size
    if 2 * array > _memory():
        return stored < array
    return _sparse_is_faster(changed, size, spins)


def _entries(couplings):
    """(row, column, value) of the nonzero couplings, row by row with ascending columns: the same arrays for an
    array and for a scipy.sparse CSR matrix with sorted columns and no stored zeros. Entries are returned as given."""
    if isinstance(couplings, tuple):
        return couplings
    if hasattr(couplings, "tocsr"):
        return np.repeat(np.arange(couplings.shape[0]), np.diff(couplings.indptr)), couplings.indices, couplings.data
    row, col = np.nonzero(couplings)
    return row, col, np.asarray(couplings[row, col], dtype=np.float64)


def _array(couplings, n):
    """The couplings as an n x n C-contiguous float64 array: an array of that kind is returned as given."""
    if isinstance(couplings, tuple):
        out = np.zeros((n, n))
        out[couplings[0], couplings[1]] = couplings[2]
        return out
    return couplings.toarray() if hasattr(couplings, "tocsr") else np.ascontiguousarray(couplings, dtype=np.float64)


def _count(couplings):
    """The number of nonzero couplings."""
    if isinstance(couplings, tuple):
        return len(couplings[2])
    return couplings.nnz if hasattr(couplings, "tocsr") else np.count_nonzero(couplings)


def _spin_form(couplings, n):
    """The couplings of a problem of n spins as the kernel chosen for them takes them, from an array, a scipy.sparse
    CSR matrix or entries: a float64 array for the array kernel, or _entries() for the sparse kernel. Whatever is
    computed from either form does not depend on the input format."""
    count = _count(couplings)
    if _use_sparse(count / n, n, True, 24 * count):   # the entries: 24 bytes per nonzero coupling
        return _entries(couplings)
    return _array(couplings, n)


def _folded(fold, states):
    """The lowest of the states, as (energy, state) pairs, after it has taken from the others every cluster of
    variables that one of them has lower (fold(a, b) does that for a from b and returns the energy change), going
    round until its energy no longer falls: its energy and the state."""
    energy, state = min(states, key=lambda pair: pair[0])
    return _fold_into(fold, energy, state, [other for _, other in states if other is not state]), state


def _fold_into(fold, energy, state, others):
    """The energy of state after it has taken from the others, in place, every cluster one of them has lower (by more
    than the rounding of the cluster's sum: the fold's rule), going round until its energy no longer falls."""
    while True:
        lower = energy + sum(fold(state, other) for other in others)
        if not lower < energy:
            return energy
        energy = lower


K = 2       # slices per doubling of beta in the scan
W = 8       # sweeps per slice: a slice is W moves per variable, one kernel call
STILL = 2   # doublings with under a tenth of the weight moving that end the scan


def _profile(walk_from, x0, scale, weight, budget):
    """The scan: from a tenth of the scale, K slices per doubling of beta, recording at each doubling the weight
    mobility, the share of the weight carried by the variables on which consecutive slices' lowest states differ, to
    STILL doublings with under a tenth moving or the budget. Returns the cold edge of each doubling, the mobility, the
    lowest state as (energy, state) and the steps used."""
    walk = walk_from(x0)
    w = walk.w
    g = 2.0 ** (1.0 / K)
    b = scale / 10.0
    total = weight.sum()
    lowest, x_low = walk.energy(), x0
    betas, mob, used, still, last = [], [], 0, 0, None
    while used + K * w <= budget and still < STILL:
        low, moved = None, []
        for _ in range(K):
            e, st = walk.step(b, b * g)
            b *= g
            if last is not None:
                moved.append(weight[st != last].sum() / total)
            last = st.copy()
            if low is None or e < low[0]:
                low = (e, st)
        used += K * w
        if low[0] < lowest:
            lowest, x_low = low
        betas.append(b)
        mob.append(float(np.mean(moved)) if moved else 0.5)
        still = still + 1 if mob[-1] * 10 <= 1 else 0
    return betas, mob, (lowest, x_low.copy()), used


def _schedule(betas, mob, slices):
    """The slices of an anneal over the scanned range shared among the doublings in proportion to the weight that
    froze in each, the mobility's fall across it (the first doubling against a free walk's half), as (beta_from,
    beta_to) per slice, geometric within a doubling. Doublings in which nothing froze get none; the melt slice from
    beta = 0 to the first edge is not included."""
    before = np.array([0.5] + mob[:-1])
    freezing = np.maximum(0.0, before - np.array(mob))
    if not freezing.sum() > 0.0:   # a scan too short to see anything freeze: the slices to its last doubling
        freezing[-1] = 1.0
    share = freezing / freezing.sum()
    counts = np.floor(share * slices).astype(int)
    for i in np.argsort(share * slices - counts)[::-1][: slices - counts.sum()]:   # the remainder to the largest fractions
        counts[i] += 1
    grid = []
    for d, c in enumerate(counts):
        if c:
            edges = betas[d] / 2.0 * 2.0 ** (np.arange(c + 1) / c)
            grid.extend(zip(edges[:-1], edges[1:]))
    return grid


def _restart(walk_from, x, grid):
    """One anneal from x: a melt slice from beta = 0 to the grid's first edge, then the grid. The lowest states of
    the two halves of its slices, as (energy, state), and the steps used."""
    walk = walk_from(x)
    lows = [walk.step(0.0, grid[0][0])] + [walk.step(b0, b1) for b0, b1 in grid]
    k = len(lows)
    states = [min(lows[: k // 2], key=lambda p: p[0]), min(lows[k // 2:], key=lambda p: p[0])]
    return [(e, st.copy()) for e, st in states], k * walk.w


def _solve(problem, budget, rng):
    """The loop on a problem object (_Spins, _Cats, _Swap): the scan from a random start, then restarts of the scan's
    length as long as the budget holds, each melted from the best state so far and annealed on the scan's schedule,
    its lowest states folded into the best state and the variables that still differ annealed again. Returns the
    best state as the kernels hold it."""
    walk_from = lambda x: problem.walk(x, rng)
    x0 = problem.start(rng)
    betas, mob, (energy, x_best), L = _profile(walk_from, x0, problem.scale(x0), problem.weight, budget)
    schedule = lambda slices: _schedule(betas, mob, slices)
    grid = schedule(max(1, L // (W * problem.n) - 1))
    anneal = lambda S, x, e, steps: problem.reanneal(S, x, e, schedule, grid[-1][1], steps, rng)
    steps_for = problem.steps_for(L)
    spent = L
    while spent + L <= budget:
        states, steps = _restart(walk_from, x_best, grid)
        states.append((energy, x_best))
        energy, x_best = _folded(problem.fold, states)
        energy, x_best, re_steps = _reannealed(anneal, problem.fold, steps_for, energy, x_best, states)
        spent += steps + re_steps   # the re-anneals are paid for in the same currency
    return x_best


class _Slicer:
    """A walk advanced a slice at a time: step(b0, b1) anneals one slice of W sweeps from beta b0 to b1 and returns
    its lowest (energy, state) over all the variables; energy() the walk's current energy; w the steps of a slice."""

    def __init__(self, step, energy, w):
        self.step, self.energy, self.w = step, energy, w


def _spin_weight(couplings, fields):
    """A spin's weight, the energy it can move: the sum of |J| over its couplings plus |h|."""
    if isinstance(couplings, tuple):
        row, _, data = couplings
        return np.bincount(row, weights=np.abs(data), minlength=len(fields)) + np.abs(fields)
    return np.abs(couplings).sum(axis=1) + np.abs(fields)


class _Spins:
    """A spin problem as the loop reads it, the couplings as _spin_form() returns them."""

    def __init__(self, couplings, fields, tabu_lambda, tabu_tenure):
        self.n = n = len(fields)
        self.couplings, self.fields = couplings, fields
        self.sparse = isinstance(couplings, tuple)
        if self.sparse:
            row, col, data = couplings
            ptr = np.concatenate(([0], np.cumsum(np.bincount(row, minlength=n)))).astype(np.int64)
            self.stored = (ptr, col.astype(np.int32), data)
            self.fold = lambda a, b: _lib.pfa_fold_sparse_spin(*self.stored, fields, n, a, b)
        else:
            self.fold = lambda a, b: _lib.pfa_fold_spin(couplings, fields, n, a, b)
        self.weight = _spin_weight(couplings, fields)
        self.lam, self.tenure = tabu_lambda, int(tabu_tenure * n)

    def start(self, rng):
        return (2.0 * rng.integers(0, 2, self.n) - 1.0).astype(np.float32)

    def half_fields(self, spins):
        """(fields + couplings @ spins) / 2 as the spin kernels keep it, and the energy."""
        spins = spins.astype(np.float64)
        if self.sparse:
            row, col, data = self.couplings
            coupled = np.bincount(row, weights=data * spins[col], minlength=self.n)
        else:
            coupled = self.couplings @ spins
        return 0.5 * (self.fields + coupled), 0.5 * spins @ coupled + self.fields @ spins

    def scale(self, spins):
        """The beta scale of a state: one over the mean |dE| of a flip."""
        phi, _ = self.half_fields(spins)
        return 1.0 / np.mean(4.0 * np.abs(phi))

    def walk(self, spins, rng):
        n, w = self.n, W * self.n
        phi, energy = self.half_fields(spins)
        energy_c = ctypes.c_double(energy)
        s = spins.astype(np.float32)

        def step(b0, b1):
            # the kernels' contract: the first half's lowest starts as the state and energy the slice starts from
            first_c, late_c, marks = ctypes.c_double(energy_c.value), ctypes.c_double(), np.empty(2)
            s_first, s_late = s.copy(), np.empty(n, dtype=np.float32)
            args = (ctypes.byref(energy_c), ctypes.byref(first_c), s_first, ctypes.byref(late_c), s_late, marks,
                    int(rng.integers(1, 2**63)), self.lam, self.tenure)
            if self.sparse:
                _lib.pfa_anneal_sparse_spin(*self.stored, phi, s, n, w, b0, b1, *args)
            else:
                noise_vec = -np.log(-np.log(rng.random(n)))
                noise = np.concatenate((noise_vec, noise_vec)).astype(np.float32)
                first = int(np.argmax(4.0 * b0 * s.astype(np.float64) * phi + noise_vec))
                _lib.pfa_anneal_spin(self.couplings, phi, s, noise, n, w, b0, b1, first, *args)
            return (first_c.value, s_first) if first_c.value <= late_c.value else (late_c.value, s_late)

        return _Slicer(step, lambda: energy_c.value, w)

    def sub(self, S, x):
        """The spins S as a problem of their own, the others fixed as in x and their couplings folded into the fields;
        the kernel is chosen again."""
        n, m = self.n, len(S)
        if self.sparse:
            row, col, data = self.couplings
            at = np.full(n, -1)
            at[S] = np.arange(m)
            in_row, in_col = at[row] >= 0, at[col] >= 0
            keep, out = in_row & in_col, in_row & ~in_col
            sub = (at[row[keep]], at[col[keep]], data[keep])
            fixed = np.bincount(at[row[out]], weights=data[out] * x[col[out]].astype(np.float64), minlength=m)
        else:
            T = np.setdiff1d(np.arange(n), S, assume_unique=True)
            sub = np.ascontiguousarray(self.couplings[np.ix_(S, S)])
            fixed = self.couplings[np.ix_(S, T)] @ x[T].astype(np.float64)
        return _Spins(_spin_form(sub, m), np.ascontiguousarray(self.fields[S] + fixed), self.lam, self.tenure / n)

    def reanneal(self, S, x, energy, schedule, cold, steps, rng):
        """The spins S annealed again on the scan's schedule with the slices steps affords: a melt slice then the
        grid, on the sub-problem. The two halves' lowest states over all the spins."""
        part = self.sub(S, x)
        walk = part.walk(x[S], rng)
        offset = energy - walk.energy()   # the fixed spins' energy
        sub_grid = schedule(max(1, steps // walk.w - 1))
        lows = [walk.step(0.0, sub_grid[0][0])] + [walk.step(a, b) for a, b in sub_grid]
        k = len(lows)
        states = []
        for e, found in (min(lows[: k // 2], key=lambda p: p[0]), min(lows[k // 2:], key=lambda p: p[0])):
            s = x.copy()
            s[S] = found
            states.append((e + offset, s))
        return states

    def steps_for(self, L):
        return lambda S, x: max(1, L * len(S) // self.n)

    def values(self, x):
        return x.astype(np.int8)


def _run_spins(couplings, fields, num_steps, seed, tabu_lambda, tabu_tenure):
    """run() for spins, on validated input, with the couplings as _spin_form() returns them."""
    problem = _Spins(couplings, fields, tabu_lambda, tabu_tenure)
    return problem.values(_solve(problem, num_steps, np.random.default_rng(seed)))


def _run_categorical(couplings, fields, sizes, num_steps, seed, tabu_lambda, tabu_tenure):
    """run() for categorical problems, on validated input (couplings as an array or a scipy.sparse CSR matrix).
    Variables with one value never move and are left out, and so are the values that elimination removes."""
    multi = sizes > 1
    args = num_steps, seed, tabu_lambda, tabu_tenure
    x = np.zeros(len(sizes), dtype=np.int64)
    if not multi.any():
        return x
    form = _Form(couplings, sizes[multi])
    alive = np.empty(form.starts[-1] + len(form.starts) - 1, dtype=np.int8)
    if not _lib.pfa_eliminate(*form.lines, np.ascontiguousarray(fields), alive):
        x[multi] = _anneal_categorical(form, fields, *args)
        return x
    couplings, fields, count, value = _reduced(form, fields, alive.astype(bool))
    at = np.cumsum(count) - count          # where each variable's allowed values start in value
    free = count > 1
    if free.any():
        at[free] += _anneal_categorical(_Form(couplings, count[free]), fields, *args)
    x[multi] = value[at]
    return x


class _Form:
    """The couplings of a categorical problem whose variables all have more than one value, as the kernel chosen
    for them reads them. starts: rows of variable r are starts[r] .. starts[r + 1] - 1. sparse: whether the sparse
    kernel runs it. couplings: _entries() for the sparse kernel, a float64 array for the array kernel. lines: how
    both pfa_eliminate() and the sparse kernel find the couplings of a row to a neighbour's rows (see
    eliminate.c): first row of each variable, number of variables, where each variable's neighbours
    start, the neighbours, where a neighbour's rows start in a line, where each variable's lines start, the length
    of its lines, and the lines, which are the sparse kernel's tables or the array itself."""

    def __init__(self, couplings, sizes):
        V = len(sizes)
        width = sizes - 1                                                     # rows per variable
        starts = np.concatenate(([0], np.cumsum(width)))
        m = int(starts[-1])
        grp = np.repeat(np.arange(V), width)
        if isinstance(couplings, np.ndarray):   # the variables with any coupling between their rows
            nonzero = couplings != 0
            count = np.count_nonzero(nonzero)
            some = np.logical_or.reduceat(np.logical_or.reduceat(nonzero, starts[:-1], axis=1), starts[:-1], axis=0)
            pairs = np.flatnonzero(some)
            del nonzero, some
        else:
            row, col, data = _entries(couplings)
            count = len(data)
            pairs = np.unique(grp[row].astype(np.int64) * V + grp[col])       # coupled (variable, neighbour), as one number
        left, right = pairs // V, pairs % V
        tables = int(np.sum(width[left] * width[right]))                      # the sparse kernel stores these in full
        self.starts, self.grp = starts, grp.astype(np.int32)
        self.sparse = _use_sparse(tables / m, m, False, 8 * tables + 24 * count)   # the tables and the entries
        vptr = np.searchsorted(left, np.arange(V + 1))
        if self.sparse:
            self.couplings = _entries(couplings)
            seg, line, rows, vals = _tables(*self.couplings, pairs, grp, starts)
        else:
            self.couplings = _array(couplings, m)
            seg, line, rows, vals = starts[right], np.full(V, m), starts[:-1] * m, self.couplings.reshape(-1)
        self.lines = (starts.astype(np.int32), V, vptr.astype(np.int32), right.astype(np.int32), seg.astype(np.int64),
                      rows.astype(np.int64), line.astype(np.int64), vals)


def _tables(row, col, data, pairs, grp, starts):
    """Couplings between rows (row, col, data) -> full tables between coupled variables, as the sparse categorical
    kernel reads them. pairs: the coupled (variable, neighbour) as variable * V + neighbour, ascending. Variable r's
    table has one line per row of r, with the couplings to every row of its first neighbour, then of its second,
    and so on. Returns where each neighbour's rows start in a line, the length of each variable's lines, where each
    variable's table starts, and the tables."""
    V = len(starts) - 1
    width = np.diff(starts)                                            # rows per variable
    left, right = pairs // V, pairs % V
    vptr = np.searchsorted(left, np.arange(V + 1))
    seg = np.cumsum(width[right]) - width[right]                       # where each neighbour's rows start in a line,
    seg -= seg[vptr[left]]                                             # counted from the variable's first neighbour
    line = np.bincount(left, weights=width[right], minlength=V).astype(np.int64)   # length of a line
    rows = np.concatenate(([0], np.cumsum(width * line)[:-1]))
    vals = np.zeros(int(np.sum(width * line)))
    e = np.searchsorted(pairs, grp[row].astype(np.int64) * V + grp[col])
    r = left[e]
    vals[rows[r] + (row - starts[r]) * line[r] + seg[e] + col - starts[right[e]]] = data
    return seg, line, rows, vals


def _reduced(form, fields, alive):
    """The problem of form on the allowed values (alive, per value with value 0 included, variable by variable), with
    the energy changed by a constant. The lowest allowed value of each variable becomes its value 0: with k_s the
    row of that value of s (no row if it is the old value 0, and its terms are 0), the couplings and fields are

        T'(a, c) = T(a, c) - (T(a, k_s) + T(k_r, c)) + T(k_r, k_s)
        f'(a) = G(a) - G(k_r),    G(a) = f(a) + sum over s of T(a, k_s)

    Variables left with one value are left out: their couplings are in the fields. Returns the couplings (entries
    or an array, as in form), the fields, the number of allowed values of every variable of form, and the old value
    of each allowed value, variable by variable."""
    starts = form.starts
    V, m = len(starts) - 1, int(starts[-1])
    width = np.diff(starts)
    slot = starts[:-1] + np.arange(V)                    # where each variable's values start in alive
    count = np.add.reduceat(alive.astype(np.int64), slot)
    on = np.flatnonzero(alive)
    value = on - np.repeat(slot, count)
    k = np.where(value[np.cumsum(count) - count] > 0, on[np.cumsum(count) - count] - np.arange(V) - 1, -1)   # row k_r
    is_k = np.zeros(m, dtype=bool)
    is_k[k[k >= 0]] = True
    keep = np.delete(alive, slot) & ~is_k                # the rows of the reduced problem
    new = np.cumsum(keep) - 1                            # their new numbers
    k_of = np.repeat(k, width)                           # per row: k of its variable
    n = int(keep.sum())
    out_fields = np.empty(n)
    _, _, vptr, adj, _, rows, line, vals = form.lines
    if form.sparse:
        wide = width[adj]
        to_row = np.arange(int(wide.sum())) - np.repeat(np.cumsum(wide) - wide - starts[adj], wide)   # row of each column
        ends = np.concatenate(([0], np.cumsum(line)))
        out = [np.empty(0, dtype=np.int64)], [np.empty(0, dtype=np.int64)], [np.empty(0)]
    else:
        to_row = np.arange(m)
        out = np.zeros((n, n))
    for r in np.flatnonzero(count > 1):
        lo, hi = starts[r], starts[r + 1]
        a = np.flatnonzero(keep[lo:hi])
        if form.sparse:
            table = vals[rows[r] : rows[r] + width[r] * line[r]].reshape(width[r], line[r])
            cols = to_row[ends[r] : ends[r + 1]]
        else:
            table, cols = form.couplings[lo:hi], to_row
        G = fields[lo:hi] + table[:, is_k[cols]].sum(axis=1)
        c = np.flatnonzero(keep[cols])
        h = np.flatnonzero(k_of[cols[c]] >= 0)                       # the columns whose variable has a k_s,
        at_k = c[h] - (cols[c[h]] - k_of[cols[c[h]]])                # and the column of k_s
        T = table[np.ix_(a, c)]
        if k[r] >= 0:
            own = table[k[r] - lo]                                   # T(k_r, .)
            rest = np.flatnonzero(k_of[cols[c]] < 0)
            T[:, h] = T[:, h] - (table[np.ix_(a, at_k)] + own[c[h]]) + own[at_k]
            T[:, rest] -= own[c[rest]]
            out_fields[new[lo + a]] = G[a] - G[k[r] - lo]
        else:
            T[:, h] -= table[np.ix_(a, at_k)]
            out_fields[new[lo + a]] = G[a]
        if form.sparse:
            i, j = np.nonzero(T)
            out[0].append(new[lo + a][i])
            out[1].append(new[cols[c]][j])
            out[2].append(T[i, j])
        else:
            out[new[lo + a[0]] : new[lo + a[-1]] + 1] = T           # c is every kept row
    if form.sparse:
        out = tuple(np.concatenate(x) for x in out)
    return out, out_fields, count, value


def _anneal_categorical(form, fields, num_steps, seed, tabu_lambda, tabu_tenure):
    """The anneal of a categorical problem in the form of _Form: the value of each variable, 0 for its value 0."""
    starts, couplings = form.starts, form.couplings
    V, m = len(starts) - 1, int(starts[-1])
    sizes = np.diff(starts) + 1

    def values(cur):   # current row of each variable (-1: value 0) -> the value of every variable
        return np.where(cur < 0, 0, cur - starts[:-1] + 1)

    if np.all(sizes == 2):
        # x = (s + 1) / 2: the spin problem with couplings C and fields 2 f + C 1 has 4 times the energy (plus a
        # constant); the scale is read, so the factor plays no part
        spin = _spin_form(couplings, m)
        sums = np.bincount(spin[0], weights=spin[2], minlength=m) if isinstance(spin, tuple) else spin.sum(axis=1)
        spins = _run_spins(spin, 2.0 * fields + sums, num_steps, seed, tabu_lambda, tabu_tenure)
        return values(np.where(spins > 0, np.arange(V), -1))

    problem = _Cats(form, np.ascontiguousarray(fields, dtype=np.float64), tabu_lambda, tabu_tenure)
    return values(_solve(problem, num_steps, np.random.default_rng(seed)))


class _Cats:
    """A categorical problem as the loop reads it, in the form of _Form (every variable with more than one value, the
    eliminated values gone); a state is the current row of each variable, -1 at value 0, as the kernels hold it."""

    def __init__(self, form, fields, tabu_lambda, tabu_tenure):
        self.form, self.fields = form, fields
        self.starts, self.grp = form.starts, form.grp
        self.n = V = len(self.starts) - 1
        self.m = int(self.starts[-1])
        self.sizes = np.diff(self.starts) + 1
        if form.sparse:
            first, _, vptr, adj, _, rows, _, vals = form.lines
            self.stored = first, vptr, adj, rows, vals
            row, _, data = form.couplings
            per_row = np.bincount(row, weights=np.abs(data), minlength=self.m)
        else:
            per_row = np.abs(form.couplings).sum(axis=1)
        # a variable's weight, the energy it can move: the sum over its rows of their |couplings| and |field|
        self.weight = np.bincount(self.grp, weights=per_row + np.abs(fields), minlength=V)
        self.fold = lambda a, b: _lib.pfa_fold_categorical(*form.lines, fields, a, b)
        self.lam, self.tenure = tabu_lambda, int(tabu_tenure * V)

    def start(self, rng):
        v = (rng.random(self.n) * self.sizes).astype(np.int64)   # a uniformly random value per variable
        return np.where(v == 0, -1, self.starts[:-1] + v - 1).astype(np.int32)

    def fields_of(self, cur):
        """The fields plus the couplings to the current values, and the energy."""
        on = cur[cur >= 0]
        if self.form.sparse:
            row, col, data = self.form.couplings
            taken = np.zeros(self.m, dtype=bool)
            taken[on] = True
            sel = taken[row]
            return (self.fields + np.bincount(col[sel], weights=data[sel], minlength=self.m),
                    self.fields[on].sum() + 0.5 * data[sel & taken[col]].sum())
        C = self.form.couplings
        return self.fields + C[:, on].sum(axis=1), self.fields[on].sum() + 0.5 * C[np.ix_(on, on)].sum()

    def scale(self, cur):
        """The beta scale of a state: one over the mean |dE| of a change of value."""
        F, _ = self.fields_of(cur)
        at = np.where(cur >= 0, F[np.maximum(cur, 0)], 0.0)      # the field of each variable's current value
        de = np.concatenate((F - at[self.grp], -at[cur >= 0]))   # to every other row, and to value 0
        de = np.abs(de[de != 0.0])
        return 1.0 / de.mean()

    def walk(self, cur, rng):
        V, m, w = self.n, self.m, W * self.n
        cur = cur.copy()
        F, energy = self.fields_of(cur)
        F = np.ascontiguousarray(F)
        energy_c = ctypes.c_double(energy)

        def step(b0, b1):
            first_c, late_c, marks = ctypes.c_double(energy_c.value), ctypes.c_double(), np.empty(2)
            c_first, c_late = cur.copy(), np.empty(V, dtype=np.int32)
            args = (ctypes.byref(energy_c), ctypes.byref(first_c), c_first, ctypes.byref(late_c), c_late, marks,
                    int(rng.integers(1, 2**63)), self.lam, self.tenure)
            if self.form.sparse:
                _lib.pfa_anneal_sparse_categorical(*self.stored, F, m, V, cur, w, b0, b1, *args)
            else:
                noise_vec = -np.log(-np.log(rng.random(m)))
                noise = np.concatenate((noise_vec, noise_vec)).astype(np.float32)
                _lib.pfa_anneal_categorical(self.form.couplings, F, m, V, self.grp, cur, noise, w, b0, b1, *args)
            return (first_c.value, c_first) if first_c.value <= late_c.value else (late_c.value, c_late)

        return _Slicer(step, lambda: energy_c.value, w)

    def reanneal(self, S, x, energy, schedule, cold, steps, rng):
        """The variables S annealed again from random values, linear from beta = 0 to the schedule's cold end."""
        return _anneal_cat(self.form, self.fields, S, x, energy, 0.0, cold, steps, rng, self.lam, self.tenure / self.n)

    def steps_for(self, L):
        return lambda S, x: max(1, L * len(S) // self.n)


def _anneal_cat(form, fields, S, cur0, energy0, beta_start, beta_max, num_steps, rng, tabu_lambda, tabu_tenure):
    """One anneal of the variables S of a categorical problem in the form of _Form, the others fixed at their rows in
    cur0 and their couplings folded into the fields, from cur0 on S (random values when cur0 is None) with energy0
    the energy of cur0: the lowest state of the first half of the steps and the lowest of the second half, as
    (energy, state) pairs over all the variables, a state being the current row of each variable (-1: value 0)."""
    starts = whole_starts = form.starts
    V, m = len(starts) - 1, int(starts[-1])
    sizes = np.diff(starts) + 1
    part = len(S) < V
    if part:
        on = cur0[np.setdiff1d(np.arange(V), S, assume_unique=True)]
        on = on[on >= 0]                                                      # rows of the fixed variables
        keep = np.flatnonzero(np.repeat(np.isin(np.arange(V), S), sizes - 1))   # rows of S
        if isinstance(form.couplings, tuple):
            row, col, data = form.couplings
            at = np.full(m, -1)
            at[keep] = np.arange(len(keep))
            taken = np.zeros(m, dtype=bool)
            taken[on] = True
            both, out = (at[row] >= 0) & (at[col] >= 0), (at[row] >= 0) & taken[col]
            sub = (at[row[both]], at[col[both]], data[both])
            fixed = np.bincount(at[row[out]], weights=data[out], minlength=len(keep))
        else:
            sub = np.ascontiguousarray(form.couplings[np.ix_(keep, keep)])
            fixed = form.couplings[np.ix_(keep, on)].sum(axis=1)
        form, fields = _Form(sub, sizes[S]), np.ascontiguousarray(fields[keep] + fixed)
        starts = form.starts
        V, m = len(S), int(starts[-1])
        sizes = sizes[S]
    grp, sparse, couplings = form.grp, form.sparse, form.couplings
    if sparse:
        row, col, data = couplings
        first, _, vptr, adj, _, rows, _, vals = form.lines
        stored = first, vptr, adj, rows, vals
    tenure = int(tabu_tenure * V)
    if cur0 is None:
        v = (rng.random(V) * sizes).astype(np.int64)                  # a uniformly random value per variable
        cur = np.where(v == 0, -1, starts[:-1] + v - 1).astype(np.int32)
    else:
        cur = cur0[S]
        if part:                                                      # the rows renumbered to the sub-problem
            cur = np.where(cur >= 0, cur - whole_starts[S] + starts[:-1], -1).astype(np.int32)
    on = cur[cur >= 0]
    if sparse:   # from the entries, so that the input format plays no part
        taken = np.zeros(m, dtype=bool)
        taken[on] = True
        sel = taken[row]
        F = fields + np.bincount(col[sel], weights=data[sel], minlength=m)
        energy = fields[on].sum() + 0.5 * data[sel & taken[col]].sum()
    else:
        F = fields + couplings[:, on].sum(axis=1)
        energy = fields[on].sum() + 0.5 * couplings[np.ix_(on, on)].sum()
    offset = 0.0 if energy0 is None else energy0 - energy       # the energy of the fixed variables
    cur_first, cur_late = cur.copy(), np.empty(V, dtype=np.int32)
    energy_c = ctypes.c_double(energy)
    first_c, late_c = ctypes.c_double(energy), ctypes.c_double()
    lowest = ctypes.byref(first_c), cur_first, ctypes.byref(late_c), cur_late
    marks = np.empty(2)   # the energy at the half and the second half's lowest at three quarters
    if sparse:
        _lib.pfa_anneal_sparse_categorical(
            *stored, F, m, V, cur, num_steps, beta_start, beta_max, ctypes.byref(energy_c), *lowest,
            marks, int(rng.integers(1, 2**63)), tabu_lambda, tenure,
        )
    else:
        noise_vec = -np.log(-np.log(rng.random(m)))
        noise_buffer = np.concatenate((noise_vec, noise_vec)).astype(np.float32)
        _lib.pfa_anneal_categorical(
            couplings, F, m, V, grp, cur, noise_buffer, num_steps, beta_start, beta_max, ctypes.byref(energy_c),
            *lowest, marks, int(rng.integers(1, 2**63)), tabu_lambda, tenure,
        )
    states = []
    for value, found in ((first_c.value, cur_first), (late_c.value, cur_late)):
        if part:
            whole = cur0.copy()
            whole[S] = np.where(found >= 0, found - starts[:-1] + whole_starts[S], -1)
        else:
            whole = found
        states.append((value + offset, whole))
    return states


def _sparse_swap_is_faster(d, n, q, pairs):
    """Which exchange kernel takes less time per step. The array kernel streams every pair, updates F, n q entries,
    and does O(n) work besides; a sparse step refreshes the single moves of the exchanged variables and their
    neighbours, (2d + 2)(q - 1) leaves, as many blocks, and their coupled pairs, (2d + 2) d. The constants, in
    seconds, are fitted to timings of both kernels (on an Apple M4)."""
    return (2 * d + 2) * (120e-9 * (q - 1) + 18e-9 * d) < 0.32e-9 * pairs + 0.17e-9 * n * q + 10e-9 * n


def _run_swap(couplings, fields, counts, table, num_steps, seed, tabu_lambda, tabu_tenure):
    """run() with counts, on validated input: the value of each variable."""
    n = len(fields)
    if np.count_nonzero(counts) < 2:                            # one value: nothing moves
        return np.full(n, np.flatnonzero(counts)[0], dtype=np.int64)
    A = couplings if isinstance(couplings, np.ndarray) else couplings.tocsr()
    problem = _Swap(A, table, fields, counts, tabu_lambda, tabu_tenure)
    return _solve(problem, num_steps, np.random.default_rng(seed)).astype(np.int64)


class _Swap:
    """Variables that exchange values, as the loop reads them; a state is the value of each variable."""

    def __init__(self, A, table, fields, counts, tabu_lambda, tabu_tenure):
        self.n = n = len(fields)
        self.q = len(counts)
        self.A, self.table, self.fields, self.counts = A, table, fields, counts
        A_fold = _array(A, n)                                   # the fold reads the array
        self.fold = lambda a, b: _lib.pfa_fold_swap(A_fold, table, fields, n, self.q, a, b)
        row_sums = np.asarray(np.abs(A).sum(axis=1)).ravel()
        self.weight = row_sums + np.abs(fields).mean(axis=1)   # a variable's weight: its |couplings| plus its mean |field|
        self.kappa = 2.0 * table - np.diag(table)[:, None] - np.diag(table)[None, :]
        self.pairs = _pairs(counts)
        self.lam, self.tabu_tenure = tabu_lambda, tabu_tenure

    def start(self, rng):
        return np.repeat(np.arange(self.q), self.counts)[rng.permutation(self.n)].astype(np.int32)

    def fields_of(self, x):
        """F[r, k] = f_r(k) + sum_s A_rs table(k, x_s), and the energy."""
        F = np.ascontiguousarray(self.fields + self.A @ self.table[x])
        rows = np.arange(self.n)
        return F, float(self.fields[rows, x].sum() + 0.5 * (F[rows, x] - self.fields[rows, x]).sum())

    def scale(self, x):
        """The beta scale of a state: one over the mean |dE| of an exchange, over the pairs at different values or a
        sample of them."""
        F, _ = self.fields_of(x)
        n = self.n
        if self.pairs <= 2_000_000:
            r, s = np.triu_indices(n, 1)
        else:
            rng = np.random.default_rng(0)
            r, s = rng.integers(0, n, 2_000_000), rng.integers(0, n, 2_000_000)
        a, b = x[r], x[s]
        keep = a != b
        r, s, a, b = r[keep], s[keep], a[keep], b[keep]
        A_rs = self.A[r, s] if isinstance(self.A, np.ndarray) else np.asarray(self.A[r, s]).ravel()
        de = (F[r, b] - F[r, a]) + (F[s, a] - F[s, b]) + A_rs * self.kappa[a, b]
        de = np.abs(de[de != 0.0])
        return 1.0 / de.mean()

    def walk(self, x, rng):
        """A walk from x advanced a slice of W sweeps at a time, on the kernel chosen for the problem; the slice's
        lowest state is returned as the value of each variable."""
        n, q, w = self.n, self.q, W * self.n
        counts, A, table = self.counts, self.A, self.table
        order = np.argsort(counts, kind="stable")                # the values by count, the largest last
        order = order[counts[order] > 0]
        block = np.empty(q, dtype=np.int64)
        block[order] = np.arange(len(order))
        sizes = counts[order]
        val = np.repeat(order, sizes).astype(np.int32)          # value of each position
        e = np.repeat(np.cumsum(sizes), sizes).astype(np.int32)
        pairs = int(np.sum(n - e))
        sparse = _sparse_swap_is_faster(_count(A) / n, n, q, pairs)
        if sparse:
            row, col, data = _entries(A)
            ptr = np.concatenate(([0], np.cumsum(np.bincount(row, minlength=n)))).astype(np.int64)
            key = np.minimum(row, col).astype(np.int64) * n + np.maximum(row, col)
            uniq, pid = np.unique(key, return_inverse=True)
            stored = (ptr, col.astype(np.int32), data, pid.astype(np.int32), len(uniq), (uniq // n).astype(np.int32),
                      (uniq % n).astype(np.int32))
        else:
            A_arr = _array(A, n)
            rs = np.concatenate(([0], np.cumsum((n - e + 63) // 64 * 64))).astype(np.int64)
            P = int(rs[-1])
            D = np.empty(P)
        tenure = np.zeros(q, dtype=np.int64)
        tenure[order] = (self.tabu_tenure * pairs / (n - sizes)).astype(np.int64)
        pos = np.argsort(block[x], kind="stable").astype(np.int32)   # the variables by block of their value
        F, energy = self.fields_of(x)
        energy_c = ctypes.c_double(energy)

        def step(b0, b1):
            first_c, late_c, marks = ctypes.c_double(energy_c.value), ctypes.c_double(), np.empty(2)
            p_first, p_late = pos.copy(), np.empty(n, dtype=np.int32)
            args = (ctypes.byref(energy_c), ctypes.byref(first_c), p_first, ctypes.byref(late_c), p_late, marks,
                    int(rng.integers(1, 2**63)), self.lam, tenure)
            if sparse:
                _lib.pfa_anneal_sparse_swap(*stored, table, n, q, val, pos, F, w, b0, b1, *args)
            else:
                noise_vec = -np.log(-np.log(rng.random(P)))
                noise = np.concatenate((noise_vec, noise_vec)).astype(np.float32)
                _lib.pfa_anneal_swap(A_arr, table, n, q, val, e, rs, pos, F, D, noise, w, b0, b1, *args)
            value, p = (first_c.value, p_first) if first_c.value <= late_c.value else (late_c.value, p_late)
            y = np.empty(n, dtype=np.int32)
            y[p] = val
            return value, y

        return _Slicer(step, lambda: energy_c.value, w)

    def reanneal(self, S, x, energy, schedule, cold, steps, rng):
        """The variables S annealed again from random values with their counts, linear from beta = 0 to the
        schedule's cold end."""
        return _anneal_swap(self.A, self.table, self.fields, self.counts, S, x, energy, 0.0, cold, steps, rng,
                            self.lam, self.tabu_tenure)

    def steps_for(self, L):
        # the steps of a re-anneal: in proportion to the pairs of S at different values, read from the state annealed
        return lambda S, x: max(1, L * _pairs(np.bincount(x[S], minlength=self.q)) // self.pairs)


def _reannealed(anneal, fold, steps_for, energy, x, states):
    """x after the variables on which it differs from each of the states have been annealed again from random values,
    the agreeing variables fixed, for steps_for(S, x) steps: in proportion to the moves of the differing variables S.
    What the anneal finds folds into x (by the fold's rule, so that a state the anneal reports a rounding lower is
    not a gain), and when that lowers x the two states of the anneal join the states to go round again, until no
    anneal brings anything. (On dense couplings the differing variables form one cluster, which the fold cannot
    take.) Returns the energy, x and the steps the anneals took."""
    pool = [y for _, y in states if y is not x]
    spent = 0
    while pool:
        S = np.flatnonzero(x != pool.pop())
        if len(S) == 0 or len(S) == len(x):   # nothing differs, or nothing is shared
            continue
        steps = steps_for(S, x)
        spent += steps
        found = [y for _, y in anneal(S, x, energy, steps)]
        lower = _fold_into(fold, energy, x, found)
        if lower < energy:
            pool.extend(found)
            energy = lower
    return energy, x, spent


def _pairs(counts):
    """The pairs of positions with different values."""
    return (int(np.sum(counts)) ** 2 - int(np.sum(np.asarray(counts, dtype=np.int64) ** 2))) // 2


def _sub(M, I, J):
    return M[np.ix_(I, J)] if isinstance(M, np.ndarray) else M[I][:, J]


def _anneal_swap(A, table, fields, counts, S, x0, energy0, beta_start, beta_max, num_steps, rng, tabu_lambda,
                 tabu_tenure):
    """One anneal of the variables S, the others fixed at their values in x0 and their couplings folded into the
    fields, from x0 on S (random values with the counts when x0 is None) with energy0 the energy of x0: the lowest
    state of the first half of the steps and the lowest of the second half, as (energy, state) pairs over all the
    variables."""
    n, q = len(fields), len(counts)
    if len(S) < n:
        T = np.setdiff1d(np.arange(n), S, assume_unique=True)
        A_S = _sub(A, S, S)
        if not isinstance(A_S, np.ndarray):
            A_S.sort_indices()
        f = np.ascontiguousarray(fields[S] + _sub(A, S, T) @ table[x0[T]])
        counts = np.bincount(x0[S], minlength=q)
    else:
        A_S, f = A, fields
    m = len(S)
    order = np.argsort(counts, kind="stable")                    # the values by count, the largest last
    order = order[counts[order] > 0]
    block = np.empty(q, dtype=np.int64)                         # block of each value
    block[order] = np.arange(len(order))
    sizes = counts[order]
    val = np.repeat(order, sizes).astype(np.int32)              # value of each position
    e = np.repeat(np.cumsum(sizes), sizes).astype(np.int32)     # first position of the next value
    pairs = int(np.sum(m - e))
    sparse = _sparse_swap_is_faster(_count(A_S) / m, m, q, pairs)
    if sparse:
        row, col, data = _entries(A_S)
        ptr = np.concatenate(([0], np.cumsum(np.bincount(row, minlength=m)))).astype(np.int64)
        key = np.minimum(row, col).astype(np.int64) * m + np.maximum(row, col)
        uniq, pid = np.unique(key, return_inverse=True)
        stored = (ptr, col.astype(np.int32), data, pid.astype(np.int32), len(uniq), (uniq // m).astype(np.int32),
                  (uniq % m).astype(np.int32))
    else:
        A_S = _array(A_S, m)
        rs = np.concatenate(([0], np.cumsum((m - e + 63) // 64 * 64))).astype(np.int64)   # rows padded to 64 pairs
        P = int(rs[-1])
    # tenure of a variable at value v: tabu_tenure times its mean number of steps between moves, which is the pairs
    # over the positions of the other values
    tenure = np.zeros(q, dtype=np.int64)
    tenure[order] = (tabu_tenure * pairs / (m - sizes)).astype(np.int64)
    if x0 is None:
        pos = rng.permutation(m).astype(np.int32)
    else:
        pos = np.argsort(block[x0[S]], kind="stable").astype(np.int32)   # the variables by block of their value
    x = np.empty(m, dtype=np.int32)
    x[pos] = val
    rows = np.arange(m)
    F = np.ascontiguousarray(f + A_S @ table[x])                 # F[r, k] = f_r(k) + sum_s A_rs B(k, x_s)
    energy = float(f[rows, x].sum() + 0.5 * (F[rows, x] - f[rows, x]).sum())
    offset = 0.0 if energy0 is None else energy0 - energy       # the energy of the fixed variables
    pos_first, pos_late = pos.copy(), np.empty(m, dtype=np.int32)
    energy_c = ctypes.c_double(energy)
    first_c, late_c = ctypes.c_double(energy), ctypes.c_double()
    lowest = ctypes.byref(first_c), pos_first, ctypes.byref(late_c), pos_late
    marks = np.empty(2)   # the energy at the half and the second half's lowest at three quarters
    if sparse:
        _lib.pfa_anneal_sparse_swap(
            *stored, table, m, q, val, pos, F, num_steps, beta_start, beta_max, ctypes.byref(energy_c), *lowest,
            marks, int(rng.integers(1, 2**63)), tabu_lambda, tenure,
        )
    else:
        D = np.empty(P)
        noise_vec = -np.log(-np.log(rng.random(P)))
        noise_buffer = np.concatenate((noise_vec, noise_vec)).astype(np.float32)
        _lib.pfa_anneal_swap(
            A_S, table, m, q, val, e, rs, pos, F, D, noise_buffer, num_steps, beta_start, beta_max,
            ctypes.byref(energy_c), *lowest, marks, int(rng.integers(1, 2**63)), tabu_lambda, tenure,
        )
    states = []
    for value, p in ((first_c.value, pos_first), (late_c.value, pos_late)):
        x = x0.copy() if x0 is not None else np.empty(n, dtype=np.int32)
        x[S[p]] = val
        states.append((value + offset, x))
    return states


CANDIDATES = 5   # candidates per city for the tour moves: the alpha-nearest
BASIS = 2        # the exchanges' depth: 2, the 3-opt basis
WALK_SHARE = 0.8   # a restart's walk takes an unvisited candidate on the best tour's edge with this probability


def tour(points: np.ndarray, num_steps: int = 1000, seed: int = 42, rounding: str | None = None) -> np.ndarray:
    """
    The shortest closed tour through points (n x 2 coordinates). The candidate edges are each city's five
    alpha-nearest, from the Held-Karp relaxation; the kernel is Lin and Kernighan's chain of sequential 3-opt exchanges
    over them at zero temperature, the Boltzmann draw's limit on this landscape (see the README); the loop is PFA's:
    restarts from a walk along the candidate edges biased to the best tour's, each descended to a local optimum and
    folded into the best tour by edge assembly, as long as the budget holds. num_steps is the budget, the number of
    chains in all; the first descent runs whole. rounding: None for Euclidean distances, "nint" for TSPLIB's EUC_2D
    (rounded to the nearest integer), "ceil" for CEIL_2D. Returns the tour as the order of the cities (int64).
    """
    points = np.ascontiguousarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 8:
        raise ValueError("points must be n x 2 coordinates of at least eight cities")
    problem = _Tour(points, {None: 0, "nint": 1, "ceil": 2}[rounding])
    rng = np.random.default_rng(seed)
    t = problem.walk(rng)
    energy, spent = problem.descend(t)
    best = problem.encode(t)
    while spent < num_steps:
        t = problem.walk(rng, best)
        e, used = problem.descend(t)
        spent += used
        energy, best = problem.fold(energy, best, e, problem.encode(t), rng)
    return problem.decode(best).astype(np.int64)


class _Tour:
    """A tour problem: the coordinates, the candidates, the kernel's calls. A tour is kept as its edge set: per city,
    its two neighbours as one code, min * n + max, invariant to the tour's rotation and reflection; the kernel works on
    the city at each position, decoded for it and encoded from it."""

    def __init__(self, points, round_mode):
        self.n = n = len(points)
        self.round_mode = round_mode
        self.x, self.y = np.ascontiguousarray(points[:, 0]), np.ascontiguousarray(points[:, 1])
        flat = np.empty(n * CANDIDATES, dtype=np.int32)
        self.bound = _lib.pfa_tour_alpha(n, self.x, self.y, round_mode, CANDIDATES, flat)   # the Held-Karp bound reached
        self.nn = flat.reshape(n, CANDIDATES)
        self.nn_flat = flat

    def distances(self, i, j):
        r = np.sqrt((self.x[i] - self.x[j]) ** 2 + (self.y[i] - self.y[j]) ** 2)
        return np.floor(r + 0.5) if self.round_mode == 1 else np.ceil(r) if self.round_mode == 2 else r

    def encode(self, t):
        """The edge-set code of a tour (city at each position)."""
        prev, nxt = np.roll(t, 1), np.roll(t, -1)
        code = np.empty(self.n, dtype=np.int64)
        code[t] = np.minimum(prev, nxt).astype(np.int64) * self.n + np.maximum(prev, nxt)
        return code

    def decode(self, code):
        """A tour (city at each position) with the edge set code."""
        n = self.n
        a, b = code // n, code % n
        t = np.empty(n, dtype=np.int32)
        prev, cur = -1, 0
        for k in range(n):
            t[k] = cur
            nxt = int(a[cur]) if a[cur] != prev else int(b[cur])
            prev, cur = cur, nxt
        return t

    def walk(self, rng, best=None):
        """A tour by a random walk along the candidate edges: from a random city, an unvisited candidate on the best
        tour's edge with probability WALK_SHARE when there is one, else a random unvisited candidate, else the nearest
        unvisited city."""
        n = self.n
        on_best = None if best is None else np.stack((best // n, best % n), axis=1)
        seen = np.zeros(n, dtype=bool)
        cur = int(rng.integers(n))
        tour = [cur]
        seen[cur] = True
        for _ in range(n - 1):
            free = [int(c) for c in self.nn[cur] if not seen[c]]
            kept = [] if on_best is None else [c for c in free if c in on_best[cur]]
            if kept and rng.random() < WALK_SHARE:
                cur = kept[int(rng.integers(len(kept)))]
            elif free:
                cur = free[int(rng.integers(len(free)))]
            else:
                rest = np.flatnonzero(~seen)
                cur = int(rest[np.argmin(self.distances(cur, rest))])
            tour.append(cur)
            seen[cur] = True
        return np.array(tour, dtype=np.int32)

    def descend(self, t):
        """The kernel from tour t (city at each position, changed in place): chains to a local optimum. Returns the
        length and the number of chains."""
        chains = ctypes.c_long(0)
        length = _lib.pfa_tour_descent(self.n, self.x, self.y, self.round_mode, CANDIDATES, self.nn_flat, t, BASIS,
                                       ctypes.byref(chains))
        return length, chains.value

    def fold(self, energy, best, e, code, rng):
        """The best tour after edge assembly with a new one: the shortest of the two and their children."""
        dist = lambda i, j: float(self.distances(np.int64(i), np.int64(j)))
        cands = [(energy, best), (e, code)]
        child = eax.fold(best, code, self.n, dist, self.nn.astype(np.int64), rng)
        if child is not None:
            cands.append(child)
        return min(cands, key=lambda p: p[0])


def _one_hot_layout(groups):
    """Variable of each slot (labels in sorted order), slots grouped by variable in input order, values per variable,
    and where each variable's slots start in that order."""
    _, gid = np.unique(np.asarray(groups), return_inverse=True)
    order = np.argsort(gid, kind="stable")
    sizes = np.bincount(gid)
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    return gid, order, sizes, starts


def from_one_hot(couplings, fields, groups):
    """
    Converts a categorical problem from one-hot form to the form run() takes with sizes.

    One-hot form: energy 0.5 * x @ couplings @ x + fields @ x over binary x with one 1 per variable, groups[i] the
    variable of slot i; couplings within a variable are ignored. Variables are numbered in sorted order of their
    labels; value 0 of each is its first slot, the others follow in input order.

    Returns:
        (couplings, fields, sizes, offset): the energy in one-hot form is the energy in this form plus offset.
    """
    couplings = np.ascontiguousarray(couplings, dtype=np.float64)
    n = len(couplings)
    fields = np.zeros(n) if fields is None else np.ascontiguousarray(fields, dtype=np.float64)
    gid, order, sizes, starts = _one_hot_layout(groups)
    if len(gid) != n:
        raise ValueError("groups must give one variable per slot")
    is_first = np.zeros(n, dtype=bool)
    is_first[starts] = True
    ref = order[starts].astype(np.int32)
    rest = order[~is_first]
    m = len(rest)
    out_couplings = np.empty((m, m))
    out_fields = np.empty(m)
    offset = _lib.pfa_from_one_hot(couplings, fields, n, len(sizes), ref, m, rest.astype(np.int32),
                                          gid[rest].astype(np.int32), out_couplings, out_fields)
    return out_couplings, out_fields, sizes, offset


def to_one_hot(x, groups):
    """The one-hot x (0/1) of a state given as one value per variable, as run() returns it with sizes."""
    _, order, _, starts = _one_hot_layout(groups)
    one_hot = np.zeros(len(order), dtype=np.int8)
    one_hot[order[starts + np.asarray(x)]] = 1
    return one_hot
