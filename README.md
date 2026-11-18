# Pretty Fast Annealer

<p align="center"><img src="logo.png" alt="PFA"></p>

PFA minimizes an energy of single-variable and pairwise terms over discrete variables. Variable r takes one of several
values, a_r, and

```
E = Σ_r f_r(a_r) + Σ_{r<s} T_rs(a_r, a_s)
```

with fields f_r and coupling tables T_rs. A step moves one variable r from a_r to another value k, which changes the
energy by

```
ΔE(r→k) = F(k) − F(a_r),        F(k) = f_r(k) + Σ_s T_rs(k, a_s)
```

F(k), the field of value k, is the energy that value k adds given the current values of the other variables. A
variable with two values is a spin, and its one move a flip. Spins are given as s ∈ {−1, +1}ⁿ with couplings J and
fields h,

```
E(s) = ½ sᵀ J s + hᵀ s,        ΔE_j = −2 s_j (h_j + Σ_l J_jl s_l)
```

the case T_rs(s_r, s_s) = J_rs s_r s_s and f_r(s_r) = h_r s_r. A problem over 0/1 variables x is the same problem in
s = 2x − 1 (`pfa.from_binary`), and constraints enter as penalties (see Constraints). When the number of variables at
each value is fixed, k of n chosen, parts of given sizes, a permutation, a step exchanges the values of two variables
instead (see Exchanges). PFA is an annealer that never rejects: every step makes one move, chosen among all moves of
all variables by its Boltzmann weight.

```python
import pfa

spins = pfa.run(couplings, fields, beta_max=1.0, num_steps=100 * len(couplings), num_reps=10)
values = pfa.run(couplings, fields, sizes=sizes, num_steps=100 * len(sizes), num_reps=10)
values = pfa.run(couplings, fields, counts=counts, table=table, num_steps=100 * len(couplings), num_reps=10)
```

## Philosophy

**Compute per solved problem.** Given unlimited compute, random guessing finds the optimum, and no algorithm is needed. The goal of a solver is to reduce the compute it takes to reach the lowest state of a given problem.

**Tuning is compute.** The work spent tuning parameters for a problem is part of the work of solving it. A good solver needs little tuning.

**Solve the problem as stated.** Mapping a problem to another form takes polynomial time, but if solving takes exponential time, a polynomial increase in size is an exponential increase in time. An optimal solver solves the problem in its native space.

**A good solver has to sweat.** The more problem types a solver has been benchmarked on, the more kinds of local optima it has seen, and the better it is on each of them.

**Solving is one stage.** Modeling, mapping into an energy, reducing the problem, solving, mapping back and deploying: in production pipelines, all of these stages matter.

## Algorithm

Each step makes one move, chosen among all moves of all variables with probability

```
p(r→k) = e^(−β ΔE(r→k)) / Σ over all moves e^(−β ΔE)
```

while the inverse temperature β rises linearly from 0 to the end of cooling, `beta_max` in the first repetition and
found by the repetitions after that (see The end of cooling). Every state has one value per variable. So a
variable that must take exactly one of several values is not written as a one-hot group of spins, where every single
flip breaks the constraint and a penalty must hold it: there is no penalty to set and no barrier between the values,
and a move changes the whole variable in one step.

**Equilibrium.** Read as a continuous-time process, each move fires at rate e^(−β ΔE) and its reverse at
e^(+β ΔE). The ratio e^(−2β ΔE) is detailed balance for e^(−2β E): run at β, PFA samples the Boltzmann distribution of
inverse temperature 2β. Skipping the waiting times also weights each state by its exit rate, Σ over its moves of
e^(−β ΔE), favoring states with many cheap exits. An optimizer doesn't care: it keeps the best state it has seen.

**Tabu.** When variable r moves at inverse temperature β_r, the ΔE of every move of r is raised by λ/β_r for the next
V/4 steps (the tenure), V the number of variables, with λ = 3: three times the temperature at which it moved.

```
p(r→k) ∝ e^(−β (ΔE(r→k) + λ/β_r))
```

Right after the move this divides the weight of r's moves by e^λ; as the anneal cools, by e^(λβ/β_r). Early on, while
β grows fast relative to β_r, the tabu is nearly hard. A move of a tabu variable still outweighs a move of a free one
when ΔE(r→k) + λ/β_r < ΔE(s→l), so a move worth more than the penalty, like stepping back down from an expensive
climb, still happens. Among moves of similar cost, the anneal picks variables it hasn't just moved. Hot, this spreads
the moves over more variables; cold, it keeps the state moving between low-energy states instead of moving the same
few variables back and forth.

λ is in units of the temperature and the tenure a fraction of V, so the rule is unchanged when the energy is rescaled
(f, T, β → αf, αT, β/α) or V grows. Both constants are tuned on benchmarks, not derived, and set by `tabu_lambda` and
`tabu_tenure`. The tabu breaks detailed balance, which an optimizer doesn't need. With it, the best end of cooling is
roughly twice what it is without, and the re-annealing and the tabu on the value left moved it colder again on
structured problems: the best end of cooling changes with every change to the walk, and it differs between problem
classes at the same scale of ΔE (on quadratic assignment, by four times between random and grid instances). So the
repetitions find it (see The end of cooling).

**Repetitions.** Repetition 1 of R starts from a random value for every variable at β = 0. Each later one, a restart,
starts from the best state so far, with new random numbers and β starting partway up the ramp:

```
repetition 1:  random values        β: 0               → beta_max
restart:       best state so far    β: u/(R−1)·β_end   → β_end
```

where β_end is the restart's end of cooling (below) and u the number of restarts so far that brought nothing, that
is, did not lower the best energy. The first restart reheats to β = 0, a second anneal from scratch that gives the
recombination two independent states. The restarts form an outer anneal: each reheats the best state, at first hot
enough to leave its basin and search widely, later only enough to refine it. The starting β rises by one step of
β_end/(R−1) each time a restart brings nothing, so the restarts stay hot for as long as hot restarts pay.

**The end of cooling.** Where the cooling should end is the one quantity the walk cannot know in advance. Too hot and
the state is still moving when the anneal stops; too cold and the second half of every anneal is spent frozen. The
right end differs between problem classes by an order of magnitude at the same scale of ΔE and moves with every change
to the walk, so the repetitions find it. Each anneal reports whether it was still improving when it stopped: its last
quarter lowered its lowest state by more than a tenth of what its third quarter did. A ratio, so the reading is the
same at every size and energy scale, where a count of idle steps is not. After an anneal that had frozen, the next ends
twice as hot, until one is seen still improving. That end and twice it form a pair, and the repetitions alternate
between the two. Once both ends have been run as restarts, the pair moves one factor towards the end whose restarts
lowered the best energy more; when that direction reverses, the best end lies within the pair, which narrows, its
factor to its square root, around the winner: a line search that homes in. The two ends are compared by what they
gained, not by what the trace says, because on a landscape whose scale changes as the state settles the trace follows
the smallest scale and the gains follow the energy. `beta_max` is only the end of the first repetition. Against the
fixed end found by a scan over `beta_max`, an oracle no unseen instance gets, the moving end ties within noise on
most problem classes, wins by a factor two on a dense glass and trails by one standard error on a sparse one; against
any fixed end off by a factor two or more it wins outright. What it does not choose is the length
of an anneal: a schedule of doubling lengths covers that axis without a parameter, but the end is found across
anneals and a geometric schedule has too few anneals that carry budget to find it.

**Elimination.** Before the anneal, the problem loses the values that no optimum needs. Value a of variable r is
removed when another value b of r has

```
f_r(a) − f_r(b) + Σ_s min_c [ T_rs(a, c) − T_rs(b, c) ]  ≥  0
```

with the sum over the neighbours s of r, the variables coupled to it, and the minimum over the values c of s still
allowed. The left side is the least that F(a) − F(b) can be, each neighbour taken at the value worst for b. So b is at
least as good as a in every state: any state that uses a has one of no higher energy that does not, and a state of the
lowest energy always remains. With ≥ rather than >, values that are never better than another go too, duplicates
included. A removed value of s drops out of the minima over c in the tests of the neighbours of s. That can only raise
those minima, so more values can become removable, and the test repeats until nothing changes.

The anneal then runs on the values left, and variables left with one value drop out. The elimination has no
parameter. How much goes depends on the problem: a value goes when its field costs more than its couplings can ever
win back against some other value. Problems given as spins are not reduced. For a spin the test would be
|h_i| ≥ Σ_j |J_ij|: a field that outweighs all the spin's couplings together.

**Recombination.** A repetition whose lowest state is above the best state can still hold some variables in a better
arrangement. Take two states a and b, and the variables on which they differ. Setting variables r and s of a to their
values in b changes its energy by

```
Δ_r + Δ_s + I_rs,        I_rs = T_rs(a_r, a_s) − T_rs(a_r, b_s) − T_rs(b_r, a_s) + T_rs(b_r, b_s)
```

with Δ_r the change of setting r alone: the two choices are independent unless I_rs ≠ 0. For spins
I_rs = 4 J_rs a_r a_s, nonzero whenever the coupling is; two coupled variables with more values can still be
independent. So the differing variables fall into clusters, the connected components of the pairs with I_rs ≠ 0, and
no such pair joins two clusters. Setting a whole cluster G to its values in b changes the energy by

```
Δ(G) = Σ_{r ∈ G} Δ_r + Σ_{r < s ∈ G} I_rs
```

whatever is done to the other clusters, so a takes every cluster with Δ(G) < 0. That is the lowest of the 2^m states
that have each of the m clusters from a or from b, found in one pass and never above either.

Each repetition records two states: the lowest of the first half of its steps and the lowest of the second half. In a
restart the first is usually the state it started from; the second is where the reheated anneal settled, a different
state of low energy. After the repetition, the lowest of three states, these two and the best state so far, becomes
the best state and takes from the other two every cluster they have lower. A repetition that ends above the best
state still contributes its good parts, and brings nothing only when the best energy stays where it was.

How much this gives depends on how two low states differ. If the couplings are sparse and fields or a few strong
couplings hold most variables in place, low states differ in many small clusters, and the best state improves cluster
by cluster. On problems of that kind (side-chain packing, MAX-2-SAT, sparse glasses with fields or with couplings on
several scales), with R = 10 and R = 30 the final energy was lower in 82 of 84 trials (seven problems, six seeds, both
values of R) than with the earlier scheme, no recombination and every restart one step colder than the last whatever
it brought. If every two variables are coupled, or the states differ on so many variables that the clusters merge,
there is one cluster, and the fold alone keeps the lower state.

**Re-annealing.** What the fold cannot take, the variables on which the states still differ, is annealed again as a
problem of its own. The agreeing variables stay fixed and their couplings to the differing ones go into the fields;
the differing variables start from random values and anneal from β = 0 to the repetition's end, with steps in proportion to
their share of the moves. What the anneal finds folds into the best state. When that lowers it, the anneal's two
states are new material: the variables on which they differ from the best state are annealed again, round after
round, until an anneal brings nothing. Two low states of a dense problem that share most variables usually hold the
shared part right and the rest in two arrangements of about the same cost, and the anneal of the rest alone, under the
shared part as fields, looks for a third. Two states that share nothing leave nothing to fix, and nothing is done.

On quadratic assignment, where every two variables are coupled and the fold alone takes nothing, the re-annealing
cut the distance to the best known values by three to five times at equal time on the instances whose low states
share many assignments (distances on a grid, structured flows), and left the random instances, whose low states share
almost nothing, within the spread between seeds. On spin glasses, dense or sparse, the results stayed within the
spread between seeds at equal steps, the re-anneals adding up to 40% to the time on the dense ones.

Both recombinations need the states in one frame. When the couplings have a symmetry (the reflections and rotations
of a grid of locations), independent anneals land in different images of the same optimum, share no variables, and
the fold and the re-anneal take nothing from each other; `run()` does not align them. Restarts from the best state
stay in its frame when the reheat is mild and leave it when it is not.

**Exchanges.** With `counts`, the number of variables at each value is fixed: k of n items chosen, a graph cut into
parts of given sizes, a permutation with one variable per value. The couplings factor, T_rs(a, b) = A_rs B(a, b),
with A between the variables and B between the values (flows and distances in quadratic assignment, edge weights and
"same part or not" in partitioning). A step exchanges the values a and b of two variables r and s, chosen among all
pairs at different values by the same rule, and changes the energy by

```
ΔE = (F_r(b) − F_r(a)) + (F_s(a) − F_s(b)) + A_rs κ(a, b),        κ(a, b) = 2 B(a, b) − B(a, a) − B(b, b)
```

the two single moves plus a correction for the pair's own term, which both F's count with the old values. The counts
never change, so there is no penalty to hold them and no barrier between feasible states. The tabu is on the value a
variable has left, not on the variable: after the exchange, r's moves to a and s's moves to b are raised by λ/β, each
for `tabu_tenure` times the mean number of steps between two moves of a variable at that value (the pairs divided by
the positions of the other values: n/2 steps in a permutation). A pair's penalty is the sum of its two reassignments'.
The two variables stay free to move elsewhere, so a long tenure does not freeze the state, it forbids the way back:
the walk between low states cannot retrace itself and has to go on. Against the tabu on the variable, at five minutes
on quadratic assignment, the random instances closed two thirds of their distance to the best known values and the
structured ones a third to two thirds; the grid instances stayed where they were.
Recombination needs clusters that keep the counts: besides the pairs with I_rs ≠ 0, the differing variables are
joined along the cycles of their exchanges, the arcs a_r → b_r on the values, where every value has as many arcs in
as out. Every cluster then moves as many variables into each value as out of it, and taking it keeps the counts. For a
permutation the cycles are those of a⁻¹b, and the result is again the lowest of the 2^m states; otherwise which arcs
form a cycle is one choice among several, which loses choices and nothing else. Elimination does not apply: a value
is dominated only against the states that keep the counts.

## Compared with simulated annealing

SA here follows Isakov et al.,
[Optimised simulated annealing for Ising spin glasses](https://arxiv.org/abs/1401.1084), built like PFA's
implementation: full float64 coupling rows, one replica, one core. The paper's tricks for low-precision couplings,
many replicas or sparse graphs (lookup tables, multi-spin coding, bipartite updates) are left out on both sides, and
so is PFA's own storage for sparse couplings.

SA proposes one spin at a time and accepts the flip with probability min(1, e^(−β ΔE_j)). As the anneal cools, it
rejects almost every proposal; PFA never rejects. The comparison is of this step: repetitions, elimination and
recombination sit outside it and would apply to either method.

Each method ran with its final inverse temperature tuned for every length of anneal, on spin problems. A PFA flip
costs **1.2–1.4×** an accepted SA flip; a rejected SA proposal costs about 1/n of an accepted flip. With flips and
rejected proposals both counted, PFA reaches the same energy in less work by:

| problems | speedup |
|---|---|
| spin glasses | **1.5–2×** |
| constrained problems with large penalties | **15–1000×**, or SA never reaches that energy |

**Cost per step.** A flip in either method updates all n local fields. Per spin, an accepted SA flip moves 24 bytes:
a float64 coupling read, a float64 field read and written. A PFA flip also reads the spin and a stored random number
(see Gumbel-max sampling), 4 bytes each: 32 bytes, 4/3 of an accepted SA flip. A rejected SA proposal reads one field
and one random number, about 1/n of a flip. Work in units of n accepted SA flips (one per spin), rejected proposals
included:

```
PFA:  (4/3) · flips / n          SA:  (accepted flips + proposals / n) / n
```

**Descent.** Metropolis accepts every downhill flip, so it takes them in sweep order, whatever their size. PFA
weights each by e^(β |ΔE_j|) and takes the biggest first. Each PFA flip does more of the descent.

**Sweeps and the tabu.** Random proposals repeat some spins and miss others: after n of them, 1/e ≈ 37% of the spins
are still untried. A sweep proposes each spin once per n proposals, and does better. PFA has no proposals to order;
its repeat-avoidance is the tabu. A sweep is a hard rule on proposals that keeps the Boltzmann distribution
stationary; the tabu is a soft rule on flips that does not.

**Constraints.** Many hard problems are an objective plus constraints written as penalties:

```
E(s) = P · C(s) + O(s)
```

C(s) ≥ 0 is zero exactly on feasible states, O(s) is the objective, and P is much larger than the scale ε of changes
in O. The hard case: every single flip from a feasible state breaks a constraint, as with one-hot or cardinality
constraints. Then every flip costs about P or more, and getting from one feasible state to another takes two flips,
one that breaks a constraint and one that repairs it.

SA accepts the breaking flip with probability e^(−β (P + ΔO_i)), so it moves between feasible states only while
β ≲ 1/P. It needs β ≳ 1/ε to tell good objective values from bad. With P/ε ≫ 1, no β does both: at β = 1/ε each
breaking flip takes about e^(P/ε) proposals, and the constraints freeze long before the objective is optimized.

PFA is blind to a common shift of all ΔE, which cancels in the normalization of p. From a feasible state, the flips
that break one constraint all carry the same P, so

```
p_i ∝ e^(−β ΔO_i)
```

The objective alone chooses which constraint to break, whatever P is, and PFA leaves every state in one step, at any
β. It can run cold enough to resolve the objective without freezing on the constraints. Any method that never rejects
has this advantage.

**Long anneals.** Late in the anneal, what counts is how far the state still moves without climbing. Held at the same
low temperature (PFA at β, SA at 2β, see Equilibrium), PFA sits lower in energy than SA and still drifts further from where it started.

## Implementation

The annealer runs in compiled C with SIMD, on one core, on any couplings and fields. It has six kernels, the compiled
anneal loops: one for spins, one for variables with more values and one for variables that exchange values (see
Exchanges), each with the couplings either as an n × n array (the array kernels) or as lists of the nonzero ones (the
sparse kernels, see Sparse couplings). Value 0 of every
variable has no field and no couplings (any problem can be shifted to that form, see `sizes` under Usage), so each of
the other values has a row of the couplings, n rows in all, and a move reads the row of the value it leaves and of
the value it enters, none for value 0. On the array kernels those are full float64 rows, zeros included, and one pass
over the n rows updates every field and selects the next move. A move therefore costs the same on every problem of a
given size. The tricks of the array kernels:

- **Gumbel-max sampling.** Drawing from p directly costs n exponentials and a normalization per step. The
  [Gumbel-max trick](https://lips.cs.princeton.edu/the-gumbel-max-trick-for-discrete-distributions/) removes both:
  with each g_k an independent standard Gumbel, g = −log(−log u) for u uniform on (0, 1),

  ```
  P( argmax_k (−β ΔE_k + g_k) = i ) = p_i
  ```

  so picking a move costs one multiply, one add and one comparison per row.
- **Noise drawn once per repetition.** At step t, row k reads g_((k + ω_t) mod n), with a fresh random offset ω_t.
  Each step on its own is exact Gumbel-max sampling; across steps the same values return shifted, a cheap stand-in
  for fresh noise. A random offset, unlike a shift by one, keeps large values from marching through neighbouring
  indices, so results don't depend on the order of the variables. The vector is stored as g followed by g, so the n
  values from any offset are one contiguous stretch: no wraparound, no modulo. Offsets come from a xorshift64*
  generator, mapped to [0, n) by a multiply and a shift, without division.
- **Scores.** Row k, a value other than 0 of variable r, is one candidate move, scored in float32 as
  −ΔE − c_r + g_k / β, with c_r = λ / β_r the tabu penalty of r, and written

  ```
  o_r + S_k F(k) + g_k / β
  ```

  If k is not the current value a_r, the move is r → k: S_k = −1 and o_r = F(a_r) − c_r, with F(0) = 0. The row of
  a_r itself stands for the move back to value 0: S_k = 0. The pass updates every F(k) with the row of the value left
  and of the value entered, and reads o_r only for variables with more than two values, as it reaches them. Rounding
  can change a pick only when two scores agree to 7 digits (2⁻²⁴ ≈ 6·10⁻⁸).
- **Spins.** A spin has one row and one move, and the spin kernel stores φ_j = (h_j + Σ_l J_jl s_l) / 2 in float64,
  so ΔE_j = −4 s_j φ_j. Dividing −β ΔE_j + g_j by 4β leaves the argmax unchanged and gives the score

  ```
  s_j φ_j + g_j / (4β)
  ```

  with spins stored as float32 ±1: no branch and no o_r. Flipping spin i changes every φ_j by −s_i J_ij, with s_i its
  value before the flip. With s_i = ±1 fixed for the pass, that is a plain subtract or add of row i, read in place: no
  multiply, and s_i is not read again. φ_i itself does not change. After N flips, rounding has added at most about
  N · 2⁻⁵³ of the largest field: negligible. A variable with two values among variables with more is scored as a
  spin, not by the rule above: S_k = +1 when k is its value and −1 when it is at value 0, with o_r = 0.
- **Argmax without indices.** A vector argmax carries an index next to every running maximum, which roughly doubles
  the selection work. The pass instead keeps only maxima and records which chunk of 64 rows (8 vectors on SVE)
  reached the largest one. Afterwards it rescores that chunk, still in cache, and returns its first row at the
  maximum: about 64/n of a pass extra. The rescore repeats the pass's float operations exactly (the build stops the
  compiler from fusing multiply-adds on its own), so it finds the same maximum, and ties go to the smallest index.
- **Tabu for free.** For a spin, the penalty, c_i = λ / (4β_i) in score units, is added into the stored field,
  φ̃_i = φ_i − s_i c_i, so that s_i φ̃_i = s_i φ_i − c_i, and taken out when the tenure of i ends or i flips again. A
  variable with two values among variables with more has it added into F the same way; a variable with more values
  has it in o_r. The pass reads nothing extra; the anneal loop adds and removes penalties
  at O(1) per step.
- **Energy tracked, not computed.** E += ΔE at each move, O(1). The lowest state of each half of a repetition is
  copied only when the anneal steps up from a new low, so a descent through many new lows costs one copy.

### Sparse couplings

When each variable has few neighbours, `run()` uses kernels that keep only the nonzero couplings, and no step passes
over all rows. The input format plays no part: the same couplings as an array or as a `scipy.sparse` matrix give the
same result. Schedule, fields, tabu and bookkeeping are those of the array kernels. The selection is not, so a seed
gives a different result on the sparse kernels than on the array kernels.

- **Fields.** A move of variable r changes F only on the rows of r's neighbours: for a spin with d neighbours, d adds
  instead of n. For variables with more values, the couplings are kept as full tables between coupled variables.
- **Selection without a pass.** Gumbel-max with stored noise needs the pass: the offset moves, so every score changes
  at every step. The sparse kernels draw from p directly instead, which costs one exponential per ΔE that changed:
  for a spin d + 1, n if every spin were coupled to every other; for a variable with more values, every row of its
  neighbours, since each ΔE of a neighbour s, F(k) − F(a_s) over the values k of s, changes with F(a_s). The weights w_k = e^(−β ΔE_k) of
  all moves are the leaves of a tree of sums with fan-out 8, every entry the sum of the 8 below it. A uniform number
  times the total, followed down the tree, lands on move k with probability

  ```
  w_k / Σ_l w_l
  ```

  A move gives new weights to the rows it changed. The sums above them are recomputed from their 8 children, so
  rounding errors do not build up. A draw or an update touches log₈ n entries.
- **β moves at every step**, and with it every weight. The tree keeps the weights of an earlier β₀, and the draw is
  corrected: move k is accepted with probability

  ```
  e^(−(β − β₀) (ΔE_k − ΔE_min))        ΔE_min ≤ every ΔE
  ```

  and otherwise drawn again. Weight times acceptance is proportional to e^(−β ΔE_k), so the move made has exactly the
  distribution of this step's β. A draw that is not accepted costs a walk down the tree, not a step.
- **Recomputing the weights.** Rejected draws become more frequent as β leaves β₀. If they cost ρ · t at the t-th
  step after all weights were computed, and computing them costs Q, a cycle of K steps costs Q/K + ρ K/2 per step:
  least when ρ K²/2 = Q. So all weights are recomputed, at the current β, when the rejected draws since the last time
  have cost as much as that takes. Weights are kept relative to the largest, and are also recomputed if their total
  leaves the range of a float64.
- **Lowest states.** They are brought up to date from a list of the variables moved since they were last written,
  not copied.

A sparse step costs a fixed part plus one exponential per changed weight; an array step costs one pass over all n
rows. `run()` takes the sparse kernel where that is the cheaper step:

```
spins:        d < n / 30 − 7        d: mean number of neighbours of a spin
more values:  v < n / 9 − 20        v: rows a move changes, the values other than 0 of the moved variable's
                                    neighbours, averaged over the n rows
```

The constants are fitted to timings of both kernels on one machine (Apple M4). For spins that is fewer than about 3%
of the couplings nonzero from a few hundred spins up; for variables with more values, about a tenth of the value
pairs coupled.

Memory comes before speed. The array kernels need the couplings as an n × n float64 array, 8 n² bytes. If that is
more than half of the machine's memory, the kernel whose couplings take less memory runs: the sparse one when its
entries, 24 bytes per nonzero coupling, plus its tables where there are any, take less than the array.

### Elimination

One function, in `eliminate.c`, reads both layouts of the couplings, the array and the tables of the sparse kernel:
in either, the couplings of a value to the values of one neighbour are one stretch of memory.

- **A failing test stops early.** The test is made for every two values a and b of one variable, and most of them
  fail. Its sum starts from an upper bound, U(a) − U(b): the fields of a and of b with every neighbour of their
  variable held at its first value still allowed. That is one sum per value, used in every test the value enters.
  Each neighbour in turn then replaces its term by its minimum, which can only lower the total. Once the total is
  below 0 the test has failed, and the remaining neighbours are skipped.
- **The neighbour that decides is read first.** The neighbour that took a test below 0 moves to the front of the
  order in which that variable's neighbours are read, where the next tests meet it.
- **Only what changed is tested again**: the variables with a neighbour that lost a value.
- **No tolerance.** The comparison is the floating-point one, so two values that tie only up to rounding can both stay.
- **The reduced problem.** Value 0 must have no field and no couplings, so the first value left of each variable,
  k_r, becomes its value 0. For a value a of r and a value c of s,

  ```
  T'_rs(a, c) = T_rs(a, c) − T_rs(a, k_s) − T_rs(k_r, c) + T_rs(k_r, k_s)
  f'_r(a)     = G(a) − G(k_r),        G(a) = f_r(a) + Σ_s T_rs(a, k_s)
  ```

  which changes the energy by a constant. The couplings of a variable left with one value are then in the fields.
  `run()` picks the kernel again for the reduced problem.

### Recombination

`fold.c` has one function for spins on the array kernel, one for spins on the sparse kernel, and one for variables
with more values, which reads both layouts as the elimination does.

- **Clusters.** Each differing variable starts as its own set, and two sets are merged whenever a variable of one
  interacts with a variable of the other (union-find). Each set then has one sum, Δ(G). The work is the couplings of
  the differing variables: their stored neighbours on the sparse kernels, their rows of the array otherwise.
- **No tolerance on the interactions.** Two differing variables interact when I_rs is not exactly 0. Where it is
  nonzero only by rounding, two clusters are merged that could have been chosen separately: a choice is lost,
  exactness is not. A cluster is taken when Δ(G) is below 0 by more than the rounding of its own sum, 2⁻⁴⁰ of the
  magnitudes summed: two arrangements of a cluster that tie exactly would otherwise be taken back and forth between
  the rounds, each time by a rounding error.
- **Two states per repetition.** Each kernel keeps the lowest state of the first and of the second half of its
  steps: at the midpoint the current state becomes the lowest so far of the second half. Where the split falls
  matters little: at a quarter, three quarters or nine tenths of the steps the results were the same within the
  spread between seeds. Without a split most of the gain is lost, because the lowest state of a whole restart is
  usually the state it started from.
- **Rounds.** The lowest of three states, the repetition's two and the state it started from, becomes the best state
  and takes clusters from the other two, round after round until its energy, a float64, no longer falls. The first
  repetition has only its two. A repetition brought nothing when that energy is not below the energy of the state it
  started from.
- **Re-anneals.** The problem of the differing variables is built from the kernel's own inputs: their couplings among
  themselves, their fields plus their couplings to the fixed variables at their values, and with `counts` the counts
  of their own values. `run()` picks the kernel for it again. The anneal returns its two states as whole states with
  the energy of the fixed part added, so the fold and the rounds see nothing special.

### Exchanges

The moves are pairs, so the exchange kernel (`swap.c`) keeps the ΔE of every pair and selects among the pairs the way
the spin kernel selects among spins.

- **Positions.** The variables sit in positions sorted by value, and a move exchanges the variables of two positions.
  The pairs of a position with the positions of the later values are one contiguous stretch, so no same-value pair is
  ever scored: k(n − k) pairs for k chosen of n, n(n − 1)/2 for a permutation. The values are ordered by count with
  the largest last, which leaves the fewest rows.
- **Two tables.** F_r(k) for every variable and value, n × q float64, and D, the ΔE of every pair, float64. After the
  exchange of positions t and u (variables r, s at values a, b), with α_w = A_wr − A_ws over the variables and
  β_k = B(k, b) − B(k, a) over the values,

  ```
  F_w(k) += α_w β_k                       every w: A_rr = A_ss = 0 makes r and s fit the rule
  D_xy   −= (α_x − α_y) (β_x − β_y)       x, y positions other than t, u, with the α, β of their variable and value
  ```

  The second is Taillard's update for quadratic assignment and holds for any symmetric B. The pairs of t and of u
  themselves are recomputed from F, O(1) each from the two columns F(a) and F(b) collected during the update of F.
- **One pass.** The update of D is applied in the next step's pass, which scores every pair −ΔE − c_x(b) − c_y(a) + g/β
  in float32 and selects as the spin pass does, with pairs in place of spins: D read and written (16 bytes), one noise
  value (4 bytes) and, from cache, the α, β and two tabu penalties of the column. One noise value per pair, a vector of the
  number of pairs stored twice. Rows are padded to multiples of 64 pairs with +∞, so a chunk never straddles two rows
  and the rescore of the winning chunk works as for spins. The recomputed pairs are stored with the update that pass
  will subtract added back; that is one rounding, and every pair is recomputed whenever one of its positions moves,
  so errors do not build up.
- **Cost.** Per step, one pass over the pairs and one rank-1 update of F, nq multiply-adds. Memory: 8 bytes per pair
  for D, 4 for its noise, 8nq for F, 4n(n + q) for the tabu penalties, and the user's A and B.
- **Tabu.** The pair of positions (t, u) moves pos[t] to val[u] and pos[u] to val[t], so the penalties of a row are two
  vectors over its columns u: R[pos[t]][u], the row's variable at the column's value, and C[val[t]][u], the column's
  variable at the row's value, floats by position (n rows of R, q of C). A penalty set or expired writes the positions
  of its value in R and one entry of C; an exchange rewrites the two columns of C. Each value keeps a queue of the
  variables that left it, with the step; the ones whose tenure is over leave at the front, unless renewed since.
- **Recombination.** `pfa_fold_swap` joins the differing variables along the cycles of their exchanges before summing
  the clusters: a walk over the unused arcs from any value closes a cycle whenever it returns to a value it has
  visited, and can only end where it started.

**Sparse couplings** (`sparse_swap.c`). When each variable is coupled to few others, an exchange changes F on the two
variables and their neighbours only, and the kernel draws the pair from trees of sums without a pass over the pairs.
The weight of a pair with A_rs = 0 factors into the weights of the two single moves, e^(−β ΔE(r→b)) e^(−β ΔE(s→a)),
so the variables at value a form the class a→b with the sum Z(a→b) of their weights, and the uncoupled pairs of the
block of values (a, b) weigh Z(a→b) Z(b→a) − S(a, b), S the same products summed over the coupled pairs of the block,
kept as a running sum. A draw picks a block by that weight, then one variable from each class by its weight, again
while the two are coupled, and after 16 such draws by enumerating the two classes. The coupled pairs at different
values have their full weights in a second tree and are drawn from it with its share of the total. The classes are
the subtrees of one tree: each class is padded to a power of 8 so that it is one entry at one level, its class sum,
and a draw within it walks down from there. β rises between rebuilds as in the other sparse kernels, with the largest
x of any pair from trees of maxima kept next to the sums. A step refreshes the leaves of the exchanged variables and
their neighbours, (2d + 2)(q − 1) with one exponential each, the blocks of the values touched, and the coupled pairs
of those variables, (2d + 2) d. `run()` takes the sparse kernel where that is cheaper than streaming the pairs:

```
(2d + 2) (120 (q − 1) + 18 d)  <  0.32 · pairs + 0.17 n q + 10 n        in nanoseconds
```

with the constants fitted to timings of both kernels on one machine (Apple M4). For few values on a sparse graph
the sparse kernel is far faster: at degree 30 about twice, at degree 3 a hundred times. For a permutation every
changed variable touches all q = n classes, so the array kernel keeps problems below about a thousand variables, and
the sparse one takes the large sparse ones.

## Usage

`make` (GCC or Clang) compiles `pfa/native/pfa.c` for the current machine. It includes the six kernels: `spin.c`,
`categorical.c` and `swap.c` for couplings as an array, and `sparse_spin.c`, `sparse_categorical.c` and
`sparse_swap.c`, which draw from the tree of sums of `sums.c`; and `eliminate.c` and `fold.c`. The three array
kernels use the one SIMD path that matches the machine; the rest is the same code on every path:

| path | files | used on |
|---|---|---|
| SVE | `spin_sve.c`, `categorical_sve.c`, `swap_sve.c` | ARM with SVE (Graviton3 and 4, NVIDIA Grace, A64FX, ...) |
| NEON | `spin_neon.c`, `categorical_neon.c`, `swap_neon.c` | every other 64-bit ARM (Apple Silicon, Graviton2, Ampere Altra, ...) |
| AVX-512 | `spin_avx512.c`, `categorical_avx512.c`, `swap_avx512.c` | x86-64 with AVX-512 |
| AVX2 | `spin_avx2.c`, `categorical_avx2.c`, `swap_avx2.c` | x86-64 with AVX2 |
| scalar | `spin_scalar.c`, `categorical_scalar.c`, `swap_scalar.c` | anything else |

`pfa.simd_path()` reports which one was built; `make -B ARCH_FLAGS="-mavx2 -mfma"` rebuilds for another
target.

`pfa.run()` returns the lowest-energy state found, as ±1 spins (int8), or with `sizes` or `counts` as the value of
each variable (int64). Its parameters:

| parameter | what it is | default | notes |
|---|---|---|---|
| `couplings` | the couplings, an n × n symmetric matrix with zero diagonal: J for spins, with `sizes` between the values, with `counts` A between the variables; an array or a `scipy.sparse` matrix | required | The format does not change the result: `run()` picks the array or the sparse kernels from the number of nonzero couplings (see Sparse couplings and Exchanges). The array kernels read a C-contiguous float64 array in place; any other input is converted once, a `scipy.sparse` matrix to n × n float64. If that array would take more than half of the machine's memory, the kernel that needs less memory runs. |
| `fields` | the fields, a length-n vector: h for spins, or with `sizes` of the values; with `counts` n × q, the field of each variable at each value | `None` (zeros) | Enter only through F, so they cost nothing per step. |
| `sizes` | the number of values of each variable | `None` (spins) | Variable r takes a value in 0 … `sizes[r]` − 1. Each value other than 0 has a row and column of `couplings` and an entry of `fields`, ordered by variable (variable 0's values 1 … `sizes[0]` − 1 first), so n = Σ (`sizes[r]` − 1). The energy is the fields of the values taken plus each coupling between two of them once: ½ xᵀ C x + fᵀ x with x_k = 1 for the values taken. Couplings between values of the same variable must be zero. Values that no optimum needs are removed first (see Elimination). |
| `counts` | the number of variables at each of q values, summing to n | `None` | Variable r takes a value in 0 … q − 1, and `counts[v]` variables are at value v throughout (see Exchanges). The energy is Σ_r `fields[r, x_r]` + Σ_{r<s} `couplings[r, s]` · `table[x_r, x_s]`. k of n: `counts` (n − k, k) with `table` [[0, 0], [0, 1]]; parts of given sizes: `table` 1 − I to count the edges between parts; a permutation: `counts` all 1, `table` the distances. |
| `table` | with `counts`, the q × q symmetric table between the values | required with `counts` | |
| `beta_max` | the end of cooling of the first repetition | `1.0` | Measured in 1/energy: scaling the energy by α scales it by 1/α, so the default suits only ΔE of order one. The later repetitions find the end themselves, a factor two per repetition from this start (see The end of cooling), so a start within a factor ten costs a few repetitions and a start off by a thousand costs ten. A linear schedule is nearly as good as optimized ones (Isakov et al.). |
| `num_steps` | steps per repetition, the length of the anneal | `1000` | The budget of one repetition: on the array kernels num_steps · n field updates, or num_steps / V moves per variable. Longer anneals track the cooling more closely and reach lower energies. |
| `num_reps` | repetitions of the anneal, R | `1` | The first starts from a random value for every variable; each later one restarts from the best state so far with β rising from u/(R−1) of its end of cooling, u being the number of restarts so far that did not lower the best energy (see Repetitions), and the end of cooling moves between repetitions (see The end of cooling). After each repetition the best state takes the clusters of variables that the repetition has lower, and the variables that still differ are annealed again (see Recombination). |
| `seed` | random seed | `42` | Sets the first repetition's state and every repetition's Gumbel noise vector and offsets, or on the sparse kernels its draws, so results are reproducible. |
| `tabu_lambda` | the tabu penalty λ, in temperatures | `3.0` | Divides the weight of a variable's moves by e^λ just after it moved (see Tabu). Rarely worth changing. |
| `tabu_tenure` | the tabu tenure, as a fraction of the number of variables V | `0.25` | A variable stays tabu for `tabu_tenure` · V steps; with `counts`, its return to the value it left is tabu for `tabu_tenure` times the mean number of steps between two moves of a variable at that value. 0 turns the tabu off. Rarely worth changing. |

A problem in one-hot form, ½ xᵀ C x + fᵀ x over binary x with one 1 per group, converts with
`pfa.from_one_hot(couplings, fields, groups)`, with `groups[i]` the variable that x_i belongs to. It returns the
couplings, fields, sizes and the energy offset between the two forms; `pfa.to_one_hot(values, groups)` converts a
result back. A problem over 0/1 variables converts to spins with `pfa.from_binary(couplings, fields, offset)`, and
`pfa.energy(state, couplings, fields, offset)` is ½ sᵀ J s + hᵀ s + offset for a ±1 or a 0/1 state.

`examples/spins.py` solves a small problem given over 0/1 variables, `examples/colouring.py` colours a graph with
several values per variable, `examples/qap.py` solves a quadratic assignment and `examples/partition.py` cuts a graph
in two equal parts, both by exchanges, and `examples/throughput.py` measures steps per second.
