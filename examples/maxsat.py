"""A random partial Max-SAT instance: hard 3-clauses that must all hold and unit soft clauses, the fewest falsified
the better, found within one budget of flips."""
import numpy as np

import pfa

rng = np.random.default_rng(0)
n, m = 200, 800
hard = [list(np.where(rng.random(3) < 0.5, 1, -1) * (rng.choice(n, 3, replace=False) + 1)) for _ in range(m)]
soft = [[int(v)] for v in rng.choice([-1, 1], n) * np.arange(1, n + 1)]

cost, x = pfa.maxsat(hard, soft, num_steps=2_000_000, seed=1)
true = lambda lit: x[abs(lit) - 1] == (lit > 0)
print(f"cost {cost}; hard clauses falsified: {sum(not any(true(l) for l in c) for c in hard)}, "
      f"soft falsified: {sum(not true(c[0]) for c in soft)}")
