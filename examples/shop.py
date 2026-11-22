"""The job shop: a random 10 x 10 instance (every job visits every machine once, in a random order, durations 1 .. 99),
solved by pfa.shop() under a budget of restarts; prints the makespan and the machine orders."""
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pfa  # noqa: E402

rng = np.random.default_rng(0)
n, m = 10, 10
machines = np.array([rng.permutation(m) for _ in range(n)])
durations = rng.integers(1, 100, size=(n, m))

makespan, orders = pfa.shop(machines, durations, num_steps=50, seed=1)
print(f"{n} jobs on {m} machines, makespan {makespan}")
for k, order in enumerate(orders):
    print(f"  machine {k}: {order}")
