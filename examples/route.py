"""Vehicle routes through random customers from a depot: pfa.route() with the budget as the number of restarts."""
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pfa  # noqa: E402

rng = np.random.default_rng(0)
points = rng.random((201, 2)) * 1000          # the depot first
demands = np.concatenate([[0], rng.integers(1, 10, 200)])
routes = pfa.route(points, demands, capacity=50, num_steps=2000, seed=1)
length = sum(np.sqrt(((points[[0] + r + [0]][1:] - points[[0] + r + [0]][:-1]) ** 2).sum(axis=1)).sum() for r in routes)
print(f"{len(points) - 1} customers, {len(routes)} routes, total length {length:.1f}")
