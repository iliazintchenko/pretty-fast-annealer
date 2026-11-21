"""The shortest tour through random points: pfa.tour() with the budget as the number of chains."""
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pfa  # noqa: E402

rng = np.random.default_rng(0)
points = rng.random((500, 2)) * 1000
tour = pfa.tour(points, num_steps=50_000, seed=1)
length = np.sqrt(((points[tour] - points[np.roll(tour, -1)]) ** 2).sum(axis=1)).sum()
print(f"{len(points)} cities, tour length {length:.1f}")
