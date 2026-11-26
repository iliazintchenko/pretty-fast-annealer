"""A small vertex cover of a random graph: the fewest vertices touching every edge, found within one budget of steps."""
import numpy as np

import pfa

rng = np.random.default_rng(0)
n, m = 300, 1500
edges = rng.integers(0, n, (m, 2))
edges = edges[edges[:, 0] != edges[:, 1]]

size, cover = pfa.cover(edges, num_steps=2_000_000, seed=1)
covered = cover[edges[:, 0]].astype(bool) | cover[edges[:, 1]].astype(bool)
print(f"cover of {size} vertices; edges uncovered: {int((~covered).sum())}")
