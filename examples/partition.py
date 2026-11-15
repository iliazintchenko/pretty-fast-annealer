"""Balanced graph bisection as variables that exchange values: every vertex takes side 0 or 1, half on each, and an
edge whose ends are on different sides costs 1. With table[a, b] = 1 when a != b the energy is the cut."""
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pfa

rng = np.random.default_rng(0)
V, degree = 1000, 6
edges = rng.integers(0, V, (V * degree // 2, 2))
edges = np.unique(np.sort(edges[edges[:, 0] != edges[:, 1]], axis=1), axis=0)
adjacency = np.zeros((V, V))
adjacency[edges[:, 0], edges[:, 1]] = adjacency[edges[:, 1], edges[:, 0]] = 1.0

side = pfa.run(adjacency, counts=[V // 2, V - V // 2], table=1.0 - np.eye(2), beta_max=3.0, num_steps=100 * V, num_reps=10)
cut = int(np.sum(side[edges[:, 0]] != side[edges[:, 1]]))
print(f"{V} vertices, {len(edges)} edges: {cut} edges cut by a balanced bisection")
