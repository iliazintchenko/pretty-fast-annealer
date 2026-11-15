"""Quadratic assignment as variables that exchange values: facility r at location x_r, every location taken by one
facility, cost sum over pairs of flow[r, s] * distance[x_r, x_s]. The instance is planted: the flows are the distances
under a hidden permutation, negated, so the lowest energy is minus the sum of the squared distances, at that
permutation (Cauchy-Schwarz) and at whatever other permutations preserve the distances."""
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pfa

rng = np.random.default_rng(0)
n = 60
points = rng.random((n, 2))
distance = np.sqrt(((points[:, None] - points[None]) ** 2).sum(axis=2))
hidden = rng.permutation(n)
flow = -distance[np.ix_(hidden, hidden)]   # flow[r, s] = -distance[hidden[r], hidden[s]]

x = pfa.run(flow, counts=np.ones(n, dtype=int), table=distance, beta_max=20.0, num_steps=20 * n * n, num_reps=10)
energy = 0.5 * (flow * distance[np.ix_(x, x)]).sum()
print(f"{n} facilities: energy {energy:.4f}, planted optimum {-0.5 * (distance**2).sum():.4f}")
