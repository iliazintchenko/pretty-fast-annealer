"""Graph colouring, natively: pfa.colour() takes the edges and a number of colours k and returns a colouring with the
fewest edges whose ends share a colour, zero when the graph is k-colourable and the search finds it. The chromatic
number is found by lowering k while a legal colouring is still found within the budget."""
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pfa

rng = np.random.default_rng(0)
V, degree = 250, 125   # a random graph of density one half, like DSJC250.5
edges = rng.integers(0, V, (V * degree // 2, 2))
edges = np.unique(np.sort(edges[edges[:, 0] != edges[:, 1]], axis=1), axis=0)

# a greedy colouring gives the first k; each lower k is tried with the same budget until one fails
colours = np.full(V, -1)
for v in range(V):
    taken = set(colours[edges[edges[:, 0] == v, 1]]) | set(colours[edges[edges[:, 1] == v, 0]])
    colours[v] = next(c for c in range(V) if c not in taken)
k = int(colours.max()) + 1
print(f"{V} vertices, {len(edges)} edges: greedy uses {k} colours")
while True:
    conflicts, colouring = pfa.colour(edges, k - 1, num_steps=20_000_000)
    if conflicts > 0:
        print(f"{k - 1} colours: {conflicts} edges with both ends the same colour after the budget; {k} colours it is")
        break
    k -= 1
    assert np.all(colouring[edges[:, 0]] != colouring[edges[:, 1]])
    print(f"{k} colours: a legal colouring")
