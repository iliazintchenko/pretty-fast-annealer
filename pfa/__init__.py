"""PFA, the Pretty Fast Annealer: minimizes energies of single-variable and pairwise terms over discrete variables,
+-1 spins, variables with several values, or variables that exchange values with a fixed count per value. run() does
the work; see its docstring and the README."""

from .anneal import colour, cover, from_one_hot, maxsat, route, run, shop, simd_path, to_one_hot, tour
from .utils import energy, from_binary
