"""Edge assembly between two tours (Nagata's EAX in its single-cycle form): the fold for tours.

Two low tours do not differ in clusters a partition could exchange: their differing edges form a graph in which every
city has as many edges of one tour as of the other, and it decomposes into AB-cycles, cycles alternating an edge of A
and an edge of B. Removing a cycle's A-edges from A and adding its B-edges gives a 2-regular graph, a set of subtours,
which are merged into one tour by the cheapest 2-exchange between the smallest subtour and any other, repeatedly. Each
AB-cycle gives one child; the shortest child is the fold's result when it is shorter than both parents. Tours are
edge-set codes as pfa.anneal._Tour keeps them: per city, its two neighbours as min * n + max."""
import numpy as np


def neighbours(code, n):
    return np.stack((code // n, code % n), axis=1)


def ab_cycles(na, nb_, n, rng):
    """The AB-cycles of the differing edges, each as a list of (city, city, which) edges, which 0 for A and 1 for B."""
    adj_a = [[] for _ in range(n)]
    adj_b = [[] for _ in range(n)]
    for v in range(n):
        for u in na[v]:
            if u not in nb_[v]:
                adj_a[v].append(int(u))
        for u in nb_[v]:
            if u not in na[v]:
                adj_b[v].append(int(u))
    cycles = []
    cities = [v for v in range(n) if adj_a[v]]
    rng.shuffle(cities)
    for start in cities:
        while adj_a[start]:
            path, which = [start], 0
            at = {start: 0}
            v = start
            while True:
                adj = adj_a if which == 0 else adj_b
                if not adj[v]:
                    break
                u = adj[v].pop(rng.integers(len(adj[v])))
                adj[u].remove(v)
                which ^= 1
                if u in at and (len(path) - at[u]) % 2 == 0:   # a closed alternating cycle from the earlier visit of u
                    k = at[u]
                    cyc = [(path[i], path[i + 1] if i + 1 < len(path) else u) for i in range(k, len(path))]
                    first_which = 0 if k % 2 == 0 else 1
                    cycles.append([(a, b, (first_which + j) % 2) for j, (a, b) in enumerate(cyc)])
                    for i in range(k):   # the edges before the cycle on the path go back to the lists
                        a, b, w = path[i], path[i + 1], i % 2
                        (adj_a if w == 0 else adj_b)[a].append(b)
                        (adj_a if w == 0 else adj_b)[b].append(a)
                    break
                at[u] = len(path)
                path.append(u)
                v = u
    return cycles


def apply_cycle(na, cyc, n):
    """A's neighbour lists after removing the cycle's A-edges and adding its B-edges: the 2-regular graph."""
    adj = [list(map(int, na[v])) for v in range(n)]
    for a, b, w in cyc:
        if w == 0:
            adj[a].remove(b)
            adj[b].remove(a)
    for a, b, w in cyc:
        if w == 1:
            adj[a].append(b)
            adj[b].append(a)
    return adj


def subtours(adj, n):
    seen = np.zeros(n, dtype=bool)
    tours = []
    for s in range(n):
        if seen[s]:
            continue
        t, prev, cur = [], -1, s
        while True:
            seen[cur] = True
            t.append(cur)
            x0, x1 = adj[cur]
            nxt = x0 if x0 != prev else x1
            if nxt == s or seen[nxt]:
                break
            prev, cur = cur, nxt
        tours.append(t)
    return tours


def merge(adj, tours, dist, nn, n):
    """Merge the subtours into one by the cheapest 2-exchange between the smallest subtour and another, repeatedly."""
    label = np.empty(n, dtype=np.int64)
    while len(tours) > 1:
        tours.sort(key=len)
        for i, t in enumerate(tours):
            label[t] = i
        small = tours[0]
        best, best_move = np.inf, None
        for u in small:
            for v in adj[u]:
                if v < u:
                    continue
                for p in nn[u]:
                    if label[p] == 0:
                        continue
                    for q in adj[p]:
                        d1 = dist(u, p) + dist(v, q) - dist(u, v) - dist(p, q)
                        d2 = dist(u, q) + dist(v, p) - dist(u, v) - dist(p, q)
                        if d1 < best:
                            best, best_move = d1, (u, v, p, q)
                        if d2 < best:
                            best, best_move = d2, (u, v, q, p)
        if best_move is None:   # no candidate reaches another subtour: any edge of another subtour
            p, q = tours[1][0], adj[tours[1][0]][0]
            u, v = small[0], adj[small[0]][0]
            best_move = (u, v, p, q)
        u, v, p, q = best_move
        adj[u].remove(v); adj[v].remove(u); adj[p].remove(q); adj[q].remove(p)
        adj[u].append(p); adj[p].append(u); adj[v].append(q); adj[q].append(v)
        tours = subtours(adj, n)
    return adj


def children(a, b, n, dist, nn, rng):
    """The children of tours a and b (edge-set codes): one per AB-cycle, as (length, edge-set code)."""
    na, nb_ = neighbours(a, n), neighbours(b, n)
    out = []
    for cyc in ab_cycles(na, nb_, n, rng):
        adj = apply_cycle(na, cyc, n)
        adj = merge(adj, subtours(adj, n), dist, nn, n)
        nbs = np.array(adj, dtype=np.int64)
        code = np.minimum(nbs[:, 0], nbs[:, 1]) * n + np.maximum(nbs[:, 0], nbs[:, 1])
        out.append((0.5 * sum(dist(v, int(u)) for v in range(n) for u in adj[v]), code))
    return out


def fold(a, b, n, dist, nn, rng):
    """The shortest of a, b and their children either way, as (length, edge-set code)."""
    cands = children(a, b, n, dist, nn, rng) + children(b, a, n, dist, nn, rng)
    return min(cands, key=lambda p: p[0]) if cands else None
