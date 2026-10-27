"""Standing and walking rules for 2-block-tall bodies on the grid (pure functions).

A cell is *free* if it is inside the grid, not solid (block or chest) and not part of another
agent's body. A cell is *standable* if it and the cell above are free and it is on the ground
(z = 0) or on top of something solid.

Walking moves one column at a time in 8 directions:
- if the feet cell in the next column is solid, the agent steps up 1 (needs headroom above
  its own head and a standable cell on top);
- otherwise, if the head cell there is free, it walks in and drops to the first support
  below, at most 3 cells down.
Diagonal steps need both orthogonal columns passable at the higher of the two heights (no
corner cutting). Straight steps cost 1, diagonal steps √2.
"""

from __future__ import annotations

import heapq
import math
from typing import Callable, Iterator

from blockswarm.blueprints import Pos

MAX_STEP_UP = 1
MAX_DROP = 3
SQRT2 = math.sqrt(2)
DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))


class Terrain:
    """Free/solid/standable queries over a grid, given predicates for solids and bodies."""

    def __init__(self, width: int, depth: int, height: int,
                 solid: Callable[[Pos], bool], occupied: Callable[[Pos], bool]):
        self.width, self.depth, self.height = width, depth, height
        self.solid = solid
        self.occupied = occupied  # other agents' body cells

    def in_bounds(self, c: Pos) -> bool:
        x, y, z = c
        return 0 <= x < self.width and 0 <= y < self.depth and 0 <= z < self.height

    def free(self, c: Pos) -> bool:
        return self.in_bounds(c) and not self.solid(c) and not self.occupied(c)

    def standable(self, c: Pos) -> bool:
        x, y, z = c
        return (self.free(c) and self.free((x, y, z + 1))
                and (z == 0 or self.solid((x, y, z - 1))))

    def _enter(self, c: Pos, nx: int, ny: int) -> int | None:
        """Feet height after stepping from c into column (nx, ny), or None if blocked."""
        x, y, z = c
        if not self.in_bounds((nx, ny, 0)):
            return None
        if not self.free((nx, ny, z)):
            if self.free((x, y, z + 2)) and self.standable((nx, ny, z + MAX_STEP_UP)):
                return z + MAX_STEP_UP
            return None
        if not self.free((nx, ny, z + 1)):
            return None
        for nz in range(z, z - MAX_DROP - 1, -1):
            if nz < 0 or not self.free((nx, ny, nz)):
                return None
            if nz == 0 or self.solid((nx, ny, nz - 1)):
                return nz
        return None

    def neighbours(self, c: Pos) -> Iterator[tuple[Pos, float]]:
        x, y, z = c
        for dx, dy in DIRS:
            nz = self._enter(c, x + dx, y + dy)
            if nz is None:
                continue
            if dx and dy:
                h = max(z, nz)
                if not all(self.free((cx, cy, hz)) for cx, cy in ((x + dx, y), (x, y + dy))
                           for hz in (h, h + 1)):
                    continue
                yield (x + dx, y + dy, nz), SQRT2
            else:
                yield (x + dx, y + dy, nz), 1.0

    def costs_from(self, start: Pos, goal: Pos | None = None) -> tuple[dict[Pos, float], dict[Pos, Pos]]:
        """Dijkstra from start (stops early at goal). Returns (cost, previous) maps."""
        dist: dict[Pos, float] = {start: 0.0}
        prev: dict[Pos, Pos] = {}
        heap: list[tuple[float, Pos]] = [(0.0, start)]
        done: set[Pos] = set()
        while heap:
            d, c = heapq.heappop(heap)
            if c in done:
                continue
            done.add(c)
            if c == goal:
                break
            for n, w in self.neighbours(c):
                nd = d + w
                if nd < dist.get(n, math.inf) - 1e-9:
                    dist[n] = nd
                    prev[n] = c
                    heapq.heappush(heap, (nd, n))
        return {c: dist[c] for c in done}, prev

    def path(self, start: Pos, goal: Pos) -> tuple[list[Pos], float] | None:
        """Shortest path (both ends included) and its length, or None if unreachable."""
        if start == goal:
            return [start], 0.0
        dist, prev = self.costs_from(start, goal)
        if goal not in dist:
            return None
        path = [goal]
        while path[-1] != start:
            path.append(prev[path[-1]])
        path.reverse()
        return path, dist[goal]


def path_length(path: list[list[int]] | list[Pos]) -> float:
    """Walk length of a path: 1 per straight step, √2 per diagonal step."""
    total = 0.0
    for a, b in zip(path, path[1:]):
        total += SQRT2 if (a[0] != b[0] and a[1] != b[1]) else 1.0
    return total
