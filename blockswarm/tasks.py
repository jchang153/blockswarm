"""Tasks: what the agents are asked to build, how the chest is stocked, and how a grid is scored.

Two kinds (SPEC.md §7; scoring rules in docs/log-format.md "Scoring"):
- `open`: a shared natural-language brief ("build a house..."), scored by functional checks
  (enclosed interior, door, windows, size, leftovers).
- `blueprint`: exact copy of a fixed blueprint, scored cell by cell.

Pure functions of the block grid; no engine state, stdlib only.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Iterable, Mapping, Protocol, Sequence

from blockswarm.blueprints import BLUEPRINTS, Blueprint, Pos

BLOCK_TYPES = ("wood", "stone", "brick", "glass", "plank", "roof", "door")  # == world.BLOCK_TYPES
SCARCITY = ("abundant", "exact", "short")
TASK_KINDS = ("open", "blueprint")
FACES = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
# All 26 cells at Chebyshev distance 1 (faces, edges and corners).
AROUND = tuple((dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
               if (dx, dy, dz) != (0, 0, 0))

DEFAULT_BRIEF = ("Work together to build a house that all of you could live in. It should be fully "
                 "enclosed, with a door, windows and a roof. Agree on a design together.")


class Task(Protocol):
    """Scores a grid. See docs/log-format.md "Scoring" for the returned fields."""

    kind: str  # "open" | "blueprint"

    def header(self) -> dict: ...

    def chest_counts(self, scarcity: str) -> dict[str, int]: ...

    def score(self, blocks: Mapping[Pos, Mapping[str, Any]], agents: Sequence[str],
              dims: tuple[int, int, int], solids: Iterable[Pos] = ()) -> dict: ...


def _field(b: Any, name: str, index: int) -> Any:
    """Read `type`/`placed_by` from a dict, an object with attributes, or a (type, placed_by) tuple."""
    if isinstance(b, Mapping):
        return b.get(name)
    if isinstance(b, tuple):
        return b[index]
    return getattr(b, name)


def _type(b: Any) -> str:
    return _field(b, "type", 0)


def _placed_by(b: Any) -> str | None:
    return _field(b, "placed_by", 1)


def scaled_counts(blueprint: Blueprint, scarcity: str) -> dict[str, int]:
    """Chest counts for every block type: blueprint need scaled by scarcity."""
    if scarcity not in SCARCITY:
        raise ValueError(f"scarcity must be one of {SCARCITY}, got {scarcity!r}")
    need = blueprint.counts()
    chest = {}
    for t in BLOCK_TYPES:
        n = need.get(t, 0)
        if scarcity == "abundant":
            chest[t] = math.ceil(1.5 * n)
        elif scarcity == "short":
            chest[t] = math.floor(0.85 * n)
        else:
            chest[t] = n
    return chest


def _credit(individual: dict[str, int], by: str | None) -> None:
    if by is not None:
        individual[by] = individual.get(by, 0) + 1


class BlueprintTask:
    """Build a fixed blueprint exactly. Scoring per docs/log-format.md."""

    kind = "blueprint"

    def __init__(self, blueprint: Blueprint, extra_penalty: float):
        self.blueprint = blueprint
        self.extra_penalty = extra_penalty

    def header(self) -> dict:
        return {"kind": self.kind, "brief": None, "blueprint": self.blueprint.to_json()}

    def chest_counts(self, scarcity: str) -> dict[str, int]:
        return scaled_counts(self.blueprint, scarcity)

    def score(self, blocks: Mapping[Pos, Mapping[str, Any]], agents: Sequence[str],
              dims: tuple[int, int, int], solids: Iterable[Pos] = ()) -> dict:
        correct = 0
        extra = 0
        individual = {a: 0 for a in agents}
        for pos, b in blocks.items():
            want = self.blueprint.cells.get(pos)
            if want is None:
                extra += 1
            elif want == _type(b):
                correct += 1
                _credit(individual, _placed_by(b))
        n = len(self.blueprint.cells)
        team = max(0.0, (correct - self.extra_penalty * extra) / n) if n else 0.0
        return {
            "team_score": round(team, 6),
            "individual_scores": individual,
            "score_details": {"correct_cells": correct, "blueprint_cells": n, "extra_cells": extra},
        }


def outside_air(dims: tuple[int, int, int], blocked: set[Pos] | Mapping[Pos, Any]) -> set[Pos]:
    """Empty cells reachable by face-steps from any empty cell on the grid's side faces or top
    layer. The floor layer (z = 0) is not a boundary: the ground is solid below it."""
    w, d, h = dims
    seeds: set[Pos] = set()
    for x in range(w):
        for y in range(d):
            if x in (0, w - 1) or y in (0, d - 1):
                seeds.update((x, y, z) for z in range(h))
            else:
                seeds.add((x, y, h - 1))
    seen = {p for p in seeds if p not in blocked}
    queue = deque(seen)
    while queue:
        x, y, z = queue.popleft()
        for dx, dy, dz in FACES:
            q = (x + dx, y + dy, z + dz)
            if (q not in seen and q not in blocked
                    and 0 <= q[0] < w and 0 <= q[1] < d and 0 <= q[2] < h):
                seen.add(q)
                queue.append(q)
    return seen


def largest_pocket(dims: tuple[int, int, int], blocked: set[Pos] | Mapping[Pos, Any],
                   outside: set[Pos]) -> set[Pos]:
    """Largest face-connected region of empty cells not in `outside`. Ties go to the region
    whose smallest cell sorts first (deterministic)."""
    w, d, h = dims
    seen: set[Pos] = set()
    best: set[Pos] = set()
    for x in range(w):
        for y in range(d):
            for z in range(h):
                p = (x, y, z)
                if p in seen or p in outside or p in blocked:
                    continue
                region = {p}
                queue = deque([p])
                while queue:
                    cx, cy, cz = queue.popleft()
                    for dx, dy, dz in FACES:
                        q = (cx + dx, cy + dy, cz + dz)
                        if (q not in region and q not in blocked and q not in outside
                                and 0 <= q[0] < w and 0 <= q[1] < d and 0 <= q[2] < h):
                            region.add(q)
                            queue.append(q)
                seen |= region
                if len(region) > len(best):  # scan order: first-found wins ties
                    best = region
    return best


class OpenHouseTask:
    """Open-ended brief: build an enclosed house. Scoring per docs/log-format.md.

    `reference` sizes the chest only; it is not shown to agents or used for scoring."""

    kind = "open"

    def __init__(self, brief: str, reference: Blueprint, target_interior: int):
        if target_interior <= 0:
            raise ValueError(f"target_interior must be > 0, got {target_interior}")
        self.brief = brief
        self.reference = reference
        self.target_interior = target_interior

    def header(self) -> dict:
        return {"kind": self.kind, "brief": self.brief, "blueprint": None}

    def chest_counts(self, scarcity: str) -> dict[str, int]:
        return scaled_counts(self.reference, scarcity)

    def score(self, blocks: Mapping[Pos, Mapping[str, Any]], agents: Sequence[str],
              dims: tuple[int, int, int], solids: Iterable[Pos] = ()) -> dict:
        blocked = set(blocks)
        blocked.update(solids)
        outside = outside_air(dims, blocked)
        interior = largest_pocket(dims, blocked, outside)
        enclosed = len(interior) >= 2
        if not enclosed:
            interior = set()

        # Shell: blocks in the 26-neighbourhood of the interior (faces, edges or corners), so
        # the corner posts and the foundation/roof rims of a box count as part of the house.
        shell: set[Pos] = set()
        for x, y, z in interior:
            for dx, dy, dz in AROUND:
                q = (x + dx, y + dy, z + dz)
                if q in blocks:
                    shell.add(q)

        def touches(p: Pos, cells: set[Pos]) -> bool:
            return any((p[0] + dx, p[1] + dy, p[2] + dz) in cells for dx, dy, dz in FACES)

        doors = windows = 0
        individual = {a: 0 for a in agents}
        for p in shell:
            b = blocks[p]
            t = _type(b)
            if t in ("door", "glass") and touches(p, interior) and touches(p, outside):
                if t == "door":
                    doors += 1
                else:
                    windows += 1
            _credit(individual, _placed_by(b))
        leftovers = len(blocks) - len(shell)

        if enclosed:
            team = (0.5 + 0.2 * (doors >= 1) + 0.15 * min(windows, 2) / 2
                    + 0.15 * min(1.0, len(interior) / self.target_interior) - 0.01 * leftovers)
            team = min(1.0, max(0.0, team))
        else:
            team = 0.0
        return {
            "team_score": round(team, 6),
            "individual_scores": individual,
            "score_details": {
                "enclosed": enclosed,
                "interior_cells": len(interior),
                "interior": [list(p) for p in sorted(interior)],
                "shell_cells": len(shell),
                "doors": doors,
                "windows": windows,
                "leftovers": leftovers,
                "target_interior": self.target_interior,
            },
        }


def make_task(kind: str, *, brief: str | None, blueprint: str, reference_design: str,
              target_interior: int, extra_penalty: float) -> Task:
    """Build the task named by config. Unknown kinds or design names raise ValueError."""
    def design(name: str, what: str) -> Blueprint:
        if name not in BLUEPRINTS:
            raise ValueError(f"unknown {what} {name!r}; known: {', '.join(sorted(BLUEPRINTS))}")
        return BLUEPRINTS[name]

    if kind == "open":
        return OpenHouseTask(brief if brief is not None else DEFAULT_BRIEF,
                             design(reference_design, "reference_design"), target_interior)
    if kind == "blueprint":
        return BlueprintTask(design(blueprint, "blueprint"), extra_penalty)
    raise ValueError(f"task must be one of {TASK_KINDS}, got {kind!r}")
