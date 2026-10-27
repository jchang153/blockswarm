"""Target structures. A blueprint maps (x, y, z) -> block type."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

Pos = tuple[int, int, int]


@dataclass(frozen=True)
class Blueprint:
    name: str
    cells: dict[Pos, str]

    def counts(self) -> Counter:
        return Counter(self.cells.values())

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "cells": [[x, y, z, t] for (x, y, z), t in sorted(self.cells.items())],
        }


def _box_house(name: str, x0: int, y0: int, w: int, d: int, wall_h: int, windows: bool) -> Blueprint:
    """Foundation, perimeter walls with a door on the front face, optional windows, flat roof."""
    cells: dict[Pos, str] = {}
    for x in range(x0, x0 + w):
        for y in range(y0, y0 + d):
            cells[(x, y, 0)] = "stone"
            cells[(x, y, wall_h + 1)] = "roof"
    door_x = x0 + w // 2
    for z in range(1, wall_h + 1):
        for x in range(x0, x0 + w):
            for y in range(y0, y0 + d):
                if x in (x0, x0 + w - 1) or y in (y0, y0 + d - 1):
                    cells[(x, y, z)] = "brick"
        cells[(door_x, y0, z)] = "door" if z <= 2 else "brick"
    if windows and wall_h >= 2:
        mid_y = y0 + d // 2
        cells[(x0, mid_y, 2)] = "glass"
        cells[(x0 + w - 1, mid_y, 2)] = "glass"
        for x in (x0 + 1, x0 + w - 2):
            if x != door_x:
                cells[(x, y0 + d - 1, 2)] = "glass"
    return Blueprint(name, cells)


BLUEPRINTS: dict[str, Blueprint] = {
    "hut_small": _box_house("hut_small", 6, 6, 3, 3, 2, windows=False),
    "house": _box_house("house", 4, 5, 7, 5, 3, windows=True),
}
