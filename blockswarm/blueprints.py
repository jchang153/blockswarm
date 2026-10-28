"""Reference structures: the scripted test bot's private design (not shown to agents or used by
the world). A blueprint maps (x, y, z) -> catalog item id (blockswarm/catalog.py)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

Pos = tuple[int, int, int]

# Part of the house -> catalog item. Chosen so the `house` costs 272 coins, within the
# default wallet for three agents (300).
MATERIALS: dict[str, str] = {
    "floor": "stone",
    "wall": "oak_planks",
    "window": "glass",
    "door": "oak_door",
    "roof": "dark_oak_planks",
}


@dataclass(frozen=True)
class Blueprint:
    name: str
    cells: dict[Pos, str]

    def counts(self) -> Counter:
        return Counter(self.cells.values())

    def cost(self) -> int:
        """Coins to buy every block of the design at catalog prices."""
        from blockswarm.catalog import CATALOG
        return sum(CATALOG[t].price * n for t, n in self.counts().items())

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "cells": [[x, y, z, t] for (x, y, z), t in sorted(self.cells.items())],
        }


def _box_house(name: str, x0: int, y0: int, w: int, d: int, wall_h: int, windows: bool,
               materials: dict[str, str] = MATERIALS) -> Blueprint:
    """Foundation, perimeter walls with a 2-high door on the front face, optional windows, flat roof."""
    m = materials
    cells: dict[Pos, str] = {}
    for x in range(x0, x0 + w):
        for y in range(y0, y0 + d):
            cells[(x, y, 0)] = m["floor"]
            cells[(x, y, wall_h + 1)] = m["roof"]
    door_x = x0 + w // 2
    for z in range(1, wall_h + 1):
        for x in range(x0, x0 + w):
            for y in range(y0, y0 + d):
                if x in (x0, x0 + w - 1) or y in (y0, y0 + d - 1):
                    cells[(x, y, z)] = m["wall"]
        cells[(door_x, y0, z)] = m["door"] if z <= 2 else m["wall"]
    if windows and wall_h >= 2:
        mid_y = y0 + d // 2
        cells[(x0, mid_y, 2)] = m["window"]
        cells[(x0 + w - 1, mid_y, 2)] = m["window"]
        for x in (x0 + 1, x0 + w - 2):
            if x != door_x:
                cells[(x, y0 + d - 1, 2)] = m["window"]
    return Blueprint(name, cells)


BLUEPRINTS: dict[str, Blueprint] = {
    "hut_small": _box_house("hut_small", 6, 6, 3, 3, 2, windows=False),
    "house": _box_house("house", 4, 5, 7, 5, 3, windows=True),
}
