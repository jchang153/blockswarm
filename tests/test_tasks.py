"""Unit tests for task scoring and chest sizing. Run: python3 -m unittest tests.test_tasks"""

from __future__ import annotations

import math
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blockswarm.blueprints import BLUEPRINTS, Blueprint, _box_house  # noqa: E402
from blockswarm.tasks import (BLOCK_TYPES, DEFAULT_BRIEF, BlueprintTask, OpenHouseTask,  # noqa: E402
                              make_task)

DIMS = (16, 16, 8)
AGENTS = ["a", "b"]
HOUSE = BLUEPRINTS["house"]  # x 4..10, y 5..9, foundation z=0, walls z=1..3, roof z=4


def build(bp: Blueprint, by: str = "a") -> dict:
    return {p: {"type": t, "placed_by": by} for p, t in bp.cells.items()}


def open_task(target: int = 12) -> OpenHouseTask:
    return OpenHouseTask(DEFAULT_BRIEF, HOUSE, target)


def score(blocks: dict, solids=(), dims=DIMS, agents=AGENTS) -> dict:
    return open_task().score(blocks, agents, dims, solids)


class OpenScoringTest(unittest.TestCase):
    def test_empty_world(self):
        s = score({})
        self.assertEqual(s["team_score"], 0.0)
        self.assertEqual(s["individual_scores"], {"a": 0, "b": 0})
        self.assertFalse(s["score_details"]["enclosed"])
        self.assertEqual(s["score_details"]["interior_cells"], 0)

    def test_exact_house(self):
        s = score(build(HOUSE))
        d = s["score_details"]
        self.assertTrue(d["enclosed"])
        # Interior: x 5..9 (5) x y 6..8 (3) x z 1..3 (3).
        self.assertEqual(d["interior_cells"], 45)
        self.assertEqual(d["interior"], sorted([x, y, z] for x in range(5, 10)
                                               for y in range(6, 9) for z in range(1, 4)))
        self.assertEqual(d["doors"], 2)      # two stacked door blocks
        self.assertEqual(d["windows"], 4)    # two side windows, two on the back wall
        self.assertEqual(d["leftovers"], 0)
        self.assertEqual(d["shell_cells"], len(HOUSE.cells))
        self.assertEqual(d["target_interior"], 12)
        self.assertEqual(s["team_score"], 1.0)
        self.assertEqual(s["individual_scores"], {"a": len(HOUSE.cells), "b": 0})

    def test_hole_in_wall(self):
        blocks = build(HOUSE)
        del blocks[(4, 6, 1)]
        s = score(blocks)
        self.assertFalse(s["score_details"]["enclosed"])
        self.assertEqual(s["team_score"], 0.0)

    def test_hole_in_roof(self):
        blocks = build(HOUSE)
        del blocks[(7, 7, 4)]
        s = score(blocks)
        self.assertFalse(s["score_details"]["enclosed"])
        self.assertEqual(s["team_score"], 0.0)

    def test_door_replaced_by_wall(self):
        blocks = build(HOUSE)
        for p in [p for p, b in blocks.items() if b["type"] == "door"]:
            blocks[p] = {"type": "brick", "placed_by": "a"}
        s = score(blocks)
        self.assertTrue(s["score_details"]["enclosed"])
        self.assertEqual(s["score_details"]["doors"], 0)
        self.assertAlmostEqual(s["team_score"], 0.8)

    def test_doorway_left_open_leaks(self):
        blocks = build(HOUSE)
        for p in [p for p, b in blocks.items() if b["type"] == "door"]:
            del blocks[p]
        self.assertEqual(score(blocks)["team_score"], 0.0)

    def test_scaffolding_is_leftover(self):
        blocks = build(HOUSE)
        for z in range(3):
            blocks[(1, 1, z)] = {"type": "wood", "placed_by": "b"}
        blocks[(3, 7, 1)] = {"type": "wood", "placed_by": "b"}  # face-touching the outer wall
        s = score(blocks)
        self.assertEqual(s["score_details"]["leftovers"], 4)
        self.assertAlmostEqual(s["team_score"], 0.96)
        self.assertEqual(s["individual_scores"]["b"], 0)

    def test_sealed_box_without_door(self):
        cells = {}
        for x in range(2, 6):
            for y in range(2, 6):
                for z in range(0, 4):
                    if x in (2, 5) or y in (2, 5) or z in (0, 3):
                        cells[(x, y, z)] = "stone"
        s = score(build(Blueprint("box", cells)))
        d = s["score_details"]
        self.assertTrue(d["enclosed"])
        self.assertEqual(d["interior_cells"], 8)
        self.assertEqual((d["doors"], d["windows"], d["leftovers"]), (0, 0, 0))
        self.assertAlmostEqual(s["team_score"], 0.6)  # 0.5 + 0.15 * 8/12

    def test_one_window_partial(self):
        blocks = build(HOUSE)
        for p in [p for p, b in blocks.items() if b["type"] == "glass"][1:]:
            blocks[p] = {"type": "brick", "placed_by": "a"}
        s = score(blocks)
        self.assertEqual(s["score_details"]["windows"], 1)
        self.assertAlmostEqual(s["team_score"], 0.925)

    def test_one_cell_pocket_not_enclosed(self):
        cells = {(5, 5, 0): "stone", (5, 5, 2): "stone", (4, 5, 1): "stone", (6, 5, 1): "stone",
                 (5, 4, 1): "stone", (5, 6, 1): "stone"}
        s = score(build(Blueprint("pocket", cells)))
        self.assertFalse(s["score_details"]["enclosed"])
        self.assertEqual(s["team_score"], 0.0)

    def test_two_rooms_largest_chosen(self):
        hut = _box_house("hut", 12, 11, 3, 3, 2, windows=False)  # interior 1x1x2
        blocks = build(HOUSE, "a")
        blocks.update(build(hut, "b"))
        s = score(blocks)
        d = s["score_details"]
        self.assertEqual(d["interior_cells"], 45)
        self.assertEqual(d["leftovers"], len(hut.cells))
        self.assertEqual(s["individual_scores"], {"a": len(HOUSE.cells), "b": 0})
        self.assertEqual(s["team_score"], round(max(0.0, 1.0 - 0.01 * len(hut.cells)), 6))

    def test_no_foundation_floor_is_ground(self):
        # Drop the foundation and lower everything by one: walls stand on the ground.
        blocks = {(x, y, z - 1): b for (x, y, z), b in build(HOUSE).items() if z > 0}
        d = score(blocks)["score_details"]
        self.assertTrue(d["enclosed"])
        self.assertEqual(d["interior_cells"], 45)  # now z 0..2
        self.assertEqual(d["leftovers"], 0)

    def test_individual_credit_only_for_shell(self):
        blocks = build(HOUSE, "a")
        for p in list(blocks)[::2]:
            blocks[p] = dict(blocks[p], placed_by="b")
        blocks[(0, 0, 0)] = {"type": "wood", "placed_by": "b"}
        s = score(blocks, agents=["a", "b", "c"])
        n = len(HOUSE.cells)
        self.assertEqual(s["individual_scores"], {"a": n - (n + 1) // 2, "b": (n + 1) // 2, "c": 0})

    def test_chest_solid_does_not_leak(self):
        blocks = build(HOUSE)
        del blocks[(4, 6, 1)]
        s = score(blocks, solids=[(4, 6, 1)])
        self.assertTrue(s["score_details"]["enclosed"])
        self.assertEqual(s["score_details"]["leftovers"], 0)
        self.assertEqual(s["score_details"]["shell_cells"], len(HOUSE.cells) - 1)
        # A chest sitting outside changes nothing.
        self.assertEqual(score(build(HOUSE), solids=[(1, 1, 0)])["team_score"], 1.0)

    def test_door_not_counted_without_outside_air(self):
        blocks = build(HOUSE)
        blocks[(7, 4, 1)] = {"type": "brick", "placed_by": "a"}  # bricks up the door's outside
        blocks[(7, 4, 2)] = {"type": "brick", "placed_by": "a"}
        d = score(blocks)["score_details"]
        self.assertEqual(d["doors"], 0)

    def test_speed(self):
        blocks = build(HOUSE)
        task = open_task()
        t0 = time.perf_counter()
        for _ in range(200):
            task.score(blocks, AGENTS, DIMS)
        self.assertLess(time.perf_counter() - t0, 5.0)
        big = task.score(blocks, AGENTS, (64, 64, 32))
        self.assertEqual(big["team_score"], 1.0)


class BlueprintScoringTest(unittest.TestCase):
    def setUp(self):
        self.bp = BLUEPRINTS["hut_small"]
        self.task = BlueprintTask(self.bp, extra_penalty=0.5)

    def test_exact(self):
        s = self.task.score(build(self.bp, "a"), AGENTS, DIMS)
        n = len(self.bp.cells)
        self.assertEqual(s["team_score"], 1.0)
        self.assertEqual(s["individual_scores"], {"a": n, "b": 0})
        self.assertEqual(s["score_details"], {"correct_cells": n, "blueprint_cells": n, "extra_cells": 0})

    def test_wrong_type_and_extra(self):
        blocks = build(self.bp, "a")
        wrong = next(iter(blocks))
        blocks[wrong] = {"type": "glass", "placed_by": "b"}
        blocks[(0, 0, 0)] = {"type": "wood", "placed_by": "b"}
        s = self.task.score(blocks, AGENTS, DIMS)
        n = len(self.bp.cells)
        self.assertEqual(s["score_details"], {"correct_cells": n - 1, "blueprint_cells": n, "extra_cells": 1})
        self.assertEqual(s["team_score"], round((n - 1 - 0.5) / n, 6))
        self.assertEqual(s["individual_scores"], {"a": n - 1, "b": 0})

    def test_floor_at_zero(self):
        blocks = {(0, y, 0): {"type": "wood", "placed_by": "a"} for y in range(10)}
        self.assertEqual(self.task.score(blocks, AGENTS, DIMS)["team_score"], 0.0)

    def test_empty(self):
        s = self.task.score({}, AGENTS, DIMS)
        self.assertEqual(s["team_score"], 0.0)
        self.assertEqual(s["individual_scores"], {"a": 0, "b": 0})


class ChestAndFactoryTest(unittest.TestCase):
    def test_chest_counts(self):
        need = HOUSE.counts()
        for task in (open_task(), BlueprintTask(HOUSE, 0.1)):
            exact = task.chest_counts("exact")
            self.assertEqual(set(exact), set(BLOCK_TYPES))
            for t in BLOCK_TYPES:
                self.assertEqual(exact[t], need.get(t, 0))
                self.assertEqual(task.chest_counts("abundant")[t], math.ceil(1.5 * need.get(t, 0)))
                self.assertEqual(task.chest_counts("short")[t], math.floor(0.85 * need.get(t, 0)))
            self.assertEqual(exact["wood"], 0)
            with self.assertRaises(ValueError):
                task.chest_counts("plenty")

    def test_make_task(self):
        t = make_task("open", brief=None, blueprint="hut_small", reference_design="house",
                      target_interior=12, extra_penalty=0.1)
        self.assertEqual(t.kind, "open")
        self.assertEqual(t.header(), {"kind": "open", "brief": DEFAULT_BRIEF, "blueprint": None})
        t = make_task("open", brief="Build a tower.", blueprint="hut_small", reference_design="house",
                      target_interior=12, extra_penalty=0.1)
        self.assertEqual(t.header()["brief"], "Build a tower.")
        t = make_task("blueprint", brief=None, blueprint="hut_small", reference_design="house",
                      target_interior=12, extra_penalty=0.1)
        self.assertEqual(t.kind, "blueprint")
        self.assertEqual(t.header(), {"kind": "blueprint", "brief": None,
                                      "blueprint": BLUEPRINTS["hut_small"].to_json()})

    def test_make_task_errors(self):
        kw = dict(brief=None, blueprint="hut_small", reference_design="house",
                  target_interior=12, extra_penalty=0.1)
        with self.assertRaises(ValueError):
            make_task("castle", **kw)
        with self.assertRaises(ValueError):
            make_task("blueprint", **dict(kw, blueprint="nope"))
        with self.assertRaises(ValueError):
            make_task("open", **dict(kw, reference_design="nope"))
        with self.assertRaises(ValueError):
            make_task("open", **dict(kw, target_interior=0))

    def test_block_types_match_world(self):
        try:
            from blockswarm import world
        except Exception as e:  # world.py may be mid-edit by another agent
            self.skipTest(f"blockswarm.world not importable: {e}")
        self.assertEqual(BLOCK_TYPES, world.BLOCK_TYPES)


if __name__ == "__main__":
    unittest.main()
