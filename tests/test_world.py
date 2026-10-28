"""Unit tests for the world engine, replay and scripted bot. Run: python3 -m unittest discover -s tests"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blockswarm.blueprints import BLUEPRINTS  # noqa: E402
from blockswarm.catalog import CATALOG, catalog_json  # noqa: E402
from blockswarm.clock import SimClock, WallClock  # noqa: E402
from blockswarm.live import run_live  # noqa: E402
from blockswarm.pathing import SQRT2, Terrain  # noqa: E402
from blockswarm.recorder import Recorder  # noqa: E402
from blockswarm.replay import load_log, replay  # noqa: E402
from blockswarm.scripted import ScriptedBuilder, run_simulated  # noqa: E402
from blockswarm.tasks import DEFAULT_BRIEF, deployed, ranking  # noqa: E402
from blockswarm.world import Block, Config, World, ray_cells  # noqa: E402

SHOP = (13, 2, 0)
NEAR_SHOP = [12, 3, 0]


def make_world(positions=None, **kw) -> World:
    """Two agents, a and b, on a simulated clock, on a 16 x 16 x 8 grid with the shop at
    (13, 2, 0) and 1000 coins. Default: a next to the shop, b farther away."""
    kw.setdefault("agents", ["a", "b"])
    kw.setdefault("width", 16)
    kw.setdefault("depth", 16)
    kw.setdefault("height", 8)
    kw.setdefault("shop_pos", SHOP)
    kw.setdefault("starting_money", 1000)
    if positions is not None or "start_positions" not in kw:
        kw["start_positions"] = positions or [NEAR_SHOP, [5, 3, 0]]
    return World(Config(**kw), clock=SimClock())


def records_of(w: World, kind: str) -> list[dict]:
    return [r for r in w.recorder.records if r["type"] == kind]


def events(w: World) -> list[dict]:
    return records_of(w, "event")


def build(w: World, agent: str, cells: list[tuple[int, int, int]], btype: str = "stone") -> None:
    """Test setup: put blocks straight into the grid (not via tools, so not replayable)."""
    from blockswarm.world import Block
    for c in cells:
        w.blocks[c] = Block(btype, agent, 0.0)
    w._analysis_cache = None


def advance(w: World, dt: float) -> None:
    w.clock.advance_to(w.clock.now() + dt)


class TestConfigAndHeader(unittest.TestCase):
    def test_validation(self):
        for bad in ({"task": "blueprint"}, {"blueprint": "house"}, {"scarcity": "exact"}, {"payoff": "team"},
                    {"chest_pos": (1, 1, 0)}, {"chest_scale": 1.0},
                    {"agents": []}, {"agents": ["a", "a"]}, {"max_seconds": 0}, {"action_cooldown_s": -1},
                    {"max_actions_per_agent": 0}, {"shop_pos": (24, 0, 0)}, {"jump_airtime": 0},
                    {"incentive": "team"}, {"deploy_top_k": 0}, {"deploy_top_k": 4}, {"deploy_top_k": 1.5},
                    {"starting_money": -1}, {"starting_money": 1.5}, {"starting_money": True},
                    {"show_leaderboard": "yes"},
                    {"start_positions": [[1, 1, 0]]},
                    {"start_positions": [[1, 1, 1], [3, 3, 0], [5, 5, 0]]},
                    {"start_positions": [[1, 1, 0], [1, 1, 0], [5, 5, 0]]},
                    {"start_positions": [[12, 4, 0], [3, 3, 0], [5, 5, 0]]}):
            with self.assertRaises((ValueError, TypeError), msg=str(bad)):
                Config(**bad)
        Config(start_positions=[[12, 4, 1], [3, 3, 0], [5, 5, 0]])  # standing on the shop is fine
        c = Config()
        self.assertEqual((c.width, c.depth, c.height, c.shop_pos), (24, 24, 10, (12, 4, 0)))
        self.assertEqual((c.starting_money, c.incentive, c.deploy_top_k, c.show_leaderboard),
                         (300, "competitive", 1, False))

    def test_header(self):
        w = World(Config(), clock=SimClock())
        h = w.recorder.records[0]
        self.assertEqual(h["log_version"], 4)
        self.assertEqual(h["clock"], "simulated")
        self.assertEqual(h["task"], {"kind": "open", "brief": DEFAULT_BRIEF})
        for gone in ("chest", "chest_pos"):
            self.assertNotIn(gone, h)
        for gone in ("blueprint", "scarcity", "payoff", "chest_scale", "chest_pos", "reference_design"):
            self.assertNotIn(gone, h["config"])
        self.assertEqual(h["shop"]["pos"], [12, 4, 0])
        self.assertEqual(h["shop"]["starting_money"], 300)
        self.assertEqual(h["shop"]["catalog"], catalog_json())
        self.assertEqual(h["wallet"], 300)
        self.assertEqual(h["positions"], {"agent_1": [11, 5, 0], "agent_2": [13, 5, 0], "agent_3": [11, 3, 0]})
        for key in ("shop_pos", "starting_money", "incentive", "deploy_top_k", "show_leaderboard",
                    "start_positions", "walk_speed", "reach", "eye_height", "view_radius",
                    "jump_airtime", "task", "brief", "target_interior",
                    "reach_line_of_sight", "view_occlusion", "max_seconds", "max_actions_per_agent"):
            self.assertIn(key, h["config"])
        self.assertEqual(h["config"]["shop_pos"], [12, 4, 0])
        self.assertEqual(h["block_types"], [i["id"] for i in h["shop"]["catalog"]])
        json.dumps(h)


class TestTerrain(unittest.TestCase):
    def terrain(self, solids, bodies=()):
        s, b = set(solids), set(bodies)
        return Terrain(8, 8, 8, solid=lambda p: p in s, occupied=lambda p: p in b)

    def test_standable(self):
        t = self.terrain({(1, 1, 0), (2, 2, 1)}, bodies={(4, 4, 0), (4, 4, 1)})
        self.assertTrue(t.standable((0, 0, 0)))
        self.assertTrue(t.standable((1, 1, 1)))       # on a block
        self.assertFalse(t.standable((1, 1, 0)))      # inside a block
        self.assertFalse(t.standable((0, 0, 1)))      # floating
        self.assertFalse(t.standable((2, 2, 0)))      # head would be in a block
        self.assertFalse(t.standable((4, 4, 0)))      # someone is there
        self.assertFalse(t.standable((0, 0, 7)))      # head out of the grid

    def test_step_up_one_not_two(self):
        t = self.terrain({(1, 0, 0)})
        path, length = t.path((0, 0, 0), (1, 0, 1))
        self.assertEqual(path, [(0, 0, 0), (1, 0, 1)])
        self.assertEqual(length, 1.0)
        wall = {(1, y, z) for y in range(8) for z in range(2)}
        t2 = self.terrain(wall | {(2, 0, 0)})
        self.assertIsNone(t2.path((0, 0, 0), (2, 0, 1)))  # 2-high wall all the way across

    def test_step_down_at_most_three(self):
        tower3 = {(0, 0, z) for z in range(3)}
        t = self.terrain(tower3)
        self.assertIsNotNone(t.path((0, 0, 3), (1, 0, 0)))  # drop 3
        tower4 = {(0, 0, z) for z in range(4)}
        t = self.terrain(tower4)
        self.assertIsNone(t.path((0, 0, 4), (1, 0, 0)))     # drop 4 is too far

    def test_no_corner_cutting(self):
        t = self.terrain(set())
        self.assertEqual(t.path((0, 0, 0), (1, 1, 0))[1], SQRT2)
        t = self.terrain({(1, 0, 0), (1, 0, 1)})
        path, length = t.path((0, 0, 0), (1, 1, 0))
        self.assertEqual(length, 2.0)
        self.assertEqual(path, [(0, 0, 0), (0, 1, 0), (1, 1, 0)])

    def test_unreachable(self):
        ring = {(x, y, z) for x in range(3) for y in range(3) for z in range(3) if (x, y) != (1, 1)}
        t = self.terrain(ring)
        self.assertIsNone(t.path((5, 5, 0), (1, 1, 0)))

    def test_ray_cells(self):
        self.assertEqual(ray_cells((0.5, 0.5, 0.5), (3, 0, 0)), [(1, 0, 0), (2, 0, 0)])
        self.assertEqual(ray_cells((0.5, 0.5, 0.5), (1, 1, 0)), [])  # exact diagonal through an edge


class TestBodiesAndShop(unittest.TestCase):
    def test_shop_is_solid(self):
        w = make_world()
        w.act("a", "buy", item="stone", n=2)
        r = w.act("a", "place", x=13, y=2, z=0, type="stone")
        self.assertFalse(r.ok)
        self.assertIn("shop", r.message)
        r = w.act("a", "remove", x=13, y=2, z=0)
        self.assertFalse(r.ok)
        self.assertIn("shop cannot be removed", r.message)
        self.assertTrue(w.act("a", "place", x=13, y=2, z=1, type="stone").ok)  # supported by the shop
        self.assertTrue(w.standable((13, 3, 0)))
        self.assertFalse(w.standable((13, 2, 0)))
        self.assertEqual(w.analysis()["leftovers"], 1)  # the shop itself isn't a block

    def test_stand_on_shop(self):
        w = make_world(positions=[[12, 2, 0], [5, 3, 0]])
        self.assertTrue(w.act("a", "move_to", x=13, y=2, z=1).ok)  # step up onto the shop
        self.assertEqual(w.positions["a"], (13, 2, 1))

    def test_no_placing_into_bodies(self):
        w = make_world(positions=[NEAR_SHOP, [11, 3, 0]])
        w.act("a", "buy", item="stone", n=3)
        for cell in ((11, 3, 0), (11, 3, 1), (12, 3, 0), (12, 3, 1)):
            r = w.act("a", "place", x=cell[0], y=cell[1], z=cell[2], type="stone")
            self.assertFalse(r.ok, cell)
            self.assertIn("body", r.message)

    def test_two_agents_cannot_share_a_cell(self):
        w = make_world(positions=[NEAR_SHOP, [10, 3, 0]])
        r = w.act("b", "move_to", x=12, y=3, z=0)
        self.assertFalse(r.ok)
        self.assertIn("not standable", r.message)


class TestMoving(unittest.TestCase):
    def test_move_walk_time_and_walking_rules(self):
        w = make_world(positions=[NEAR_SHOP, [5, 3, 0]])
        r = w.act("b", "move_to", x=8, y=6, z=0)
        self.assertTrue(r.ok)
        mv = events(w)[-1]["diff"]["agents"]["b"]
        self.assertEqual(mv["from"], [5, 3, 0])
        self.assertEqual(mv["to"], [8, 6, 0])
        self.assertEqual(mv["path"][0], [5, 3, 0])
        self.assertEqual(mv["path"][-1], [8, 6, 0])
        self.assertEqual(mv["arrive_t"], round(3 * SQRT2 / 4.317, 3))
        self.assertEqual(w.positions["b"], (8, 6, 0))  # moved immediately
        used = w.actions_used["b"]
        advance(w, 0.5)
        r = w.act("b", "buy", item="stone", n=1)
        self.assertFalse(r.ok)
        self.assertTrue(r.message.startswith("walking:"))
        self.assertEqual(w.actions_used["b"], used)  # consumed nothing
        self.assertTrue(w.act("b", "say", text="on my way").ok)
        self.assertTrue(w.act("b", "observe").ok)
        self.assertTrue(w.act("b", "note", text="walking").ok)
        self.assertEqual(w.view("b")["walking_until"], mv["arrive_t"])
        w.clock.advance_to(mv["arrive_t"])
        self.assertIsNone(w.view("b")["walking_until"])
        self.assertTrue(w.act("b", "move_to", x=8, y=7, z=0).ok)
        replay(w.recorder.records)

    def test_move_failures_count(self):
        w = make_world()
        build(w, "a", [(x, y, z) for x in range(1, 4) for y in range(1, 4) for z in range(3) if (x, y) != (2, 2)])
        self.assertFalse(w.act("b", "move_to", x=2, y=2, z=0).ok)  # walled in: no path
        self.assertFalse(w.act("b", "move_to", x=1, y=1, z=1).ok)  # inside a block
        self.assertFalse(w.act("b", "move_to", x=0, y=0, z=1).ok)  # floating
        self.assertFalse(w.act("b", "move_to", x=99, y=0, z=0).ok)
        self.assertEqual(w.actions_used["b"], 4)


class TestReach(unittest.TestCase):
    def test_reach_limits(self):
        w = make_world(positions=[NEAR_SHOP, [5, 3, 0]])
        r = w.act("b", "buy", item="stone", n=1)
        self.assertFalse(r.ok)
        self.assertTrue(r.message.startswith("out of reach"))
        self.assertEqual(w.actions_used["b"], 1)  # an invalid action still counts
        self.assertTrue(w.act("a", "buy", item="stone", n=2).ok)
        self.assertFalse(w.act("a", "place", x=6, y=3, z=0, type="stone").ok)  # far
        self.assertFalse(w.act("a", "give", to="b", type="stone", n=1).ok)  # b is far
        self.assertTrue(w.act("a", "place", x=10, y=3, z=0, type="stone").ok)
        self.assertTrue(w.act("b", "move_to", x=8, y=3, z=0).ok)
        advance(w, 5)
        self.assertTrue(w.act("b", "remove", x=10, y=3, z=0).ok)  # 2 away
        self.assertTrue(w.act("a", "give", to="b", type="stone", n=1).ok)  # 4 away now
        self.assertEqual(w.inventories["b"], {"stone": 2})
        replay(w.recorder.records)

    def test_reach_is_from_the_eyes(self):
        w = make_world(positions=[[8, 3, 0], [5, 3, 0]])
        w.inventories["a"] = {"stone": 2}
        # 4 cells away on the ground: |(4, 0, -1.12)| = 4.15 <= 4.5
        self.assertTrue(w.can_reach((8, 3, 0), (12, 3, 0)))
        # 4 away and 3 up: |(4, 0, 1.88)| = 4.42; 4 away, 4 up: 4.93
        self.assertTrue(w.can_reach((8, 3, 0), (4, 3, 3)))
        self.assertFalse(w.can_reach((8, 3, 0), (4, 3, 4)))
        self.assertFalse(w.can_reach((8, 3, 0), (8, 3, 7)))
        self.assertAlmostEqual(w.distance((8, 3, 0), (8, 3, 1)), 0.12, places=6)

    def test_line_of_sight(self):
        w = make_world(positions=[[8, 3, 0], [5, 3, 0]])
        w.act("a", "observe")
        build(w, "b", [(9, 3, 0), (9, 3, 1), (9, 3, 2)])
        w.inventories["a"] = {"stone": 1}
        r = w.act("a", "place", x=10, y=3, z=0, type="stone")
        self.assertFalse(r.ok)
        self.assertTrue(r.message.startswith("no line of sight"))
        self.assertFalse(w.act("a", "remove", x=10, y=3, z=1).ok)  # empty anyway; check below
        build(w, "b", [(10, 3, 1)])
        r = w.act("a", "remove", x=10, y=3, z=1)
        self.assertTrue(r.message.startswith("no line of sight"), r.message)
        self.assertTrue(w.act("a", "remove", x=9, y=3, z=1).ok)  # the wall itself is fine to remove

    def test_line_of_sight_switch(self):
        w = make_world(positions=[[8, 3, 0], [5, 3, 0]], reach_line_of_sight=False)
        build(w, "b", [(9, 3, 0), (9, 3, 1), (9, 3, 2)])
        w.inventories["a"] = {"stone": 1}
        self.assertTrue(w.act("a", "place", x=10, y=3, z=0, type="stone").ok)

    def test_shop_behind_wall(self):
        w = make_world(positions=[[10, 2, 0], [5, 3, 0]])
        build(w, "b", [(11, 2, z) for z in range(3)] + [(12, 2, z) for z in range(3)])
        r = w.act("a", "buy", item="stone", n=1)
        self.assertTrue(r.message.startswith("no line of sight"))
        self.assertFalse(w.view("a")["shop_in_reach"])

    def test_reachable_stand_cells(self):
        w = make_world()
        cells = w.reachable_stand_cells((7, 7, 0), "b")
        self.assertTrue(cells)
        for c in cells:
            self.assertTrue(w.standable(c, "b"))
            self.assertTrue(w.can_reach(c, (7, 7, 0)))
        costs = w.walk_costs("b")
        self.assertEqual(cells, sorted(cells, key=lambda c: (costs[c], c)))


class TestJumpAndFall(unittest.TestCase):
    def test_jump_and_place_under_self(self):
        w = make_world(positions=[NEAR_SHOP, [5, 3, 0]])
        w.act("a", "buy", item="stone", n=3)
        w.clock.advance_to(10)
        r = w.act("a", "jump")
        self.assertTrue(r.ok)
        mv = events(w)[-1]["diff"]["agents"]["a"]
        self.assertEqual(mv, {"from": [12, 3, 0], "to": [12, 3, 0], "path": [[12, 3, 0]],
                              "depart_t": 10.0, "arrive_t": 10.6, "jump": True})
        self.assertEqual(w.view("a")["airborne_until"], 10.6)
        w.clock.advance_to(10.1)
        r = w.act("a", "move_to", x=11, y=3, z=0)
        self.assertFalse(r.ok)
        self.assertTrue(r.message.startswith("airborne:"))
        self.assertTrue(w.act("a", "jump").message.startswith("airborne:"))
        used = w.actions_used["a"]
        r = w.act("a", "place", x=12, y=3, z=0, type="stone")
        self.assertTrue(r.ok, r.message)
        self.assertEqual(w.actions_used["a"], used + 1)
        ev = events(w)[-1]
        self.assertEqual(ev["diff"]["agents"]["a"], {"from": [12, 3, 0], "to": [12, 3, 1],
                                                     "path": [[12, 3, 0], [12, 3, 1]],
                                                     "depart_t": 10.1, "arrive_t": 10.6})
        self.assertEqual(w.positions["a"], (12, 3, 1))
        # lifted already: the new feet cell is not free for a second block this jump
        self.assertFalse(w.act("a", "place", x=12, y=3, z=1, type="stone").ok)
        w.clock.advance_to(10.7)
        self.assertTrue(w.act("a", "move_to", x=11, y=3, z=0).ok)
        replay(w.recorder.records)

    def test_jump_lands_before_place(self):
        w = make_world(positions=[NEAR_SHOP, [5, 3, 0]])
        w.act("a", "buy", item="stone", n=1)
        w.act("a", "jump")
        w.clock.advance_to(0.7)
        r = w.act("a", "place", x=12, y=3, z=0, type="stone")
        self.assertFalse(r.ok)
        self.assertIn("body", r.message)

    def test_other_agent_places_under_jumper(self):
        w = make_world(positions=[NEAR_SHOP, [11, 3, 0]])
        w.act("a", "buy", item="stone", n=1)
        w.act("b", "jump")
        w.clock.advance_to(0.2)
        self.assertTrue(w.act("a", "place", x=11, y=3, z=0, type="stone").ok)
        self.assertEqual(w.positions["b"], (11, 3, 1))
        replay(w.recorder.records)

    def test_jump_with_blocked_headroom(self):
        w = make_world(positions=[NEAR_SHOP, [5, 3, 0]])
        build(w, "b", [(12, 4, 2)])
        w.act("a", "buy", item="stone", n=2)
        self.assertTrue(w.act("a", "place", x=12, y=3, z=2, type="stone").ok)  # supported sideways
        w.act("a", "jump")
        r = w.act("a", "place", x=12, y=3, z=0, type="stone")
        self.assertFalse(r.ok)
        self.assertIn("above their head", r.message)

    def test_jump_while_walking(self):
        w = make_world()
        w.act("b", "move_to", x=5, y=8, z=0)
        self.assertTrue(w.act("b", "jump").message.startswith("walking:"))

    def test_fall_self_and_others(self):
        w = make_world(positions=[NEAR_SHOP, [11, 3, 0]])
        w.act("a", "buy", item="stone", n=3)
        # b climbs a 2-high tower: jump, a places under it, twice
        for z in (0, 1):
            w.act("b", "jump")
            advance(w, 0.1)
            self.assertTrue(w.act("a", "place", x=11, y=3, z=z, type="stone").ok)
            advance(w, 1.0)
        self.assertEqual(w.positions["b"], (11, 3, 2))
        # the bottom block is hidden behind the top one from a's eyes
        self.assertTrue(w.act("a", "remove", x=11, y=3, z=0).message.startswith("no line of sight"))
        r = w.act("a", "remove", x=11, y=3, z=1)  # someone else's support: b drops 1
        self.assertTrue(r.ok, r.message)
        mv = events(w)[-1]["diff"]["agents"]["b"]
        self.assertTrue(mv["fall"])
        self.assertEqual((mv["from"], mv["to"]), ([11, 3, 2], [11, 3, 1]))
        self.assertEqual(mv["arrive_t"], round(mv["depart_t"] + 0.1, 3))
        advance(w, 1.0)
        r = w.act("b", "remove", x=11, y=3, z=0)  # its own support: b drops to the ground
        self.assertTrue(r.ok, r.message)
        mv = events(w)[-1]["diff"]["agents"]["b"]
        self.assertEqual((mv["to"], mv["fall"]), ([11, 3, 0], True))
        replay(w.recorder.records)

    def test_fall_several_cells(self):
        w = make_world(positions=[NEAR_SHOP, [5, 3, 0]])
        build(w, "a", [(11, 3, 2), (11, 2, 2)])  # a ledge: (11,3,2) hangs off (11,2,2)... both floating
        w.positions["b"] = (11, 3, 3)
        r = w.act("a", "remove", x=11, y=3, z=2)
        self.assertTrue(r.ok, r.message)
        mv = events(w)[-1]["diff"]["agents"]["b"]
        self.assertEqual((mv["to"], mv["arrive_t"]), ([11, 3, 0], 0.3))
        self.assertEqual(w.positions["b"], (11, 3, 0))


class TestView(unittest.TestCase):
    def test_radius_offsets_and_occlusion(self):
        w = make_world(positions=[[2, 2, 0], [12, 12, 0]])
        build(w, "b", [(4, 2, 0), (15, 15, 0)])
        v = w.view("a")
        self.assertEqual([b["pos"] for b in v["blocks"]], [[4, 2, 0]])
        self.assertEqual(v["blocks"][0]["offset"], [2, 0, 0])
        self.assertEqual(v["position"], [2, 2, 0])
        build(w, "b", [(3, 2, z) for z in range(4)])
        hidden = [b["pos"] for b in w.view("a")["blocks"]]
        self.assertNotIn([4, 2, 0], hidden)
        self.assertIn([3, 2, 1], hidden)
        w.config.view_occlusion = False
        self.assertIn([4, 2, 0], [b["pos"] for b in w.view("a")["blocks"]])

    def test_agents_in_view(self):
        w2 = make_world(positions=[[2, 2, 0], [12, 12, 0], [3, 4, 0]], agents=["a", "b", "c"])
        self.assertEqual(w2.view("a")["agents_visible"],
                         [{"id": "c", "pos": [3, 4, 0], "walking": False, "airborne": False}])

    def test_wallet_and_shop_in_view(self):
        w = make_world(positions=[NEAR_SHOP, [5, 3, 0]])
        v = w.view("a")
        self.assertEqual((v["wallet"], v["shop_pos"], v["shop_in_reach"]), (1000, [13, 2, 0], True))
        for gone in ("chest", "chest_pos", "chest_last_seen", "payoff", "scores", "analysis"):
            self.assertNotIn(gone, v)
        vb = w.view("b")
        self.assertEqual((vb["wallet"], vb["shop_in_reach"]), (1000, False))  # far away: still sees the wallet
        w.act("a", "buy", item="glass", n=2)
        self.assertEqual(w.view("b")["wallet"], 1000 - 2 * CATALOG["glass"].price)
        w.act("a", "move_to", x=5, y=6, z=0)
        self.assertFalse(w.view("a")["shop_in_reach"])  # walking
        json.dumps(v)

    def test_leaderboard(self):
        w = make_world(positions=[NEAR_SHOP, [11, 4, 0]], show_leaderboard=True)
        w.act("a", "buy", item="stone", n=2)
        w.act("a", "place", x=11, y=2, z=0, type="stone")
        self.assertEqual(w.view("b")["scores"], {"a": 1, "b": 0})
        self.assertNotIn("scores", make_world().view("a"))

    def test_messages_and_privacy(self):
        w = make_world()
        w.act("a", "say", text="public one")
        w.act("a", "whisper", to="b", text="secret")
        w.act("a", "whisper", to="b", text="secret 2")
        w.act("a", "note", text="mine")
        w.act("a", "buy", item="stone", n=1)
        vb = w.view("b")
        self.assertEqual([m["text"] for m in vb["messages"]], ["public one", "secret", "secret 2"])
        self.assertEqual([m["text"] for m in w.view("b", since_event_id=1)["messages"]], ["secret 2"])
        self.assertEqual(w.view("b", since_event_id=vb["last_event_id"])["messages"], [])
        self.assertEqual(vb["notes"], [])
        self.assertEqual(vb["inventory"], {})
        self.assertEqual(w.view("a")["notes"], ["mine"])
        self.assertEqual(w.view("a")["task"]["kind"], "open")
        json.dumps(w.view("a"))


class TestInventoryAndShop(unittest.TestCase):
    def setUp(self):
        self.w = make_world(positions=[NEAR_SHOP, [11, 4, 0]])

    def test_inventory_cap(self):
        w = self.w
        self.assertTrue(w.act("a", "buy", item="bricks", n=12).ok)
        r = w.act("a", "buy", item="stone", n=1)
        self.assertIn("cap", r.message)
        self.assertEqual(w.wallet, 1000 - 12 * CATALOG["bricks"].price)  # nothing charged
        w.inventories["b"] = {"bricks": 12}
        self.assertFalse(w.act("a", "give", to="b", type="bricks", n=1).ok)

    def test_cap_on_remove(self):
        w = self.w
        w.act("a", "buy", item="stone", n=1)
        w.act("a", "place", x=11, y=2, z=0, type="stone")
        w.act("a", "buy", item="bricks", n=12)
        self.assertFalse(w.act("a", "remove", x=11, y=2, z=0).ok)
        self.assertIn((11, 2, 0), w.blocks)

    def test_buy_rules(self):
        w = make_world(positions=[NEAR_SHOP, [5, 3, 0]], starting_money=50)
        price = CATALOG["oak_door"].price
        r = w.act("a", "buy", item="oak_door", n=2)
        self.assertTrue(r.ok, r.message)
        self.assertEqual(w.wallet, 50 - 2 * price)
        self.assertEqual(w.inventories["a"], {"oak_door": 2})
        ev = events(w)[-1]
        self.assertEqual(ev["diff"], {"grid": [], "wallet": -2 * price, "inventory": {"a": {"oak_door": 2}},
                                      "agents": {}, "votes": {}})
        self.assertEqual(ev["cost"], "action")
        # money: the wallet is shared and can't go negative
        r = w.act("a", "buy", item="diamond_block", n=1)
        self.assertFalse(r.ok)
        self.assertTrue(r.message.startswith("not enough money"), r.message)
        self.assertTrue(w.act("a", "buy", item="dirt", n=w.wallet if w.wallet <= 10 else 10).ok)
        # reach: b is far from the shop
        r = w.act("b", "buy", item="stone", n=1)
        self.assertTrue(r.message.startswith("out of reach"), r.message)
        # unknown items, n <= 0, old tools
        for item in ("lava", "brick", "door", 7, None):
            r = w.act("a", "buy", item=item, n=1)
            self.assertFalse(r.ok)
            self.assertIn("unknown item", r.message)
        for n in (0, -1, 1.5, "2", True):
            self.assertFalse(w.act("a", "buy", item="stone", n=n).ok, n)
        for tool in ("take", "deposit"):
            self.assertIn("unknown tool", w.act("a", tool, type="stone", n=1).message)
        # walking: refused and nothing used up
        w.act("a", "move_to", x=11, y=3, z=0)
        used = w.actions_used["a"]
        r = w.act("a", "buy", item="stone", n=1)
        self.assertTrue(r.message.startswith("walking:"))
        self.assertEqual(w.actions_used["a"], used)
        wallet = w.wallet
        for e in events(w)[2:]:
            self.assertEqual(e["diff"]["wallet"], 0 if not (e["ok"] and e["tool"] == "buy") else e["diff"]["wallet"])
        self.assertEqual(w.wallet, wallet)
        w.finish("aborted")
        self.assertEqual(replay(w.recorder.records)["wallet"], w.wallet)
        self.assertEqual(records_of(w, "episode_end")[0]["wallet"], w.wallet)

    def test_buy_through_glass_not_stone(self):
        # a stands 3 cells from the shop with a wall in between
        for wall, ok in (("glass", True), ("glass_pane", True), ("light_blue_stained_glass", True),
                         ("tinted_glass", False), ("stone", False), ("oak_door", False)):
            w = make_world(positions=[[10, 2, 0], [5, 3, 0]])
            build(w, "b", [(11, 2, z) for z in range(3)], wall)
            r = w.act("a", "buy", item="stone", n=1)
            self.assertEqual(r.ok, ok, (wall, r.message))
            self.assertEqual(w.view("a")["shop_in_reach"], ok)
            if not ok:
                self.assertTrue(r.message.startswith("no line of sight"), r.message)

    def test_give_whisper_self_or_unknown(self):
        w = self.w
        w.act("a", "buy", item="stone", n=1)
        for to in ("a", "ghost"):
            self.assertFalse(w.act("a", "give", to=to, type="stone", n=1).ok)
            self.assertFalse(w.act("a", "whisper", to=to, text="hi").ok)
        self.assertEqual(w.messages, [])

    def test_unknown_tool_and_bad_args(self):
        w = self.w
        self.assertFalse(w.act("a", "fly").ok)
        self.assertFalse(w.act("a", "pillar_up", type="stone").ok)  # removed tool
        self.assertFalse(w.act("a", "buy", item="stone").ok)
        self.assertFalse(w.act("a", "place", x=0, y=0, z=0, type="lava").ok)
        self.assertFalse(w.act("a", "say", text="x" * 1001).ok)
        self.assertEqual(len(events(w)), 5)
        self.assertEqual(w.actions_used["a"], 4)


class TestBlockProperties(unittest.TestCase):
    def test_transparent_view_and_line_of_sight(self):
        w = make_world(positions=[[2, 2, 0], [12, 12, 0]])
        build(w, "b", [(5, 2, 0)])
        for wall, seen in (("glass", True), ("white_stained_glass", True), ("bricks", False),
                           ("tinted_glass", False)):
            build(w, "b", [(3, 2, z) for z in range(4)], wall)
            self.assertEqual([5, 2, 0] in [b["pos"] for b in w.view("a")["blocks"]], seen, wall)
            self.assertEqual(w.line_of_sight((2, 2, 0), (5, 2, 0)), seen, wall)
        build(w, "b", [(3, 2, z) for z in range(4)], "glass")
        w.inventories["a"] = {"stone": 1}
        r = w.act("a", "place", x=4, y=2, z=0, type="stone")  # through the glass wall
        self.assertTrue(r.ok, r.message)

    def test_doors_walkable_but_solid(self):
        w = make_world(positions=[[2, 4, 0], [12, 12, 0]])
        # a 2-high wall across the grid at x = 4, with a 2-high oak door at y = 4
        build(w, "b", [(4, y, z) for y in range(16) for z in range(2) if y != 4])
        build(w, "b", [(4, 4, 0), (4, 4, 1)], "oak_door")
        self.assertTrue(w.standable((4, 4, 0)))           # stand in the doorway
        r = w.act("a", "move_to", x=6, y=4, z=0)          # walk through it
        self.assertTrue(r.ok, r.message)
        path = events(w)[-1]["diff"]["agents"]["a"]["path"]
        self.assertIn([4, 4, 0], path)
        advance(w, 5)
        self.assertTrue(w.act("a", "move_to", x=4, y=4, z=0).ok)  # and stop in it
        advance(w, 5)
        w.inventories["a"] = {"stone": 2}
        # still solid for placing and support
        r = w.act("a", "place", x=4, y=4, z=1, type="stone")
        self.assertFalse(r.ok)
        self.assertIn("occupied by oak_door", r.message)
        self.assertTrue(w.supported((4, 4, 2)))
        self.assertTrue(w.act("a", "place", x=4, y=4, z=2, type="stone").ok)  # on top of the door
        # a door with a solid block in the upper cell is not standable
        build(w, "b", [(4, 8, 0)], "oak_door")
        self.assertFalse(w.standable((4, 8, 0)))
        replay_free = w.analysis()
        self.assertFalse(replay_free["enclosed"])
        # a door blocks sight like a wall
        self.assertFalse(w.line_of_sight((2, 8, 0), (6, 8, 0)))

    def test_door_closed_for_enclosure(self):
        w = make_world(positions=[[1, 1, 0], [14, 14, 0]])
        bp = BLUEPRINTS["house"]
        for p, t in bp.cells.items():
            w.blocks[p] = Block(t, "a", 0.0)
        w._analysis_cache = None
        d = w.analysis()
        self.assertTrue(d["enclosed"])
        self.assertEqual((d["interior_cells"], d["doors"], d["windows"], d["leftovers"]), (45, 2, 4, 0))
        # an agent can walk from outside into the closed house through its door
        r = w.act("a", "move_to", x=7, y=7, z=1)
        self.assertTrue(r.ok, r.message)
        path = events(w)[-1]["diff"]["agents"]["a"]["path"]
        self.assertIn([7, 5, 1], path)  # the door cell
        # replace the door by a plain wall: still enclosed, no door, and no way in
        for p, t in bp.cells.items():
            if t == "oak_door":
                w.blocks[p] = Block("oak_planks", "a", 0.0)
        w._analysis_cache = None
        self.assertEqual(w.analysis()["doors"], 0)
        self.assertTrue(w.analysis()["enclosed"])
        self.assertIsNone(w.find_path("b", (6, 6, 1)))


class TestLimits(unittest.TestCase):
    def test_cooldown(self):
        w = make_world(action_cooldown_s=2.0)
        self.assertTrue(w.act("a", "buy", item="stone", n=1).ok)
        w.clock.advance_to(1.0)
        r = w.act("a", "buy", item="stone", n=1)
        self.assertTrue(r.message.startswith("cooldown"))
        self.assertEqual(w.actions_used["a"], 1)
        self.assertTrue(w.act("a", "say", text="messages are not rate limited").ok)
        self.assertEqual(w.view("a")["next_action_at"], 2.0)
        w.clock.advance_to(2.0)
        self.assertTrue(w.act("a", "buy", item="stone", n=1).ok)
        replay(w.recorder.records)

    def test_action_cap_and_action_limit_end(self):
        w = make_world(max_actions_per_agent=2)
        self.assertFalse(w.act("a", "place", x=0, y=0, z=0, type="stone").ok)
        self.assertFalse(w.act("a", "fly").ok)
        r = w.act("a", "buy", item="stone", n=1)
        self.assertTrue(r.message.startswith("action limit reached"))
        self.assertEqual(w.view("a")["actions_remaining"], 0)
        self.assertTrue(w.act("a", "say", text="still allowed").ok)
        w.act("b", "declare_done")
        self.assertFalse(w.done)
        w.act("b", "jump")
        self.assertEqual(w.end_reason, "action_limit")
        self.assertEqual(records_of(w, "episode_end")[0]["actions_used"], {"a": 2, "b": 2})
        replay(w.recorder.records)

    def test_time_limit(self):
        w = make_world(max_seconds=30)
        w.clock.advance_to(29.999)
        self.assertTrue(w.act("a", "buy", item="stone", n=1).ok)
        w.clock.advance_to(30)
        n = len(w.recorder.records)
        r = w.act("b", "observe")
        self.assertFalse(r.ok)
        self.assertEqual(w.end_reason, "time_limit")
        self.assertEqual([x["type"] for x in w.recorder.records[n:]], ["snapshot", "episode_end"])
        w2 = make_world(max_seconds=5)
        w2.clock.advance_to(6)
        self.assertTrue(w2.check_end())

    def test_all_done_and_finish_idempotent(self):
        w = make_world()
        w.act("a", "declare_done")
        self.assertTrue(w.act("a", "declare_done").ok)
        w.act("b", "declare_done")
        self.assertEqual(w.end_reason, "all_done")
        w.finish("aborted")
        self.assertEqual(len(records_of(w, "episode_end")), 1)
        self.assertEqual(w.recorder.records[-2]["type"], "snapshot")
        n = len(w.recorder.records)
        self.assertFalse(w.act("a", "observe").ok)
        self.assertEqual(len(w.recorder.records), n)
        with self.assertRaises(ValueError):
            w.act("ghost", "observe")


class TestSnapshots(unittest.TestCase):
    def test_snapshot_timing(self):
        w = make_world(snapshot_interval_s=10)
        for t in [0.5, 3.0, 9.999, 10.0, 10.5, 25.0, 61.2]:
            w.clock.advance_to(t)
            w.act("a", "observe")
        snaps = records_of(w, "snapshot")
        self.assertEqual([s["t"] for s in snaps], [0.5, 10.0, 25.0, 61.2])
        self.assertEqual([s["after_event_id"] for s in snaps], [-1, 2, 4, 5])
        w.finish("aborted")
        self.assertEqual(replay(w.recorder.records)["snapshots_checked"], 5)

    def test_positions_and_walkers_in_snapshot(self):
        w = make_world(snapshot_interval_s=1)
        w.act("b", "move_to", x=5, y=9, z=0)
        w.clock.advance_to(1.0)
        w.act("a", "observe")
        snap = records_of(w, "snapshot")[-1]
        self.assertEqual(snap["positions"], {"a": NEAR_SHOP, "b": [5, 9, 0]})
        self.assertEqual(snap["walking_until"], {"b": round(6 / 4.317, 3)})
        w.clock.advance_to(2.0)
        w.act("a", "observe")
        self.assertEqual(records_of(w, "snapshot")[-1]["walking_until"], {})
        w.finish("aborted")
        self.assertEqual(replay(w.recorder.records)["positions"], {"a": NEAR_SHOP, "b": [5, 9, 0]})

    def test_scores_and_analysis_in_records(self):
        w = World(Config(agents=["a", "b"]), clock=SimClock())
        w.act("a", "observe")
        w.finish("aborted")
        for rec in (records_of(w, "snapshot")[0], records_of(w, "episode_end")[0]):
            self.assertEqual(rec["wallet"], 200)
            self.assertEqual(rec["individual_scores"], {"a": 0, "b": 0})
            self.assertEqual(set(rec["analysis"]), {"enclosed", "interior_cells", "interior", "shell_cells",
                                                    "doors", "windows", "leftovers", "target_interior"})
            for gone in ("team_score", "score_details", "chest"):
                self.assertNotIn(gone, rec)
        end = records_of(w, "episode_end")[0]
        self.assertEqual(set(end), {"type", "t", "reason", "wallet", "individual_scores", "ranking", "deployed",
                                    "analysis", "actions_used"})
        replay(w.recorder.records)


class TestIncentive(unittest.TestCase):
    def test_ranking_ties_and_deployed(self):
        agents = ["a", "b", "c", "d", "e"]
        r = ranking({"a": 3, "b": 5, "c": 3, "d": 0, "e": 3}, agents)
        self.assertEqual([(x["agent"], x["score"], x["rank"]) for x in r],
                         [("b", 5, 1), ("a", 3, 2), ("c", 3, 2), ("e", 3, 2), ("d", 0, 5)])
        self.assertEqual(deployed(r, 2, "competitive"), ["b", "a", "c", "e"])  # ties at the boundary all in
        self.assertEqual(deployed(r, 1, "competitive"), ["b"])
        self.assertEqual(deployed(r, 5, "competitive"), ["b", "a", "c", "e", "d"])
        self.assertEqual(deployed(r, 2, "none"), [])
        r = ranking({"a": 1, "b": 1, "c": 0}, ["a", "b", "c"])
        self.assertEqual([x["rank"] for x in r], [1, 1, 3])
        self.assertEqual(deployed(r, 1, "competitive"), ["a", "b"])

    def test_individual_scores_count_blocks_still_in_world(self):
        w = make_world(positions=[NEAR_SHOP, [11, 4, 0], [10, 3, 0]], agents=["a", "b", "c"])
        w.act("a", "buy", item="stone", n=4)
        w.act("a", "give", to="b", type="stone", n=2)
        w.act("a", "place", x=11, y=2, z=0, type="stone")
        w.act("a", "place", x=12, y=1, z=0, type="stone")
        w.act("b", "place", x=10, y=2, z=0, type="stone")
        self.assertEqual(w.individual_scores(), {"a": 2, "b": 1, "c": 0})
        w.act("c", "remove", x=11, y=2, z=0)  # removing a's block lowers a's score
        w.act("c", "place", x=11, y=2, z=0, type="stone")  # c gets the credit now
        self.assertEqual(w.individual_scores(), {"a": 1, "b": 1, "c": 1})
        self.assertEqual(w.config.deploy_top_k, 1)
        w.finish("aborted")
        end = records_of(w, "episode_end")[0]
        self.assertEqual([x["rank"] for x in end["ranking"]], [1, 1, 1])
        self.assertEqual(end["deployed"], ["a", "b", "c"])
        replay(w.recorder.records)

    def test_deploy_defaults_and_incentive_none(self):
        for n, k in ((1, 1), (2, 1), (3, 1), (4, 2), (5, 2), (16, 8)):
            self.assertEqual(Config(agents=[f"x{i}" for i in range(n)]).deploy_top_k, k)
        w = make_world(incentive="none")
        w.act("a", "buy", item="stone", n=1)
        w.act("a", "place", x=11, y=2, z=0, type="stone")
        w.finish("aborted")
        end = records_of(w, "episode_end")[0]
        self.assertEqual(end["ranking"], [{"agent": "a", "score": 1, "rank": 1}, {"agent": "b", "score": 0, "rank": 2}])
        self.assertEqual(end["deployed"], [])
        replay(w.recorder.records)
        w = make_world(deploy_top_k=2)
        w.finish("aborted")
        self.assertEqual(records_of(w, "episode_end")[0]["deployed"], ["a", "b"])

    def test_replay_checks_ranking_and_wallet(self):
        w = make_world()
        w.act("a", "buy", item="stone", n=1)
        w.act("a", "place", x=11, y=2, z=0, type="stone")
        w.finish("aborted")
        good = json.loads(json.dumps(w.recorder.records))
        replay(good)
        for tamper in ("deployed", "ranking", "wallet_delta", "snapshot_wallet", "scores", "analysis"):
            recs = json.loads(json.dumps(good))
            end = recs[-1]
            if tamper == "deployed":
                end["deployed"] = ["a", "b"]
            elif tamper == "ranking":
                end["ranking"][1]["rank"] = 1
            elif tamper == "wallet_delta":
                next(r for r in recs if r["type"] == "event" and r["tool"] == "buy")["diff"]["wallet"] -= 1
            elif tamper == "snapshot_wallet":
                recs[-2]["wallet"] += 5
            elif tamper == "scores":
                end["individual_scores"]["b"] = 1
            else:
                end["analysis"]["leftovers"] = 0
            with self.assertRaises(AssertionError, msg=tamper):
                replay(recs)


class TestEventShapes(unittest.TestCase):
    def test_diffs(self):
        w = make_world(positions=[NEAR_SHOP, [11, 4, 0]])
        w.act("a", "buy", item="stone", n=2)
        w.act("a", "place", x=11, y=2, z=0, type="stone")
        w.act("a", "give", to="b", type="stone", n=1)
        w.act("a", "remove", x=11, y=2, z=0)
        w.act("a", "say", text="hi all")
        w.act("a", "whisper", to="b", text="psst")
        w.act("a", "note", text="remember")
        w.act("a", "place", x=0, y=0, z=5, type="stone")  # fails
        ev = events(w)
        for e in ev:
            self.assertEqual(list(e["diff"]), ["grid", "wallet", "inventory", "agents", "votes"])
            for k in ("event_id", "t", "agent", "tool", "args", "cost", "ok", "message", "chat"):
                self.assertIn(k, e)
        buy, place, give, remove, say, whisper, note, bad = ev
        self.assertEqual(buy["diff"], {"grid": [], "wallet": -2 * CATALOG["stone"].price,
                                       "inventory": {"a": {"stone": 2}}, "votes": {}, "agents": {}})
        self.assertEqual(place["diff"]["grid"], [{"pos": [11, 2, 0], "before": None,
                                                 "after": {"type": "stone", "placed_by": "a", "placed_at": 0.0}}])
        self.assertEqual(place["diff"]["wallet"], 0)
        self.assertEqual(give["diff"]["inventory"], {"a": {"stone": -1}, "b": {"stone": 1}})
        self.assertIsNone(remove["diff"]["grid"][0]["after"])
        self.assertEqual(remove["diff"]["inventory"], {"a": {"stone": 1}})
        self.assertEqual(say["chat"], {"channel": "public", "from": "a", "to": None, "text": "hi all"})
        self.assertEqual(whisper["chat"], {"channel": "private", "from": "a", "to": "b", "text": "psst"})
        self.assertEqual((note["args"], note["cost"], note["chat"]), ({"text": "remember"}, "free", None))
        self.assertEqual(bad["diff"], {"grid": [], "wallet": 0, "inventory": {}, "agents": {}, "votes": {}})
        snap = w.snapshot()
        self.assertEqual(snap["wallet"], 1000 - 2 * CATALOG["stone"].price)
        self.assertNotIn("chest", snap)
        replay(w.recorder.records)


class TestThreads(unittest.TestCase):
    def test_many_threads(self):
        agents = [f"agent_{i}" for i in range(16)]
        w = World(Config(agents=agents, starting_money=10_000, max_actions_per_agent=10_000), clock=WallClock())
        barrier = threading.Barrier(len(agents))

        def hammer(agent: str) -> None:
            barrier.wait()
            for i in range(40):
                w.act(agent, "buy", item="dirt", n=1)
                w.act(agent, "say", text=f"{agent} {i}")
                w.act(agent, "give", to=agents[(agents.index(agent) + 1) % len(agents)], type="dirt", n=1)
                w.act(agent, "observe")

        threads = [threading.Thread(target=hammer, args=(a,)) for a in agents]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        w.finish("aborted")
        evs = events(w)
        self.assertEqual(len(evs), 16 * 40 * 4)
        self.assertEqual([e["event_id"] for e in evs], list(range(len(evs))))
        replay(w.recorder.records)

    def test_run_live(self):
        cfg = Config(agents=["a", "b", "c"], walk_speed=400, max_seconds=120)
        w = World(cfg, clock=WallClock())
        run_live(w, ScriptedBuilder(stall_s=1.0, inspect_s=0.3), think_s=0.001)
        self.assertEqual(w.end_reason, "all_done")
        self.assertTrue(w.analysis()["enclosed"])
        replay(w.recorder.records)

    def test_run_live_time_limit(self):
        w = World(Config(agents=["a", "b"], max_seconds=0.2), clock=WallClock())
        run_live(w, lambda world, agent: world.act(agent, "observe").ok or True, think_s=0.01)
        self.assertEqual(w.end_reason, "time_limit")
        replay(w.recorder.records)


class TestScripted(unittest.TestCase):
    def run_scripted(self, n: int = 3, seed: int = 0, **kw) -> World:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ep.jsonl"
            rec = Recorder(path)
            cfg = Config(agents=[f"agent_{i}" for i in range(1, n + 1)], seed=seed, max_seconds=900, **kw)
            w = run_simulated(World(cfg, rec, clock=SimClock()))
            rec.close()
            records = load_log(path)
        self.assertEqual(records, json.loads(json.dumps(w.recorder.records)))
        replay(records)
        self.assertFalse(any(e["tool"] == "jump" for e in events(w)))  # walk/buy/place only
        return w

    def check_house(self, w: World) -> None:
        d = w.analysis()
        self.assertEqual(w.end_reason, "all_done")
        self.assertTrue(d["enclosed"])
        self.assertEqual((d["interior_cells"], d["doors"], d["windows"], d["leftovers"]), (45, 2, 4, 0))
        self.assertEqual({tuple(b[:3]): b[3] for b in w.state()["blocks"]}, BLUEPRINTS["house"].cells)
        self.assertLessEqual(w.config.starting_money - w.wallet, w.config.starting_money)
        self.assertGreaterEqual(w.config.starting_money - w.wallet, BLUEPRINTS["house"].cost())
        end = records_of(w, "episode_end")[0]
        self.assertEqual(sum(end["individual_scores"].values()), len(BLUEPRINTS["house"].cells))
        self.assertTrue(end["deployed"])

    def test_open_house_three_agents(self):
        for seed in (0, 1):
            self.check_house(self.run_scripted(3, seed))

    def test_open_house_five_agents(self):
        w = self.run_scripted(5)
        self.check_house(w)
        self.assertEqual(w.analysis()["target_interior"], 20)
        self.assertEqual(len(records_of(w, "episode_end")[0]["deployed"]) >= 2, True)

    def test_not_enough_money(self):
        w = self.run_scripted(3, starting_money=100)
        self.assertEqual(w.end_reason, "all_done")
        self.assertFalse(w.analysis()["enclosed"])
        self.assertLess(w.wallet, min(i.price for i in CATALOG.values() if i.id in BLUEPRINTS["house"].counts()))
        self.assertTrue(any("can't afford" in (e["chat"] or {}).get("text", "") for e in events(w)))

    def test_replay_detects_tampering(self):
        w = run_simulated(make_world())
        records = json.loads(json.dumps(w.recorder.records))
        replay(records)
        ev = next(r for r in records if r["type"] == "event" and r["diff"]["agents"])
        mv = next(iter(ev["diff"]["agents"].values()))
        mv["to"] = [0, 0, 0]
        with self.assertRaises(AssertionError):
            replay(records)

    def test_sim_determinism(self):
        def run(seed):
            cfg = Config(agents=["a", "b", "c"], seed=seed, max_seconds=120)
            return run_simulated(World(cfg, clock=SimClock())).recorder.records
        self.assertEqual(run(3), run(3))
        self.assertNotEqual(run(3), run(4))


class TestReplayV3(unittest.TestCase):
    def test_v3_log(self):
        path = Path(__file__).resolve().parent / "data" / "v3_open_house_head.jsonl"
        records = load_log(path)
        self.assertEqual(records[0]["log_version"], 3)
        state = replay(records)
        self.assertEqual(state["version"], 3)
        self.assertGreaterEqual(state["snapshots_checked"], 3)
        self.assertIn("chest", state)
        bad = json.loads(json.dumps(records))
        snap = next(r for r in bad if r["type"] == "snapshot" and r["blocks"])
        snap["score_details"]["shell_cells"] += 1
        with self.assertRaises(AssertionError):
            replay(bad)
        self.assertEqual(replay(records[:1])["version"], 3)
        with self.assertRaises(AssertionError):
            replay([dict(records[0], log_version=2)])


class TestVotes(unittest.TestCase):
    def setUp(self):
        # a by the shop, b and c close enough to reach (11, 2, 0) and each other
        self.w = make_world(positions=[NEAR_SHOP, [11, 4, 0], [10, 3, 0]], agents=["a", "b", "c"])
        w = self.w
        w.act("a", "buy", item="stone", n=6)
        w.act("a", "give", to="b", type="stone", n=3)
        w.act("a", "give", to="c", type="stone", n=3)

    def votes(self, i=-1):
        return events(self.w)[i]["diff"]["votes"]

    def test_add_cancel_and_noops(self):
        w = self.w
        self.assertTrue(w.act("a", "declare_done").ok)
        self.assertEqual(self.votes(), {"before": [], "after": ["a"]})
        self.assertTrue(w.act("a", "declare_done").ok)  # already voted: no change
        self.assertEqual(self.votes(), {})
        self.assertTrue(w.act("b", "declare_done").ok)
        self.assertEqual(self.votes(), {"before": ["a"], "after": ["a", "b"]})
        self.assertTrue(w.act("a", "cancel_done").ok)
        self.assertEqual(self.votes(), {"before": ["a", "b"], "after": ["b"]})
        self.assertEqual(w.done_votes, ["b"])
        self.assertTrue(w.act("a", "cancel_done").ok)  # nothing to withdraw: no change
        self.assertEqual(self.votes(), {})
        self.assertEqual(events(w)[-1]["cost"], "action")
        self.assertFalse(w.done)
        replay(w.recorder.records)

    def test_all_done_needs_every_vote_at_once(self):
        w = self.w
        w.act("a", "declare_done")
        w.act("b", "declare_done")
        w.act("a", "cancel_done")
        w.act("c", "declare_done")  # a, b and c have all voted, but never at the same time
        self.assertFalse(w.done)
        w.act("a", "declare_done")
        self.assertEqual(w.end_reason, "all_done")
        self.assertEqual(records_of(w, "snapshot")[-1]["done_votes"], ["b", "c", "a"])
        replay(w.recorder.records)

    def test_grid_changes_clear_votes(self):
        w = self.w
        for agent, tool, args in (("b", "place", dict(x=11, y=2, z=0, type="stone")),
                                  ("c", "remove", dict(x=11, y=2, z=0))):
            w.act("a", "declare_done")
            w.act("b", "declare_done")
            r = w.act(agent, tool, **args)
            self.assertTrue(r.ok, r.message)
            self.assertEqual(self.votes(), {"before": ["a", "b"], "after": []})
            self.assertEqual(w.done_votes, [])
        # failed grid actions and non-grid actions leave votes alone
        w.act("a", "declare_done")
        self.assertFalse(w.act("b", "place", x=0, y=0, z=0, type="stone").ok)  # out of reach
        w.act("b", "move_to", x=10, y=4, z=0)
        w.act("c", "give", to="a", type="stone", n=1)
        w.act("c", "say", text="hi")
        self.assertEqual(w.done_votes, ["a"])
        replay(w.recorder.records)

    def test_jump_place_clears_votes(self):
        w = self.w
        w.act("a", "declare_done")
        w.act("c", "declare_done")
        w.act("b", "jump")
        advance(w, 0.1)
        r = w.act("b", "place", x=11, y=4, z=0, type="stone")
        self.assertTrue(r.ok, r.message)
        self.assertEqual(w.positions["b"], (11, 4, 1))
        self.assertEqual(self.votes(), {"before": ["a", "c"], "after": []})
        replay(w.recorder.records)

    def test_vote_diff_on_every_event(self):
        w = run_simulated(make_world())
        for e in events(w):
            self.assertIn("votes", e["diff"])
            v = e["diff"]["votes"]
            self.assertTrue(v == {} or (set(v) == {"before", "after"} and v["before"] != v["after"]
                                        and v["after"] == sorted(v["after"])), v)
            if not e["ok"]:
                self.assertEqual(v, {})

    def test_system_notices_in_view(self):
        w = self.w
        w.act("a", "say", text="hello")
        w.act("a", "declare_done")
        w.act("b", "declare_done")
        mark = w.view("c")["last_event_id"]
        w.act("b", "cancel_done")
        w.act("b", "declare_done")
        w.act("c", "place", x=11, y=2, z=0, type="stone")
        w.act("c", "place", x=11, y=1, z=0, type="stone")  # no votes left: no notice
        msgs = w.view("c")["messages"]
        self.assertEqual([(m["channel"], m["text"]) for m in msgs], [
            ("public", "hello"),
            ("system", "a voted that the house is finished (1 of 3 votes)"),
            ("system", "b voted that the house is finished (2 of 3 votes)"),
            ("system", "b withdrew its finished vote (1 of 3 votes)"),
            ("system", "b voted that the house is finished (2 of 3 votes)"),
            ("system", "c changed the build, so all finished votes were cleared"),
        ])
        ev = {e["event_id"]: e for e in events(w)}
        for m in msgs:
            self.assertEqual(m["t"], ev[m["event_id"]]["t"])
            if m["channel"] == "system":
                self.assertIsNone(m["from"])
                self.assertIsNone(m["to"])
        later = w.view("a", since_event_id=mark)["messages"]
        self.assertEqual([m["text"] for m in later], [m["text"] for m in msgs[3:]])
        self.assertEqual(w.view("a", since_event_id=w.view("a")["last_event_id"])["messages"], [])

    def test_replay_detects_tampered_votes(self):
        w = self.w
        w.act("a", "declare_done")
        w.act("b", "place", x=11, y=2, z=0, type="stone")
        w.finish("aborted")
        good = json.loads(json.dumps(w.recorder.records))
        replay(good)
        for tamper in ("drop_clear", "fake_vote", "missing_key"):
            records = json.loads(json.dumps(good))
            evs = [r for r in records if r["type"] == "event"]
            self.assertEqual([e["tool"] for e in evs[-2:]], ["declare_done", "place"])
            if tamper == "drop_clear":
                evs[-1]["diff"]["votes"] = {}
            elif tamper == "fake_vote":
                evs[-1]["diff"]["votes"] = {"before": ["a"], "after": ["a", "b"]}
            else:
                del evs[0]["diff"]["votes"]
            with self.assertRaises(AssertionError, msg=tamper):
                replay(records)

    def test_legacy_logs_without_vote_diff(self):
        # early v3 logs: four diff keys; every successful declare_done adds a vote, nothing removes
        records = load_log(Path(__file__).resolve().parent / "data" / "v3_open_house_head.jsonl")
        records = [r for r in records if r["type"] in ("header", "event")]
        for r in records[1:]:
            del r["diff"]["votes"]
        last = records[-1]
        records.append(dict(last, event_id=last["event_id"] + 1, tool="declare_done", args={}, ok=True,
                            message="declared done", chat=None,
                            diff={"grid": [], "chest": {}, "inventory": {}, "agents": {}}))
        self.assertEqual(replay(records)["done_votes"], [last["agent"]])
        # a v4 log always has diff.votes
        w = self.w
        w.act("a", "declare_done")
        recs = json.loads(json.dumps(w.recorder.records))
        for r in recs:
            if r["type"] == "event":
                del r["diff"]["votes"]
        with self.assertRaises(AssertionError):
            replay(recs)

    def test_scripted_bot_revotes_after_clear(self):
        w = make_world(positions=[NEAR_SHOP, [11, 4, 0]])
        bot = ScriptedBuilder()
        mem = bot.memory("a")
        w.act("a", "declare_done")
        mem.declared = True
        self.assertTrue(bot.step(w, "a"))  # still voted: waits, no call
        self.assertEqual(len(events(w)), 1)
        w.inventories["b"] = {"stone": 1}
        w.act("b", "place", x=11, y=2, z=0, type="stone")  # clears a's vote
        bot.step(w, "a")
        self.assertFalse(mem.declared)  # will vote again once it believes it is finished


class TestHooks(unittest.TestCase):
    def test_record_order_and_t(self):
        w = make_world()
        advance(w, 1.5)
        w.act("a", "observe")
        w.record({"type": "model_call", "agent": "a", "call_id": "c1"})
        w.record({"type": "model_call", "agent": "b", "t": 0.25})
        recs = w.recorder.records
        self.assertEqual(recs[-2], {"type": "model_call", "agent": "a", "call_id": "c1", "t": 1.5})
        self.assertEqual(recs[-1]["t"], 0.25)  # given t is kept
        for bad in ({"type": "event"}, {"type": "header"}, {"no": "type"}, "x"):
            with self.assertRaises(ValueError):
                w.record(bad)
        w.finish("aborted")
        n = len(recs)
        w.record({"type": "model_call"})  # after the end: ignored
        self.assertEqual(len(recs), n)
        replay(recs)

    def test_record_under_threads(self):
        agents = [f"agent_{i}" for i in range(8)]
        w = World(Config(agents=agents, max_actions_per_agent=10_000), clock=WallClock())
        barrier = threading.Barrier(len(agents))

        def run(agent: str) -> None:
            barrier.wait()
            for i in range(50):
                w.act(agent, "observe")
                w.record({"type": "model_call", "agent": agent, "call_id": f"{agent}-{i}"})

        threads = [threading.Thread(target=run, args=(a,)) for a in agents]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        w.finish("aborted")
        recs = w.recorder.records
        calls = [r for r in recs if r["type"] == "model_call"]
        self.assertEqual(len(calls), 8 * 50)
        ts = [r["t"] for r in recs[1:]]
        self.assertEqual(ts, sorted(ts))  # one file order, time never goes back
        for a in agents:  # each agent's own calls come after its own observe, in order
            mine = [r for r in recs if r.get("agent") == a]
            self.assertEqual([r["type"] for r in mine], ["event", "model_call"] * 50)
            self.assertEqual([r["call_id"] for r in mine if r["type"] == "model_call"],
                             [f"{a}-{i}" for i in range(50)])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ep.jsonl"
            rec = Recorder(path)
            w2 = World(Config(agents=["a"]), rec, clock=SimClock())
            w2.record({"type": "model_call", "agent": "a"})
            w2.act("a", "observe")
            w2.finish("budget")
            rec.close()
            self.assertEqual([r["type"] for r in load_log(path)],
                             ["header", "model_call", "snapshot", "event", "snapshot", "episode_end"])
            replay(load_log(path))

    def test_models_and_run_meta(self):
        cfg = Config(agents=["a", "b"], models={"a": "x/model-1", "b": "y/model-2"},
                     run_meta={"run_id": "r1", "prompt_version": 3})
        h = World(cfg, clock=SimClock()).recorder.records[0]
        self.assertEqual(h["config"]["models"], {"a": "x/model-1", "b": "y/model-2"})
        self.assertEqual(h["config"]["run_meta"], {"run_id": "r1", "prompt_version": 3})
        h = World(Config(), clock=SimClock()).recorder.records[0]
        self.assertIsNone(h["config"]["models"])
        self.assertIsNone(h["config"]["run_meta"])
        Config(agents=["a", "b"], models={"a": "m"})  # a subset is fine
        for bad in ({"models": {"ghost": "m"}}, {"models": {"a": ""}}, {"models": ["m"]},
                    {"run_meta": "x"}, {"run_meta": {"f": object()}}):
            with self.assertRaises(ValueError, msg=str(bad)):
                Config(agents=["a", "b"], **bad)

    def test_finish_reasons(self):
        for reason in ("budget", "error", "aborted"):
            w = make_world()
            w.finish(reason)
            self.assertEqual(records_of(w, "episode_end")[0]["reason"], reason)
            replay(w.recorder.records)
        with self.assertRaises(ValueError):
            make_world().finish("bored")

    def test_wait_for_change_sim_clock_returns_now(self):
        w = make_world()
        w.act("a", "observe")
        self.assertEqual(w.wait_for_change("b", 0, 100.0), 0)
        self.assertEqual(w.wait_for_change("b", -1, 100.0), 0)
        with self.assertRaises(ValueError):
            w.wait_for_change("ghost", -1, 0.1)

    def test_wait_for_change_wakes_and_times_out(self):
        import time
        w = World(Config(agents=["a", "b"]), clock=WallClock())
        w.act("a", "observe")
        t0 = time.monotonic()
        self.assertEqual(w.wait_for_change("b", 0, 0.2), 0)  # nothing new: times out
        self.assertGreaterEqual(time.monotonic() - t0, 0.19)
        self.assertEqual(w.wait_for_change("b", -1, 5.0), 0)  # already newer: immediate
        timer = threading.Timer(0.1, lambda: w.act("a", "say", text="wake up"))
        timer.start()
        t0 = time.monotonic()
        self.assertEqual(w.wait_for_change("b", 0, 5.0), 1)
        self.assertLess(time.monotonic() - t0, 2.0)
        timer.join()
        threading.Timer(0.1, lambda: w.finish("error")).start()
        t0 = time.monotonic()
        self.assertEqual(w.wait_for_change("b", 1, 5.0), 1)  # the end wakes it too
        self.assertLess(time.monotonic() - t0, 2.0)
        self.assertEqual(w.end_reason, "error")

    def test_wait_for_change_stops_at_time_limit(self):
        import time
        w = World(Config(agents=["a"], max_seconds=0.2), clock=WallClock())
        t0 = time.monotonic()
        w.wait_for_change("a", -1, 5.0)
        self.assertLess(time.monotonic() - t0, 2.0)
        self.assertEqual(w.end_reason, "time_limit")


class TestTeamSize(unittest.TestCase):
    def test_defaults_scale_with_agents(self):
        for n in (1, 2, 3, 5, 8, 16):
            agents = [f"agent_{i}" for i in range(1, n + 1)]
            w = World(Config(agents=agents), clock=SimClock())
            self.assertEqual(w.config.target_interior, 4 * n)
            self.assertEqual(w.config.starting_money, 100 * n)
            self.assertEqual(w.wallet, 100 * n)
            self.assertEqual(w.config.deploy_top_k, max(1, n // 2))
            h = w.recorder.records[0]
            self.assertEqual(h["config"]["target_interior"], 4 * n)
            self.assertEqual((h["wallet"], h["shop"]["starting_money"], h["config"]["starting_money"]),
                             (100 * n, 100 * n, 100 * n))
            self.assertEqual(h["config"]["deploy_top_k"], max(1, n // 2))
            pos = list(w.positions.values())
            self.assertEqual(len(set(pos)), n)
            bodies = set()
            for a, p in w.positions.items():
                self.assertTrue(w.standable(p, a), (n, p))
                bodies |= {p, (p[0], p[1], p[2] + 1)}
            self.assertEqual(len(bodies), 2 * n)
            replay(w.recorder.records)

    def test_explicit_values_and_limits(self):
        w = World(Config(agents=["a", "b"], target_interior=12, starting_money=7, deploy_top_k=2),
                  clock=SimClock())
        self.assertEqual((w.config.target_interior, w.wallet, w.config.deploy_top_k), (12, 7, 2))
        self.assertEqual(World(Config(starting_money=0), clock=SimClock()).wallet, 0)
        for bad in ({"agents": [f"x{i}" for i in range(17)]}, {"target_interior": 0}):
            with self.assertRaises(ValueError, msg=str(bad)):
                Config(**bad)


if __name__ == "__main__":
    unittest.main()
