"""Tests for continuous walking (docs/walking.md): a walker's position, body, eyes and reach
follow it cell by cell along its path; destinations are reserved; walks are truncated when the
grid changes ahead and end in a fall when the support under the current cell goes; snapshots
carry `walks`; replay.py replays and verifies all of it. Teleport mode stays as before.
Run: python3 -m unittest tests.test_walking"""

from __future__ import annotations

import copy
import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blockswarm.clock import SimClock, WallClock  # noqa: E402
from blockswarm.pathing import walk_index, walk_times  # noqa: E402
from blockswarm.recorder import Recorder  # noqa: E402
from blockswarm.replay import replay  # noqa: E402
from blockswarm.scripted import run_simulated  # noqa: E402
from blockswarm.world import Config, World  # noqa: E402

SPEED = 4.317
# b walks (5,3,0) -> (5,10,0) straight along y (7 steps; the only shortest path)
TIMES = [round(i / SPEED, 3) for i in range(8)]  # [0.0, 0.232, 0.463, 0.695, 0.927, 1.158, 1.39, 1.621]


def world(positions=None, agents=("a", "b"), **kw) -> World:
    """16 x 16 x 8 grid, shop at (9, 8, 0), simulated clock. Default: a at (8, 7, 0) next to the
    shop (in reach of column x = 5 around y = 5..9), b at (5, 3, 0)."""
    kw.setdefault("width", 16)
    kw.setdefault("depth", 16)
    kw.setdefault("height", 8)
    kw.setdefault("shop_pos", (9, 8, 0))
    kw.setdefault("starting_money", 1000)
    kw["start_positions"] = positions or [[8, 7, 0], [5, 3, 0]]
    return World(Config(agents=list(agents), **kw), clock=SimClock())


def at(w: World, t: float) -> None:
    w.clock.advance_to(t)


def events(w: World) -> list[dict]:
    return [r for r in w.recorder.records if r["type"] == "event"]


class TestConfig(unittest.TestCase):
    def test_default_and_header(self):
        self.assertEqual(Config().walking, "continuous")
        with self.assertRaises(ValueError):
            Config(walking="run")
        h = World(Config(), clock=SimClock()).recorder.records[0]
        self.assertEqual(h["config"]["walking"], "continuous")
        # teleport logs are what the engine wrote before: no walking key (readers default to teleport)
        h = World(Config(walking="teleport"), clock=SimClock()).recorder.records[0]
        self.assertNotIn("walking", h["config"])

    def test_walk_helpers(self):
        path = [(0, 0, 0), (1, 0, 0), (2, 1, 0)]
        times = walk_times(path, 2.0, 2.0)
        self.assertEqual(times, [2.0, 2.5, round(2.0 + (1 + 2 ** 0.5) / 2, 3)])
        self.assertEqual([walk_index(times, t) for t in (1.0, 2.0, 2.499, 2.5, 3.0, 9.0)], [0, 0, 0, 1, 1, 2])


class TestWalkingPosition(unittest.TestCase):
    def test_cell_by_cell(self):
        w = world()
        r = w.act("b", "move_to", x=5, y=10, z=0)
        self.assertTrue(r.ok)
        mv = events(w)[-1]["diff"]["agents"]["b"]  # move_to entry unchanged
        self.assertEqual((mv["from"], mv["to"], mv["depart_t"], mv["arrive_t"]), ([5, 3, 0], [5, 10, 0], 0.0, TIMES[-1]))
        self.assertEqual(w.positions["b"], (5, 3, 0))  # not at the destination yet
        for i, t in enumerate(TIMES):
            if t > 0:
                at(w, round(t - 0.001, 3))
                self.assertEqual(w.positions["b"], (5, 3 + i - 1, 0), t)
            at(w, t)
            self.assertEqual(w.positions["b"], (5, 3 + i, 0), t)
            self.assertEqual(w.current_cell("b"), (5, 3 + i, 0))
        self.assertEqual(w.walks, {})
        self.assertFalse(w.is_walking("b"))
        replay(w.recorder.records)

    def test_view_and_visibility_follow_the_walker(self):
        w = world()
        w.act("b", "move_to", x=5, y=10, z=0)
        at(w, 0.5)
        v = w.view("b")
        self.assertEqual(v["position"], [5, 5, 0])
        self.assertEqual(v["eye"], [5.5, 5.5, 0.5 + 1.12])
        self.assertEqual(v["destination"], [5, 10, 0])
        self.assertEqual(v["walking_until"], TIMES[-1])
        seen = {e["id"]: e for e in w.view("a")["agents_visible"]}
        self.assertEqual(seen["b"]["pos"], [5, 5, 0])
        self.assertTrue(seen["b"]["walking"])
        self.assertIsNone(w.view("a")["destination"])
        at(w, 2.0)
        self.assertIsNone(w.view("b")["destination"])
        self.assertEqual(w.view("b")["position"], [5, 10, 0])
        # teleport views have no destination key
        self.assertNotIn("destination", world(walking="teleport").view("a"))

    def test_paths_and_costs_start_at_the_current_cell(self):
        w = world()
        w.act("b", "move_to", x=5, y=10, z=0)
        at(w, 0.7)
        path, _ = w.find_path("b", (5, 12, 0))
        self.assertEqual(path[0], (5, 6, 0))
        self.assertEqual(w.walk_costs("b")[(5, 6, 0)], 0.0)
        self.assertNotIn("b", w.cut_off_if_solid([(5, 6, 2)]))

    def test_body_moves_with_the_walker(self):
        w = world()
        w.act("a", "buy", item="stone", n=4)
        w.act("b", "move_to", x=5, y=10, z=0)
        at(w, 0.5)  # b at (5, 5, 0)
        r = w.act("a", "place", x=5, y=5, z=0, type="stone")
        self.assertFalse(r.ok)
        self.assertIn("body", r.message)
        self.assertTrue(w.act("a", "place", x=5, y=4, z=0, type="stone").ok)  # left behind: free
        self.assertEqual(events(w)[-1]["diff"]["agents"], {})  # behind it: no truncation
        replay(w.recorder.records)

    def test_snapshot_mid_walk(self):
        w = world(snapshot_interval_s=1)
        w.act("b", "move_to", x=5, y=10, z=0)
        at(w, 1.0)
        w.act("a", "observe")
        snap = [r for r in w.recorder.records if r["type"] == "snapshot"][-1]
        self.assertEqual(snap["positions"], {"a": [8, 7, 0], "b": [5, 7, 0]})
        self.assertEqual(snap["walking_until"], {"b": TIMES[-1]})
        self.assertEqual(snap["walks"], {"b": {"path": [[5, y, 0] for y in range(3, 11)], "depart_t": 0.0,
                                               "arrive_t": TIMES[-1]}})
        at(w, 2.0)
        w.act("a", "observe")
        snap = [r for r in w.recorder.records if r["type"] == "snapshot"][-1]
        self.assertEqual((snap["walks"], snap["positions"]["b"]), ({}, [5, 10, 0]))
        w.finish("aborted")
        self.assertEqual(replay(w.recorder.records)["positions"]["b"], [5, 10, 0])
        # replay recomputes the current cell: a snapshot claiming the destination is caught
        recs = copy.deepcopy(w.recorder.records)
        next(r for r in recs if r["type"] == "snapshot" and r["walks"])["positions"]["b"] = [5, 10, 0]
        with self.assertRaises(AssertionError):
            replay(recs)
        recs = copy.deepcopy(w.recorder.records)
        next(r for r in recs if r["type"] == "snapshot" and r["walks"])["walks"] = {}
        with self.assertRaises(AssertionError):
            replay(recs)

    def test_new_walk_after_arrival(self):
        w = world()
        w.act("b", "move_to", x=5, y=10, z=0)
        at(w, 1.0)
        self.assertTrue(w.act("b", "move_to", x=5, y=5, z=0).message.startswith("walking:"))
        at(w, TIMES[-1])
        self.assertTrue(w.act("b", "move_to", x=5, y=8, z=0).ok)
        self.assertEqual(events(w)[-1]["diff"]["agents"]["b"]["from"], [5, 10, 0])
        at(w, 5.0)
        self.assertEqual(w.positions["b"], (5, 8, 0))
        replay(w.recorder.records)


class TestReservation(unittest.TestCase):
    def test_destination_reserved_not_the_path(self):
        w = world(positions=[[8, 7, 0], [5, 3, 0], [5, 13, 0]], agents=("a", "b", "c"))
        w.act("a", "buy", item="stone", n=4)
        w.act("b", "move_to", x=5, y=10, z=0)
        at(w, 0.1)
        r = w.act("c", "move_to", x=5, y=10, z=0)
        self.assertFalse(r.ok)
        self.assertTrue(r.message.startswith("(5, 10, 0) is reserved: b is walking to (5, 10, 0)"), r.message)
        for z in (0, 1):
            r = w.act("a", "place", x=5, y=10, z=z, type="stone")
            self.assertFalse(r.ok)
            self.assertIn("reserved", r.message)
        self.assertNotIn((5, 10, 0), w.reachable_stand_cells((5, 9, 0), "c"))
        self.assertIn((5, 10, 0), w.reachable_stand_cells((5, 9, 0), "b"))  # its own destination
        # c may walk through b's destination (and through b itself): walkers pass through each other
        self.assertTrue(w.act("c", "move_to", x=5, y=8, z=0).ok)
        self.assertIn([5, 10, 0], events(w)[-1]["diff"]["agents"]["c"]["path"])
        at(w, 3.0)
        self.assertEqual((w.positions["b"], w.positions["c"]), ((5, 10, 0), (5, 8, 0)))
        # after arrival the reservation is gone (b's body is there now)
        self.assertEqual(w.reserved_cells(), {})
        replay(w.recorder.records)


class TestTruncation(unittest.TestCase):
    def test_block_ahead_stops_the_walk(self):
        w = world()
        w.act("a", "buy", item="stone", n=4)
        w.act("b", "move_to", x=5, y=10, z=0)
        at(w, 0.5)  # b at (5, 5, 0), index 2
        r = w.act("a", "place", x=5, y=8, z=0, type="stone")
        self.assertTrue(r.ok)
        stop = events(w)[-1]["diff"]["agents"]["b"]
        self.assertEqual(stop, {"from": [5, 5, 0], "to": [5, 7, 0], "path": [[5, 5, 0], [5, 6, 0], [5, 7, 0]],
                                "depart_t": 0.0, "arrive_t": TIMES[4], "stopped": True, "path_offset": 2})
        self.assertEqual(w.walking_until["b"], TIMES[4])
        self.assertEqual(w.view("b")["destination"], [5, 7, 0])
        at(w, 0.8)
        self.assertEqual(w.positions["b"], (5, 6, 0))
        self.assertTrue(w.act("b", "say", text="hm").ok)
        self.assertTrue(w.act("b", "jump").message.startswith("walking:"))
        at(w, TIMES[4])
        self.assertEqual(w.positions["b"], (5, 7, 0))
        self.assertTrue(w.act("b", "jump").ok)
        w.finish("aborted")
        replay(w.recorder.records)
        # tampering with the truncation is caught
        for change in ({"arrive_t": TIMES[5]}, {"path_offset": 1}, {"depart_t": 0.5},
                       {"to": [5, 8, 0], "path": [[5, 5, 0], [5, 6, 0], [5, 7, 0], [5, 8, 0]]}):
            recs = copy.deepcopy(w.recorder.records)
            ev = next(r for r in recs if r["type"] == "event" and r["tool"] == "place" and r["ok"])
            ev["diff"]["agents"]["b"].update(change)
            with self.assertRaises(AssertionError, msg=str(change)):
                replay(recs)
        recs = copy.deepcopy(w.recorder.records)
        ev = next(r for r in recs if r["type"] == "event" and r["tool"] == "place" and r["ok"])
        ev["diff"]["agents"] = {}  # without the truncation the replayed walk goes on
        with self.assertRaises(AssertionError):
            replay(recs)

    def test_block_right_ahead_stops_now(self):
        w = world()
        w.act("a", "buy", item="stone", n=4)
        w.act("b", "move_to", x=5, y=10, z=0)
        at(w, 0.5)  # b at (5, 5, 0)
        self.assertTrue(w.act("a", "place", x=5, y=6, z=1, type="stone").ok is False)  # unsupported
        self.assertTrue(w.act("a", "place", x=5, y=6, z=0, type="stone").ok)
        stop = events(w)[-1]["diff"]["agents"]["b"]
        self.assertEqual((stop["to"], stop["path"], stop["arrive_t"], stop["path_offset"]),
                         ([5, 5, 0], [[5, 5, 0]], 0.5, 2))
        self.assertEqual(w.walks, {})
        self.assertFalse(w.is_walking("b"))
        self.assertTrue(w.act("b", "jump").ok)  # free to act at once
        replay(w.recorder.records)

    def test_headroom_and_steps_count(self):
        # a block at head height over a future cell makes it unstandable: truncation too
        w = world()
        w.act("a", "buy", item="stone", n=4)
        w.act("a", "place", x=4, y=8, z=0, type="stone")
        w.act("b", "move_to", x=5, y=10, z=0)  # path still straight along x = 5
        self.assertEqual(events(w)[-1]["diff"]["agents"]["b"]["path"], [[5, y, 0] for y in range(3, 11)])
        at(w, 0.3)
        self.assertTrue(w.act("a", "place", x=4, y=8, z=1, type="stone").ok)  # beside the path: nothing
        self.assertEqual(events(w)[-1]["diff"]["agents"], {})
        self.assertTrue(w.act("a", "place", x=5, y=8, z=1, type="stone").ok)  # supported by (4, 8, 1)
        self.assertEqual(events(w)[-1]["diff"]["agents"]["b"]["to"], [5, 7, 0])
        replay(w.recorder.records)

    def test_back_off_from_a_standing_agent(self):
        # c steps onto b's path after b planned it; a block right after c's cell would stop b in
        # c's body, so b stops one cell earlier
        w = world(positions=[[8, 7, 0], [5, 3, 0], [6, 7, 0]], agents=("a", "b", "c"))
        w.act("a", "buy", item="stone", n=4)
        w.act("b", "move_to", x=5, y=10, z=0)
        at(w, 0.1)
        self.assertTrue(w.act("c", "move_to", x=5, y=7, z=0).ok)  # on b's path, not b's destination
        at(w, 0.5)
        self.assertTrue(w.act("a", "place", x=5, y=8, z=0, type="stone").ok)
        stop = events(w)[-1]["diff"]["agents"]["b"]
        self.assertEqual((stop["to"], stop["arrive_t"]), ([5, 6, 0], TIMES[3]))
        at(w, 2.0)
        self.assertEqual((w.positions["b"], w.positions["c"]), ((5, 6, 0), (5, 7, 0)))
        replay(w.recorder.records)

    def test_remove_support_ahead_truncates_and_under_falls(self):
        def setup():
            w = world()
            w.act("a", "buy", item="stone", n=4)
            w.act("a", "place", x=5, y=6, z=0, type="stone")  # a step on b's way
            w.act("b", "move_to", x=5, y=9, z=0)
            path = events(w)[-1]["diff"]["agents"]["b"]["path"]
            self.assertEqual(path, [[5, 3, 0], [5, 4, 0], [5, 5, 0], [5, 6, 1], [5, 7, 0], [5, 8, 0], [5, 9, 0]])
            return w

        w = setup()
        at(w, 0.5)  # b at (5, 5, 0): the cell on the step is ahead
        self.assertTrue(w.act("a", "remove", x=5, y=6, z=0).ok)
        stop = events(w)[-1]["diff"]["agents"]["b"]
        self.assertEqual((stop["to"], stop["arrive_t"], stop["stopped"]), ([5, 5, 0], 0.5, True))
        replay(w.recorder.records)

        w = setup()
        at(w, 0.8)  # b on the step at (5, 6, 1)
        self.assertEqual(w.positions["b"], (5, 6, 1))
        self.assertTrue(w.act("a", "remove", x=5, y=6, z=0).ok)
        fall = events(w)[-1]["diff"]["agents"]["b"]
        self.assertEqual(fall, {"from": [5, 6, 1], "to": [5, 6, 0], "path": [[5, 6, 1], [5, 6, 0]],
                                "depart_t": 0.8, "arrive_t": 0.9, "fall": True})
        self.assertEqual(w.walks, {})
        at(w, 3.0)
        self.assertEqual(w.positions["b"], (5, 6, 0))  # the walk ended with the fall
        replay(w.recorder.records)


class TestCombatAndGive(unittest.TestCase):
    def script(self, walking: str) -> World:
        # a at (7, 8, 0) by the shop at (9, 8, 0); b walks (5, 3, 0) -> (5, 13, 0) past it
        w = world(positions=[[7, 8, 0], [5, 3, 0]], combat=True, walking=walking)
        w.act("a", "buy", item="stone", n=2)
        w.act("b", "move_to", x=5, y=13, z=0)
        at(w, 0.3)
        w.act("a", "attack", target="b")  # b at (5, 4, 0): out of reach
        at(w, 1.2)  # b at (5, 8, 0), right beside a
        w.act("a", "give", to="b", type="stone", n=1)
        w.act("a", "attack", target="b")
        return w

    def test_reach_targets_follow_the_walker(self):
        w = self.script("continuous")
        ok = [(e["tool"], e["ok"]) for e in events(w)[2:]]
        self.assertEqual(ok, [("attack", False), ("give", True), ("attack", True)])
        self.assertIn("out of reach", events(w)[2]["message"])
        self.assertEqual(w.health["b"], 13)
        replay(w.recorder.records)
        w = self.script("teleport")  # b was already at its destination, out of reach
        self.assertEqual([(e["tool"], e["ok"]) for e in events(w)[2:]],
                         [("attack", False), ("give", False), ("attack", False)])
        replay(w.recorder.records)

    def test_killed_mid_walk(self):
        w = world(positions=[[7, 8, 0], [5, 3, 0]], combat=True, sword_damage=20, snapshot_interval_s=1)
        w.act("b", "move_to", x=5, y=12, z=0)
        at(w, 1.2)
        self.assertTrue(w.act("a", "attack", target="b").ok)
        self.assertEqual(events(w)[-1]["diff"]["agents"]["b"], {"dead": True, "at": [5, 8, 0]})
        self.assertEqual((w.walks, w.walking_until["b"]), ({}, 0.0))
        self.assertEqual(w.reserved_cells(), {})
        at(w, 5.0)
        self.assertEqual(w.positions["b"], (5, 8, 0))
        w.act("a", "observe")
        w.finish("aborted")
        out = replay(w.recorder.records)
        self.assertEqual(out["positions"]["b"], [5, 8, 0])


class TestTeleportUnchanged(unittest.TestCase):
    def test_teleport_moves_at_once(self):
        w = world(walking="teleport", snapshot_interval_s=1)
        w.act("a", "buy", item="stone", n=4)
        w.act("b", "move_to", x=5, y=10, z=0)
        self.assertEqual(w.positions["b"], (5, 10, 0))
        self.assertEqual(w.walks, {})
        at(w, 0.5)
        self.assertTrue(w.act("a", "place", x=5, y=8, z=0, type="stone").ok)  # no truncation
        self.assertEqual(events(w)[-1]["diff"]["agents"], {})
        at(w, 1.0)
        w.act("a", "observe")
        snap = [r for r in w.recorder.records if r["type"] == "snapshot"][-1]
        self.assertNotIn("walks", snap)
        replay(w.recorder.records)

    def test_scripted_runs_replay_in_both_modes(self):
        for mode in ("teleport", "continuous"):
            cfg = Config(agents=["agent_1", "agent_2", "agent_3"], seed=1, max_seconds=300, walking=mode)
            w = run_simulated(World(cfg, recorder=Recorder(), clock=SimClock()))
            out = replay(w.recorder.records)
            self.assertGreater(out["snapshots_checked"], 2, mode)


class TestThreads(unittest.TestCase):
    def test_walking_under_threads(self):
        agents = [f"agent_{i}" for i in range(8)]
        w = World(Config(agents=agents, starting_money=10_000, max_actions_per_agent=10_000, walk_speed=40,
                         snapshot_interval_s=0.05), clock=WallClock())
        barrier = threading.Barrier(len(agents))

        def run(agent: str) -> None:
            i = agents.index(agent)
            barrier.wait()
            for k in range(30):
                w.act(agent, "move_to", x=4 + (i + k) % 12, y=3 + (i * 3 + k) % 9, z=0)
                w.act(agent, "buy", item="dirt", n=1)
                w.act(agent, "place", x=4 + (i * 5 + k) % 12, y=4 + (k * 7) % 9, z=0, type="dirt")
                w.act(agent, "remove", x=4 + (i * 5 + k + 3) % 12, y=4 + (k * 7 + 2) % 9, z=0)
                _ = w.positions[agent], w.view(agent)

        threads = [threading.Thread(target=run, args=(a,)) for a in agents]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        w.finish("aborted")
        replay(w.recorder.records)


if __name__ == "__main__":
    unittest.main()
