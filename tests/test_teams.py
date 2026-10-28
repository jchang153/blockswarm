"""Tests for the team battle variant (docs/team-battle.md): teams, per-team wallets and votes,
chat channels, swords and health, death, log v5 and its replay.
Run: python3 -m unittest tests.test_teams"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from blockswarm.catalog import CATALOG  # noqa: E402
from blockswarm.clock import SimClock  # noqa: E402
from blockswarm.replay import load_log, replay  # noqa: E402
from blockswarm.tasks import TEAM_BATTLE_BRIEF, side_team, team_summary  # noqa: E402
from blockswarm.world import DEAD_MSG, SWORD, TOOLS, Block, Config, World  # noqa: E402
from make_v5_fixture import FIXTURE, build_fixture  # noqa: E402

TEAMS = {"red": ["r1", "r2"], "blue": ["b1", "b2"]}
SHOPS = {"red": [3, 4, 0], "blue": [12, 4, 0]}


def team_world(positions=None, **kw) -> World:
    """2 v 2 on a 16 x 8 x 6 platform split at x = 8; shops red (3,4,0), blue (12,4,0).
    Default positions: r1 (2,5,0), r2 (4,5,0), b1 (11,5,0), b2 (13,5,0)."""
    kw.setdefault("width", 16)
    kw.setdefault("depth", 8)
    kw.setdefault("height", 6)
    kw.setdefault("agents", ["r1", "r2", "b1", "b2"])
    kw.setdefault("teams", TEAMS)
    kw.setdefault("shops", SHOPS)
    if positions is not None:
        kw["start_positions"] = positions
    return World(Config(**kw), clock=SimClock())


def events(w: World) -> list[dict]:
    return [r for r in w.recorder.records if r["type"] == "event"]


def at(w: World, t: float) -> None:
    w.clock.advance_to(t)


class TestTeamConfig(unittest.TestCase):
    def test_defaults(self):
        red = [f"red_{i}" for i in range(1, 6)]
        blue = [f"blue_{i}" for i in range(1, 6)]
        c = Config(width=40, depth=20, height=10, agents=red + blue, teams={"red": red, "blue": blue},
                   shops={"red": [6, 10, 0], "blue": [33, 10, 0]})
        self.assertEqual((c.variant, c.divider_x, c.combat, c.incentive, c.starting_money, c.shop_pos),
                         ("team_battle", 20, True, "none", 500, (6, 10, 0)))
        self.assertEqual((c.max_health, c.sword_damage, c.attack_reach, c.attack_cooldown_s), (20, 7, 3.0, 0.6))
        w = World(c, clock=SimClock())
        self.assertEqual(w.wallets, {"red": 500, "blue": 500})
        self.assertIsNone(w.wallet)
        for a in red:
            self.assertLess(w.positions[a][0], 20)
            self.assertLessEqual(max(abs(w.positions[a][0] - 6), abs(w.positions[a][1] - 10)), 2)
        for a in blue:
            self.assertGreaterEqual(w.positions[a][0], 20)
        self.assertEqual(len(set(w.positions.values())), 10)
        v4 = Config()
        self.assertEqual((v4.variant, v4.teams, v4.combat, v4.divider_x, v4.shops), ("shop", None, False, None, None))

    def test_validation(self):
        base = dict(agents=["r1", "r2", "b1", "b2"], teams=TEAMS, shops=SHOPS, width=16, depth=8, height=6)
        for bad in ({"teams": {"red": ["r1", "r2", "b1", "b2"]}},
                    {"teams": {"red": ["r1"], "blue": ["b1", "b2"], "green": ["r2"]}},
                    {"teams": {"red": ["r1", "r2"], "blue": ["b1"]}},              # b2 in no team
                    {"teams": {"red": ["b1", "b2"], "blue": ["r1", "r2"]}},        # agents not red then blue
                    {"teams": {"red": [], "blue": ["r1", "r2", "b1", "b2"]}},
                    {"shops": None}, {"shops": {"red": [3, 4, 0]}}, {"shops": {"red": [3, 4, 0], "blue": [3, 4, 0]}},
                    {"shops": {"red": [30, 4, 0], "blue": [12, 4, 0]}},
                    {"divider_x": 0}, {"divider_x": 16}, {"divider_x": 2.5},
                    {"variant": "shop"}, {"variant": "war"}, {"combat": "yes"},
                    {"max_health": 0}, {"sword_damage": 0}, {"attack_reach": 0}, {"attack_cooldown_s": -1}):
            with self.assertRaises((ValueError, TypeError), msg=str(bad)):
                Config(**{**base, **bad})
        for bad in ({"variant": "team_battle"}, {"shops": SHOPS}, {"divider_x": 5}):
            with self.assertRaises(ValueError, msg=str(bad)):
                Config(**bad)
        Config(**base, combat=False)
        Config(combat=True)  # combat without teams is allowed
        # standing on either shop is fine for start positions
        Config(**base, start_positions=[[3, 4, 1], [2, 5, 0], [12, 4, 1], [13, 5, 0]])

    def test_header(self):
        w = team_world()
        h = w.recorder.records[0]
        self.assertEqual(h["log_version"], 5)
        self.assertEqual(h["variant"], "team_battle")
        self.assertEqual(h["teams"], TEAMS)
        self.assertEqual(h["shops"], SHOPS)
        self.assertEqual(h["wallets"], {"red": 200, "blue": 200})
        self.assertIsNone(h["wallet"])
        self.assertEqual(h["divider_x"], 8)
        self.assertEqual(h["combat"], {"max_health": 20, "sword_damage": 7, "attack_reach": 3.0,
                                       "attack_cooldown_s": 0.6})
        self.assertEqual(h["health"], {a: 20 for a in w.agents})
        self.assertEqual(h["equipment"], {a: [SWORD] for a in w.agents})
        self.assertEqual(h["shop"]["pos"], [3, 4, 0])
        self.assertEqual(h["shop"]["starting_money"], 200)
        self.assertEqual(h["config"]["incentive"], "none")
        self.assertEqual(h["task"], {"kind": "open", "brief": TEAM_BATTLE_BRIEF})
        self.assertEqual(team_world(brief="custom").task["brief"], "custom")
        self.assertEqual(w.positions, {"r1": (2, 5, 0), "r2": (4, 5, 0), "b1": (11, 5, 0), "b2": (13, 5, 0)})


class TestWalletsAndShops(unittest.TestCase):
    def test_buy_charges_the_buyers_team_at_any_shop(self):
        w = team_world(positions=[[2, 5, 0], [10, 5, 0], [11, 5, 0], [13, 5, 0]])
        price = CATALOG["stone"].price
        self.assertTrue(w.act("r1", "buy", item="stone", n=2).ok)
        self.assertTrue(w.act("r2", "buy", item="stone", n=3).ok)  # at blue's shop, red's wallet
        self.assertEqual(w.wallets, {"red": 200 - 5 * price, "blue": 200})
        self.assertEqual(w.wallet_of("r2"), 200 - 5 * price)
        self.assertEqual(events(w)[1]["diff"]["wallet"], {"red": -3 * price})
        r = w.act("b1", "buy", item="stone", n=1)
        self.assertTrue(r.ok)
        self.assertEqual(w.wallets["blue"], 200 - price)
        self.assertEqual(w.act("r1", "place", x=0, y=0, z=0, type="stone").ok, False)
        self.assertEqual(events(w)[-1]["diff"]["wallet"], {})
        replay(w.recorder.records)

    def test_not_enough_money_is_per_team(self):
        w = team_world(starting_money=3)
        self.assertFalse(w.act("r1", "buy", item="stone", n=100).ok)
        self.assertIn("wallet has 3", events(w)[-1]["message"])

    def test_shops_are_solid(self):
        w = team_world(positions=[[2, 5, 0], [4, 5, 0], [11, 5, 0], [13, 5, 0]])
        self.assertTrue(w.solid((12, 4, 0)))
        w.act("b1", "buy", item="stone", n=1)
        r = w.act("b1", "place", x=12, y=4, z=0, type="stone")
        self.assertFalse(r.ok)
        self.assertIn("is the shop", r.message)
        self.assertFalse(w.act("b1", "remove", x=12, y=4, z=0).ok)


class TestTeamVotesAndChat(unittest.TestCase):
    def test_votes_are_per_team_and_cleared_by_side(self):
        w = team_world()
        for a in ("r1", "b1"):
            w.act(a, "buy", item="stone", n=2)
        w.act("r1", "declare_done")
        w.act("b1", "declare_done")
        self.assertEqual(w.team_votes, {"red": ["r1"], "blue": ["b1"]})
        self.assertEqual(w.view("r2")["done_votes"], ["r1"])
        self.assertEqual(events(w)[-1]["diff"]["votes"], {"blue": {"before": [], "after": ["b1"]}})
        w.act("b1", "place", x=11, y=6, z=0, type="stone")  # blue side: only blue's votes clear
        self.assertEqual(w.team_votes, {"red": ["r1"], "blue": []})
        self.assertEqual(events(w)[-1]["diff"]["votes"], {"blue": {"before": ["b1"], "after": []}})
        msgs = [m["text"] for m in w.view("b2")["messages"]]
        self.assertIn("b1 changed the build on your team's side, so all your team's finished votes were cleared",
                      msgs)
        self.assertFalse(any("changed the build" in m["text"] for m in w.view("r1")["messages"]))
        self.assertFalse(any("b1 voted" in m["text"] for m in w.view("r1")["messages"]))
        w.act("r2", "declare_done")
        self.assertEqual(w.team_votes["red"], ["r1", "r2"])
        w.act("b2", "declare_done")
        self.assertFalse(w.done)
        w.act("b1", "declare_done")
        self.assertEqual(w.end_reason, "all_done")
        end = replay(w.recorder.records)
        self.assertEqual(end["done_votes"], {"red": ["r1", "r2"], "blue": ["b2", "b1"]})
        snap = [r for r in w.recorder.records if r["type"] == "snapshot"][-1]
        self.assertEqual(snap["done_votes"], {"red": ["r1", "r2"], "blue": ["b2", "b1"]})

    def test_divider_side(self):
        self.assertEqual([side_team(x, 8, ["red", "blue"]) for x in (0, 7, 8, 15)], ["red", "red", "blue", "blue"])
        w = team_world()
        self.assertEqual((w.side_of(7), w.side_of(8), w.team_of("b2")), ("red", "blue", "blue"))

    def test_channels(self):
        w = team_world()
        w.act("r1", "say", text="red only")
        w.act("b1", "say_all", text="everyone")
        w.act("r2", "whisper", to="b2", text="secret")
        ev = events(w)
        self.assertEqual(ev[0]["chat"], {"channel": "team", "from": "r1", "to": None, "text": "red only",
                                         "team": "red"})
        self.assertEqual(ev[0]["message"], "said (team)")
        self.assertEqual(ev[1]["chat"], {"channel": "public", "from": "b1", "to": None, "text": "everyone"})
        self.assertEqual(ev[2]["chat"]["channel"], "private")
        seen = {a: [m["text"] for m in w.view(a)["messages"]] for a in w.agents}
        self.assertEqual(seen, {"r1": ["red only", "everyone"], "r2": ["red only", "everyone", "secret"],
                                "b1": ["everyone"], "b2": ["everyone", "secret"]})

    def test_say_all_and_attack_need_the_variant(self):
        w = World(Config(agents=["a", "b"]), clock=SimClock())
        self.assertIn("say_all", TOOLS)
        self.assertIn("attack", TOOLS)
        self.assertNotIn("say_all", w.tools)
        self.assertNotIn("attack", w.tools)
        r = w.act("a", "say_all", text="hi")
        self.assertFalse(r.ok)
        self.assertIn("unknown tool", r.message)
        self.assertFalse(w.act("a", "attack", target="b").ok)
        self.assertNotIn("health", w.view("a"))
        self.assertNotIn("team", w.view("a"))
        replay(w.recorder.records)


class TestView(unittest.TestCase):
    def test_view_keys(self):
        w = team_world()
        v = w.view("b1")
        self.assertEqual(v["team"], "blue")
        self.assertEqual(v["teams"], TEAMS)
        self.assertEqual(v["wallet"], 200)
        self.assertNotIn("wallets", v)
        self.assertEqual(v["shops"], SHOPS)
        self.assertEqual(v["shop_pos"], [12, 4, 0])
        self.assertEqual(v["divider_x"], 8)
        self.assertEqual(v["shops_in_reach"], {"red": False, "blue": True})
        self.assertTrue(v["shop_in_reach"])
        self.assertEqual(v["side"], {"own": {"x_min": 8, "x_max": 15}, "other": {"x_min": 0, "x_max": 7},
                                     "current": "blue"})
        self.assertEqual((v["health"], v["max_health"], v["equipment"], v["dead"]), (20, 20, [SWORD], []))
        self.assertEqual({o["id"]: o["team"] for o in v["agents_visible"]}, {"r2": "red", "b2": "blue"})
        rv = w.view("r1")
        self.assertEqual(rv["side"]["own"], {"x_min": 0, "x_max": 7})
        self.assertEqual({o["id"]: o["team"] for o in rv["agents_visible"]}, {"r2": "red"})
        json.dumps(v)


class TestCombat(unittest.TestCase):
    def setUp(self):
        # r1 next to b1 across the line; r2 and b2 far away
        self.w = team_world(positions=[[7, 5, 0], [2, 2, 0], [8, 5, 0], [14, 2, 0]])

    def test_hit_cooldown_and_kill(self):
        w = self.w
        at(w, 1.0)
        r = w.act("r1", "attack", target="b1")
        self.assertEqual((r.ok, r.message), (True, "hit b1: health 13/20"))
        self.assertEqual(events(w)[-1]["diff"]["combat"], {"attacker": "r1", "target": "b1", "damage": 7,
                                                           "health_before": 20, "health_after": 13,
                                                           "killed": False})
        self.assertEqual(w.view("b1")["health"], 13)
        self.assertEqual([m["text"] for m in w.view("b1")["messages"]], ["you were hit by r1 (health 13/20)"])
        self.assertEqual(w.view("b2")["messages"], [])
        at(w, 1.5)
        r = w.act("r1", "attack", target="b1")
        self.assertFalse(r.ok)
        self.assertTrue(r.message.startswith("cooldown"))
        self.assertEqual(w.actions_used["r1"], 1)  # not counted
        at(w, 1.6)
        self.assertTrue(w.act("r1", "attack", target="b1").ok)
        at(w, 2.2)
        r = w.act("r1", "attack", target="b1")
        self.assertEqual(r.message, "killed b1")
        self.assertFalse(w.is_alive("b1"))
        self.assertEqual(w.alive_agents(), ["r1", "r2", "b2"])
        self.assertEqual(w.health["b1"], 0)
        self.assertEqual(w.kills, [{"t": 2.2, "attacker": "r1", "target": "b1"}])
        self.assertIn("b1 was killed by r1", [m["text"] for m in w.view("r2")["messages"]])
        self.assertNotIn("b1", [o["id"] for o in w.view("r1")["agents_visible"]])
        self.assertEqual(w.view("r1")["dead"], ["b1"])
        # the dead can't act (nothing logged) and can't be targeted
        n = len(w.recorder.records)
        self.assertEqual(w.act("b1", "say", text="hi"), w.act("b1", "observe"))
        self.assertEqual(w.act("b1", "say", text="hi").message, DEAD_MSG)
        self.assertEqual(len(w.recorder.records), n)
        at(w, 3.0)
        r = w.act("r1", "attack", target="b1")
        self.assertEqual((r.ok, r.message), (False, "b1 is dead"))
        self.assertFalse(w.act("r1", "give", to="b1", type="stone", n=1).ok)
        # the body is gone: its cells are free
        self.assertTrue(w.standable((8, 5, 0), "r1"))
        replay(w.recorder.records)

    def test_death_drops_inventory_and_vote(self):
        w = team_world(positions=[[10, 5, 0], [2, 2, 0], [11, 5, 0], [14, 2, 0]], attack_cooldown_s=0)
        self.assertTrue(w.act("b1", "buy", item="stone", n=3).ok)
        w.act("b1", "declare_done")
        for _ in range(3):
            w.act("r1", "attack", target="b1")
        ev = events(w)[-1]
        self.assertEqual(ev["diff"]["inventory"], {"b1": {"stone": -3}})
        self.assertEqual(ev["diff"]["agents"], {"b1": {"dead": True, "at": [11, 5, 0]}})
        self.assertEqual(ev["diff"]["votes"], {"blue": {"before": ["b1"], "after": []}})
        self.assertEqual(w.inventories["b1"], {})
        self.assertEqual(w.wallets["blue"], 200 - 3 * CATALOG["stone"].price)  # no refund
        # b2 is the only living blue: its vote plus both reds' ends the episode
        w.act("r1", "declare_done")
        w.act("r2", "declare_done")
        self.assertFalse(w.done)
        w.act("b2", "declare_done")
        self.assertEqual(w.end_reason, "all_done")
        end = [r for r in w.recorder.records if r["type"] == "episode_end"][0]
        self.assertEqual(end["dead"], ["b1"])
        self.assertEqual(end["teams"]["blue"]["alive"], ["b2"])
        self.assertEqual(end["teams"]["blue"]["dead"], ["b1"])
        self.assertEqual((end["ranking"], end["deployed"]), ([], []))
        replay(w.recorder.records)

    def test_reach_and_line_of_sight(self):
        w = team_world(positions=[[5, 5, 0], [2, 2, 0], [8, 5, 0], [14, 2, 0]], attack_cooldown_s=0)
        r = w.act("r1", "attack", target="b1")
        self.assertFalse(r.ok)
        self.assertTrue(r.message.startswith("out of reach"), r.message)
        self.assertIn("(reach 3.0)", r.message)
        w = team_world(positions=[[6, 5, 0], [2, 2, 0], [8, 5, 0], [14, 2, 0]], attack_cooldown_s=0)
        self.assertTrue(w.act("r1", "attack", target="b1").ok)  # 2 cells away: head cell within 3.0
        for z in (0, 1, 2):
            w.blocks[(7, 5, z)] = Block("stone", "r2", 0.0)
        at(w, 1.0)
        r = w.act("r1", "attack", target="b1")
        self.assertFalse(r.ok)
        self.assertTrue(r.message.startswith("no line of sight"), r.message)
        r = w.act("r1", "attack", target="r1")
        self.assertEqual(r.message, "cannot target yourself")

    def test_walking_attacker_and_friendly_fire(self):
        w = self.w
        w.act("r1", "move_to", x=6, y=5, z=0)
        r = w.act("r1", "attack", target="b1")
        self.assertTrue(r.message.startswith("walking"))
        at(w, 1.0)
        w2 = team_world(positions=[[7, 5, 0], [6, 5, 0], [8, 5, 0], [14, 2, 0]])
        self.assertTrue(w2.act("r1", "attack", target="r2").ok)  # teammates can be hit
        self.assertEqual(w2.health["r2"], 13)

    def test_all_dead(self):
        w = World(Config(agents=["a", "b"], combat=True, attack_cooldown_s=0, start_positions=[[11, 5, 0], [12, 5, 0]]),
                  clock=SimClock())
        for _ in range(3):
            w.act("a", "attack", target="b")
        self.assertFalse(w.done)  # a is alive
        self.assertEqual(w.alive_agents(), ["a"])
        w.finish("aborted")
        replay(w.recorder.records)
        # every agent dead -> all_dead (only reachable by killing the last one; simulate via _kill)
        w = World(Config(agents=["a", "b", "c"], combat=True, attack_cooldown_s=0,
                         start_positions=[[11, 5, 0], [12, 5, 0], [13, 5, 0]]), clock=SimClock())
        for _ in range(3):
            w.act("a", "attack", target="b")
        for _ in range(3):
            w.act("c", "attack", target="a")
        self.assertEqual(w.alive_agents(), ["c"])
        self.assertFalse(w.done)
        w.dead.append("c")  # the last survivor can't be killed by anyone; force it for the end check
        self.assertTrue(w.check_end())
        self.assertEqual(w.end_reason, "all_dead")


class TestTeamStatsAndReplay(unittest.TestCase):
    def test_team_summary(self):
        blocks = {(1, 1, 0): {"type": "stone", "placed_by": "r1"}, (9, 1, 0): {"type": "stone", "placed_by": "r2"},
                  (10, 1, 0): {"type": "stone", "placed_by": "b1"}}
        s = team_summary(blocks, (16, 8, 6), TEAMS, SHOPS, 8, {"b2"}, {"red": 4, "blue": 1}, {"blue": 2})
        self.assertEqual(s["red"]["blocks_on_side"], 1)
        self.assertEqual(s["blue"]["blocks_on_side"], 2)
        self.assertEqual(s["red"]["blocks_by_team_on_other_side"], 1)
        self.assertEqual(s["blue"]["blocks_by_team_on_other_side"], 0)
        self.assertEqual((s["red"]["blocks_placed_by_team"], s["red"]["team_blocks_removed_by_other_team"]), (4, 0))
        self.assertEqual(s["blue"]["team_blocks_removed_by_other_team"], 2)
        self.assertEqual((s["blue"]["alive"], s["blue"]["dead"]), (["b1"], ["b2"]))
        self.assertEqual(s["red"]["analysis"]["leftovers"], 1)

    def test_fixture_replays_and_is_up_to_date(self):
        records = load_log(FIXTURE)
        out = replay(records)
        self.assertEqual(out["version"], 5)
        self.assertEqual(out["dead"], ["red_2"])
        self.assertEqual(out["end"]["reason"], "all_done")
        self.assertGreaterEqual(out["snapshots_checked"], 2)
        tools = {r["tool"] for r in records if r["type"] == "event"}
        self.assertTrue({"buy", "place", "remove", "say", "say_all", "whisper", "attack", "declare_done"} <= tools)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "fresh.jsonl"
            build_fixture(p)
            self.assertEqual(p.read_text(), FIXTURE.read_text(),
                             "tests/data/v5_team_battle_sample.jsonl is stale: run python3 -m tests.make_v5_fixture")

    def test_replay_detects_tampering(self):
        good = load_log(FIXTURE)
        replay(good)

        def tampered(fn):
            recs = copy.deepcopy(good)
            fn(recs)
            return recs

        def first(recs, pred):
            return next(r for r in recs if r["type"] == "event" and pred(r))

        cases = {
            "damage": lambda rs: first(rs, lambda r: r["tool"] == "attack" and r["ok"])["diff"]["combat"]
            .update(health_after=14),
            "team wallet": lambda rs: first(rs, lambda r: r["tool"] == "buy")["diff"].update(wallet={"blue": -8}),
            "votes side": lambda rs: first(rs, lambda r: r["tool"] == "remove")["diff"].update(votes={}),
            "dead acts": lambda rs: first(rs, lambda r: r["tool"] == "say" and r["t"] > 7.4).update(agent="red_2"),
            "kills": lambda rs: rs[-1].update(kills=[]),
            "teams": lambda rs: rs[-1]["teams"]["red"].update(blocks_placed_by_team=0),
            "health": lambda rs: rs[-1]["health"].update(red_2=6),
        }
        for name, fn in cases.items():
            with self.assertRaises(AssertionError, msg=name):
                replay(tampered(fn))


if __name__ == "__main__":
    unittest.main()
