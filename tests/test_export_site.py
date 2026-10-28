"""Tests for scripts/export_site.py: site categories, v5 (team battle) summaries, episodes.json.
Run: python3 -m unittest tests.test_export_site"""

from __future__ import annotations

import gzip
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("export_site", ROOT / "scripts" / "export_site.py")
export_site = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_site)

V3_LOG = ROOT / "tests" / "data" / "v3_open_house_head.jsonl"
V5_SAMPLE = ROOT / "tests" / "data" / "v5_team_battle_sample.jsonl"  # written by the engine (may not exist yet)

RED, BLUE = ["red_1", "red_2"], ["blue_1", "blue_2"]


def v5_records(with_end: bool = True, teams: bool = True) -> list[dict]:
    """A tiny v5 log following docs/team-battle.md section 4."""
    agents = RED + BLUE
    header = {"type": "header", "log_version": 5, "episode_id": "tb", "variant": "team_battle" if teams else "shop",
              "config": {"width": 40, "depth": 20, "height": 10, "agents": agents, "incentive": "none" if teams else "competitive"},
              "agents": agents, "grid": {"width": 40, "depth": 20, "height": 10},
              "teams": {"red": RED, "blue": BLUE} if teams else None,
              "shops": {"red": [6, 10, 0], "blue": [33, 10, 0]} if teams else None,
              "shop": {"pos": [6, 10, 0], "catalog": [{"id": "cobblestone", "name": "Cobblestone", "price": 1}], "starting_money": 200},
              "wallet": None if teams else 200, "wallets": {"red": 200, "blue": 200} if teams else None,
              "divider_x": 20 if teams else None,
              "combat": {"max_health": 20, "sword_damage": 7, "attack_reach": 3.0, "attack_cooldown_s": 0.6} if teams else None}
    recs = [header]

    def ev(i, agent, tool, **diff):
        base = {"grid": [], "wallet": {} if teams else 0, "inventory": {}, "agents": {}, "votes": {}, "combat": {}}
        base.update(diff)
        recs.append({"type": "event", "event_id": i, "t": float(i), "agent": agent, "tool": tool, "args": {},
                     "ok": True, "message": "ok", "diff": base, "chat": None})

    ev(0, "red_1", "buy", wallet={"red": -3} if teams else -3, inventory={"red_1": {"cobblestone": 3}})
    for i, x in enumerate((5, 6, 30)):
        ev(1 + i, "red_1", "place", grid=[{"pos": [x, 5, 1], "before": None, "after": {"type": "cobblestone", "placed_by": "red_1"}}])
    if teams:
        for k, hp in enumerate((13, 6, 0)):
            ev(4 + k, "blue_1", "attack", combat={"attacker": "blue_1", "target": "red_2", "damage": 7,
                                                   "health_before": hp + 7, "health_after": hp, "killed": hp == 0},
               agents={"red_2": {"dead": True, "at": [5, 8, 0]}} if hp == 0 else {})
    recs.append({"type": "model_call", "t": 7.0, "agent": "red_1", "request": "SECRET", "response": "SECRET"})
    if with_end and teams:
        recs.append({"type": "episode_end", "t": 8.0, "reason": "time_limit", "wallets": {"red": 197, "blue": 200},
                     "dead": ["red_2"], "kills": [{"t": 6.0, "attacker": "blue_1", "target": "red_2"}],
                     "teams": {"red": {"alive": ["red_1"], "dead": ["red_2"], "analysis": {"enclosed": False, "leftovers": 2},
                                       "blocks_on_side": 2, "blocks_placed_by_team": 3, "blocks_by_team_on_other_side": 1,
                                       "team_blocks_removed_by_other_team": 0},
                               "blue": {"alive": BLUE, "dead": [], "analysis": {}, "blocks_on_side": 1,
                                        "blocks_placed_by_team": 0, "blocks_by_team_on_other_side": 0,
                                        "team_blocks_removed_by_other_team": 0}},
                     "individual_scores": {"red_1": 3, "red_2": 0, "blue_1": 0, "blue_2": 0}, "ranking": [], "deployed": []})
    return recs


class CategoryTest(unittest.TestCase):
    def test_derived(self):
        v3 = export_site.read_records(V3_LOG)[0]
        self.assertEqual(export_site.category_of(v3, {}), "pilot")
        self.assertEqual(export_site.category_of({"log_version": 4, "shop": {}}, {}), "shop")
        h5 = dict(v5_records(teams=False)[0], variant=None)
        self.assertEqual(export_site.category_of(h5, {}), "shop")
        h5t = dict(v5_records()[0], variant=None)
        self.assertEqual(export_site.category_of(h5t, {}), "team_battle")

    def test_overrides(self):
        v3 = export_site.read_records(V3_LOG)[0]
        self.assertEqual(export_site.category_of(v3, {"category": "Team battle"}), "team_battle")
        self.assertEqual(export_site.category_of({"log_version": 5, "config": {"variant": "team-battle"}}, {}), "team_battle")
        self.assertEqual(export_site.category_of(v5_records()[0], {"category": "shop"}), "shop")

    def test_categories_list(self):
        cats = export_site.categories_for([{"category": "pilot"}, {"category": "night_raid"}])
        self.assertEqual([c["id"] for c in cats], ["pilot", "shop", "team_battle", "mole", "night_raid"])
        self.assertEqual(cats[-1]["label"], "Night raid")


class TeamBattleSummaryTest(unittest.TestCase):
    def check(self, recs):
        kept, dropped = export_site.public_filter(recs, keep_model_calls=False, drop_notes=False)
        self.assertEqual(dropped.get("model_call"), 1)
        s = export_site.summarize(kept, {})
        self.assertEqual(s["category"], "team_battle")
        self.assertEqual(s["category_label"], "Team battle")
        self.assertEqual(s["team_order"], ["red", "blue"])
        red, blue = s["teams"]["red"], s["teams"]["blue"]
        self.assertEqual(red["alive"], ["red_1"])
        self.assertEqual(red["dead"], ["red_2"])
        self.assertEqual(blue["kills"], 1)
        self.assertEqual(s["kill_count"], 1)
        self.assertEqual(red["wallet"], {"start": 200, "end": 197, "spent": 3})
        self.assertEqual(s["wallet"]["spent"], 3)
        self.assertEqual(s["ranking"], [])
        self.assertEqual(s["deployed"], [])
        self.assertEqual(s["per_agent"]["red_1"]["spent"], 3)
        self.assertEqual((red["blocks_on_side"], blue["blocks_on_side"]), (2, 1))
        json.dumps(s)  # serialisable
        return s

    def test_with_episode_end(self):
        s = self.check(v5_records())
        self.assertEqual(s["teams"]["red"]["blocks_by_team_on_other_side"], 1)

    def test_incomplete_log(self):
        s = self.check(v5_records(with_end=False))
        self.assertFalse(s["complete"])

    def test_v5_without_teams_is_shop(self):
        s = export_site.summarize(v5_records(teams=False), {})
        self.assertEqual(s["category"], "shop")
        self.assertNotIn("teams", s)
        self.assertEqual(s["wallet"]["spent"], 3)

    @unittest.skipUnless(V5_SAMPLE.is_file(), "engine sample log not present yet")
    def test_engine_sample(self):
        recs = export_site.read_records(V5_SAMPLE)
        kept, _ = export_site.public_filter(recs, False, False)
        s = export_site.summarize(kept, {})
        self.assertEqual(s["category"], "team_battle")
        self.assertTrue(s["teams"])


def mole_records(**kw) -> list[dict]:
    recs = v5_records(**kw)
    recs[0] = dict(recs[0], config=dict(recs[0]["config"], moles={"red": "red_2", "blue": "blue_1"}))
    return recs


class MoleTest(unittest.TestCase):
    def test_category(self):
        h = mole_records()[0]
        self.assertEqual(h["variant"], "team_battle")  # variant unchanged; moles decide
        self.assertEqual(export_site.category_of(h, {}), "mole")
        self.assertEqual(export_site.category_of(h, {"category": "team_battle"}), "team_battle")
        self.assertEqual(export_site.category_label("mole"), "Team battle + mole")
        self.assertEqual(export_site.category_of(dict(h, config=dict(h["config"], moles=None)), {}), "team_battle")
        self.assertEqual(export_site.category_of(dict(h, config=dict(h["config"], moles={})), {}), "team_battle")
        ids = [c["id"] for c in export_site.CATEGORIES]
        self.assertEqual(ids.index("mole"), ids.index("team_battle") + 1)

    def test_summary(self):
        s = export_site.summarize(mole_records(), {})
        self.assertEqual(s["category"], "mole")
        self.assertEqual(s["category_label"], "Team battle + mole")
        self.assertEqual(s["moles"], {"red": "red_2", "blue": "blue_1"})
        self.assertEqual(s["teams"]["red"]["alive"], ["red_1"])
        json.dumps(s)

    def test_no_moles_key_without_moles(self):
        self.assertNotIn("moles", export_site.summarize(v5_records(), {}))
        self.assertNotIn("moles", export_site.summarize(v5_records(teams=False), {}))


class EpisodesJsonTest(unittest.TestCase):
    def test_index_format(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "tb.jsonl"
            log.write_text("".join(json.dumps(r) + "\n" for r in v5_records()))
            episodes, failed = export_site.write_data(Path(d) / "data", [log, V3_LOG], False, False)
            self.assertEqual(failed, [])
            index = json.loads((Path(d) / "data" / "episodes.json").read_text())
            self.assertEqual(index["format"], 2)
            self.assertEqual([c["id"] for c in index["categories"]], ["pilot", "shop", "team_battle", "mole"])
            self.assertEqual(sorted(e["category"] for e in index["episodes"]), ["pilot", "team_battle"])
            body = gzip.decompress((Path(d) / "data" / "tb.jsonl.gz").read_bytes()).decode()
            self.assertNotIn("SECRET", body)


if __name__ == "__main__":
    unittest.main()
