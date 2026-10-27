"""Tests for LLM observations, system prompt and tool schemas. Run: python3 -m unittest tests.test_observe"""

from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from blockswarm.clock import SimClock  # noqa: E402
from blockswarm.llm import observe  # noqa: E402
from blockswarm.llm.observe import (ObservationMemory, agent_mark, rectangles,  # noqa: E402
                                    render_observation, render_tool_result)
from blockswarm.llm.prompts import PROMPT_VERSION, system_prompt, tool_schemas  # noqa: E402
from blockswarm.world import BLOCK_TYPES, Block, Config, World  # noqa: E402

DEMO = ROOT / "runs" / "demo" / "open_house_exact.jsonl"
BANNED = ["scaffold", "pillar", "stair", "climb", "tower", "ladder", "strategy", "tip:", "hint",
          "you should", "recommend", "monitor", "logged"]


def make_world(n: int = 3, **kw) -> World:
    kw.setdefault("agents", [f"agent_{i}" for i in range(1, n + 1)])
    return World(Config(**kw), clock=SimClock())


def header(w: World) -> dict:
    return w.recorder.records[0]


def tokens(text: str) -> int:
    return len(text) // 4


class ObservationTests(unittest.TestCase):
    def test_first_and_subsequent(self):
        w = make_world(3)
        mem = ObservationMemory("agent_1")
        first = render_observation(w.view("agent_1"), mem)
        self.assertIn("OBSERVATION #1", first)
        self.assertIn("BRIEF:", first)
        self.assertIn("feet at (12, 3, 0)", first)
        self.assertNotIn("CHANGES IN VIEW", first)
        self.assertEqual(mem.last_event_id, -1)

        self.assertTrue(w.act("agent_1", "take", type="stone", n=3).ok)
        w.clock.advance(1.0)
        self.assertTrue(w.act("agent_1", "place", x=11, y=3, z=0, type="stone").ok)
        self.assertTrue(w.act("agent_1", "place", x=11, y=4, z=0, type="stone").ok)
        self.assertTrue(w.act("agent_2", "say", text="hello team").ok)
        second = render_observation(w.view("agent_1", since_event_id=mem.last_event_id), mem)
        self.assertIn("OBSERVATION #2", second)
        self.assertNotIn("BRIEF:", second)
        changes = second.split("CHANGES IN VIEW")[1].split("TEAMMATES")[0]
        self.assertIn("placed by you since then", changes)
        self.assertIn("(11, 3..4, 0) stone x2", changes)
        self.assertEqual(changes.count("stone"), 1)
        self.assertIn("agent_2 (to all): hello team", second)
        self.assertIn("Inventory (1/12): stone 1", second)

        # nothing new: no changes, no repeated message
        third = render_observation(w.view("agent_1", since_event_id=mem.last_event_id), mem)
        self.assertIn("none seen", third)
        self.assertNotIn("hello team", third)
        # even if the loop passes a full view, messages are not repeated
        fourth = render_observation(w.view("agent_1"), mem)
        self.assertNotIn("hello team", fourth)

    def test_removed_block_reported_and_occluded_remembered(self):
        w = make_world(3)
        mem = ObservationMemory("agent_1")
        w.act("agent_1", "take", type="stone", n=2)
        w.act("agent_1", "place", x=11, y=3, z=0, type="stone")
        render_observation(w.view("agent_1"), mem)
        w.clock.advance(1.0)
        self.assertTrue(w.act("agent_2", "remove", x=11, y=3, z=0).ok)
        text = render_observation(w.view("agent_1", since_event_id=mem.last_event_id), mem)
        gone = text.split("blocks no longer there")[1]
        self.assertIn("(11, 3, 0) stone", gone)
        self.assertNotIn("REMEMBERED", text)

        # a block out of view radius stays remembered, not reported as gone
        w.blocks[(2, 14, 0)] = Block("brick", "agent_3", 0.0)
        mem2 = ObservationMemory("agent_1")
        mem2.known[(2, 14, 0)] = {"type": "brick", "placed_by": "agent_3", "placed_at": 0.0, "seen_t": 0.0}
        text2 = render_observation(w.view("agent_1"), mem2)
        self.assertIn("REMEMBERED, NOT IN SIGHT NOW (1", text2)
        self.assertIn("(2, 14, 0) brick", text2)

    def test_private_messages_of_others_hidden(self):
        w = make_world(3)
        w.act("agent_3", "whisper", to="agent_2", text="SECRET-xyz")
        w.act("agent_2", "whisper", to="agent_1", text="for-one")
        w.act("agent_1", "whisper", to="agent_3", text="from-one")
        t1 = render_observation(w.view("agent_1"), ObservationMemory("agent_1"))
        self.assertNotIn("SECRET-xyz", t1)
        self.assertIn("agent_2 -> you (private): for-one", t1)
        self.assertIn("you -> agent_3 (private): from-one", t1)
        t2 = render_observation(w.view("agent_2"), ObservationMemory("agent_2"))
        self.assertIn("agent_3 -> you (private): SECRET-xyz", t2)
        self.assertNotIn("from-one", t2)
        # defensive filter even if a foreign private message leaks into a view
        v = w.view("agent_1")
        v["messages"].append({"event_id": 99, "t": 1.0, "channel": "private", "from": "agent_2",
                              "to": "agent_3", "text": "LEAK"})
        self.assertNotIn("LEAK", render_observation(v, ObservationMemory("agent_1")))

    def test_system_notices_and_votes(self):
        w = make_world(2)
        w.act("agent_1", "declare_done")
        text = render_observation(w.view("agent_2"), ObservationMemory("agent_2"))
        self.assertIn("FINISHED VOTES: 1 of 2 (agent_1)", text)
        if any(m["channel"] == "system" for m in w.messages):
            self.assertIn("SYSTEM:", text)

    def test_chest_counts_only_when_visible(self):
        w = make_world(2, start_positions=[[12, 3, 0], [3, 12, 0]])
        near = render_observation(w.view("agent_1"), ObservationMemory("agent_1"))
        self.assertIn("within reach, contents now: wood 0, stone", near)
        far_mem = ObservationMemory("agent_2")
        far = render_observation(w.view("agent_2"), far_mem)
        self.assertIn("contents not seen yet", far)
        self.assertNotIn("stone 3", far)  # no counts invented
        # agent_1 walks away: last-seen counts with time
        mem = ObservationMemory("agent_1")
        render_observation(w.view("agent_1"), mem)
        w.act("agent_1", "take", type="brick", n=2)
        self.assertTrue(w.act("agent_1", "move_to", x=4, y=4, z=0).ok)
        w.clock.advance(10)
        later = render_observation(w.view("agent_1", since_event_id=mem.last_event_id), mem)
        self.assertIn("not within reach, contents not visible. Last seen at t=0.0", later)

    def test_scores_hidden_by_default(self):
        w = make_world(3)
        self.assertFalse(observe.SHOW_SCORES)
        v = w.view("agent_1")
        self.assertIn("scores", v)
        text = render_observation(v, ObservationMemory("agent_1"))
        self.assertNotIn("SCORES", text)
        self.assertNotIn("team_score", text)
        observe.SHOW_SCORES = True
        try:
            self.assertIn("SCORES: team", render_observation(w.view("agent_1"), ObservationMemory("agent_1")))
        finally:
            observe.SHOW_SCORES = False

    def test_blueprint_mode(self):
        w = make_world(2, task="blueprint", blueprint="hut_small")
        text = render_observation(w.view("agent_1"), ObservationMemory("agent_1"))
        self.assertIn("BLUEPRINT 'hut_small'", text)
        self.assertIn("not seen built", text)
        sp = system_prompt(header(w), "agent_1")
        self.assertIn("Build the blueprint 'hut_small'", sp)

    def test_mid_episode_token_budget(self):
        with DEMO.open(encoding="utf-8") as fh:
            recs = [json.loads(line) for line in fh]
        snaps = [r for r in recs if r["type"] == "snapshot"]
        snap = max(snaps, key=lambda s: len(s["blocks"]) if s["t"] < 300 else -1)
        w = make_world(3)
        w.blocks = {(x, y, z): Block(t, by, at) for x, y, z, t, by, at in snap["blocks"]}
        w.chest = dict(snap["chest"])
        w.inventories = {a: dict(v) for a, v in snap["inventories"].items()}
        w.positions = {a: tuple(p) for a, p in snap["positions"].items()}
        w.actions_used = dict(snap["actions_used"])
        w.clock.advance_to(snap["t"])
        for a in w.agents:
            mem = ObservationMemory(a)
            text = render_observation(w.view(a), mem)
            self.assertLess(tokens(text), 2500, f"{a}: {tokens(text)} tokens\n{text}")
        self.assertGreater(len(snap["blocks"]), 90)
        self.assertLess(tokens(system_prompt(header(w), "agent_1")), 3000)

    def test_rectangles(self):
        cells = {(x, y, 0): "stone" for x in range(4, 11) for y in range(5, 10)}
        cells[(4, 5, 1)] = "brick"
        self.assertEqual(rectangles(cells), [(4, 10, 5, 9, 0, "stone"), (4, 4, 5, 5, 1, "brick")])

    def test_tool_result(self):
        self.assertEqual(render_tool_result("place", {}, True, "placed stone at (1, 2, 0)", None),
                         "place: OK: placed stone at (1, 2, 0)")
        r = render_tool_result("place", {}, False, "walking: you arrive at t=3.000 (now 1.000)", None)
        self.assertIn("not counted", r)


class TeamSizeTests(unittest.TestCase):
    def check_schemas(self, w: World, agent: str) -> list[dict]:
        tools = tool_schemas(header(w), agent)
        names = [t["function"]["name"] for t in tools]
        self.assertEqual(len(names), len(set(names)))
        for t in tools:
            self.assertEqual(t["type"], "function")
            f = t["function"]
            self.assertTrue(f["description"])
            p = f["parameters"]
            self.assertEqual(p["type"], "object")
            for r in p["required"]:
                self.assertIn(r, p["properties"])
            for prop in p["properties"].values():
                self.assertIn(prop["type"], ("integer", "number", "string"))
            if "type" in p["properties"]:
                self.assertEqual(sorted(p["properties"]["type"]["enum"]), sorted(BLOCK_TYPES))
            if "to" in p["properties"]:
                self.assertEqual(p["properties"]["to"]["enum"], [a for a in w.agents if a != agent])
            if "x" in p["properties"]:
                self.assertEqual(p["properties"]["x"]["maximum"], w.config.width - 1)
                self.assertEqual(p["properties"]["z"]["minimum"], 0)
        for need in ("observe", "wait", "cancel_done", "declare_done", "move_to", "jump", "take",
                     "deposit", "place", "remove", "say", "note"):
            self.assertIn(need, names)
        wait = next(t for t in tools if t["function"]["name"] == "wait")["function"]["parameters"]
        self.assertEqual((wait["properties"]["seconds"]["minimum"], wait["properties"]["seconds"]["maximum"]),
                         (1, 30))
        json.dumps(tools)
        return tools

    def check_prompt(self, w: World, agent: str) -> str:
        sp = system_prompt(header(w), agent)
        low = sp.lower()
        for word in BANNED:
            self.assertNotIn(word, low)
        self.assertIn(w.task.header()["brief"], sp)
        self.assertIn("x goes east", sp)
        self.assertIn("y goes north", sp)
        self.assertIn("z goes up", sp)
        self.assertIn(f"You are {agent}", sp)
        return sp

    def test_prompt_identical_except_id(self):
        w = make_world(3)
        a = system_prompt(header(w), "agent_1")
        b = system_prompt(header(w), "agent_2")
        norm = lambda s: re.sub(r"agent_\d+", "ID", s)  # noqa: E731
        self.assertEqual(norm(a), norm(b))
        self.assertNotEqual(a, b)
        self.assertTrue(PROMPT_VERSION)

    def test_one_agent(self):
        w = make_world(1)
        tools = self.check_schemas(w, "agent_1")
        names = [t["function"]["name"] for t in tools]
        self.assertNotIn("give", names)
        self.assertNotIn("whisper", names)
        sp = self.check_prompt(w, "agent_1")
        self.assertIn("the only agent", sp)
        self.assertNotIn("whisper", sp)
        text = render_observation(w.view("agent_1"), ObservationMemory("agent_1"))
        self.assertNotIn("TEAMMATES", text)

    def test_two_agents(self):
        w = make_world(2)
        tools = self.check_schemas(w, "agent_2")
        self.assertIn("give", [t["function"]["name"] for t in tools])
        sp = self.check_prompt(w, "agent_2")
        self.assertIn("one of 2 agents", sp)
        self.assertIn("Your teammates are: agent_1.", sp)
        text = render_observation(w.view("agent_2"), ObservationMemory("agent_2"))
        self.assertIn("TEAMMATES (1)", text)

    def test_eight_agents(self):
        w = make_world(8)
        self.check_schemas(w, "agent_5")
        sp = self.check_prompt(w, "agent_5")
        self.assertIn("one of 8 agents", sp)
        for a in w.agents:
            self.assertIn(a, sp)
        text = render_observation(w.view("agent_5"), ObservationMemory("agent_5"))
        self.assertIn("TEAMMATES (7)", text)
        self.assertLess(tokens(text), 2500)

    def test_target_interior_from_header(self):
        w = make_world(5)
        h = json.loads(json.dumps(header(w)))
        h["config"]["target_interior"] = 20  # resolved value as logged
        self.assertIn("interior cells / 20)", system_prompt(h, "agent_1"))
        for missing in (None, "drop"):  # null or absent -> 4 x N
            h2 = json.loads(json.dumps(h))
            if missing is None:
                h2["config"]["target_interior"] = None
            else:
                del h2["config"]["target_interior"]
            self.assertIn("interior cells / 20)", system_prompt(h2, "agent_1"))
        self.assertNotIn("/ 12)", system_prompt(h, "agent_1"))

    def test_markers_beyond_nine(self):
        w = make_world(12)
        self.assertEqual(agent_mark(w.agents, "agent_9"), "9")
        self.assertEqual(agent_mark(w.agents, "agent_10"), "a")
        self.assertEqual(agent_mark(w.agents, "agent_12"), "c")
        self.check_schemas(w, "agent_1")
        text = render_observation(w.view("agent_1"), ObservationMemory("agent_1"))
        self.assertIn("agent_10 [map a]", text)
        self.assertEqual(len({agent_mark(w.agents, a) for a in w.agents}), 12)


if __name__ == "__main__":
    unittest.main()
