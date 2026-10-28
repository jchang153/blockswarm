"""Tests for the LLM orchestrator (client, budget, agent loop, runner). No network: every
model call goes to MockClient or to OpenRouterClient with a fake `urlopen`."""

from __future__ import annotations

import io
import json
import socket
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from blockswarm.clock import WallClock
from blockswarm.llm.agent import LLMAgent, LLMConfig
from blockswarm.llm.budget import Budget, UnknownPrice, price_for
from blockswarm.llm.client import BudgetRefused, MockClient, OpenRouterClient, mock_result, parse_tool_calls, tool_call
from blockswarm.llm.runner import WORLD_KEYS, ConfigError, load_dotenv, resolve_config, run_episode
from blockswarm.recorder import Recorder
from blockswarm.world import Config, World

FAKE_KEY = "sk-or-v1-FAKEKEY-0123456789abcdef-never-real"
MODEL = "mock/model"
PRICES = {MODEL: {"input_per_m": 1.0, "output_per_m": 2.0}}


def write_config(d: Path, **over) -> Path:
    cfg = {"episode_id": "t", "n_agents": 3, "model": MODEL, "budget_usd": 1.0, "max_seconds": 2.0,
           "llm": {"max_tokens": 256, "history_steps": 3, "stagger_s": 0.2, "error_pause_s": 0.05,
                   "prices": PRICES},
           "mock": {"latency_s": [0.03, 0.08]}}
    for k, v in over.items():
        if k == "llm":
            cfg["llm"] = {**cfg["llm"], **v}
        else:
            cfg[k] = v
    p = d / "cfg.json"
    p.write_text(json.dumps(cfg))
    return p


def read_log(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "episode.jsonl").read_text().splitlines()]


def make_world(n: int = 2, max_seconds: float = 10.0) -> World:
    cfg = Config(agents=[f"agent_{i}" for i in range(1, n + 1)], max_seconds=max_seconds)
    return World(cfg, Recorder(), clock=WallClock())


def make_agent(world: World, agent: str, client, budget: Budget | None = None, **llm) -> LLMAgent:
    cfg = LLMConfig.from_dict({"max_tokens": 128, "prices": PRICES, "stagger_s": 0, **llm})
    return LLMAgent(world, agent, client, MODEL, budget or Budget(10.0), cfg)


def tool_msgs(messages: list[dict]) -> list[str]:
    return [m["content"] for m in messages if m["role"] == "tool"]


class FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def ok_body(calls=(), cost=0.001) -> bytes:
    return json.dumps({
        "id": "gen-1", "model": MODEL, "provider": "Fake",
        "choices": [{"finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": f"c{i}", "type": "function",
                            "function": {"name": n, "arguments": a}} for i, (n, a) in enumerate(calls)]}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 10, "cost": cost,
                  "prompt_tokens_details": {"cached_tokens": 40}}}).encode()


class TestClient(unittest.TestCase):
    def test_parse_tool_calls_malformed(self):
        tcs = parse_tool_calls([{"id": "a", "function": {"name": "place", "arguments": "{\"x\": 1,"}},
                                {"id": "b", "function": {"name": "jump", "arguments": ""}},
                                {"id": "c", "function": {"name": "say", "arguments": "[1]"}}])
        self.assertIsNone(tcs[0]["arguments"])
        self.assertEqual(tcs[0]["raw_arguments"], "{\"x\": 1,")
        self.assertEqual(tcs[1]["arguments"], {})
        self.assertIsNone(tcs[2]["arguments"])

    def test_success_parse_and_headers(self):
        seen = {}

        def urlopen(req, timeout):
            seen["auth"] = req.get_header("Authorization")
            seen["body"] = json.loads(req.data)
            return FakeResp(ok_body([("jump", "{}"), ("place", '{"x":1,"y":2,"z":0,"type":"stone"}')]))

        c = OpenRouterClient(FAKE_KEY, base_url="http://fake.invalid", urlopen=urlopen, sleep=lambda s: None)
        r = c.chat(MODEL, [{"role": "user", "content": "hi"}], [{"type": "function", "function": {"name": "jump"}}],
                   max_tokens=10, temperature=0.5, provider={"require_parameters": True})
        self.assertIsNone(r.error)
        self.assertEqual([t["name"] for t in r.tool_calls], ["jump", "place"])
        self.assertEqual(r.tool_calls[1]["arguments"]["type"], "stone")
        self.assertEqual(r.usage["cost_usd"], 0.001)
        self.assertEqual(r.usage["cached_tokens"], 40)
        self.assertEqual(seen["auth"], f"Bearer {FAKE_KEY}")
        self.assertEqual(seen["body"]["usage"], {"include": True})
        self.assertEqual(seen["body"]["provider"], {"require_parameters": True})
        self.assertNotIn(FAKE_KEY, json.dumps(r.to_log()) + json.dumps(r.raw) + repr(c))

    def test_retries_then_gives_up_and_no_retry_on_401(self):
        calls = []

        def urlopen(req, timeout):
            calls.append(1)
            raise urllib.error.HTTPError(req.full_url, 503, "busy", {}, io.BytesIO(b"overloaded"))

        c = OpenRouterClient(FAKE_KEY, base_url="http://fake.invalid", max_retries=3, urlopen=urlopen,
                             sleep=lambda s: None)
        r = c.chat(MODEL, [], None, max_tokens=10, temperature=None)
        self.assertEqual(len(calls), 4)
        self.assertIn("503", r.error)
        calls.clear()

        def urlopen401(req, timeout):
            calls.append(1)
            raise urllib.error.HTTPError(req.full_url, 401, "no", {}, io.BytesIO(f"bad key {FAKE_KEY}".encode()))

        c = OpenRouterClient(FAKE_KEY, base_url="http://fake.invalid", urlopen=urlopen401, sleep=lambda s: None)
        r = c.chat(MODEL, [], None, max_tokens=10, temperature=None)
        self.assertEqual(len(calls), 1)
        self.assertNotIn(FAKE_KEY, r.error)
        self.assertIn("[REDACTED]", r.error)

    def test_timeout_retry_reserves_per_attempt(self):
        n = [0]

        def urlopen(req, timeout):
            n[0] += 1
            if n[0] == 1:
                raise urllib.error.URLError(socket.timeout("timed out"))
            return FakeResp(ok_body([("jump", "{}")], cost=0.002))

        b = Budget(1.0)
        c = OpenRouterClient(FAKE_KEY, base_url="http://fake.invalid", urlopen=urlopen, sleep=lambda s: None)
        g = b.guard(MODEL, 0.01, PRICES[MODEL])
        r = c.chat(MODEL, [], None, max_tokens=10, temperature=None, guard=g)
        self.assertIsNone(r.error)
        self.assertEqual(r.attempts, 2)
        # the timed-out attempt keeps its full reservation (maybe billed); the second is actual cost
        self.assertAlmostEqual(b.spent, 0.012)
        self.assertEqual(b.reserved, 0.0)


class TestBudget(unittest.TestCase):
    def test_reserve_commit_refuse_and_ledger(self):
        with tempfile.TemporaryDirectory() as d:
            b = Budget(0.10, Path(d) / "ledger.jsonl")
            r1 = b.reserve(0.06)
            with self.assertRaises(BudgetRefused):
                b.reserve(0.05)
            self.assertTrue(b.exhausted)
            b.commit(r1, 0.01)
            self.assertAlmostEqual(b.spent, 0.01)
            self.assertAlmostEqual(b.remaining, 0.09)
            b.close()
            kinds = [json.loads(line)["kind"] for line in (Path(d) / "ledger.jsonl").read_text().splitlines()]
            self.assertEqual(kinds, ["open", "reserve", "refuse", "commit", "close"])

    def test_overrun_and_uncertain(self):
        b = Budget(1.0)
        r = b.reserve(0.01)
        b.commit(r, 0.02)
        self.assertEqual(b.overruns, 1)
        self.assertFalse(b.exhausted)  # an overrun is logged but doesn't stop the run
        self.assertAlmostEqual(b.spent, 0.02)
        r = b.reserve(0.5)
        b.commit(r, 0.99)               # overrun that reaches the limit does stop it
        self.assertTrue(b.exhausted)
        b2 = Budget(1.0)
        r = b2.reserve(0.3)
        b2.commit(r, None, "maybe_billed")
        self.assertAlmostEqual(b2.spent, 0.3)
        self.assertAlmostEqual(b2.uncertain_usd, 0.3)

    def test_threaded_never_exceeds(self):
        b = Budget(1.0)
        granted = []

        def worker():
            for _ in range(50):
                try:
                    rid = b.reserve(0.013)
                except BudgetRefused:
                    return
                granted.append(rid)
                b.commit(rid, None)

        ths = [threading.Thread(target=worker) for _ in range(8)]
        [t.start() for t in ths]
        [t.join() for t in ths]
        self.assertLessEqual(b.spent, 1.0)
        self.assertEqual(len(granted), int(1.0 / 0.013))

    def test_prices(self):
        with self.assertRaises(UnknownPrice):
            price_for({}, "x/y")
        self.assertTrue(price_for({}, "x/y", allow_unknown=True)["unknown"])


class TestConfig(unittest.TestCase):
    def test_agents_and_models(self):
        base = {"model": MODEL, "budget_usd": 1, "llm": {"prices": PRICES}}
        rc = resolve_config({**base, "n_agents": 5})
        self.assertEqual(rc["agents"], [f"agent_{i}" for i in range(1, 6)])
        self.assertEqual(rc["world_cfg"].target_interior, 20)
        with self.assertRaises(ConfigError):
            resolve_config({**base, "n_agents": 2, "agents": ["a", "b", "c"]})
        rc = resolve_config({**base, "agents": ["a", "b"], "n_agents": 2})
        self.assertEqual(rc["agents"], ["a", "b"])
        rc = resolve_config({"models": [MODEL, MODEL], "n_agents": 2, "budget_usd": 1, "llm": {"prices": PRICES}})
        self.assertEqual(rc["models"], {"agent_1": MODEL, "agent_2": MODEL})
        with self.assertRaises(ConfigError):
            resolve_config({"models": [MODEL], "n_agents": 2, "budget_usd": 1, "llm": {"prices": PRICES}})
        rc = resolve_config({**base, "n_agents": 3}, n_agents=4)
        self.assertEqual(len(rc["agents"]), 4)

    def test_refusals(self):
        with self.assertRaises(ConfigError):  # placeholder model
            resolve_config({"model": "CHOOSE_MODEL", "budget_usd": 1, "llm": {"prices": {}}})
        with self.assertRaises(ConfigError):  # no price
            resolve_config({"model": "a/b", "budget_usd": 1})
        resolve_config({"model": "a/b", "budget_usd": 1}, allow_unknown_price=True)
        with self.assertRaises(ConfigError):  # no budget
            resolve_config({"model": MODEL, "llm": {"prices": PRICES}})
        with self.assertRaises(ConfigError):  # typo
            resolve_config({"model": MODEL, "budget_usd": 1, "llm": {"prices": PRICES}, "max_secnds": 3})

    def test_dotenv_does_not_override(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / ".env"
            p.write_text("# c\nOPENROUTER_API_KEY='abc'\nOTHER=1\n")
            env = {"OTHER": "keep"}
            load_dotenv(p, env)
            self.assertEqual(env, {"OTHER": "keep", "OPENROUTER_API_KEY": "abc"})


class TestAgent(unittest.TestCase):
    def run_agents(self, world, agents, timeout=10):
        ths = [threading.Thread(target=a.run, daemon=True) for a in agents]
        [t.start() for t in ths]
        [t.join(timeout) for t in ths]
        self.assertFalse(any(t.is_alive() for t in ths))

    def test_jump_then_place_back_to_back(self):
        w = make_world(1)
        x, y, z = w.positions["agent_1"]
        client = MockClient({"agent_1": [
            mock_result(tool_call("buy", item="stone", n=1)),
            mock_result(tool_call("jump"), tool_call("place", x=x, y=y, z=z, type="stone")),
        ]}, latency_s=0.5)  # model latency must not matter: both run inside one response
        ag = make_agent(w, "agent_1", client, max_model_calls_per_agent=2)
        self.run_agents(w, [ag])
        ev = [r for r in w.recorder.records if r["type"] == "event"]
        self.assertEqual([(e["tool"], e["ok"]) for e in ev], [("buy", True), ("jump", True), ("place", True)])
        self.assertEqual(w.positions["agent_1"], (x, y, z + 1))

    def test_invalid_tool_calls(self):
        w = make_world(1)
        bad_json = {"id": "bad", "name": "place", "arguments": None, "raw_arguments": "{\"x\": 1,"}
        client = MockClient({"agent_1": [
            mock_result(tool_call("place", x="a", y=1, z=0, type="stone"), bad_json,
                        tool_call("fly", up=True), tool_call("wait", seconds=100)),
            mock_result(tool_call("note", text="ok")),
        ]})
        ag = make_agent(w, "agent_1", client, max_model_calls_per_agent=2)
        self.run_agents(w, [ag])
        ev = [r for r in w.recorder.records if r["type"] == "event"]
        self.assertEqual([(e["tool"], e["ok"]) for e in ev], [("place", False), ("place", False), ("note", True)])
        self.assertIn("_raw_arguments", ev[1]["args"])
        msgs = tool_msgs(client.calls[1]["messages"])
        self.assertEqual(len(msgs), 4)
        self.assertIn("integer", msgs[0])
        self.assertIn("JSON", msgs[1])
        self.assertIn("unknown tool", msgs[2])
        self.assertIn("between 1 and 30", msgs[3])
        # replayed assistant message has valid JSON arguments even for the malformed call
        asst = [m for m in client.calls[1]["messages"] if m["role"] == "assistant"][0]
        for tc in asst["tool_calls"]:
            json.loads(tc["function"]["arguments"])

    def test_wait_returns_on_other_agents_event(self):
        w = make_world(2)
        client = MockClient({"agent_1": [mock_result(tool_call("wait", seconds=30))],
                             "agent_2": [mock_result(tool_call("say", text="hello"))]})
        a1 = make_agent(w, "agent_1", client, max_model_calls_per_agent=1)
        a2 = make_agent(w, "agent_2", client, max_model_calls_per_agent=1)
        a2.start_delay_s = 0.5
        t0 = time.monotonic()
        self.run_agents(w, [a1, a2])
        self.assertLess(time.monotonic() - t0, 5)
        res = [r for r in w.recorder.records if r["type"] == "tool_results" and r["agent"] == "agent_1"][0]
        self.assertIn("sent a message", res["results"][0]["content"])

    def test_wait_ignores_unnoticeable_events(self):
        w = make_world(2)
        client = MockClient({"agent_1": [mock_result(tool_call("wait", seconds=1))],
                             "agent_2": [mock_result(tool_call("place", x=0, y=0, z=5, type="stone")),
                                         mock_result(tool_call("note", text="private"))]})
        a1 = make_agent(w, "agent_1", client, max_model_calls_per_agent=1)
        a2 = make_agent(w, "agent_2", client, max_model_calls_per_agent=2)
        a2.start_delay_s = 0.2
        self.run_agents(w, [a1, a2])
        res = [r for r in w.recorder.records if r["type"] == "tool_results" and r["agent"] == "agent_1"][0]
        self.assertIn("nothing new", res["results"][0]["content"])

    def test_history_window(self):
        w = make_world(1)
        client = MockClient({"agent_1": [mock_result(tool_call("note", text=f"n{i}")) for i in range(8)]})
        ag = make_agent(w, "agent_1", client, max_model_calls_per_agent=8, history_steps=2)
        self.run_agents(w, [ag])
        self.assertEqual(len(client.calls), 8)
        for i, c in enumerate(client.calls):
            users = [m for m in c["messages"] if m["role"] == "user"]
            self.assertEqual(len(users), min(i, 2) + 1)
            self.assertEqual(c["messages"][0]["role"], "system")
        # chunked dropping keeps at most history_steps
        w = make_world(1)
        client = MockClient({"agent_1": [mock_result(tool_call("note", text=f"n{i}")) for i in range(8)]})
        ag = make_agent(w, "agent_1", client, max_model_calls_per_agent=8, history_steps=4, history_chunk=3)
        self.run_agents(w, [ag])
        sizes = [sum(m["role"] == "user" for m in c["messages"]) - 1 for c in client.calls]
        self.assertEqual(sizes, [0, 1, 2, 3, 4, 2, 3, 4])

    def test_empty_replies_nudge_and_implicit_wait(self):
        w = make_world(1)
        client = MockClient({"agent_1": [mock_result(content="thinking..."), mock_result(content="hmm"),
                                         mock_result(tool_call("note", text="x"))]})
        ag = make_agent(w, "agent_1", client, max_model_calls_per_agent=3, empty_steps_before_wait=2,
                        implicit_wait_s=0.2)
        self.run_agents(w, [ag])
        self.assertIn("no tool call", client.calls[1]["messages"][-1]["content"])
        self.assertEqual(ag.stats["implicit_waits"], 1)
        self.assertTrue(any(r["type"] == "implicit_wait" for r in w.recorder.records))


RED = [f"red_{i}" for i in range(1, 6)]
BLUE = [f"blue_{i}" for i in range(1, 6)]
TEAM_KEYS = {"variant": "team_battle", "teams": {"red": RED, "blue": BLUE},
             "shops": {"red": [6, 10, 0], "blue": [33, 10, 0]}, "divider_x": 20, "width": 40, "depth": 20,
             "height": 10}


def world_has_teams() -> bool:
    return "teams" in getattr(Config, "__dataclass_fields__", {})


class TestDeadAgent(unittest.TestCase):
    """A killed agent's loop stops (World.is_alive false), before a call or mid-response."""

    def test_loop_stops_when_killed(self):
        w = make_world(2)
        alive = {"agent_1": True, "agent_2": True}
        w.is_alive = lambda a: alive[a]  # stands in for the engine's team-battle death

        def script(state, messages):
            if state["call_index"] == 0:  # first call: the agent is killed during it
                alive[state["agent"]] = False
            return mock_result(tool_call("note", text="a"), tool_call("note", text="b"))

        client = MockClient(script)
        ag = make_agent(w, "agent_1", client, max_model_calls_per_agent=5)
        th = threading.Thread(target=ag.run, daemon=True)
        th.start()
        th.join(10)
        self.assertFalse(th.is_alive())
        self.assertEqual(ag.stop_reason, "dead")
        self.assertEqual(len(client.calls), 1)
        self.assertFalse([r for r in w.recorder.records if r["type"] == "event"])  # no tool ran
        res = [r for r in w.recorder.records if r["type"] == "tool_results"][0]["results"]
        self.assertTrue(all(not r["ok"] and "killed" in r["content"] for r in res))
        stop = [r for r in w.recorder.records if r["type"] == "agent_stop"]
        self.assertEqual([r["reason"] for r in stop], ["dead"])
        # an agent that is already dead makes no model call at all
        client2 = MockClient(script)
        ag2 = make_agent(w, "agent_1", client2)
        self.assertEqual(ag2.run(), "dead")
        self.assertEqual(client2.calls, [])
        self.assertFalse(w.done)  # other agents go on

    @unittest.skipUnless(world_has_teams(), "world.py has no team battle yet")
    def test_killed_in_real_world(self):
        from blockswarm.clock import SimClock
        cfg = Config(agents=RED + BLUE, start_positions=[[2 + 2 * i, 2, 0] for i in range(5)]
                     + [[2 + 2 * i, 3, 0] for i in range(5)], attack_cooldown_s=0.0, **TEAM_KEYS)
        w = World(cfg, Recorder(), clock=SimClock())
        for _ in range(3):
            self.assertTrue(w.act("blue_1", "attack", target="red_1").ok)
        self.assertFalse(w.is_alive("red_1"))
        client = MockClient([])
        ag = LLMAgent(w, "red_1", client, MODEL, Budget(1.0),
                      LLMConfig.from_dict({"max_tokens": 64, "prices": PRICES, "stagger_s": 0}))
        self.assertEqual(ag.run(), "dead")
        self.assertEqual(client.calls, [])


@unittest.skipUnless(world_has_teams(), "world.py has no team battle yet")
class TestTeamWait(unittest.TestCase):
    def test_wait_wakes_on_own_team_message_and_hits(self):
        cfg = Config(agents=RED + BLUE, start_positions=[[2 + 2 * i, 2, 0] for i in range(5)]
                     + [[2 + 2 * i, 3, 0] for i in range(5)], **TEAM_KEYS)
        w = World(cfg, Recorder(), clock=WallClock())
        ag = make_agent(w, "red_1", MockClient([]))
        start = w.last_event_id()
        self.assertTrue(w.act("blue_1", "say", text="blue team only").ok)
        self.assertIsNone(ag._noticeable(start, w.last_event_id()))
        start = w.last_event_id()
        self.assertTrue(w.act("red_2", "say", text="red team").ok)
        self.assertEqual(ag._noticeable(start, w.last_event_id()), "red_2 sent a message")
        start = w.last_event_id()
        self.assertTrue(w.act("blue_1", "attack", target="red_1").ok)
        self.assertEqual(ag._noticeable(start, w.last_event_id()), "you were hit by blue_1")


class TestTeamConfig(unittest.TestCase):
    @unittest.skipUnless(world_has_teams(), "world.py has no team battle yet")
    def test_llm_by_model_overrides_per_team(self):
        prices = {"m/a": {"input_per_m": 1, "output_per_m": 1}, "m/b": {"input_per_m": 1, "output_per_m": 1}}
        raw = {"budget_usd": 1.0, "team_models": {"red": "m/a", "blue": "m/b"},
               "llm": {"prices": prices, "temperature": None, "history_steps": 4},
               "llm_by_model": {"m/a": {"max_tokens": 4096, "reasoning": {"effort": "low"}},
                                "m/b": {"reasoning": {"enabled": False}}}, **TEAM_KEYS}
        rc = resolve_config(raw)
        red, blue = rc["llm_for"]["m/a"], rc["llm_for"]["m/b"]
        self.assertEqual((red.max_tokens, red.reasoning), (4096, {"effort": "low"}))
        self.assertEqual((blue.max_tokens, blue.reasoning), (2048, {"enabled": False}))
        self.assertEqual((red.history_steps, blue.history_steps), (4, 4))  # shared settings kept
        self.assertIsNone(blue.temperature)
        with self.assertRaises(ConfigError):
            resolve_config({**raw, "llm_by_model": {"m/zzz": {"max_tokens": 10}}})
        for f in sorted((Path(__file__).resolve().parents[1] / "configs").glob("vs*.json")):  # shipped matchups
            rc = resolve_config(json.loads(f.read_text()))
            self.assertEqual(len(set(rc["models"].values())), 2, f.name)

    @unittest.skipUnless(world_has_teams(), "world.py has no team battle yet")
    def test_teams_define_agents_and_team_models(self):
        raw = {"budget_usd": 1.0, "team_models": {"red": "m/a", "blue": "m/b"},
               "llm": {"prices": {"m/a": {"input_per_m": 1, "output_per_m": 1},
                                  "m/b": {"input_per_m": 1, "output_per_m": 1}}}, **TEAM_KEYS}
        rc = resolve_config(raw)
        self.assertEqual(rc["agents"], RED + BLUE)
        self.assertEqual({rc["models"][a] for a in RED}, {"m/a"})
        self.assertEqual({rc["models"][a] for a in BLUE}, {"m/b"})
        self.assertEqual(rc["world_cfg"].teams, {"red": RED, "blue": BLUE})
        self.assertTrue(rc["world_cfg"].brief.startswith("Your team and the other team are each building"))
        for f in ("shops", "divider_x", "variant"):
            self.assertIn(f, rc["world_kwargs"])
        same = {k: v for k, v in raw.items() if k != "team_models"}
        rc = resolve_config({**same, "model": MODEL, "llm": {"prices": PRICES}})
        self.assertEqual(set(rc["models"].values()), {MODEL})

    def test_team_config_refusals(self):
        base = {"budget_usd": 1.0, "llm": {"prices": PRICES}, **TEAM_KEYS}
        with self.assertRaises(ConfigError):
            resolve_config({**base, "model": MODEL, "team_models": {"red": MODEL, "blue": MODEL}})
        with self.assertRaises(ConfigError):
            resolve_config({**base, "team_models": {"red": MODEL}})
        with self.assertRaises(ConfigError):
            resolve_config({**base, "model": MODEL}, n_agents=4)
        with self.assertRaises(ConfigError):
            resolve_config({**base, "model": MODEL, "agents": BLUE + RED})
        with self.assertRaises(ConfigError):
            resolve_config({"budget_usd": 1.0, "model": MODEL, "team_models": {"a": MODEL}})

    def test_shipped_team_configs(self):
        root = Path(__file__).resolve().parents[1] / "configs"
        for name in ("teams-mock", "teams-dsv4flash", "teams-glm53flash", "teams-gpt6luna", "teams-haiku55"):
            with self.subTest(name=name):
                raw = json.loads((root / f"{name}.json").read_text())
                self.assertEqual(raw["episode_id"], name)
                self.assertEqual(raw["variant"], "team_battle")
                self.assertEqual(raw["teams"], {"red": RED, "blue": BLUE})
                self.assertEqual(raw["shops"], {"red": [6, 10, 0], "blue": [33, 10, 0]})
                self.assertEqual((raw["width"], raw["depth"], raw["height"], raw["divider_x"]), (40, 20, 10, 20))
                self.assertEqual(raw["walking"], "continuous")  # docs/walking.md
                if name != "teams-mock":
                    self.assertEqual((raw["max_seconds"], raw["budget_usd"]), (1800, 20.0))
                    pilot = json.loads((root / f"pilot2-{name[6:]}.json").read_text())
                    self.assertEqual(raw["llm"], pilot["llm"])
                    self.assertEqual(raw["model"], pilot["model"])
                if world_has_teams():
                    rc = resolve_config(raw, mock=True)
                    self.assertEqual(rc["agents"], RED + BLUE)
                    if "walking" in WORLD_KEYS:
                        self.assertEqual(rc["world_kwargs"]["walking"], "continuous")


    def test_shipped_mole_configs(self):
        """configs/mole-vsN-*.json = configs/vsN-*.json plus `moles` (docs/mole.md section 4)."""
        root = Path(__file__).resolve().parents[1] / "configs"
        files = sorted(root.glob("mole-vs*.json"))
        self.assertEqual(len(files), 6)

        def boom(*a, **k):
            raise AssertionError("network used in dry run")

        for f in files:
            with self.subTest(name=f.stem):
                raw = json.loads(f.read_text())
                vs = json.loads((root / f"{f.stem[5:]}.json").read_text())
                self.assertEqual(raw["episode_id"], f.stem)
                strip = lambda c: {k: v for k, v in c.items() if k not in ("comment", "episode_id", "moles")}  # noqa: E731
                self.assertEqual(strip(raw), strip(vs))
                self.assertEqual(sorted(raw["moles"]), sorted(raw["teams"]))
                for team, mole in raw["moles"].items():
                    self.assertIn(mole, raw["teams"][team])
                rc = resolve_config(raw)
                self.assertEqual(rc["world_cfg"].moles, raw["moles"])
                with mock.patch.object(urllib.request, "urlopen", boom), \
                        mock.patch.object(socket, "create_connection", boom), \
                        mock.patch.object(socket.socket, "connect", boom):
                    res = run_episode(f, dry_run=True, quiet=True, env={"OPENROUTER_API_KEY": FAKE_KEY})
                self.assertTrue(res["ok"])


class TestRunner(unittest.TestCase):
    def check_episode(self, run_dir: Path, n: int) -> list[dict]:
        recs = read_log(run_dir)
        self.assertEqual(recs[0]["type"], "header")
        self.assertEqual(recs[-1]["type"], "episode_end")
        self.assertEqual(len(recs[0]["agents"]), n)
        ts = [r["t"] for r in recs if "t" in r]
        self.assertEqual(ts, sorted(ts))
        seen_call: dict[str, int] = {}
        for r in recs:
            if r["type"] == "model_call":
                seen_call[r["agent"]] = seen_call.get(r["agent"], 0) + 1
                self.assertEqual(r["request"]["messages"][0]["content_ref"], "agent_prompt")
            elif r["type"] == "event":
                # every event follows (in file order) a model_call of the same agent
                self.assertGreater(seen_call.get(r["agent"], 0), 0)
        self.assertEqual(set(seen_call), set(recs[0]["agents"]))
        ids = [r["call_id"] for r in recs if r["type"] == "model_call" and r["agent"] == "agent_1"]
        self.assertEqual(ids, sorted(ids))
        man = json.loads((run_dir / "manifest.json").read_text())
        self.assertEqual(man["status"], "finished")
        self.assertEqual(man["n_agents"], n)
        self.assertTrue((run_dir / "ledger.jsonl").exists())
        return recs

    def test_mock_episode_n2_and_n5(self):
        for n in (2, 5):
            with self.subTest(n=n), tempfile.TemporaryDirectory() as d:
                cfg = write_config(Path(d), n_agents=n)
                res = run_episode(cfg, run_dir=Path(d) / "run", mock=True, quiet=True)
                self.assertEqual(res["end_reason"], "time_limit")
                recs = self.check_episode(Path(d) / "run", n)
                self.assertTrue(any(r["type"] == "event" and r["ok"] for r in recs))
                self.assertEqual(recs[0]["config"]["target_interior"], 4 * n)
                self.assertEqual(recs[0]["config"]["models"], {f"agent_{i}": MODEL for i in range(1, n + 1)})

    @unittest.skipUnless(world_has_teams(), "world.py has no team battle yet")
    def test_mock_team_battle_episode(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = write_config(Path(d), n_agents=None, max_seconds=6.0, budget_usd=5.0, **TEAM_KEYS)
            raw = json.loads(cfg.read_text())
            del raw["n_agents"]
            cfg.write_text(json.dumps(raw))
            res = run_episode(cfg, run_dir=Path(d) / "run", mock=True, quiet=True)
            recs = self.check_episode(Path(d) / "run", 10)
            self.assertEqual(recs[0]["teams"], {"red": RED, "blue": BLUE})
            prompts = [r for r in recs if r["type"] == "agent_prompt"]
            self.assertEqual(len(prompts), 10)
            self.assertTrue(all("Whichever team builds the best house" in r["system_prompt"] for r in prompts))
            self.assertIn("Whichever team builds the best house", recs[0]["task"]["brief"])
            self.assertTrue(all("attack" in [t["function"]["name"] for t in r["tools"]] for r in prompts))
            self.assertTrue(any(r["type"] == "event" and r["tool"] == "buy" and r["ok"] for r in recs))
            self.assertIn(res["end_reason"], ("time_limit", "all_done", "all_dead"))

    def test_budget_exhaustion_stops_all(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = write_config(Path(d), budget_usd=0.01, max_seconds=30)
            t0 = time.monotonic()
            res = run_episode(cfg, run_dir=Path(d) / "run", mock=True, quiet=True)
            self.assertLess(time.monotonic() - t0, 20)
            self.assertEqual(res["end_reason"], "budget")
            self.assertLessEqual(res["cost"]["spent_usd"], 0.01)
            self.assertEqual(res["cost"]["open_reservations"], 0)
            self.assertTrue(all(a["stop_reason"] in ("budget", "episode_over", "stopped")
                                for a in res["agents"].values()))
            kinds = [json.loads(line)["kind"] for line in (Path(d) / "run" / "ledger.jsonl").read_text().splitlines()]
            self.assertIn("refuse", kinds)

    def test_api_errors_finish_error_and_no_key_leak(self):
        attempts = []

        def urlopen(req, timeout):
            attempts.append(1)
            if len(attempts) % 2:
                raise urllib.error.HTTPError(req.full_url, 502, "bad gateway", {},
                                             io.BytesIO(f"upstream echoed Bearer {FAKE_KEY}".encode()))
            raise urllib.error.URLError(socket.timeout("timed out"))

        client = OpenRouterClient(FAKE_KEY, base_url="http://fake.invalid", max_retries=2, urlopen=urlopen,
                                  sleep=lambda s: None)
        with tempfile.TemporaryDirectory() as d:
            cfg = write_config(Path(d), n_agents=2, max_seconds=30, llm={"max_consecutive_errors": 2})
            res = run_episode(cfg, run_dir=Path(d) / "run", client=client, quiet=True,
                              env={"OPENROUTER_API_KEY": FAKE_KEY})
            self.assertEqual(res["end_reason"], "error")
            recs = read_log(Path(d) / "run")
            errs = [r for r in recs if r["type"] == "model_call"]
            self.assertTrue(errs and all(r["error"] for r in errs))
            self.assertTrue(all(len(r["attempts"]) == 3 for r in errs))
            self.assertGreater(res["cost"]["uncertain_usd"], 0)  # 5xx/timeouts keep their reservations
            for f in (Path(d) / "run").rglob("*"):
                if f.is_file():
                    self.assertNotIn(FAKE_KEY, f.read_text(errors="replace"), f.name)
                    self.assertNotIn("FAKEKEY", f.read_text(errors="replace"), f.name)

    def test_dry_run_makes_no_network_calls(self):
        def boom(*a, **k):
            raise AssertionError("network used in dry run")

        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(urllib.request, "urlopen", boom), \
                mock.patch.object(socket, "create_connection", boom), \
                mock.patch.object(socket.socket, "connect", boom):
            cfg = write_config(Path(d))
            res = run_episode(cfg, run_dir=Path(d) / "run", dry_run=True, quiet=True,
                              env={"OPENROUTER_API_KEY": FAKE_KEY})
            self.assertTrue(res["ok"])
            self.assertFalse((Path(d) / "run").exists())
            plan = res["plan"]
            self.assertEqual(plan["n_agents"], 3)
            res = run_episode(cfg, dry_run=True, quiet=True, env={}, n_agents=6)
            self.assertFalse(res["ok"])  # no key
            self.assertEqual(res["plan"]["n_agents"], 6)
            self.assertAlmostEqual(res["plan"]["max_spend_if_uncapped_usd~"],
                                   2 * plan["max_spend_if_uncapped_usd~"], places=3)

    def test_real_run_requires_key(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = write_config(Path(d))
            with self.assertRaises(ConfigError):
                run_episode(cfg, run_dir=Path(d) / "run", quiet=True, env={})


if __name__ == "__main__":
    unittest.main()
