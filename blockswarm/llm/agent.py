"""One LLM agent's live loop: observe -> call the model -> run its tool calls -> repeat.

Each agent runs `LLMAgent.run()` in its own thread against the shared, wall-clock World.

Context per model call: the system prompt, then the last `history_steps` steps, then a fresh
user message with the current observation. A step is [user observation, assistant reply,
tool results...]. The first observation is a full view; later ones pass
`since_event_id=memory.last_event_id` so only new messages are shown (blocks, inventory and
positions are always current). With `history_chunk` > 1, old steps are dropped that many at
a time, which keeps the message prefix stable for longer (better provider prompt caching).

All tool calls in one response are executed sequentially and immediately (e.g. jump then
place lands within the airtime). World tools go to `world.act`; `observe` and `wait` are
handled here (observe is also logged as a world event). `wait(seconds)` returns early only
on something the agent could notice that happens after the wait starts: another agent's
message to it, a vote change, or a successful block/position change within its view
radius. It returns a short note; the next step's observation shows the details.

Logging (into the episode log via `world.record`, see docs/log-format.md):
- `agent_prompt`: once per agent, the full system prompt and tool schemas (with sha256).
- `model_call`: every model call, when its response arrives: full request messages (the
  system message is replaced by a reference to `agent_prompt`), tool names, response
  (content, reasoning, tool calls incl. raw argument strings), usage/cost, latency, error.
- `tool_results`: the text returned to the model for each tool call of that response.
- `agent_stop`: why the agent's loop ended.
Team battle: an agent that has been killed (`World.is_alive` false) stops its loop with reason
"dead" before its next model call and does not run the rest of a response's tool calls.
`wait` also wakes on a team message to the agent's team and on an attack on or near it.
Records produced after the episode ended (the world refuses them) go to `late_sink`.
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from blockswarm.llm.budget import Budget, estimate_cost, estimate_prompt_tokens, price_for
from blockswarm.world import TOOLS as WORLD_TOOLS
from blockswarm.world import World

ORCHESTRATOR_TOOLS = ("observe", "wait")
WAIT_MIN_S, WAIT_MAX_S = 1, 30
NUDGE = ("(You made no tool call in your last reply. Text outside tool calls is private: nobody "
         "sees it and it changes nothing in the world. Use the tools to act, talk or wait.)")


@dataclass
class LLMConfig:
    max_tokens: int = 2048
    temperature: float | None = 1.0
    reasoning: dict | None = None
    provider: dict | None = None
    extra_body: dict | None = None
    parallel_tool_calls: bool = True
    history_steps: int = 6
    history_chunk: int = 1
    max_model_calls_per_agent: int = 200
    max_consecutive_errors: int = 3
    error_pause_s: float = 5.0
    empty_steps_before_wait: int = 2
    implicit_wait_s: float = 5.0
    replay_reasoning: bool = True
    prices: dict = field(default_factory=dict)
    allow_unknown_price: bool = False
    price_safety: float = 1.25
    stagger_s: float = 2.0

    @classmethod
    def from_dict(cls, d: dict | None) -> "LLMConfig":
        d = dict(d or {})
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"unknown llm settings {unknown}; known: {sorted(known)}")
        cfg = cls(**d)
        if cfg.max_tokens < 1 or cfg.history_steps < 1 or cfg.max_model_calls_per_agent < 1:
            raise ValueError("max_tokens, history_steps and max_model_calls_per_agent must be >= 1")
        if not 1 <= cfg.history_chunk <= cfg.history_steps:
            raise ValueError("history_chunk must be in [1, history_steps]")
        if cfg.max_consecutive_errors < 1 or cfg.empty_steps_before_wait < 1:
            raise ValueError("max_consecutive_errors and empty_steps_before_wait must be >= 1")
        if cfg.price_safety < 1.0:
            raise ValueError("price_safety must be >= 1.0")
        return cfg


def sha256(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def _default_prompts():
    from blockswarm.llm import prompts
    return prompts


def _default_observe():
    from blockswarm.llm import observe
    return observe


class LLMAgent:
    def __init__(self, world: World, agent_id: str, client: Any, model: str, budget: Budget,
                 cfg: LLMConfig, *, prompts: Any = None, observe: Any = None,
                 late_sink: Callable[[dict], None] | None = None, start_delay_s: float = 0.0):
        self.world = world
        self.agent = agent_id
        self.client = client
        self.model = model
        self.budget = budget
        self.cfg = cfg
        self.prompts = prompts or _default_prompts()
        self.observe = observe or _default_observe()
        self.late_sink = late_sink
        self.start_delay_s = start_delay_s
        try:
            self.memory = self.observe.ObservationMemory(agent_id, view_occlusion=world.config.view_occlusion)
        except TypeError:
            self.memory = self.observe.ObservationMemory(agent_id)
        self.price = price_for(cfg.prices, model, cfg.allow_unknown_price)
        header = world.recorder.records[0]
        self.system_prompt: str = self.prompts.system_prompt(header, agent_id)
        try:
            self.tools: list[dict] = self.prompts.tool_schemas(header, agent_id)
        except TypeError:
            self.tools = self.prompts.tool_schemas(header)
        self.tool_names = [t["function"]["name"] for t in self.tools]
        self.world_tool_names = {n for n in self.tool_names if n in WORLD_TOOLS and n not in ORCHESTRATOR_TOOLS}
        self.history: list[list[dict]] = []
        self.calls = 0
        self.errors = 0
        self.consecutive_errors = 0
        self.consecutive_empty = 0
        self.stop_reason: str | None = None
        self.stats = {"calls": 0, "errors": 0, "tool_calls": 0, "empty_replies": 0, "prompt_tokens": 0,
                      "completion_tokens": 0, "cached_tokens": 0, "reasoning_tokens": 0, "cost_usd": 0.0,
                      "charged_usd": 0.0, "implicit_waits": 0}
        self._pending: list[str] = []  # observation texts not yet delivered (after an API error)
        self._stop = threading.Event()

    # ------------------------------------------------------------------ logging

    def _record(self, rec: dict) -> None:
        # Hold the world lock so "is the episode over?" and the write are atomic: World.record
        # silently drops records after the end, and those must go to the late sink instead.
        with self.world._lock:
            if not self.world.done:
                self.world.record(rec)
                return
        if self.late_sink:
            self.late_sink({"t_wall": time.time(), **rec})

    def _log_prompt(self) -> None:
        self._record({"type": "agent_prompt", "agent": self.agent, "model": self.model,
                      "system_prompt": self.system_prompt, "system_prompt_sha256": sha256(self.system_prompt),
                      "tools": self.tools, "tools_sha256": sha256(self.tools),
                      "prompt_version": getattr(self.prompts, "PROMPT_VERSION", None)})

    # ------------------------------------------------------------------ loop

    def stop(self) -> None:
        self._stop.set()

    def alive(self) -> bool:
        """False once the world says this agent was killed (team battle); True without combat."""
        is_alive = getattr(self.world, "is_alive", None)
        return True if is_alive is None else bool(is_alive(self.agent))

    def _sleep(self, s: float) -> None:
        end = time.monotonic() + s
        while not self.world.done and not self._stop.is_set() and time.monotonic() < end:
            time.sleep(min(0.05, max(0.0, end - time.monotonic())))

    def run(self) -> str:
        try:
            self._log_prompt()
            if self.start_delay_s:
                self._sleep(self.start_delay_s)
            while True:
                if self.world.check_end() or self._stop.is_set():
                    self.stop_reason = self.stop_reason or ("stopped" if self._stop.is_set() else "episode_over")
                    break
                if not self.alive():
                    self.stop_reason = "dead"
                    break
                if self.calls >= self.cfg.max_model_calls_per_agent:
                    self.stop_reason = "max_model_calls"
                    break
                if not self.step():
                    break
        except Exception as e:  # noqa: BLE001 - logged, then the episode ends as "error"
            self.stop_reason = "exception"
            self._record({"type": "orchestrator_error", "agent": self.agent, "error": repr(e),
                          "traceback": traceback.format_exc()})
            self.world.finish("error")
        if not self.world.done:  # after the end, the runner's manifest has the stop reason and stats
            self._record({"type": "agent_stop", "agent": self.agent, "reason": self.stop_reason,
                          "stats": dict(self.stats)})
        return self.stop_reason or "unknown"

    def build_messages(self, obs_text: str) -> list[dict]:
        msgs: list[dict] = [{"role": "system", "content": self.system_prompt}]
        for step in self.history:
            msgs.extend(step)
        msgs.append({"role": "user", "content": obs_text})
        return msgs

    def _observation(self) -> str:
        since = None if self.memory.last_event_id is None else self.memory.last_event_id
        view = self.world.view(self.agent, since_event_id=since)
        text = self.observe.render_observation(view, self.memory)
        if self.memory.last_event_id is None or self.memory.last_event_id < view["last_event_id"]:
            self.memory.last_event_id = view["last_event_id"]
        return text

    def step(self) -> bool:
        """One model call and its tool calls. Returns False when the loop should stop."""
        obs_parts = self._pending + [self._observation()]
        obs_text = "\n\n".join(([NUDGE] if self.consecutive_empty else []) + obs_parts)
        messages = self.build_messages(obs_text)
        self.calls += 1
        call_id = f"{self.agent}-{self.calls:04d}"
        est = estimate_cost(self.price, estimate_prompt_tokens(messages, self.tools), self.cfg.max_tokens,
                            self.cfg.price_safety)
        guard = self.budget.guard(self.model, est, self.price, {"agent": self.agent, "call_id": call_id})
        t_req = self.world.now()
        res = self.client.chat(self.model, messages, self.tools, max_tokens=self.cfg.max_tokens,
                               temperature=self.cfg.temperature, reasoning=self.cfg.reasoning,
                               provider=self.cfg.provider, parallel_tool_calls=self.cfg.parallel_tool_calls,
                               extra_body=self.cfg.extra_body, guard=guard, agent=self.agent)
        self._account(res, guard)
        self._log_call(call_id, t_req, messages, res, guard)

        if res.budget_exhausted:
            self.stop_reason = "budget"
            self.world.finish("budget")
            return False
        if res.error is not None:
            self.errors += 1
            self.consecutive_errors += 1
            self.stats["errors"] += 1
            self._pending = obs_parts  # deliver these observations with the next call
            if self.consecutive_errors >= self.cfg.max_consecutive_errors:
                self.stop_reason = "api_errors"
                self.world.finish("error")
                return False
            self._sleep(self.cfg.error_pause_s)
            return True
        self.consecutive_errors = 0
        self._pending = []

        step: list[dict] = [{"role": "user", "content": obs_text}]
        assistant = self._assistant_message(res)
        if assistant is not None:
            step.append(assistant)
        results: list[dict] = []
        if res.tool_calls:
            self.consecutive_empty = 0
            for tc in res.tool_calls:
                if not self.alive():
                    text, ok = "not executed: you were killed", False
                    results.append({"tool_call_id": tc["id"], "name": tc["name"], "ok": ok, "content": text})
                    step.append({"role": "tool", "tool_call_id": tc["id"], "content": text})
                    continue
                text, ok = self._execute(tc)
                results.append({"tool_call_id": tc["id"], "name": tc["name"], "ok": ok, "content": text})
                step.append({"role": "tool", "tool_call_id": tc["id"], "content": text})
            self._record({"type": "tool_results", "agent": self.agent, "call_id": call_id, "results": results})
        else:
            self.consecutive_empty += 1
            self.stats["empty_replies"] += 1
            if self.consecutive_empty >= self.cfg.empty_steps_before_wait:
                self.stats["implicit_waits"] += 1
                since = self.memory.last_event_id if self.memory.last_event_id is not None else -1
                latest = self.world.wait_for_change(self.agent, since, self.cfg.implicit_wait_s)
                self._record({"type": "implicit_wait", "agent": self.agent, "call_id": call_id,
                              "seconds": self.cfg.implicit_wait_s, "changed": latest > since})
        self._push_history(step)
        if self.budget.exhausted:
            self.stop_reason = "budget"
            self.world.finish("budget")
            return False
        return True

    def _push_history(self, step: list[dict]) -> None:
        self.history.append(step)
        if len(self.history) > self.cfg.history_steps:
            drop = max(len(self.history) - self.cfg.history_steps, self.cfg.history_chunk)
            del self.history[:drop]

    def _assistant_message(self, res) -> dict | None:
        if not res.tool_calls and not (res.content or "").strip():
            return None  # empty reply: nothing valid to replay
        msg: dict = {"role": "assistant", "content": res.content or None}
        if res.tool_calls:
            msg["tool_calls"] = [{"id": tc["id"], "type": "function",
                                  "function": {"name": tc["name"],
                                               "arguments": tc["raw_arguments"] if tc["arguments"] is not None else "{}"}}
                                 for tc in res.tool_calls]
        if self.cfg.replay_reasoning and res.reasoning_details:
            msg["reasoning_details"] = res.reasoning_details
        return msg

    # ------------------------------------------------------------------ tools

    def _execute(self, tc: dict) -> tuple[str, bool]:
        name, args, raw = tc["name"], tc["arguments"], tc["raw_arguments"]
        self.stats["tool_calls"] += 1
        render = self.observe.render_tool_result
        if self.world.done:
            return "not executed: the episode is over", False
        if name not in self.tool_names:
            return render(name, args or {}, False, f"unknown tool {name!r}; tools: {', '.join(self.tool_names)}", None), False
        if args is None:
            msg = f"arguments are not a valid JSON object: {raw[:300]!r}"
            if name in self.world_tool_names:  # log the failed call as a world event
                self.world.act(self.agent, name, _raw_arguments=raw)
            return render(name, {}, False, msg, None), False
        if name == "observe":
            if args:
                return render(name, args, False, "observe takes no arguments", None), False
            self.world.act(self.agent, "observe")
            return self._observation(), True
        if name == "wait":
            s = args.get("seconds")
            if set(args) - {"seconds"} or isinstance(s, bool) or not isinstance(s, (int, float)) \
                    or not WAIT_MIN_S <= s <= WAIT_MAX_S:
                return render(name, args, False, f"wait needs seconds between {WAIT_MIN_S} and {WAIT_MAX_S}", None), False
            t0 = self.world.now()
            woke = self.wait_relevant(float(s))
            waited = round(self.world.now() - t0, 2)
            if self.world.done:
                msg = f"waited {waited} s; the episode is over"
            elif woke:
                msg = f"waited {waited} s; stopped early: {woke}"
            else:
                msg = f"waited {waited} s; nothing new that you can see happened"
            return render(name, args, True, msg, {"waited_s": waited, "woke_on": woke}), True
        res = self.world.act(self.agent, name, **args)
        return render(name, args, res.ok, res.message, res.data), res.ok

    def wait_relevant(self, seconds: float) -> str | None:
        """Block up to `seconds` until another agent does something this agent could notice:
        a message to it (public, whisper to it, vote notice) or a successful change of blocks
        or agent positions within its view radius. Events from before the wait started don't
        count (the next observation shows them). Returns a short reason, or None on timeout."""
        w = self.world
        deadline = time.monotonic() + seconds
        latest = w.last_event_id()
        while not w.done:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            new = w.wait_for_change(self.agent, latest, remaining)
            if new <= latest:
                continue
            why = self._noticeable(latest, new)
            latest = new
            if why:
                return why
        return None

    def _noticeable(self, after_id: int, upto_id: int) -> str | None:
        w = self.world
        recs = w.recorder.records
        events = []
        for r in reversed(recs):  # new events are at the end of the log
            if r.get("type") == "event":
                if r["event_id"] <= after_id:
                    break
                if r["event_id"] <= upto_id:
                    events.append(r)
        with w._lock:
            me = w.positions[self.agent]
            radius = w.config.view_radius
            my_team = next((t for t, m in (getattr(w.config, "teams", None) or {}).items() if self.agent in m), None)
        near = lambda p: w.distance(me, tuple(p)) <= radius  # noqa: E731
        for e in reversed(events):
            if e["agent"] == self.agent or not e["ok"]:
                continue
            chat = e.get("chat")
            if chat and (chat["channel"] == "public" or chat.get("to") == self.agent
                         or (chat["channel"] == "team" and my_team is not None and chat.get("team") == my_team)):
                return f"{e['agent']} sent a message"
            diff = e.get("diff") or {}
            fight = diff.get("combat") or {}
            if fight:
                if fight.get("target") == self.agent:
                    return f"you were hit by {fight.get('attacker')}"
                if fight.get("killed"):
                    return f"{fight.get('target')} was killed"
                if any(near(w.positions[a]) for a in (fight.get("attacker"), fight.get("target"))
                       if a in w.positions):
                    return f"{fight.get('attacker')} attacked {fight.get('target')} near you"
            diff = e.get("diff") or {}
            votes = diff.get("votes")
            if votes and (my_team is None or "before" in votes or my_team in votes):
                return "the finished votes changed"
            if any(near(g["pos"]) for g in diff.get("grid", [])):
                return f"{e['agent']} changed a block near you"
            if any(near(m["to"]) or near(m["from"]) for m in diff.get("agents", {}).values()):
                return f"{e['agent']} moved near you"
        return None

    # ------------------------------------------------------------------ accounting

    def _account(self, res, guard) -> None:
        self.stats["calls"] += 1
        self.stats["charged_usd"] = round(self.stats["charged_usd"] + guard.charged, 8)
        for k in ("prompt_tokens", "completion_tokens", "cached_tokens", "reasoning_tokens"):
            v = res.usage.get(k) if res.usage else None
            if isinstance(v, int):
                self.stats[k] += v
        c = res.usage.get("cost_usd") if res.usage else None
        if isinstance(c, (int, float)):
            self.stats["cost_usd"] = round(self.stats["cost_usd"] + c, 8)

    def _log_call(self, call_id: str, t_req: float, messages: list[dict], res, guard) -> None:
        logged = copy.deepcopy(messages)
        logged[0] = {"role": "system", "content_ref": "agent_prompt",
                     "sha256": sha256(self.system_prompt)}
        self._record({
            "type": "model_call", "agent": self.agent, "call_id": call_id, "model": self.model,
            "t_request": t_req,
            "request": {"messages": logged, "tools": self.tool_names, "max_tokens": self.cfg.max_tokens,
                        "temperature": self.cfg.temperature, "reasoning": self.cfg.reasoning,
                        "provider": self.cfg.provider, "history_steps_in_context": len(self.history)},
            "response": {"content": res.content, "reasoning": res.reasoning,
                         "reasoning_details": res.reasoning_details,
                         "tool_calls": res.tool_calls, "finish_reason": res.finish_reason,
                         "id": res.response_id, "provider": res.provider, "served_model": res.served_model},
            "usage": res.usage, "charged_usd": round(guard.charged, 8), "attempts": guard.attempts,
            "latency_s": res.latency_s, "error": res.error,
        })
