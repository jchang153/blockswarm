"""Run one live LLM episode from a JSON config: world + one LLMAgent thread per agent.

Outputs go to runs/<run_id>/ (see docs/running.md):
  episode.jsonl       the episode log (world events + model_call/tool_results/... records)
  config.json         the config as given (plus CLI overrides under "_overrides")
  manifest.json       git commit, prompt version, models, timing, cost, end reason, python
  ledger.jsonl        every budget reservation / commit (no secrets, no prompts)
  late_records.jsonl  model calls whose responses arrived after the episode ended (if any)

Config keys: any World Config field (width, starting_money, incentive, max_seconds, ...), plus
  "n_agents": N (ids agent_1..agent_N) and/or "agents": [...] (must agree),
  "model": "id" or "models": {agent: id} | [id, ...] (one per agent),
  team battle (docs/team-battle.md): the world keys "variant", "teams" ({team: [agent ids]}; the
  agent list is then the teams' members in order, and agents/n_agents may be omitted), "shops",
  "divider_x", "combat", "max_health", "sword_damage", "attack_reach", "attack_cooldown_s", and
  "team_models": {team: model id} instead of "model"/"models" (e.g. one model per team),
  "llm": LLMConfig fields (max_tokens, temperature, reasoning, provider, history_steps,
         max_model_calls_per_agent, prices, ...),
  "budget_usd", "episode_id", "client": {"timeout_s", "max_retries"},
  "mock": {"latency_s": [lo, hi], "seed"} (only used with mock=True),
  and free-form "comment" / "model_candidates" / keys starting with "_" (ignored).
"""

from __future__ import annotations

import json
import math
import os
import platform
import random
import subprocess
import sys
import threading
import time
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from blockswarm.blueprints import BLUEPRINTS
from blockswarm.catalog import CATALOG
from blockswarm.clock import WallClock
from blockswarm.llm.agent import LLMAgent, LLMConfig
from blockswarm.llm.budget import Budget, UnknownPrice, estimate_cost, price_for
from blockswarm.llm.prompts import TEAM_BRIEF
from blockswarm.llm.client import ChatResult, MockClient, OpenRouterClient, mock_result, tool_call
from blockswarm.recorder import Recorder
from blockswarm.world import Config, World

REPO = Path(__file__).resolve().parents[2]
PLACEHOLDER_MODEL = "CHOOSE_MODEL"
META_KEYS = {"episode_id", "model", "models", "team_models", "llm", "budget_usd", "n_agents", "agents",
             "client", "mock", "comment", "model_candidates"}
WORLD_KEYS = {f.name for f in fields(Config)} - {"agents", "models", "run_meta"}
# Rough prompt size used for the dry-run spend estimate (system + tools + window), in tokens.
TYPICAL_PROMPT_TOKENS = 8000


class ConfigError(ValueError):
    pass


# ---------------------------------------------------------------------------- config


def resolve_config(raw: dict, *, n_agents: int | None = None, budget_usd: float | None = None,
                   allow_unknown_price: bool = False, mock: bool = False) -> dict:
    """Validate a run config; returns {world_cfg, agents, models, llm, budget_usd, ...}."""
    if not isinstance(raw, dict):
        raise ConfigError("config must be a JSON object")
    unknown = sorted(k for k in raw if k not in META_KEYS and k not in WORLD_KEYS and not k.startswith("_"))
    if unknown:
        raise ConfigError(f"unknown config keys {unknown}")

    # agents
    agents = raw.get("agents")
    n = raw.get("n_agents")
    teams = raw.get("teams")
    if teams is not None:
        if not isinstance(teams, dict) or not teams or not all(
                isinstance(t, str) and t and isinstance(m, list) and m and all(isinstance(a, str) and a for a in m)
                for t, m in teams.items()):
            raise ConfigError("teams must be a dict team -> non-empty list of agent ids")
        if n_agents is not None:
            raise ConfigError("--n-agents cannot be used with teams; edit the teams in the config")
        members = [a for m in teams.values() for a in m]
        if agents is not None and agents != members:
            raise ConfigError(f"agents {agents} must equal the teams' members in order {members}")
        if n is not None and n != len(members):
            raise ConfigError(f"n_agents={n} disagrees with the teams' {len(members)} members")
        agents, n = members, None
    if n_agents is not None:  # CLI override wins over both
        n, agents = n_agents, None
    if n is not None and (isinstance(n, bool) or not isinstance(n, int) or n < 1):
        raise ConfigError(f"n_agents must be a positive integer, got {n!r}")
    if agents is not None:
        if not isinstance(agents, list) or not all(isinstance(a, str) and a for a in agents):
            raise ConfigError("agents must be a list of agent id strings")
        if n is not None and n != len(agents):
            raise ConfigError(f"n_agents={n} disagrees with agents list of length {len(agents)}")
    elif n is not None:
        agents = [f"agent_{i}" for i in range(1, n + 1)]
    else:
        agents = list(Config().agents)

    # models
    if sum(k in raw for k in ("model", "models", "team_models")) > 1:
        raise ConfigError("give only one of 'model', 'models' and 'team_models'")
    if "team_models" in raw:
        tm = raw["team_models"]
        if teams is None:
            raise ConfigError("team_models needs teams")
        if not isinstance(tm, dict) or set(tm) != set(teams):
            raise ConfigError(f"team_models keys must equal the teams {sorted(teams)}")
        models = {a: tm[t] for t, m in teams.items() for a in m}
    elif "models" in raw:
        m = raw["models"]
        if isinstance(m, list):
            if len(m) != len(agents):
                raise ConfigError(f"models list has {len(m)} entries for {len(agents)} agents")
            models = dict(zip(agents, m))
        elif isinstance(m, dict):
            if set(m) != set(agents):
                raise ConfigError(f"models dict keys {sorted(m)} must equal agents {agents}")
            models = {a: m[a] for a in agents}
        else:
            raise ConfigError("models must be a dict agent -> model or a list")
    elif "model" in raw:
        models = {a: raw["model"] for a in agents}
    else:
        raise ConfigError("config needs 'model' or 'models'")
    if not all(isinstance(v, str) and v.strip() for v in models.values()):
        raise ConfigError("model ids must be non-empty strings")
    if not mock and any(v == PLACEHOLDER_MODEL for v in models.values()):
        raise ConfigError(f"model is still the placeholder {PLACEHOLDER_MODEL!r}; choose a model id "
                          "(see model_candidates in the config) and add its prices")

    try:
        llm = LLMConfig.from_dict(raw.get("llm"))
    except (TypeError, ValueError) as e:
        raise ConfigError(f"llm: {e}") from None
    llm.allow_unknown_price = llm.allow_unknown_price or allow_unknown_price or mock
    prices: dict[str, dict] = {}
    for model in sorted(set(models.values())):
        try:
            prices[model] = price_for(llm.prices, model, llm.allow_unknown_price)
        except UnknownPrice as e:
            raise ConfigError(str(e)) from None

    b = budget_usd if budget_usd is not None else raw.get("budget_usd")
    if isinstance(b, bool) or not isinstance(b, (int, float)) or not b > 0:
        raise ConfigError(f"budget_usd must be a number > 0, got {b!r}")

    world_kwargs = {k: raw[k] for k in WORLD_KEYS if k in raw}
    if teams is not None and raw.get("brief") is None:  # the team-battle brief, logged in the header
        world_kwargs["brief"] = TEAM_BRIEF
    try:
        world_cfg = Config(agents=agents, models=dict(models), **world_kwargs)
    except (TypeError, ValueError) as e:
        raise ConfigError(f"world config: {e}") from None

    client = dict(raw.get("client") or {})
    if set(client) - {"timeout_s", "max_retries"}:
        raise ConfigError("client accepts only timeout_s and max_retries")
    return {"agents": agents, "models": models, "llm": llm, "prices": prices, "budget_usd": float(b),
            "world_kwargs": world_kwargs, "world_cfg": world_cfg, "episode_id": raw.get("episode_id"),
            "client": client, "mock": dict(raw.get("mock") or {})}


def spend_plan(rc: dict) -> dict:
    """Worst-case and typical spend figures for the dry run."""
    llm: LLMConfig = rc["llm"]
    per_agent = []
    for a in rc["agents"]:
        p = rc["prices"][rc["models"][a]]
        out_only = estimate_cost(p, 0, llm.max_tokens, 1.0)
        typical = estimate_cost(p, TYPICAL_PROMPT_TOKENS, llm.max_tokens, llm.price_safety)
        per_agent.append({"agent": a, "model": rc["models"][a], "price": p,
                          "per_call_reservation_usd~": typical, "per_call_output_cap_usd": out_only})
    n = len(rc["agents"])
    worst = sum(x["per_call_reservation_usd~"] for x in per_agent) * llm.max_model_calls_per_agent
    min_res = min(x["per_call_reservation_usd~"] for x in per_agent)
    return {"n_agents": n, "per_agent": per_agent,
            "max_model_calls_per_agent": llm.max_model_calls_per_agent,
            "max_spend_if_uncapped_usd~": round(worst, 4),
            "formula": f"{n} agents x {llm.max_model_calls_per_agent} calls x per-call reservation "
                       f"(~{TYPICAL_PROMPT_TOKENS} prompt tokens + {llm.max_tokens} max output, x{llm.price_safety})",
            "budget_usd": rc["budget_usd"],
            "budget_binds": worst > rc["budget_usd"],
            "calls_covered_by_budget~": int(rc["budget_usd"] / min_res) if min_res > 0 else None}


# ---------------------------------------------------------------------------- mock policy


class MockPolicy:
    """Random-but-valid scripted 'model' for offline end-to-end runs (reads the world directly).

    Fetches blocks, walks near the reference house, places blocks on empty supported cells of
    that design, sometimes jumps and places under itself, says things, waits while walking.
    Usage tokens are estimated from message size; cost from the configured prices.

    Team battle (world config `teams`): each team builds the reference house shifted next to its
    own shop, buys at its own shop from its team wallet, and the policy also uses every new
    mechanic now and then so offline runs exercise them for the viewer: team `say` and `say_all`,
    raids (walk to the other side and remove one of the other team's blocks) and `attack` on an
    agent within sword reach (mostly the other team's, rarely a teammate). This is a test
    fixture, not a model of behaviour.
    """

    def __init__(self, world: World, prices: dict[str, dict], seed: int = 0):
        self.world = world
        self.prices = prices
        self.seed = seed
        self.design = sorted(BLUEPRINTS["house"].cells)
        self.design_types = sorted(set(BLUEPRINTS["house"].cells.values()))
        self.teams: dict[str, list[str]] = dict(getattr(world.config, "teams", None) or {})
        self.team_design: dict[str, list[tuple[int, int, int]]] = {t: self._shifted_design(t) for t in self.teams}

    # -- team battle helpers
    def team_of(self, agent: str | None) -> str | None:
        return next((t for t, m in self.teams.items() if agent in m), None)

    def _shops(self) -> dict[str, tuple[int, int, int]]:
        shops = getattr(self.world.config, "shops", None) or {}
        return {t: tuple(p) for t, p in shops.items()}

    def _divider(self) -> int:
        d = getattr(self.world.config, "divider_x", None)
        return int(d) if d is not None else self.world.config.width // 2

    def _on_side(self, team: str, x: int) -> bool:
        first = next(iter(self.teams))
        return (x < self._divider()) == (team == first)

    def _shifted_design(self, team: str) -> list[tuple[int, int, int]]:
        """The reference house moved next to the team's shop and inside its side of the line."""
        cfg = self.world.config
        shop = self._shops().get(team) or tuple(cfg.shop_pos)
        xs = [c[0] for c in self.design]
        ys = [c[1] for c in self.design]
        wx, wy = max(xs) - min(xs), max(ys) - min(ys)
        lo, hi = (0, self._divider() - 1) if team == next(iter(self.teams)) else (self._divider(), cfg.width - 1)
        x0 = min(max(shop[0] - wx // 2, lo), hi - wx)
        y0 = shop[1] + 2 if shop[1] + 2 + wy < cfg.depth else max(0, shop[1] - 2 - wy)
        out = [(x - min(xs) + x0, y - min(ys) + y0, z) for x, y, z in self.design]
        return sorted(c for c in out if cfg.in_bounds(c) and c != shop)

    def _alive(self, agent: str) -> bool:
        f = getattr(self.world, "is_alive", None)
        return True if f is None else bool(f(agent))

    def _sword_reach(self, pos, target: str) -> bool:
        w = self.world
        if target not in w.positions:
            return False
        reach = float(getattr(w.config, "attack_reach", 3.0) or 3.0)
        tx, ty, tz = w.positions[target]
        eye = w.eye(pos)
        for cell in sorted(((tx, ty, tz), (tx, ty, tz + 1)),
                           key=lambda c: math.dist(eye, (c[0] + 0.5, c[1] + 0.5, c[2] + 0.5)))[:1]:
            if math.dist(eye, (cell[0] + 0.5, cell[1] + 0.5, cell[2] + 0.5)) > reach:
                return False
            return not w.config.reach_line_of_sight or w.line_of_sight(pos, cell)
        return False

    def __call__(self, state: dict, messages: list) -> ChatResult:
        agent = state["agent"]
        rng = state.setdefault("rng", random.Random(f"{self.seed}-{agent}"))
        if self.teams:
            calls = self._decide_team(agent, rng, state)
            r = rng.random()
            if r < 0.05:
                calls.append(tool_call("say", text=f"{agent} to my team: working on our house, t={self.world.now():.0f}"))
            elif r < 0.08:
                calls.append(tool_call("say_all", text=f"{agent} to everyone: hello, t={self.world.now():.0f}"))
        else:
            calls = self._decide(agent, rng)
            if rng.random() < 0.08:
                calls.append(tool_call("say", text=f"{agent}: working on the house, t={self.world.now():.0f}"))
        chars = len(json.dumps(messages))
        pt, ctoks = chars // 4, 40 + 20 * len(calls)
        p = self.prices.get(state["model"]) or {"input_per_m": 0.0, "output_per_m": 0.0}
        cost = round((pt * p["input_per_m"] + ctoks * p["output_per_m"]) / 1e6, 8)
        content = None if rng.random() < 0.7 else "Planning my next move."
        return mock_result(*calls, content=content, cost_usd=cost, prompt_tokens=pt, completion_tokens=ctoks)

    def _decide(self, agent: str, rng: random.Random) -> list[dict]:
        w = self.world
        with w._lock:
            t = w.now()
            if w.is_walking(agent, t):
                return [tool_call("wait", seconds=max(1, min(30, int(w.walking_until[agent] - t) + 1)))]
            pos = w.positions[agent]
            inv = dict(w.inventories[agent])
            if not inv:
                if w.can_reach(pos, w.shop_pos):
                    cheap = [(CATALOG[k].price, k) for k in sorted(set(self.design_types))]
                    avail = [(p, k) for p, k in cheap if p <= w.wallet]
                    if not avail:
                        return [tool_call("declare_done")]
                    p, k = rng.choice(avail)
                    return [tool_call("buy", item=k, n=max(1, min(w.wallet // p, rng.randint(2, 5))))]
                stands = w.reachable_stand_cells(w.shop_pos, agent)
                if not stands:
                    return [tool_call("wait", seconds=2)]
                return [tool_call("move_to", x=stands[0][0], y=stands[0][1], z=stands[0][2])]
            btype = rng.choice(sorted(inv))
            bodies = w.body_cells()
            cands = [c for c in self.design if c not in w.blocks and c != w.shop_pos and c not in bodies
                     and w.supported(c)]
            if not cands:
                return [tool_call("declare_done")]
            if rng.random() < 0.1 and pos[2] < 2 and w.supported(pos):
                return [tool_call("jump"), tool_call("place", x=pos[0], y=pos[1], z=pos[2], type=btype)]
            near = [c for c in cands if w.can_reach(pos, c)]
            if near:
                c = rng.choice(near[:6])
                return [tool_call("place", x=c[0], y=c[1], z=c[2], type=btype)]
            c = rng.choice(cands[:12])
            stands = w.reachable_stand_cells(c, agent)
            if not stands:
                return [tool_call("observe")]
            s = stands[0]
            return [tool_call("move_to", x=s[0], y=s[1], z=s[2])]


    def _decide_team(self, agent: str, rng: random.Random, state: dict) -> list[dict]:
        w = self.world
        team = self.team_of(agent)
        other = next(t for t in self.teams if t != team)
        shop = self._shops().get(team) or tuple(w.config.shop_pos)
        wallets = getattr(w, "wallets", None) or {}
        with w._lock:
            t = w.now()
            if w.is_walking(agent, t):
                return [tool_call("wait", seconds=max(1, min(30, int(w.walking_until[agent] - t) + 1)))]
            pos = w.positions[agent]
            inv = dict(w.inventories[agent])
            held = sum(inv.values())
            # sword: anyone within reach, mostly the other team
            if rng.random() < 0.15:
                near = [a for a in w.agents if a != agent and self._alive(a) and self._sword_reach(pos, a)]
                opp = [a for a in near if self.team_of(a) == other]
                if opp:
                    return [tool_call("attack", target=rng.choice(opp))]
                if near and rng.random() < 0.1:
                    return [tool_call("attack", target=rng.choice(near))]
            # raid: cross the line and remove one of the other team's blocks
            if not state.get("raid") and rng.random() < 0.03:
                state["raid"] = True
            if state.get("raid"):
                theirs = sorted(p for p, b in w.blocks.items()
                                if not self._on_side(team, p[0]) and self.team_of(b.placed_by) == other)
                if not theirs or held >= w.config.inventory_cap:
                    state["raid"] = False
                else:
                    near = [p for p in theirs if w.can_reach(pos, p)]
                    if near:
                        state["raid"] = rng.random() < 0.3
                        c = rng.choice(near)
                        return [tool_call("remove", x=c[0], y=c[1], z=c[2])]
                    c = rng.choice(theirs[:10])
                    stands = w.reachable_stand_cells(c, agent)
                    if stands:
                        s = stands[0]
                        return [tool_call("move_to", x=s[0], y=s[1], z=s[2])]
                    state["raid"] = False
            # build the team's house on its own side, buying at the team's own shop
            if not inv:
                if w.can_reach(pos, shop):
                    money = wallets.get(team, 0)
                    avail = [(CATALOG[k].price, k) for k in self.design_types if CATALOG[k].price <= money]
                    if not avail:
                        return [tool_call("declare_done")]
                    p, k = rng.choice(avail)
                    return [tool_call("buy", item=k, n=max(1, min(money // p, rng.randint(2, 5))))]
                stands = w.reachable_stand_cells(shop, agent)
                if not stands:
                    return [tool_call("wait", seconds=2)]
                return [tool_call("move_to", x=stands[0][0], y=stands[0][1], z=stands[0][2])]
            btype = rng.choice(sorted(inv))
            bodies = w.body_cells()
            cands = [c for c in self.team_design[team] if c not in w.blocks and c not in bodies and w.supported(c)]
            if not cands:
                return [tool_call("declare_done")]
            near = [c for c in cands if w.can_reach(pos, c)]
            if near:
                c = rng.choice(near[:6])
                return [tool_call("place", x=c[0], y=c[1], z=c[2], type=btype)]
            c = rng.choice(cands[:12])
            stands = w.reachable_stand_cells(c, agent)
            if not stands:
                return [tool_call("observe")]
            s = stands[0]
            return [tool_call("move_to", x=s[0], y=s[1], z=s[2])]


# ---------------------------------------------------------------------------- helpers


def load_dotenv(path: str | Path, environ: dict | None = None) -> list[str]:
    """Minimal .env parser: KEY=VALUE lines; never overrides existing variables. Returns keys set."""
    env = os.environ if environ is None else environ
    p = Path(path)
    set_keys: list[str] = []
    if not p.is_file():
        return set_keys
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
            v = v[1:-1]
        if k and k not in env:
            env[k] = v
            set_keys.append(k)
    return set_keys


def git_info() -> dict:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:  # noqa: BLE001
            return None
    status = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status) if status is not None else None}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _prompt_version() -> str | None:
    try:
        from blockswarm.llm import prompts
        return getattr(prompts, "PROMPT_VERSION", None)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------- run


def run_episode(config_path: str | Path, *, run_dir: str | Path | None = None, dry_run: bool = False,
                mock: bool = False, budget_usd: float | None = None, n_agents: int | None = None,
                allow_unknown_price: bool = False, client: Any = None, env: dict | None = None,
                quiet: bool = False, settle_timeout_s: float = 30.0) -> dict:
    """Run (or dry-run) one episode. Returns a summary dict (also written into manifest.json)."""
    say = (lambda *a: None) if quiet else (lambda *a: print(*a, flush=True))
    env = os.environ if env is None else env
    config_path = Path(config_path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    rc = resolve_config(raw, n_agents=n_agents, budget_usd=budget_usd,
                        allow_unknown_price=allow_unknown_price, mock=mock)
    llm: LLMConfig = rc["llm"]
    started = datetime.now(timezone.utc)
    episode_id = rc["episode_id"] or config_path.stem
    run_id = f"{episode_id}-{started.strftime('%Y%m%dT%H%M%SZ')}" + ("-mock" if mock else "")
    out = Path(run_dir) if run_dir else REPO / "runs" / run_id
    key_present = bool(env.get("OPENROUTER_API_KEY"))
    plan = spend_plan(rc)

    if dry_run:
        say(f"DRY RUN (no model calls) for {config_path}")
        say(f"  agents ({len(rc['agents'])}): {', '.join(rc['agents'])}")
        for a in plan["per_agent"]:
            say(f"  {a['agent']}: {a['model']}  price in/out per 1M: ${a['price']['input_per_m']}/"
                f"${a['price']['output_per_m']}{' (UNKNOWN, fallback)' if a['price'].get('unknown') else ''}  "
                f"reservation/call ~${a['per_call_reservation_usd~']:.4f}")
        c = rc["world_cfg"]
        say(f"  world: task={c.task} grid={c.width}x{c.depth}x{c.height} shop_pos={list(c.shop_pos)} "
            f"starting_money={c.starting_money} incentive={c.incentive} deploy_top_k={c.deploy_top_k} "
            f"show_leaderboard={c.show_leaderboard} max_seconds={c.max_seconds}")
        if getattr(c, "teams", None):
            say(f"  teams: " + "; ".join(f"{t}: {', '.join(m)}" for t, m in c.teams.items())
                + f"  shops={getattr(c, 'shops', None)} divider_x={getattr(c, 'divider_x', None)} "
                f"variant={getattr(c, 'variant', None)}")
        say(f"  llm: max_tokens={llm.max_tokens} temperature={llm.temperature} reasoning={llm.reasoning} "
            f"history_steps={llm.history_steps} max_calls/agent={llm.max_model_calls_per_agent}")
        say(f"  max spend without budget: ~${plan['max_spend_if_uncapped_usd~']:.2f} = {plan['formula']}")
        say(f"  budget: ${rc['budget_usd']:.2f} (hard cap; {'binds' if plan['budget_binds'] else 'does not bind'}; "
            f"covers ~{plan['calls_covered_by_budget~']} calls at full reservation)")
        say(f"  OPENROUTER_API_KEY present: {key_present}{'' if key_present or mock else '  <-- REQUIRED for a real run'}")
        say(f"  outputs would go to: {out}")
        return {"dry_run": True, "ok": key_present or mock, "key_present": key_present, "run_dir": str(out),
                "plan": plan, "agents": rc["agents"], "models": rc["models"]}

    if client is None and not mock and not key_present:
        raise ConfigError("OPENROUTER_API_KEY is not set (put it in .env or the environment)")
    if out.exists() and any(out.iterdir()):
        raise ConfigError(f"run dir {out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)
    cfg_copy = dict(raw)
    overrides = {k: v for k, v in {"n_agents": n_agents, "budget_usd": budget_usd, "mock": mock or None,
                                   "allow_unknown_price": allow_unknown_price or None}.items() if v is not None}
    if overrides:
        cfg_copy["_overrides"] = overrides
    _write_json(out / "config.json", cfg_copy)

    budget = Budget(rc["budget_usd"], out / "ledger.jsonl")
    late_lock = threading.Lock()
    late_path = out / "late_records.jsonl"
    late_count = [0]

    def late_sink(rec: dict) -> None:
        with late_lock:
            with late_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
            late_count[0] += 1

    git = git_info()
    run_meta = {"run_id": run_id, "prompt_version": _prompt_version(), "git_commit": git["commit"],
                "git_dirty": git["dirty"], "mock": mock, "budget_usd": rc["budget_usd"],
                "llm": {k: getattr(llm, k) for k in ("max_tokens", "temperature", "reasoning", "provider",
                                                     "history_steps", "history_chunk",
                                                     "max_model_calls_per_agent")}}
    manifest: dict = {"run_id": run_id, "episode_id": episode_id, "status": "running", "mock": mock,
                      "config_path": str(config_path), "git": git, "prompt_version": run_meta["prompt_version"],
                      "models": rc["models"], "n_agents": len(rc["agents"]), "budget_usd": rc["budget_usd"],
                      "llm": {**run_meta["llm"], "prices": rc["prices"], "price_safety": llm.price_safety},
                      "python": sys.version.split()[0], "platform": platform.platform(),
                      "start_utc": started.isoformat(timespec="seconds"), "spend_plan": plan}
    _write_json(out / "manifest.json", manifest)

    world_cfg = Config(agents=rc["agents"], models=dict(rc["models"]), run_meta=run_meta, **rc["world_kwargs"])
    recorder = Recorder(out / "episode.jsonl")
    world = World(world_cfg, recorder, episode_id=episode_id, clock=WallClock())

    if client is None:
        if mock:
            m = rc["mock"]
            lat = m.get("latency_s", [0.3, 1.0])
            client = MockClient(MockPolicy(world, rc["prices"], seed=m.get("seed", world_cfg.seed)),
                                latency_s=tuple(lat) if isinstance(lat, list) else lat, seed=world_cfg.seed)
        else:
            client = OpenRouterClient(env["OPENROUTER_API_KEY"], **rc["client"])

    rng = random.Random(world_cfg.seed)
    agents = [LLMAgent(world, a, client, rc["models"][a], budget, llm, late_sink=late_sink,
                       start_delay_s=rng.uniform(0, llm.stagger_s)) for a in rc["agents"]]
    results: dict[str, str] = {}

    def worker(ag: LLMAgent) -> None:
        results[ag.agent] = ag.run()

    threads = [threading.Thread(target=worker, args=(ag,), name=f"llm-{ag.agent}", daemon=True) for ag in agents]
    say(f"run {run_id}: {len(agents)} agents, models {sorted(set(rc['models'].values()))}, "
        f"budget ${rc['budget_usd']:.2f}, max {world_cfg.max_seconds:.0f} s")
    say(f"  log: {out / 'episode.jsonl'}")
    say(f"  watch live: python3 scripts/serve_viewer.py {_rel(out / 'episode.jsonl')} --follow")
    interrupted = False
    try:
        for th in threads:
            th.start()
        while any(th.is_alive() for th in threads) and not world.check_end():
            time.sleep(0.1)
    except KeyboardInterrupt:
        interrupted = True
        say("\ninterrupted: ending the episode (aborted) and waiting for in-flight calls ...")
        world.finish("aborted")
    if not world.done:  # every agent stopped on its own (e.g. max_model_calls)
        world.finish("aborted")
    for ag in agents:
        ag.stop()
    deadline = time.monotonic() + settle_timeout_s
    try:
        for th in threads:
            th.join(max(0.0, deadline - time.monotonic()))
    except KeyboardInterrupt:
        say("not waiting any longer for in-flight calls")
    recorder.close()
    budget.close()

    ended = datetime.now(timezone.utc)
    end = recorder.records[-1] if recorder.records and recorder.records[-1].get("type") == "episode_end" else {}
    counts: dict[str, int] = {}
    for r in recorder.records:
        counts[r["type"]] = counts.get(r["type"], 0) + 1
    bsum = budget.summary()
    manifest.update({
        "status": "finished", "end_utc": ended.isoformat(timespec="seconds"),
        "wall_seconds": round((ended - started).total_seconds(), 1), "episode_seconds": end.get("t"),
        "end_reason": world.end_reason, "interrupted": interrupted,
        "wallet": end.get("wallet"), "individual_scores": end.get("individual_scores"),
        "ranking": end.get("ranking"), "deployed": end.get("deployed"),
        "analysis": end.get("analysis"), "actions_used": end.get("actions_used"),
        "cost": {**bsum, "provider_reported_usd": round(sum(a.stats["cost_usd"] for a in agents), 8)},
        "agents": {a.agent: {"model": a.model, "stop_reason": results.get(a.agent, "still_running"), **a.stats}
                   for a in agents},
        **{k: end[k] for k in ("variant", "wallets", "health", "dead", "kills", "teams") if k in end},
        "record_counts": counts, "late_records": late_count[0],
        "threads_still_running": [th.name for th in threads if th.is_alive()],
    })
    _write_json(out / "manifest.json", manifest)
    say(f"\nepisode over: {world.end_reason}  (episode t={end.get('t')} s, wall {manifest['wall_seconds']} s)")
    if end.get("teams"):
        say(f"  wallets left {end.get('wallets')}  dead {end.get('dead')}  kills {len(end.get('kills') or [])}  "
            f"individual {end.get('individual_scores')}")
    else:
        say(f"  wallet left {end.get('wallet')}  individual {end.get('individual_scores')}  "
            f"deployed {end.get('deployed')}")
    say(f"  cost ${bsum['spent_usd']:.4f} of ${bsum['limit_usd']:.2f} (provider-reported "
        f"${manifest['cost']['provider_reported_usd']:.4f}; uncertain ${bsum['uncertain_usd']:.4f}; "
        f"open reservations {bsum['open_reservations']})")
    for a in agents:
        s = a.stats
        say(f"  {a.agent}: {s['calls']} calls, {s['tool_calls']} tool calls, {s['errors']} errors, "
            f"stop={results.get(a.agent, 'still_running')}")
    say(f"  records: {counts}" + (f"; {late_count[0]} late records in late_records.jsonl" if late_count[0] else ""))
    say(f"  outputs: {out}")
    say(f"  replay: python3 scripts/serve_viewer.py {_rel(out / 'episode.jsonl')}")
    return {"run_dir": str(out), "interrupted": interrupted, **{k: manifest[k] for k in ("run_id", "end_reason", "deployed", "cost",
                                                               "agents", "record_counts", "late_records")}}


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(p)
