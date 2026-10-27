"""Deterministic, honest, cooperative scripted policy (no LLM), and a simulated-time runner.

All scripted agents share one internal design: the task blueprint in blueprint mode, or the
`house` blueprint in open mode (the world does not know about it). Each step makes at most
one tool call. Like an LLM agent, the policy plans from what it has *seen* (its views, which
are limited by distance and line of sight) and from the view at its previous step, so other
agents may have taken the blocks or filled the cell by the time it acts; those calls fail
(and are logged) and it re-plans. For geometry (paths, where to stand) it uses the world's
helper queries, which an LLM agent would get as tools.

Finished votes: once it believes the build is finished (or it is stuck), it says so and calls
`declare_done`, then waits without acting. Any change to the build clears every vote, so if
another agent places or removes a block afterwards, it looks again and votes again later.

This is a test/baseline bot, not a model of good play: it only walks, takes, places and
talks (no climbing or scaffolding). Build order: foundation, walls with the doorway left
open, then the roof (outer ring first) from wherever a standing spot reaches it, which for
the house is from inside, walking in through the doorway; the door goes in last, and never
while that would wall someone in.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field

from blockswarm.blueprints import BLUEPRINTS, Pos
from blockswarm.clock import SimClock
from blockswarm.world import NEIGHBOURS, World

OPEN_DESIGN = "house"
PROGRESS_EVERY_S = 60.0
STALL_STEPS = 4      # own steps in a row with no visible change before giving up when stuck
TOP_TARGETS = 6      # how many candidate cells to consider when choosing where to stand
WAIT = ("wait", {})  # sentinel: do nothing this step, but don't count as stuck


@dataclass
class _Memory:
    """Per-agent state kept between steps (what an agent could keep in its notes)."""
    view: dict | None = None
    known: dict = field(default_factory=dict)        # pos -> type of blocks seen
    fingerprint: tuple | None = None
    stalled_steps: int = 0
    planned: bool = False
    announced_out: set = field(default_factory=set)
    last_progress_t: float = 0.0
    declare_said: bool = False
    declared: bool = False
    placed: int = 0
    inspected: dict = field(default_factory=dict)    # pos -> t last inspected
    inspect_count: dict = field(default_factory=dict)  # pos -> times inspected
    last_change_t: float = 0.0                       # when scores/chest last changed


class ScriptedBuilder:
    """Builds the shared design. `step(world, agent)` makes at most one tool call."""

    def __init__(self, stall_s: float = 15.0, inspect_s: float = 30.0) -> None:
        self._mem: dict[str, _Memory] = {}
        self.stall_s = stall_s      # episode seconds with no noticeable progress before giving up
        self.inspect_s = inspect_s  # min episode seconds between looks at the same cell

    def memory(self, agent: str) -> _Memory:
        return self._mem.setdefault(agent, _Memory())

    def step(self, world: World, agent: str) -> bool:
        """Make at most one tool call. Returns False once this agent has nothing left to do."""
        mem = self.memory(agent)
        if world.done:
            return False
        if mem.declared:
            # Voted finished: wait without acting (any change to the build would clear every
            # vote). If someone changed the build since, my vote was cleared: look again and
            # vote again once I believe it is finished (or stalled) again.
            if agent in world.view(agent)["done_votes"]:
                return True
            mem.declared = False
            mem.stalled_steps = 0
            mem.view = None
        if mem.view is None or mem.view["walking_until"] is not None:
            # first step, or my last look was mid-walk: look around on arrival
            mem.view = world.view(agent)
            self._learn(world, agent, mem, mem.view)
        call = self._decide(world, agent, mem.view, mem)
        if call is not None:
            tool, args = call
            res = world.act(agent, tool, **args)
            self._after(world, agent, mem, tool, args, res.ok, res.message)
        fresh = world.view(agent)
        self._learn(world, agent, mem, fresh)
        # Progress I can notice: what I know is built, and the chest counts I last saw.
        fp = (tuple(sorted(mem.known.items())),
              tuple(sorted((fresh["chest_last_seen"] or {}).get("counts", {}).items())))
        if fp == mem.fingerprint:
            mem.stalled_steps += 1
        else:
            mem.stalled_steps = 0
            mem.last_change_t = fresh["t"]
        mem.fingerprint = fp
        mem.view = fresh
        return True

    # ------------------------------------------------------------------ knowledge

    @staticmethod
    def design(v: dict) -> dict[Pos, str]:
        bp = v["task"].get("blueprint")
        if bp:
            return {(x, y, z): t for x, y, z, t in bp["cells"]}
        return dict(BLUEPRINTS[OPEN_DESIGN].cells)

    def _learn(self, world: World, agent: str, mem: _Memory, v: dict) -> None:
        seen = {tuple(b["pos"]): b["type"] for b in v["blocks"]}
        pos = tuple(v["position"])
        for p in [p for p in mem.known if p not in seen]:
            if world.visible(pos, p):  # in sight but not listed: it is gone
                del mem.known[p]
        mem.known.update(seen)

    def _after(self, world: World, agent: str, mem: _Memory, tool: str, args: dict, ok: bool, msg: str) -> None:
        if tool == "place":
            p = (args["x"], args["y"], args["z"])
            if ok:
                mem.placed += 1
                mem.known[p] = args["type"]
            elif "occupied by" in msg and "body" not in msg:
                mem.known[p] = msg.split("occupied by ")[1].split(" ")[0]
        elif tool == "declare_done" and ok:
            mem.declared = True

    def _missing(self, v: dict, mem: _Memory) -> dict[Pos, str]:
        return {p: t for p, t in self.design(v).items() if mem.known.get(p) != t}

    # ------------------------------------------------------------------ decisions

    def _decide(self, world: World, agent: str, v: dict, mem: _Memory) -> tuple[str, dict] | None:
        if v["walking_until"] is not None and v["walking_until"] > world.now():
            return None
        if not mem.planned:
            mem.planned = True
            return "say", {"text": self._plan_text(v)}
        design = self.design(v)
        missing = self._missing(v, mem)
        inv = v["inventory"]

        if not missing:
            return self._declare(mem, "The house is complete as far as I can see. Declaring done.")

        need = Counter(missing.values())
        surplus = {t: n for t, n in inv.items() if need[t] == 0}
        if surplus:
            if v["chest"] is not None:
                t = sorted(surplus)[0]
                return "deposit", {"type": t, "n": surplus[t]}
            return self._go_to_chest(world, agent)

        place = self._place_or_walk(world, agent, mem, missing, inv, design)
        if place is WAIT:
            return None
        if place:
            return place

        fetch = self._fetch(world, agent, v, mem, missing, inv, design)
        if fetch:
            return fetch

        inspect = self._inspect(world, agent, v, mem, missing)
        if inspect is WAIT:  # wait by the chest, out of everyone's way
            return self._go_to_chest(world, agent)
        if inspect:
            return inspect

        if v["chest"] is None:
            return self._go_to_chest(world, agent)
        for t in sorted(need):
            if t not in mem.announced_out and v["chest"].get(t, 0) == 0 and inv.get(t, 0) == 0:
                mem.announced_out.add(t)
                return "say", {"text": f"I'm out of {t} and the chest has none; "
                                       f"{need[t]} {t} cell(s) of our design still look missing to me."}
        # Nothing placeable, fetchable or worth inspecting; give up once nothing changes either.
        if mem.stalled_steps >= STALL_STEPS and v["t"] - mem.last_change_t >= self.stall_s:
            held = ", ".join(f"{n} {t}" for t, n in sorted(inv.items()))
            return self._declare(mem, "Nothing more I can do. As far as I can see these are still missing: "
                                 + ", ".join(f"{need[t]} {t} (chest has {v['chest'].get(t, 0)})"
                                             for t in sorted(need))
                                 + (f"; I hold {held} but can't place it anywhere I can reach" if held else "")
                                 + ". Declaring done.")
        if v["t"] - mem.last_progress_t >= PROGRESS_EVERY_S:
            mem.last_progress_t = v["t"]
            done_cells = len(design) - len(missing)
            return "say", {"text": f"Progress as far as I can see: {done_cells}/{len(design)} cells of "
                                   f"our design are built; I have placed {mem.placed}."}
        return None

    def _declare(self, mem: _Memory, text: str) -> tuple[str, dict]:
        if not mem.declare_said:
            mem.declare_said = True
            return "say", {"text": text}
        return "declare_done", {}

    def _plan_text(self, v: dict) -> str:
        idx = v["agents"].index(v["you"])
        n = len(v["agents"])
        what = ("the blueprint" if v["task"].get("blueprint")
                else f"a {OPEN_DESIGN} like the standard 7x5 one at x 4-10, y 5-9 (walls 3 high, door at the front)")
        return (f"Plan: let's build {what}. Foundation first, then walls with the doorway left open, "
                f"then the roof from inside, door last. I'll take cells where (x + y) % {n} == {idx} "
                "and help anywhere else; I only take blocks I am about to place.")

    # ---- placing

    def _footprint(self, design: dict[Pos, str]) -> set[tuple[int, int]]:
        return {(x, y) for x, y, _ in design}

    def _targets(self, world: World, agent: str, mem: _Memory, missing: dict[Pos, str],
                 design: dict[Pos, str]) -> list[tuple[Pos, str]]:
        """Placeable missing cells in build order: lowest z, outer ring first, own share, door last."""
        solid = set(mem.known) | {world.chest_pos}
        bodies = world.body_cells()
        foot = self._footprint(design)
        xs = [x for x, _ in foot]
        ys = [y for _, y in foot]
        idx = world.agents.index(agent)
        n = len(world.agents)
        only_doors = all(t == "door" for t in missing.values())

        def supported(p: Pos) -> bool:
            x, y, z = p
            return z == 0 or any((x + dx, y + dy, z + dz) in solid for dx, dy, dz in NEIGHBOURS)

        cells = [(p, t) for p, t in missing.items()
                 if p not in bodies and supported(p)
                 and (t != "door" or only_doors)]

        def key(c):
            (x, y, z), t = c
            edge = x in (min(xs), max(xs)) or y in (min(ys), max(ys))
            return (z, not edge, (x + y) % n != idx, (x, y, z))

        return sorted(cells, key=key)

    def _safe(self, world: World, cell: Pos) -> bool:
        return not world.cut_off_if_solid([cell])

    def _place_from_here(self, world: World, agent: str, mem: _Memory, missing: dict[Pos, str],
                         inv: dict, design: dict[Pos, str]) -> tuple[str, dict] | None:
        pos = world.positions[agent]
        for p, t in self._targets(world, agent, mem, missing, design):
            if inv.get(t, 0) > 0 and world.can_reach(pos, p) and self._safe(world, p):
                return "place", {"x": p[0], "y": p[1], "z": p[2], "type": t}
        return None

    def _place_or_walk(self, world: World, agent: str, mem: _Memory, missing: dict[Pos, str],
                       inv: dict, design: dict[Pos, str]) -> tuple[str, dict] | None:
        here = self._place_from_here(world, agent, mem, missing, inv, design)
        if here:
            return here
        targets = [(p, t) for p, t in self._targets(world, agent, mem, missing, design) if inv.get(t, 0) > 0]
        if not targets:
            return None
        top = targets[:TOP_TARGETS]
        reach_sets = {p: set(self._stand_cells(world, agent, p, design)) for p, _ in top}
        costs = world.walk_costs(agent)
        candidates = {c for c in set().union(*reach_sets.values()) if c in costs}
        if candidates:

            def score(c):
                return (-sum(c in s for s in reach_sets.values()),
                        min(i for i, (p, _) in enumerate(top) if c in reach_sets[p]), costs[c], c)
            best = min(candidates, key=score)
            if best == world.positions[agent] and agent in world.cut_off_if_solid([top[0][0]]):
                # placing from here would wall me in: stand outside the house instead
                foot = self._footprint(design)
                outside = [c for c in candidates if (c[0], c[1]) not in foot]
                if outside:
                    best = min(outside, key=score)
            if best != world.positions[agent]:
                return "move_to", {"x": best[0], "y": best[1], "z": best[2]}
            # Here already, but placing would wall someone in: step back to the chest so I'm
            # not in their way out, and wait for them.
            return self._go_to_chest(world, agent) or WAIT
        return None

    def _stand_cells(self, world: World, agent: str, target: Pos, design: dict[Pos, str]) -> list[Pos]:
        """Walkable standing spots that reach `target`, preferring ones outside design cells."""
        cells = world.reachable_stand_cells(target, agent)
        clear = [c for c in cells if c not in design and (c[0], c[1], c[2] + 1) not in design]
        return clear or cells

    # ---- fetching and inspecting

    def _go_to_chest(self, world: World, agent: str) -> tuple[str, dict] | None:
        cells = world.reachable_stand_cells(world.chest_pos, agent)
        if not cells:
            return None
        c = cells[0]
        if c == world.positions[agent]:
            return None
        return "move_to", {"x": c[0], "y": c[1], "z": c[2]}

    def _fetch(self, world: World, agent: str, v: dict, mem: _Memory, missing: dict[Pos, str],
               inv: dict, design: dict[Pos, str]) -> tuple[str, dict] | None:
        room = v["inventory_cap"] - sum(inv.values())
        if room <= 0:
            return None
        idx = world.agents.index(agent)
        n = len(world.agents)
        order = sorted(missing.items(), key=lambda c: (c[1] == "door", c[0][2], c[0]))
        mine = Counter(t for (x, y, _), t in missing.items() if (x + y) % n == idx)
        allneed = Counter(missing.values())
        types = list(dict.fromkeys(t for _, t in order))
        chest = v["chest"]
        seen = (v["chest_last_seen"] or {}).get("counts")
        for t in types:
            want = (mine[t] or allneed[t]) - inv.get(t, 0)
            if want <= 0:
                continue
            if chest is not None:
                if chest.get(t, 0) > 0:
                    return "take", {"type": t, "n": min(want, room, chest[t])}
            elif seen is None or seen.get(t, 0) > 0:
                return self._go_to_chest(world, agent)
        return None

    def _inspect(self, world: World, agent: str, v: dict, mem: _Memory,
                 missing: dict[Pos, str]) -> tuple[str, dict] | None:
        """Walk to where a believed-missing cell can be seen, to refresh what I know."""
        now = v["t"]
        design = self.design(v)
        held_types = {t for t, n in v["inventory"].items() if n > 0 and t in missing.values()}
        others = [p for p, t in missing.items() if t not in held_types]
        # Holding blocks I can't place while only a few other cells look missing: probably my
        # picture is stale (built out of my sight). Look again a few times before giving up.
        holding = bool(held_types) and len(others) <= 4
        waiting = False
        for p in sorted(missing, key=lambda q: (q[2], q)):
            last = mem.inspected.get(p)
            if last is not None:
                if now - last < self.inspect_s:
                    waiting = waiting or (holding and p in others and mem.inspect_count.get(p, 0) < 3)
                    continue  # looked recently
                # Look again if I noticed a change since, or (a few times) if I'm holding
                # blocks I can't place: someone may have built it out of my sight.
                if last >= mem.last_change_t and not (holding and mem.inspect_count.get(p, 0) < 3):
                    continue
            cells = self._stand_cells(world, agent, p, design)
            if not cells:
                # no spot to look from right now (maybe someone is standing there)
                waiting = waiting or (holding and p in others and mem.inspect_count.get(p, 0) < 3)
                continue
            mem.inspected[p] = now
            mem.inspect_count[p] = mem.inspect_count.get(p, 0) + 1
            if cells[0] != world.positions[agent]:
                return "move_to", {"x": cells[0][0], "y": cells[0][1], "z": cells[0][2]}
        return WAIT if waiting else None


def run_simulated(world: World, policy: ScriptedBuilder | None = None,
                  think_s: tuple[float, float] = (2.0, 8.0), seed: int | None = None) -> World:
    """Discrete-event simulation on a SimClock: each agent's next call happens at
    now + a seeded uniform "think time" (or when it arrives, if later); the earliest goes
    next (ties by agent id)."""
    if not isinstance(world.clock, SimClock):
        raise ValueError("run_simulated needs a World built with a SimClock")
    policy = policy or ScriptedBuilder()
    rng = random.Random(world.config.seed if seed is None else seed)
    next_t = {a: rng.uniform(*think_s) for a in world.agents}
    while not world.done and next_t:
        t, agent = min((t, a) for a, t in next_t.items())
        world.clock.advance_to(t)
        if world.check_end():
            break
        if policy.step(world, agent):
            next_t[agent] = max(t + rng.uniform(*think_s), world.walking_until[agent])
        else:
            del next_t[agent]
    if not world.done:
        world.check_end()
    if not world.done:
        world.finish("aborted")  # every agent stopped but the episode did not end by itself
    return world
