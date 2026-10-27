"""World engine: grid, physical chest, embodied agents, rules, live timing, scoring and logging.

Agents act live and concurrently: any agent may call `act` at any time. Calls are applied
one at a time under a lock, and that commit order is the `event_id` order. Every tool call
is written to the episode log as one `event` record (see docs/log-format.md).
Scoring and chest sizing come from the task (blockswarm/tasks.py).
The engine is deterministic given the clock readings and has no network or LLM code.
"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Optional, Protocol

from blockswarm.blueprints import BLUEPRINTS, Pos
from blockswarm.clock import WallClock
from blockswarm.pathing import Terrain
from blockswarm.recorder import LOG_VERSION, Recorder
from blockswarm.tasks import make_task

BLOCK_TYPES = ("wood", "stone", "brick", "glass", "plank", "roof", "door")
SCARCITY = ("abundant", "exact", "short")
PAYOFFS = ("team", "individual", "mixed")
TASK_KINDS = ("open", "blueprint")
NEIGHBOURS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
# Default start offsets around the chest: diagonals first, then the sides (then wider rings).
START_RING = ((-1, 1), (1, 1), (-1, -1), (1, -1), (0, 1), (0, -1), (-1, 0), (1, 0))
FALL_S_PER_CELL = 0.1

# Messages of rejected action calls that consume nothing (not counted in actions_used).
# replay.py relies on these prefixes to recount actions_used from the log.
LIMIT_MSG = "action limit reached"
COOLDOWN_MSG = "cooldown"
WALKING_MSG = "walking"
AIRBORNE_MSG = "airborne"

# tool -> (cost, ordered arg spec). Arg kinds: int, block, agent, text.
TOOLS: dict[str, tuple[str, tuple[tuple[str, str], ...]]] = {
    "observe": ("free", ()),
    "move_to": ("action", (("x", "int"), ("y", "int"), ("z", "int"))),
    "jump": ("action", ()),
    "take": ("action", (("type", "block"), ("n", "int"))),
    "deposit": ("action", (("type", "block"), ("n", "int"))),
    "give": ("action", (("to", "agent"), ("type", "block"), ("n", "int"))),
    "place": ("action", (("x", "int"), ("y", "int"), ("z", "int"), ("type", "block"))),
    "remove": ("action", (("x", "int"), ("y", "int"), ("z", "int"))),
    "say": ("message", (("text", "text"),)),
    "whisper": ("message", (("to", "agent"), ("text", "text"))),
    "note": ("free", (("text", "text"),)),
    "declare_done": ("action", ()),
    "cancel_done": ("action", ()),
}

END_REASONS = ("all_done", "time_limit", "action_limit", "aborted", "budget", "error")
# Record types the engine writes itself; World.record() refuses them.
ENGINE_RECORD_TYPES = ("header", "event", "snapshot", "episode_end")
DIFF_KEYS = ("grid", "chest", "inventory", "agents", "votes")


class Clock(Protocol):
    kind: str

    def now(self) -> float: ...


@dataclass
class Config:
    width: int = 16
    depth: int = 16
    height: int = 8
    agents: list[str] = field(default_factory=lambda: ["agent_1", "agent_2", "agent_3"])
    blueprint: str = "house"
    scarcity: str = "exact"
    inventory_cap: int = 12
    max_seconds: float = 600
    max_actions_per_agent: int = 300
    action_cooldown_s: float = 0.0
    snapshot_interval_s: float = 10
    payoff: str = "team"
    mixed_alpha: float = 0.5
    extra_penalty: float = 0.1
    seed: int = 0
    max_message_chars: int = 1000
    chest_pos: tuple[int, int, int] = (13, 2, 0)
    start_positions: Optional[list[list[int]]] = None
    walk_speed: float = 4.317
    reach: float = 4.5
    eye_height: float = 1.62
    view_radius: float = 8
    jump_airtime: float = 0.6
    task: str = "open"
    brief: Optional[str] = None
    reference_design: str = "house"
    target_interior: Optional[int] = None        # None -> 4 x number of agents
    reach_line_of_sight: bool = True
    view_occlusion: bool = True
    models: Optional[dict[str, str]] = None      # agent id -> model name (logged as-is)
    run_meta: Optional[dict] = None              # free-form JSON metadata (run id, prompt version, ...)
    chest_scale: Optional[float] = None          # None -> agents / 3 (open task), 1.0 (blueprint)

    MAX_AGENTS = 16

    def __post_init__(self) -> None:
        self.agents = list(self.agents)
        self.chest_pos = tuple(self.chest_pos)  # type: ignore[assignment]
        if self.start_positions is not None:
            self.start_positions = [list(p) for p in self.start_positions]
        if self.task not in TASK_KINDS:
            raise ValueError(f"task must be one of {TASK_KINDS}, got {self.task!r}")
        if self.blueprint not in BLUEPRINTS:
            raise ValueError(f"unknown blueprint {self.blueprint!r}; known: {sorted(BLUEPRINTS)}")
        if self.reference_design not in BLUEPRINTS:
            raise ValueError(f"unknown reference_design {self.reference_design!r}")
        if self.scarcity not in SCARCITY:
            raise ValueError(f"scarcity must be one of {SCARCITY}, got {self.scarcity!r}")
        if self.payoff not in PAYOFFS:
            raise ValueError(f"payoff must be one of {PAYOFFS}, got {self.payoff!r}")
        if not 1 <= len(self.agents) <= self.MAX_AGENTS:
            raise ValueError(f"need 1 to {self.MAX_AGENTS} agents, got {len(self.agents)}")
        n = len(self.agents)
        # Defaults that scale with the team size; the resolved values are logged in the header.
        if self.target_interior is None:
            self.target_interior = 4 * n
        if self.chest_scale is None:
            # A blueprint is the same structure for any team size, so its chest isn't scaled.
            self.chest_scale = 1.0 if self.task == "blueprint" else n / 3
        if isinstance(self.chest_scale, bool) or not isinstance(self.chest_scale, (int, float)) \
                or not self.chest_scale > 0:
            raise ValueError(f"chest_scale must be a number > 0, got {self.chest_scale!r}")
        self.chest_scale = float(self.chest_scale)
        if len(set(self.agents)) != len(self.agents):
            raise ValueError(f"agent ids must be unique: {self.agents}")
        if min(self.width, self.depth, self.height) < 2:
            raise ValueError("grid dimensions must be >= 2")
        if self.inventory_cap < 1 or self.max_actions_per_agent < 1 or self.target_interior < 1:
            raise ValueError("inventory_cap, max_actions_per_agent and target_interior must be >= 1")
        if self.max_seconds <= 0 or self.snapshot_interval_s <= 0 or self.action_cooldown_s < 0:
            raise ValueError("max_seconds and snapshot_interval_s must be > 0, action_cooldown_s >= 0")
        if self.walk_speed <= 0 or self.reach <= 0 or self.view_radius <= 0 or self.eye_height < 0:
            raise ValueError("walk_speed, reach and view_radius must be > 0, eye_height >= 0")
        if self.jump_airtime <= 0:
            raise ValueError("jump_airtime must be > 0")
        if not 0.0 <= self.mixed_alpha <= 1.0:
            raise ValueError("mixed_alpha must be in [0, 1]")
        if len(self.chest_pos) != 3 or not self.in_bounds(self.chest_pos):
            raise ValueError(f"chest_pos {self.chest_pos} is outside the grid")
        if self.task == "blueprint":
            for (x, y, z), t in BLUEPRINTS[self.blueprint].cells.items():
                if not self.in_bounds((x, y, z)):
                    raise ValueError(f"blueprint {self.blueprint!r} cell {(x, y, z)} is outside the grid")
                if t not in BLOCK_TYPES:
                    raise ValueError(f"blueprint {self.blueprint!r} uses unknown block type {t!r}")
            if self.chest_pos in BLUEPRINTS[self.blueprint].cells:
                raise ValueError(f"chest_pos {self.chest_pos} overlaps the blueprint")
        if self.start_positions is not None:
            if len(self.start_positions) != len(self.agents):
                raise ValueError("start_positions needs one position per agent")
            bodies: set = set()
            for p in self.start_positions:
                x, y, z = p
                feet, head = (x, y, z), (x, y, z + 1)
                if not (self.in_bounds(feet) and self.in_bounds(head)):
                    raise ValueError(f"start position {p} is outside the grid")
                if self.chest_pos in (feet, head) or not (z == 0 or (x, y, z - 1) == self.chest_pos):
                    raise ValueError(f"start position {p} is not standable")
                if feet in bodies or head in bodies:
                    raise ValueError(f"start positions overlap at {p}")
                bodies |= {feet, head}
        if self.models is not None:
            if not isinstance(self.models, dict):
                raise ValueError("models must be a dict of agent id -> model name")
            self.models = dict(self.models)
            unknown = sorted(set(self.models) - set(self.agents))
            if unknown:
                raise ValueError(f"models has unknown agent ids {unknown}; agents: {self.agents}")
            if not all(isinstance(m, str) and m for m in self.models.values()):
                raise ValueError("models values must be non-empty strings")
        if self.run_meta is not None:
            if not isinstance(self.run_meta, dict):
                raise ValueError("run_meta must be a dict")
            try:
                json.dumps(self.run_meta)
            except (TypeError, ValueError) as e:
                raise ValueError(f"run_meta must be JSON-serializable: {e}") from None

    def in_bounds(self, c: Iterable[int]) -> bool:
        x, y, z = c
        return 0 <= x < self.width and 0 <= y < self.depth and 0 <= z < self.height


@dataclass
class ToolResult:
    ok: bool
    message: str
    data: dict | None = None


@dataclass
class Block:
    type: str
    placed_by: str
    placed_at: float

    def to_json(self) -> dict:
        return {"type": self.type, "placed_by": self.placed_by, "placed_at": self.placed_at}


class _Fail(Exception):
    """Raised inside a tool handler to reject the action with a message."""


def _jsonable(value: Any) -> Any:
    """Coerce tool args to something json.dumps accepts (args come from model output)."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return json.loads(json.dumps(value, default=repr))


def ray_cells(start: tuple[float, float, float], target: Pos) -> list[Pos]:
    """Cells crossed by the segment from `start` to the centre of `target` (Amanatides–Woo),
    excluding the start cell and the target cell. Where the ray passes exactly through an
    edge or corner it steps diagonally (the cells beside the crack are not crossed)."""
    end = (target[0] + 0.5, target[1] + 0.5, target[2] + 0.5)
    cell = [math.floor(c) for c in start]
    step, t_max, t_delta = [0, 0, 0], [math.inf] * 3, [math.inf] * 3
    for i in range(3):
        d = end[i] - start[i]
        if d > 0:
            step[i], t_max[i], t_delta[i] = 1, (cell[i] + 1 - start[i]) / d, 1 / d
        elif d < 0:
            step[i], t_max[i], t_delta[i] = -1, (start[i] - cell[i]) / -d, 1 / -d
    out: list[Pos] = []
    for _ in range(1000):
        if tuple(cell) == tuple(target):
            return out
        m = min(t_max)
        if m > 1.0 + 1e-9:
            return out
        for i in range(3):
            if t_max[i] - m <= 1e-9:
                cell[i] += step[i]
                t_max[i] += t_delta[i]
        if tuple(cell) != tuple(target):
            out.append((cell[0], cell[1], cell[2]))
    return out


def scale_count(n: int, scale: float) -> int:
    """Chest count for one block type scaled by `chest_scale`: rounded up, zero stays zero."""
    return math.ceil(round(n * scale, 9)) if n > 0 else 0


class World:
    """One live, embodied episode. Any agent may call `act` at any time; each call is atomic."""

    def __init__(self, config: Config, recorder: Recorder | None = None, episode_id: str | None = None,
                 clock: Clock | None = None):
        self.config = config
        self.recorder = recorder if recorder is not None else Recorder()
        self.clock: Clock = clock if clock is not None else WallClock()
        name = config.blueprint if config.task == "blueprint" else config.reference_design
        self.episode_id = episode_id or f"{config.task}-{name}-{config.scarcity}-seed{config.seed}"
        self.task = make_task(config.task, brief=config.brief, blueprint=config.blueprint,
                              reference_design=config.reference_design,
                              target_interior=config.target_interior, extra_penalty=config.extra_penalty)
        self.agents: list[str] = list(config.agents)
        self.chest_pos: Pos = tuple(config.chest_pos)  # type: ignore[assignment]
        self.blocks: dict[Pos, Block] = {}
        base = self.task.chest_counts(config.scarcity)
        self.chest: dict[str, int] = {t: scale_count(int(base.get(t, 0)), config.chest_scale)  # type: ignore[arg-type]
                                      for t in BLOCK_TYPES}
        self.inventories: dict[str, dict[str, int]] = {a: {} for a in self.agents}
        self.notes: dict[str, list[str]] = {a: [] for a in self.agents}
        self.messages: list[dict] = []
        self.done_votes: list[str] = []
        self.actions_used: dict[str, int] = {a: 0 for a in self.agents}
        self.positions: dict[str, Pos] = dict(zip(self.agents, self._start_positions()))
        self.walking_until: dict[str, float] = {a: 0.0 for a in self.agents}
        self.airborne_until: dict[str, float] = {a: 0.0 for a in self.agents}
        self._lifted: set[str] = set()  # airborne agents already lifted by a block this jump
        self._chest_seen: dict[str, dict | None] = {a: None for a in self.agents}
        self._last_action_t: dict[str, float | None] = {a: None for a in self.agents}
        self._lock = threading.RLock()
        self._changed = threading.Condition(self._lock)  # notified on every committed event and at the end
        self._t = 0.0
        self._next_event_id = 0
        self._snapshot_seq = 0
        self._next_snapshot_t = 0.0
        self._scores_cache: dict | None = None
        self._done = False
        self._end_reason: str | None = None
        self.recorder.write({
            "type": "header",
            "log_version": LOG_VERSION,
            "episode_id": self.episode_id,
            "config": json.loads(json.dumps(asdict(config))),
            "task": self.task.header(),
            "clock": getattr(self.clock, "kind", "wall"),
            "grid": {"width": config.width, "depth": config.depth, "height": config.height},
            "agents": list(self.agents),
            "chest_pos": list(self.chest_pos),
            "positions": {a: list(p) for a, p in self.positions.items()},
            "block_types": list(BLOCK_TYPES),
            "chest": dict(self.chest),
            "inventories": {a: {} for a in self.agents},
        })

    def _start_positions(self) -> list[Pos]:
        c = self.config
        if c.start_positions is not None:
            return [tuple(p) for p in c.start_positions]  # type: ignore[misc]
        cx, cy, cz = self.chest_pos
        z = 0
        out: list[Pos] = []
        offsets = list(START_RING)
        for r in range(2, max(c.width, c.depth)):
            ring = [(dx, dy) for dy in range(r, -r - 1, -1) for dx in range(-r, r + 1)
                    if max(abs(dx), abs(dy)) == r]
            offsets += ring
        footprint = {(x, y) for x, y, _ in self.task_cells()}
        for dx, dy in offsets:
            p = (cx + dx, cy + dy, z)
            if (c.in_bounds(p) and c.in_bounds((p[0], p[1], 1)) and (p[0], p[1]) not in footprint
                    and (p[0], p[1]) != (cx, cy)):
                out.append(p)
            if len(out) == len(self.agents):
                return out
        raise ValueError("could not find default start positions; set start_positions")

    def task_cells(self) -> list[tuple[int, int, int]]:
        """Blueprint cells in blueprint mode (empty in open mode)."""
        bp = self.task.header().get("blueprint")
        return [(x, y, z) for x, y, z, _ in bp["cells"]] if bp else []

    # ------------------------------------------------------------------ time and ending

    @property
    def done(self) -> bool:
        return self._done

    @property
    def end_reason(self) -> str | None:
        return self._end_reason

    def now(self) -> float:
        """Episode time in seconds, rounded to ms and never decreasing."""
        with self._lock:
            self._t = max(self._t, round(self.clock.now(), 3))
            return self._t

    def check_end(self) -> bool:
        """End the episode if an end condition holds. Returns `done`."""
        with self._lock:
            if self._done:
                return True
            if all(a in self.done_votes for a in self.agents):
                self.finish("all_done")
            elif self.now() >= self.config.max_seconds:
                self.finish("time_limit")
            elif all(n >= self.config.max_actions_per_agent for n in self.actions_used.values()):
                self.finish("action_limit")
            return self._done

    def finish(self, reason: str) -> None:
        """End the episode: final snapshot, then `episode_end`. Only the first call has effect.

        `reason` is one of END_REASONS: all_done, time_limit, action_limit (set by the engine),
        aborted, budget (the orchestrator's spending limit) or error (an orchestrator failure).
        """
        if reason not in END_REASONS:
            raise ValueError(f"unknown end reason {reason!r}; known: {END_REASONS}")
        with self._lock:
            if self._done:
                return
            self._done = True
            self._end_reason = reason
            t = self.now()
            self._write_snapshot(t)
            self.recorder.write({"type": "episode_end", "t": t, "reason": reason, **self.scores(),
                                 "actions_used": dict(self.actions_used)})
            self._changed.notify_all()

    # ------------------------------------------------------------------ orchestrator hooks

    def record(self, record: dict) -> None:
        """Write an extra log record (e.g. `{"type": "model_call", ...}`) in file order with events.

        Thread-safe (same lock as events). Adds `t` (current episode time) if missing. Does
        nothing once the episode has ended. Engine record types (header, event, snapshot,
        episode_end) are refused with ValueError, since they would break replay.
        """
        if not isinstance(record, dict) or not isinstance(record.get("type"), str):
            raise ValueError("record must be a dict with a string 'type'")
        if record["type"] in ENGINE_RECORD_TYPES:
            raise ValueError(f"record type {record['type']!r} is reserved for the engine")
        with self._lock:
            if self._done:
                return
            rec = dict(record)
            if "t" not in rec:
                rec["t"] = self.now()
            self.recorder.write(rec)

    def last_event_id(self) -> int:
        """Id of the latest committed event (-1 if none yet)."""
        with self._lock:
            return self._next_event_id - 1

    def wait_for_change(self, agent: str, since_event_id: int, timeout_s: float) -> int:
        """Block until an event with id > `since_event_id` exists, `timeout_s` (wall seconds)
        passes, or the episode ends; return the latest event id.

        Any agent's event counts (including `agent`'s own). With a simulated clock nothing can
        happen while the caller blocks, so this returns immediately. With a wall clock, the wait
        is also cut short at `max_seconds`, when the episode is ended for the time limit.
        """
        if agent not in self.inventories:
            raise ValueError(f"unknown agent {agent!r}")
        with self._changed:
            if getattr(self.clock, "kind", "wall") == "simulated":
                return self._next_event_id - 1
            deadline = time.monotonic() + max(0.0, float(timeout_s))
            while not self._done and self._next_event_id - 1 <= since_event_id:
                if self.check_end():  # e.g. the time limit passed while waiting
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                to_limit = max(0.0, self.config.max_seconds - self.clock.now())
                self._changed.wait(min(remaining, to_limit + 0.002))
            return self._next_event_id - 1

    # ------------------------------------------------------------------ state

    def state(self) -> dict:
        """Blocks, chest, inventories, actions, votes, positions and walkers in log format."""
        with self._lock:
            t = self._t
            return {
                "blocks": [[x, y, z, b.type, b.placed_by, b.placed_at]
                           for (x, y, z), b in sorted(self.blocks.items())],
                "chest": dict(self.chest),
                "inventories": {a: dict(sorted(self.inventories[a].items())) for a in self.agents},
                "actions_used": dict(self.actions_used),
                "done_votes": list(self.done_votes),
                "positions": {a: list(p) for a, p in self.positions.items()},
                "walking_until": {a: w for a, w in self.walking_until.items() if w > t},
            }

    def snapshot(self) -> dict:
        """Current full state as a `snapshot` record (not written to the log)."""
        with self._lock:
            return self._snapshot_record(self.now())

    def _snapshot_record(self, t: float) -> dict:
        return {"type": "snapshot", "seq": self._snapshot_seq, "t": t,
                "after_event_id": self._next_event_id - 1, **self.scores(), **self.state()}

    def _write_snapshot(self, t: float) -> None:
        self.recorder.write(self._snapshot_record(t))
        self._snapshot_seq += 1
        interval = self.config.snapshot_interval_s
        self._next_snapshot_t = (math.floor(t / interval) + 1) * interval

    def scores(self) -> dict:
        """team_score, individual_scores and score_details from the task (cached per grid)."""
        with self._lock:
            if self._scores_cache is None:
                c = self.config
                s = self.task.score({p: b.to_json() for p, b in self.blocks.items()}, self.agents,
                                    (c.width, c.depth, c.height), [self.chest_pos])
                self._scores_cache = {"team_score": s["team_score"],
                                      "individual_scores": s["individual_scores"],
                                      "score_details": s["score_details"]}
            return json.loads(json.dumps(self._scores_cache))

    def held(self, agent: str) -> int:
        return sum(self.inventories[agent].values())

    # ------------------------------------------------------------------ geometry

    def solid(self, c: Pos) -> bool:
        return c in self.blocks or c == self.chest_pos

    def body_cells(self, exclude: str | None = None) -> dict[Pos, str]:
        """Cells occupied by agents' bodies (an airborne agent's feet cell is free)."""
        out: dict[Pos, str] = {}
        for a, (x, y, z) in self.positions.items():
            if a != exclude:
                if not self._feet_free(a):
                    out[(x, y, z)] = a
                out[(x, y, z + 1)] = a
        return out

    def is_airborne(self, agent: str, t: float | None = None) -> bool:
        return self.airborne_until[agent] > (self._t if t is None else t)

    def _feet_free(self, agent: str) -> bool:
        return self.is_airborne(agent) and agent not in self._lifted

    def terrain(self, agent: str | None = None, extra_solid: Iterable[Pos] = ()) -> Terrain:
        """Standing/walking rules with `agent`'s own body ignored."""
        bodies = self.body_cells(exclude=agent)
        extra = set(extra_solid)
        c = self.config
        return Terrain(c.width, c.depth, c.height,
                       solid=lambda p: p in self.blocks or p == self.chest_pos or p in extra,
                       occupied=lambda p: p in bodies)

    def standable(self, pos: Pos, agent: str | None = None) -> bool:
        with self._lock:
            return self.terrain(agent).standable(tuple(pos))  # type: ignore[arg-type]

    def eye(self, pos: Pos) -> tuple[float, float, float]:
        return (pos[0] + 0.5, pos[1] + 0.5, pos[2] + self.config.eye_height)

    def distance(self, from_pos: Pos, target: Pos) -> float:
        e = self.eye(from_pos)
        return math.dist(e, (target[0] + 0.5, target[1] + 0.5, target[2] + 0.5))

    def line_of_sight(self, from_pos: Pos, target: Pos, extra_solid: Iterable[Pos] = ()) -> bool:
        """Clear ray from the eye at `from_pos` to the centre of `target` (target itself may be solid)."""
        extra = set(extra_solid)
        return not any(self.solid(c) or c in extra for c in ray_cells(self.eye(from_pos), target))

    def can_reach(self, from_pos: Pos, target: Pos, extra_solid: Iterable[Pos] = ()) -> bool:
        """Within reach of the eye at `from_pos` (and in line of sight, if that rule is on)."""
        with self._lock:
            if self.distance(from_pos, target) > self.config.reach:
                return False
            return not self.config.reach_line_of_sight or self.line_of_sight(from_pos, target, extra_solid)

    def reachable_stand_cells(self, target: Pos, agent: str | None = None) -> list[Pos]:
        """Standable cells from which `target` is within reach (and line of sight).
        With `agent`, its own body is ignored and only cells it can walk to are returned,
        sorted by walk length; otherwise sorted by distance to the target."""
        with self._lock:
            tx, ty, tz = target
            r = int(math.ceil(self.config.reach)) + 1
            ter = self.terrain(agent)
            cells = []
            for x in range(tx - r, tx + r + 1):
                for y in range(ty - r, ty + r + 1):
                    for z in range(max(0, tz - r - 1), tz + r + 1):
                        p = (x, y, z)
                        if ter.standable(p) and self.can_reach(p, target):
                            cells.append(p)
            if agent is None:
                return sorted(cells, key=lambda p: (self.distance(p, target), p))
            costs = self.walk_costs(agent)
            return sorted((p for p in cells if p in costs), key=lambda p: (costs[p], p))

    def walk_costs(self, agent: str, extra_solid: Iterable[Pos] = ()) -> dict[Pos, float]:
        """Walk length from the agent's position to every cell it can reach."""
        with self._lock:
            return self.terrain(agent, extra_solid).costs_from(self.positions[agent])[0]

    def find_path(self, agent: str, dest: Pos) -> tuple[list[Pos], float] | None:
        with self._lock:
            ter = self.terrain(agent)
            if not ter.standable(tuple(dest)):  # type: ignore[arg-type]
                return None
            return ter.path(self.positions[agent], tuple(dest))  # type: ignore[arg-type]

    def cut_off_if_solid(self, cells: Iterable[Pos]) -> list[str]:
        """Agents that could no longer walk to a spot within reach of the chest if `cells`
        became solid (other agents' bodies are ignored, since they move). Internal helper for
        the scripted bot and tests, not an agent tool."""
        with self._lock:
            extra = set(cells)
            c = self.config
            ter = Terrain(c.width, c.depth, c.height,
                          solid=lambda p: p in self.blocks or p == self.chest_pos or p in extra,
                          occupied=lambda p: False)
            r = int(math.ceil(c.reach)) + 1
            cx, cy, cz = self.chest_pos
            goals = {(x, y, z) for x in range(cx - r, cx + r + 1) for y in range(cy - r, cy + r + 1)
                     for z in range(max(0, cz - r - 1), cz + r + 1)
                     if ter.standable((x, y, z)) and self.can_reach((x, y, z), self.chest_pos, extra)}
            out = []
            for a in self.agents:
                x, y, z = self.positions[a]
                if (x, y, z) in extra or (x, y, z + 1) in extra:
                    out.append(a)
                    continue
                costs = ter.costs_from((x, y, z))[0]
                if not any(g in costs for g in goals):
                    out.append(a)
            return out

    def is_walking(self, agent: str, t: float | None = None) -> bool:
        return self.walking_until[agent] > (self._t if t is None else t)

    def _sees_chest(self, agent: str) -> bool:
        return not self.is_walking(agent) and self.can_reach(self.positions[agent], self.chest_pos)

    def visible(self, from_pos: Pos, target: Pos) -> bool:
        """Whether `target` shows up in a view from `from_pos` (radius and occlusion rules)."""
        if self.distance(from_pos, target) > self.config.view_radius:
            return False
        return not self.config.view_occlusion or self.line_of_sight(from_pos, target)

    # ------------------------------------------------------------------ view

    def view(self, agent: str, since_event_id: int | None = None) -> dict:
        """Everything `agent` may see, JSON-serializable, for building prompts.

        Blocks and agents within view_radius of the eyes (and in sight, with view_occlusion);
        chest counts only while within reach of the chest, else the counts last seen.
        Messages are public ones, private ones to/from `agent`, and system notices about
        finished votes (channel "system", from/to null); with `since_event_id`, only those with
        a larger event_id. `last_event_id` can be passed back next time.
        """
        if agent not in self.inventories:
            raise ValueError(f"unknown agent {agent!r}")
        with self._lock:
            t = self.now()
            pos = self.positions[agent]
            last = self._last_action_t[agent]
            cooldown_until = 0.0 if last is None else last + self.config.action_cooldown_s
            used = self.actions_used[agent]
            if self._sees_chest(agent):
                self._chest_seen[agent] = {"counts": dict(self.chest), "t": t}
                chest = dict(self.chest)
            else:
                chest = None
            blocks = [{"pos": [x, y, z], "offset": [x - pos[0], y - pos[1], z - pos[2]],
                       "type": b.type, "placed_by": b.placed_by, "placed_at": b.placed_at}
                      for (x, y, z), b in sorted(self.blocks.items()) if self.visible(pos, (x, y, z))]
            others = []
            for a in self.agents:
                p = self.positions[a]
                if a != agent and (self.visible(pos, p) or self.visible(pos, (p[0], p[1], p[2] + 1))):
                    others.append({"id": a, "pos": list(p), "walking": self.is_walking(a, t),
                                   "airborne": self.is_airborne(a, t)})
            walking = self.walking_until[agent]
            airborne = self.airborne_until[agent]
            return {
                "t": t,
                "max_seconds": self.config.max_seconds,
                "done": self._done,
                "you": agent,
                "agents": list(self.agents),
                "payoff": self.config.payoff,
                "task": self.task.header(),
                "position": list(pos),
                "eye": list(self.eye(pos)),
                "walking_until": walking if walking > t else None,
                "airborne_until": airborne if airborne > t else None,
                "reach": self.config.reach,
                "view_radius": self.config.view_radius,
                "inventory": dict(sorted(self.inventories[agent].items())),
                "inventory_cap": self.config.inventory_cap,
                "actions_used": used,
                "actions_remaining": max(0, self.config.max_actions_per_agent - used),
                "next_action_at": max(t, cooldown_until, walking),
                "chest_pos": list(self.chest_pos),
                "chest": chest,
                "chest_last_seen": json.loads(json.dumps(self._chest_seen[agent])),
                "grid": {"width": self.config.width, "depth": self.config.depth,
                         "height": self.config.height},
                "blocks": blocks,
                "agents_visible": others,
                "messages": [dict(m) for m in self.messages
                             if (m["channel"] in ("public", "system") or agent in (m["from"], m["to"]))
                             and (since_event_id is None or m["event_id"] > since_event_id)],
                "last_event_id": self._next_event_id - 1,
                "notes": list(self.notes[agent]),
                "done_votes": list(self.done_votes),
                "scores": self.scores(),
            }

    # ------------------------------------------------------------------ actions

    def act(self, agent: str, tool: str, **args: Any) -> ToolResult:
        """Apply one tool call atomically and write one `event`.

        After the episode has ended this returns a failed result and logs nothing.
        Raises ValueError only for an unknown agent id (a caller bug, not an agent mistake).
        """
        if agent not in self.inventories:
            raise ValueError(f"unknown agent {agent!r}")
        with self._lock:
            if self.check_end():
                return ToolResult(False, f"episode is over ({self._end_reason})")
            t = self.now()
            if t >= self._next_snapshot_t:
                self._write_snapshot(t)
            result = self._apply(agent, tool, args, t)
            if self._sees_chest(agent):
                self._chest_seen[agent] = {"counts": dict(self.chest), "t": t}
            self.check_end()
            return result

    def _apply(self, agent: str, tool: str, args: dict, t: float) -> ToolResult:
        cost, spec = TOOLS.get(tool, ("action", ()))
        diff: dict = {"grid": [], "chest": {}, "inventory": {}, "agents": {}, "votes": {}}
        chat: dict | None = None
        data: dict | None = None
        ok = False
        last = self._last_action_t[agent]
        if cost == "action" and self.actions_used[agent] >= self.config.max_actions_per_agent:
            message = f"{LIMIT_MSG}: you have used all {self.config.max_actions_per_agent} actions"
        elif cost == "action" and self.walking_until[agent] > t:
            message = f"{WALKING_MSG}: you arrive at t={self.walking_until[agent]:.3f} (now {t:.3f})"
        elif tool in ("move_to", "jump") and self.airborne_until[agent] > t:
            message = f"{AIRBORNE_MSG}: you land at t={self.airborne_until[agent]:.3f} (now {t:.3f})"
        elif cost == "action" and last is not None and t - last < self.config.action_cooldown_s:
            message = (f"{COOLDOWN_MSG}: next action allowed at t={last + self.config.action_cooldown_s:.3f}"
                       f" (now {t:.3f})")
        else:
            if cost == "action":
                self.actions_used[agent] += 1
                self._last_action_t[agent] = t
            work: dict = {"grid": [], "chest": {}, "inventory": {}, "agents": {}, "votes": {}}
            votes_before = sorted(self.done_votes)
            try:
                if tool not in TOOLS:
                    raise _Fail(f"unknown tool {tool!r}; tools: {', '.join(TOOLS)}")
                clean = self._validate_args(agent, tool, spec, args)
                handler: Callable[..., tuple[str, dict | None, dict | None]] = getattr(self, f"_tool_{tool}")
                message, chat, data = handler(agent, work, t, **clean)
                ok, diff = True, work
                if work["grid"]:
                    self._scores_cache = None
                    if self.done_votes:
                        # any change to the build clears every finished vote
                        self.done_votes.clear()
                        self._notice(t, f"{agent} changed the build, so all finished votes were cleared")
                if sorted(self.done_votes) != votes_before:
                    work["votes"] = {"before": votes_before, "after": sorted(self.done_votes)}
            except _Fail as e:
                message, chat, data = str(e), None, None

        self.recorder.write({
            "type": "event",
            "event_id": self._next_event_id,
            "t": t,
            "agent": agent,
            "tool": tool,
            "args": _jsonable(dict(args)),
            "cost": cost,
            "ok": ok,
            "message": message,
            "diff": diff,
            "chat": chat,
        })
        self._next_event_id += 1
        self._changed.notify_all()
        return ToolResult(ok, message, data)

    def _validate_args(self, agent: str, tool: str, spec: tuple, args: dict) -> dict:
        names = [n for n, _ in spec]
        unknown = sorted(set(args) - set(names))
        missing = [n for n in names if n not in args]
        if unknown or missing:
            sig = f"{tool}({', '.join(names)})"
            raise _Fail(f"bad arguments for {sig}: "
                        + "; ".join(([f"missing {missing}"] if missing else [])
                                    + ([f"unexpected {unknown}"] if unknown else [])))
        clean = {}
        for name, kind in spec:
            v = args[name]
            if kind == "int":
                if not isinstance(v, int) or isinstance(v, bool):
                    raise _Fail(f"{name} must be an integer, got {v!r}")
            elif kind == "block":
                if v not in BLOCK_TYPES:
                    raise _Fail(f"unknown block type {v!r}; types: {', '.join(BLOCK_TYPES)}")
            elif kind == "agent":
                if v not in self.inventories:
                    raise _Fail(f"unknown agent {v!r}")
                if v == agent:
                    raise _Fail("cannot target yourself")
            elif kind == "text":
                if not isinstance(v, str) or not v.strip():
                    raise _Fail(f"{name} must be a non-empty string")
                if len(v) > self.config.max_message_chars:
                    raise _Fail(f"{name} is {len(v)} chars; max is {self.config.max_message_chars}")
            clean[name] = v
        if "n" in clean and clean["n"] < 1:
            raise _Fail(f"n must be >= 1, got {clean['n']}")
        return clean

    # Each handler validates fully before mutating, records deltas in `work`, and returns
    # (message, chat, data). Raising _Fail leaves the world unchanged.

    def _inv_add(self, work: dict, agent: str, btype: str, n: int) -> None:
        inv = self.inventories[agent]
        inv[btype] = inv.get(btype, 0) + n
        if inv[btype] == 0:
            del inv[btype]
        d = work["inventory"].setdefault(agent, {})
        d[btype] = d.get(btype, 0) + n

    def _chest_add(self, work: dict, btype: str, n: int) -> None:
        self.chest[btype] = self.chest.get(btype, 0) + n
        work["chest"][btype] = work["chest"].get(btype, 0) + n

    def _move(self, work: dict, agent: str, path: list[Pos], depart: float, arrive: float,
              kind: str = "walk") -> None:
        """Record a position change. kind: walk (busy until arrive), fall, jump or lift."""
        entry = {"from": list(path[0]), "to": list(path[-1]), "path": [list(p) for p in path],
                 "depart_t": depart, "arrive_t": arrive}
        if kind in ("fall", "jump"):
            entry[kind] = True
        work["agents"][agent] = entry
        self.positions[agent] = path[-1]
        if kind in ("walk", "fall"):
            self.walking_until[agent] = arrive
        if kind == "fall":
            self.airborne_until[agent] = 0.0
            self._lifted.discard(agent)
        elif kind == "jump":
            self.airborne_until[agent] = arrive
            self._lifted.discard(agent)
        elif kind == "lift":
            self._lifted.add(agent)

    def _need_held(self, agent: str, btype: str, n: int) -> None:
        have = self.inventories[agent].get(btype, 0)
        if have < n:
            raise _Fail(f"you have {have} {btype}, need {n}")

    def _need_room(self, agent: str, n: int, who: str = "you") -> None:
        free = self.config.inventory_cap - self.held(agent)
        if n > free:
            raise _Fail(f"inventory cap {self.config.inventory_cap}: {who} can hold only {free} more")

    def _check_pos(self, x: int, y: int, z: int) -> Pos:
        c = self.config
        if not c.in_bounds((x, y, z)):
            raise _Fail(f"({x}, {y}, {z}) is out of bounds "
                        f"(x<{c.width}, y<{c.depth}, z<{c.height})")
        return (x, y, z)

    def _need_reach(self, agent: str, target: Pos, what: str) -> None:
        pos = self.positions[agent]
        d = self.distance(pos, target)
        if d > self.config.reach:
            raise _Fail(f"out of reach: {what} at {target} is {d:.2f} from your eyes (reach {self.config.reach})")
        if self.config.reach_line_of_sight and not self.line_of_sight(pos, target):
            blocker = next(c for c in ray_cells(self.eye(pos), target) if self.solid(c))
            raise _Fail(f"no line of sight to {what} at {target}: blocked at {blocker}")

    def supported(self, pos: Pos) -> bool:
        x, y, z = pos
        return z == 0 or any(self.solid((x + dx, y + dy, z + dz)) for dx, dy, dz in NEIGHBOURS)

    def _tool_observe(self, agent: str, work: dict, t: float):
        return "observed", None, self.view(agent)

    def _tool_move_to(self, agent: str, work: dict, t: float, x: int, y: int, z: int):
        dest = self._check_pos(x, y, z)
        ter = self.terrain(agent)
        if not ter.standable(dest):
            raise _Fail(f"{dest} is not standable (needs an empty 2-high space on the ground, "
                        "a block or the chest, not occupied by an agent)")
        found = ter.path(self.positions[agent], dest)
        if found is None:
            raise _Fail(f"no path from {self.positions[agent]} to {dest}")
        path, length = found
        arrive = round(t + length / self.config.walk_speed, 3)
        self._move(work, agent, path, t, arrive)
        return f"walking to {dest}: {length:.2f} blocks, arrive at t={arrive:.3f}", None, None

    def _tool_jump(self, agent: str, work: dict, t: float):
        pos = self.positions[agent]
        land = round(t + self.config.jump_airtime, 3)
        self._move(work, agent, [pos], t, land, kind="jump")
        return f"jumped: airborne until t={land:.3f}", None, None

    def _tool_take(self, agent: str, work: dict, t: float, type: str, n: int):
        self._need_reach(agent, self.chest_pos, "the chest")
        if self.chest.get(type, 0) < n:
            raise _Fail(f"chest has {self.chest.get(type, 0)} {type}, asked for {n}")
        self._need_room(agent, n)
        self._chest_add(work, type, -n)
        self._inv_add(work, agent, type, n)
        return f"took {n} {type} from the chest", None, None

    def _tool_deposit(self, agent: str, work: dict, t: float, type: str, n: int):
        self._need_reach(agent, self.chest_pos, "the chest")
        self._need_held(agent, type, n)
        self._inv_add(work, agent, type, -n)
        self._chest_add(work, type, n)
        return f"deposited {n} {type} into the chest", None, None

    def _tool_give(self, agent: str, work: dict, t: float, to: str, type: str, n: int):
        tx, ty, tz = self.positions[to]
        cells = [(tx, ty, tz), (tx, ty, tz + 1)]
        target = min(cells, key=lambda c: (self.distance(self.positions[agent], c), c))
        self._need_reach(agent, target, to)
        self._need_held(agent, type, n)
        self._need_room(to, n, who=to)
        self._inv_add(work, agent, type, -n)
        self._inv_add(work, to, type, n)
        return f"gave {n} {type} to {to}", None, None

    def _tool_place(self, agent: str, work: dict, t: float, x: int, y: int, z: int, type: str):
        pos = self._check_pos(x, y, z)
        if pos == self.chest_pos:
            raise _Fail(f"{pos} is the chest")
        if pos in self.blocks:
            b = self.blocks[pos]
            raise _Fail(f"{pos} is occupied by {b.type} (placed by {b.placed_by})")
        bodies = self.body_cells()
        if pos in bodies:
            raise _Fail(f"{pos} is occupied by {bodies[pos]}'s body")
        jumper = next((a for a in self.agents if self.positions[a] == pos and self._feet_free(a)), None)
        if jumper is not None:
            x, y, z = pos
            above = (x, y, z + 2)
            if not self.config.in_bounds(above) or self.solid(above) or above in bodies:
                raise _Fail(f"{pos} is under airborne {jumper}, and the cell above their head {above} "
                            "is not free")
        self._need_reach(agent, pos, "cell")
        if not self.supported(pos):
            raise _Fail(f"{pos} is unsupported: needs z=0 or a face-adjacent block")
        self._need_held(agent, type, 1)
        self._inv_add(work, agent, type, -1)
        block = Block(type, agent, t)
        self.blocks[pos] = block
        work["grid"].append({"pos": list(pos), "before": None, "after": block.to_json()})
        if jumper is not None:
            up = (pos[0], pos[1], pos[2] + 1)
            self._move(work, jumper, [pos, up], t, self.airborne_until[jumper], kind="lift")
            return f"placed {type} at {pos}; {jumper} lands on it at {up}", None, None
        return f"placed {type} at {pos}", None, None

    def _tool_remove(self, agent: str, work: dict, t: float, x: int, y: int, z: int):
        pos = self._check_pos(x, y, z)
        if pos == self.chest_pos:
            raise _Fail("the chest cannot be removed")
        block = self.blocks.get(pos)
        if block is None:
            raise _Fail(f"{pos} is empty")
        self._need_reach(agent, pos, "block")
        self._need_room(agent, 1)
        del self.blocks[pos]
        self._inv_add(work, agent, block.type, 1)
        work["grid"].append({"pos": list(pos), "before": block.to_json(), "after": None})
        self._apply_falls(work, t)
        owner = "your" if block.placed_by == agent else f"{block.placed_by}'s"
        return f"removed {owner} {block.type} at {pos}", None, None

    def _apply_falls(self, work: dict, t: float) -> None:
        """Agents left without support drop straight down to the next standable cell."""
        for a in self.agents:
            x, y, z = self.positions[a]
            if z == 0 or self.solid((x, y, z - 1)):
                continue
            ter = self.terrain(a)
            nz = z - 1
            while nz > 0 and not ter.standable((x, y, nz)):
                nz -= 1
            path = [(x, y, z), (x, y, nz)]
            self._move(work, a, path, t, round(t + FALL_S_PER_CELL * (z - nz), 3), kind="fall")

    def _post(self, agent: str, to: str | None, text: str, t: float) -> dict:
        chat = {"channel": "public" if to is None else "private", "from": agent, "to": to, "text": text}
        self.messages.append({"event_id": self._next_event_id, "t": t, **chat})
        return chat

    def _tool_say(self, agent: str, work: dict, t: float, text: str):
        return "said (public)", self._post(agent, None, text, t), None

    def _tool_whisper(self, agent: str, work: dict, t: float, to: str, text: str):
        return f"whispered to {to}", self._post(agent, to, text, t), None

    def _tool_note(self, agent: str, work: dict, t: float, text: str):
        self.notes[agent].append(text)
        return "noted", None, None

    def _notice(self, t: float, text: str) -> None:
        """System notice shown to every agent in view() messages (derived from the event, not logged separately)."""
        self.messages.append({"event_id": self._next_event_id, "t": t, "channel": "system",
                              "from": None, "to": None, "text": text})

    def _vote_count(self) -> str:
        return f"{len(self.done_votes)} of {len(self.agents)} votes"

    def _tool_declare_done(self, agent: str, work: dict, t: float):
        if agent in self.done_votes:
            return f"you had already voted that the house is finished ({self._vote_count()})", None, None
        self.done_votes.append(agent)
        self._notice(t, f"{agent} voted that the house is finished ({self._vote_count()})")
        return f"declared done ({len(self.done_votes)}/{len(self.agents)} votes)", None, None

    def _tool_cancel_done(self, agent: str, work: dict, t: float):
        if agent not in self.done_votes:
            return f"you had not voted that the house is finished ({self._vote_count()})", None, None
        self.done_votes.remove(agent)
        self._notice(t, f"{agent} withdrew its finished vote ({self._vote_count()})")
        return f"withdrew your finished vote ({len(self.done_votes)}/{len(self.agents)} votes)", None, None
