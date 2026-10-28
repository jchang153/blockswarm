"""Turn a `World.view()` dict into the text an LLM agent reads each turn.

Design (see also prompts.py):
- Exact coordinates first: every visible block is listed with copyable absolute coordinates,
  grouped by layer z and merged into rectangles ("(4..10, 5, 1) brick" = x 4 to 10, y 5, z 1).
  LLMs reconstruct 3D layouts poorly from raw lists, so we also give precomputed facts
  (offsets and distances to agents and the shop, what changed since the last look) and a
  small top-down ASCII map per layer, labelled as an approximate picture only. Block types are
  catalog item ids, so the map uses a legend built per observation for the ids it draws.
- Blocks are listed per layer and per placer ("[agent_2] (4..6, 2, 1) oak_planks x3"); the
  world shows `placed_by` for every visible block.
- No invented facts: "now empty" is reported only for a remembered block whose cell is within
  view radius with a clear line from the eyes (judged from the non-transparent blocks this
  agent knows of); otherwise the block stays in a "remembered, not in sight" list with its
  last-seen time. The wallet balance is shown as the view gives it (always visible).
- No strategy: no to-do list, no design, no suggestions.
- The brief is in the system prompt and repeated only in the first observation.
- Scores appear only when the view has them (config show_leaderboard).
- The agent's own tool results are not repeated here; the loop returns them as tool results
  (render_tool_result).
- Team battle (view has `teams`; docs/team-battle.md): adds the agent's team and side, the line,
  health and equipment, the team wallet, both shops, teammates and the other team's agents
  (with their team; killed agents marked, from `dead` in the view or the kill notices), message
  channel labels ([team], [all], [private], [system]) and the team's finished votes. Nothing of
  this appears without `teams`, so v4 observations are unchanged.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Iterable

from blockswarm.catalog import CATALOG
from blockswarm.tasks import DEFAULT_BRIEF
from blockswarm.world import ray_cells

MAX_NOTES_CHARS = 2000  # newest notes shown up to this many characters in total
MAX_MAP_LAYERS = 8

AGENT_MARKS = "123456789abcdefghijklmnopqrstuvwxyz"
# Map letters for item ids, assigned per observation in order of first appearance on the map.
# Kept disjoint from agent marks (digits, lowercase), '@', '$' and '.'.
ITEM_MARKS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ#%&*=<>~^!?"
OTHER_MARK = "+"  # any further item ids, if more distinct ids are drawn than ITEM_MARKS holds
SHOP_MARK = "$"

Pos = tuple[int, int, int]


def agent_mark(agents: list[str], agent: str) -> str:
    """Single-character map marker for an agent: 1-9, then a-z (by order in the team list)."""
    i = agents.index(agent) if agent in agents else -1
    return AGENT_MARKS[i] if 0 <= i < len(AGENT_MARKS) else "?"


def fmt_pos(p: Iterable[int]) -> str:
    x, y, z = p
    return f"({x}, {y}, {z})"


def _fmt_off(off: Iterable[int]) -> str:
    return "(" + ", ".join(f"{v:+d}" for v in off) + ")"


def _span(a: int, b: int) -> str:
    return str(a) if a == b else f"{a}..{b}"


def rectangles(cells: dict[Pos, str]) -> list[tuple[int, int, int, int, int, str]]:
    """Cover same-type cells of each layer with rectangles (x0, x1, y0, y1, z, type), greedy:
    extend along x, then along y while the whole x-run matches. Sorted by z, y, x."""
    left = dict(cells)
    out = []
    for (x, y, z) in sorted(cells, key=lambda p: (p[2], p[1], p[0])):
        t = left.get((x, y, z))
        if t is None:
            continue
        x1 = x
        while left.get((x1 + 1, y, z)) == t:
            x1 += 1
        y1 = y
        while all(left.get((xx, y1 + 1, z)) == t for xx in range(x, x1 + 1)):
            y1 += 1
        for yy in range(y, y1 + 1):
            for xx in range(x, x1 + 1):
                del left[(xx, yy, z)]
        out.append((x, x1, y, y1, z, t))
    return out


def format_cells(cells: dict[Pos, str], indent: str = "  ",
                 owners: dict[Pos, str] | None = None) -> list[str]:
    """One line per layer: 'z=1 (12): (4..10, 5, 1) bricks x7; (4, 6..8, 1) bricks x3'.
    With `owners` ({pos: label}) each layer is split by placer:
    'z=1 (12): [agent_2] (4..10, 5, 1) bricks x7 | [you] (4, 6..8, 1) bricks x3'."""
    if owners is None:
        groups = {"": cells}
    else:
        groups = {}
        for p, t in cells.items():
            groups.setdefault(owners.get(p) or "?", {})[p] = t
    by_z: dict[int, dict[str, list[str]]] = {}
    count: dict[int, int] = {}
    for label in sorted(groups, key=lambda o: (o == "you", o == "?", o)):
        for x0, x1, y0, y1, z, t in rectangles(groups[label]):
            n = (x1 - x0 + 1) * (y1 - y0 + 1)
            by_z.setdefault(z, {}).setdefault(label, []).append(
                f"({_span(x0, x1)}, {_span(y0, y1)}, {z}) {t}" + (f" x{n}" if n > 1 else ""))
            count[z] = count.get(z, 0) + n
    out = []
    for z, parts in sorted(by_z.items()):
        body = " | ".join((f"[{o}] " if o else "") + "; ".join(r) for o, r in parts.items())
        out.append(f"{indent}z={z} ({count[z]}): {body}")
    return out


def is_opaque(btype: str) -> bool:
    """Whether a block of this type blocks sight (catalog `transparent`; unknown ids block it)."""
    item = CATALOG.get(btype)
    return not (item is not None and item.transparent)


def _group_by(cells: dict[Pos, tuple[str, str]]) -> dict[str, dict[Pos, str]]:
    """{pos: (type, who)} -> {who: {pos: type}}."""
    out: dict[str, dict[Pos, str]] = {}
    for p, (t, who) in cells.items():
        out.setdefault(who, {})[p] = t
    return out


def _counts(d: dict[str, int] | None, all_types: bool = False) -> str:
    if not d:
        return "nothing"
    items = [(k, v) for k, v in d.items() if v or all_types]
    return ", ".join(f"{k} {v}" for k, v in items) if items else "nothing"


class ObservationMemory:
    """What one agent has seen so far: blocks (with last-seen time), other agents' last seen
    positions, which messages were shown, and the last event id (for `since_event_id`)."""

    def __init__(self, agent_id: str, view_occlusion: bool = True):
        self.agent_id = agent_id
        self.view_occlusion = view_occlusion
        self.last_event_id: int | None = None
        self.last_t: float | None = None
        self.count = 0
        # pos -> {"type", "placed_by", "placed_at", "seen_t"}
        self.known: dict[Pos, dict[str, Any]] = {}
        self.agents_seen: dict[str, tuple[list[int], float]] = {}
        self._msg_keys: set = set()
        # team battle: killed agents -> (killer or None, t), from the view or kill notices
        self.dead: dict[str, tuple[str | None, float | None]] = {}


KILL_RE = re.compile(r"^(\S+) was killed by (\S+?)\.?$")


def _in_clear_view(eye: tuple[float, float, float], cell: Pos, radius: float,
                   solids: set[Pos], occlusion: bool) -> bool:
    centre = (cell[0] + 0.5, cell[1] + 0.5, cell[2] + 0.5)
    if math.dist(eye, centre) > radius:
        return False
    return not occlusion or not any(c in solids for c in ray_cells(eye, cell))


def _message_line(m: dict, me: str) -> str | None:
    ch, frm, to = m.get("channel"), m.get("from"), m.get("to")
    t = m.get("t", 0.0)
    text = m.get("text", "")
    if ch == "public":
        who = "you" if frm == me else frm
        return f"[t={t:.1f}] {who} (to all): {text}"
    if ch == "private":
        if me not in (frm, to):
            return None  # never show other agents' private messages
        if frm == me:
            return f"[t={t:.1f}] you -> {to} (private): {text}"
        return f"[t={t:.1f}] {frm} -> you (private): {text}"
    if ch == "system":
        if to not in (None, me):
            return None
        return f"[t={t:.1f}] SYSTEM: {text}"
    return None


def _team_message_line(m: dict, me: str, my_team: str | None) -> str | None:
    """Team-battle message line with a channel label: [team], [all], [private] or [system]."""
    ch, frm, to = m.get("channel"), m.get("from"), m.get("to")
    t = m.get("t", 0.0)
    text = m.get("text", "")
    who = "you" if frm == me else frm
    if ch == "team":
        if m.get("team") not in (None, my_team):
            return None  # another team's channel
        return f"[t={t:.1f}] [team] {who}: {text}"
    if ch == "public":
        return f"[t={t:.1f}] [all] {who}: {text}"
    if ch == "private":
        if me not in (frm, to):
            return None
        if frm == me:
            return f"[t={t:.1f}] [private] you -> {to}: {text}"
        return f"[t={t:.1f}] [private] {frm} -> you: {text}"
    if ch == "system":
        if to not in (None, me) or m.get("team") not in (None, my_team):
            return None
        return f"[t={t:.1f}] [system] {text}"
    return None


def _team_of(teams: dict, agent: str) -> str | None:
    for team, members in teams.items():
        if agent in members:
            return team
    return None


def _shop_line(label: str, spos: Pos, pos: Pos, eye: tuple, in_reach: Any) -> str:
    sdist = math.dist(eye, (spos[0] + 0.5, spos[1] + 0.5, spos[2] + 0.5))
    reach_txt = {True: "within reach", False: "not within reach"}.get(in_reach, "")
    return (f"{label} at {fmt_pos(spos)}, offset {_fmt_off(c - p for c, p in zip(spos, pos))}, "
            f"{sdist:.1f} from your eyes" + (f"; {reach_txt}" if reach_txt else ""))


def shop_pos(view: dict) -> Pos:
    """The shop cell from a view (`shop_pos`, or `shop.pos`)."""
    p = view.get("shop_pos")
    if p is None:
        p = (view.get("shop") or {}).get("pos")
    return tuple(p)  # type: ignore[return-value]


def _who(placed_by: str | None, me: str) -> str:
    return "you" if placed_by == me else (placed_by or "?")


def _leaderboard(scores: Any, agents: list[str]) -> list[tuple[str, int]]:
    """[(agent, count)] from a view's `scores` ({agent: n} or {"individual_scores": {...}}),
    in team order; [] when absent."""
    if not isinstance(scores, dict):
        return []
    if "team_score" in scores:  # v3 views carried the team grade; never shown
        return []
    ind = scores.get("individual_scores", scores)
    if not isinstance(ind, dict):
        return []
    order = [a for a in agents if a in ind] + sorted(a for a in ind if a not in agents)
    return [(a, ind[a]) for a in order if isinstance(ind[a], (int, float))]


def render_observation(view: dict, memory: ObservationMemory) -> str:
    """Text observation for the model. Updates `memory` (call once per view)."""
    me = view["you"]
    agents: list[str] = list(view.get("agents", [me]))
    t = float(view["t"])
    pos = tuple(view["position"])
    eye = tuple(view["eye"])
    radius = float(view["view_radius"])
    first = memory.count == 0
    out: list[str] = []

    # ---- header
    status = []
    if view.get("done"):
        status.append("EPISODE OVER")
    if view.get("walking_until") is not None:
        status.append(f"walking, arrive at t={view['walking_until']:.1f}")
    if view.get("airborne_until") is not None:
        status.append(f"airborne, land at t={view['airborne_until']:.1f}")
    nxt = view.get("next_action_at")
    if nxt is not None and nxt > t + 1e-9 and view.get("walking_until") is None:
        status.append(f"next action allowed at t={nxt:.1f}")
    out.append(f"=== OBSERVATION #{memory.count + 1} | t={t:.1f}s of {view['max_seconds']:g}s"
               f" ({max(0.0, view['max_seconds'] - t):.0f}s left) ===")
    teams: dict = {k: list(v) for k, v in (view.get("teams") or {}).items()}
    my_team = (view.get("team") or _team_of(teams, me)) if teams else None
    if teams:
        for m in view.get("messages", []):  # kill notices, also from messages not shown again
            if m.get("channel") == "system":
                km = KILL_RE.match(m.get("text", ""))
                if km and km.group(1) in agents:
                    memory.dead.setdefault(km.group(1), (km.group(2), m.get("t")))
        for a in view.get("dead") or []:
            memory.dead.setdefault(a, (None, None))
        if me in memory.dead:
            status.insert(0, "KILLED (you can no longer act)")
    out.append(f"You: {me}" + (f" (team {my_team})" if teams else "") + f" | feet at {fmt_pos(pos)} (eyes at z={eye[2]:.2f})"
               + (" | " + "; ".join(status) if status else " | standing"))
    if teams:
        dx = int(view.get("divider_x") if view.get("divider_x") is not None else view["grid"]["width"] // 2)
        order = list(teams)
        side = order[0] if pos[0] < dx else (order[1] if len(order) > 1 else order[0])
        out.append(f"Side: you are on the {side} side ({'your team' if side == my_team else 'the other team'}'s "
                   f"side); the line runs between x = {dx - 1} and x = {dx} ({order[0]} side x < {dx}"
                   + (f", {order[1]} side x >= {dx}" if len(order) > 1 else "") + ")")
        if view.get("health") is not None:
            eq = view.get("equipment") or []
            out.append(f"Health: {view['health']}/{view.get('max_health', 20)}"
                       + (f" | Equipment: {', '.join(eq)}" if eq else ""))
    out.append(f"Actions used: {view['actions_used']} ({view['actions_remaining']} left)")
    inv = view.get("inventory") or {}
    held = sum(inv.values())
    out.append(f"Inventory ({held}/{view['inventory_cap']}): {_counts(inv)}")

    # ---- wallet and shop
    if teams:
        if view.get("wallet") is not None:
            out.append(f"Team wallet ({my_team}; shared by your team; no refunds): {view['wallet']} coins")
        shops = view.get("shops") or {}
        sir = view.get("shops_in_reach")
        for team, sp in shops.items():
            flag = sir.get(team) if isinstance(sir, dict) else None
            label = f"Shop of the {team} team" + (" (yours)" if team == my_team else "")
            out.append(_shop_line(label, tuple(sp), pos, eye, flag))
        if not isinstance(sir, dict) and view.get("shop_in_reach") is not None:
            out.append("A shop is within reach" if view["shop_in_reach"] else "No shop is within reach")
        shop_cells = {tuple(p) for p in shops.values()} or {shop_pos(view)}
    else:
        if view.get("wallet") is not None:
            out.append(f"Wallet ({'shared by the whole team' if len(agents) > 1 else 'yours'}; no refunds): "
                       f"{view['wallet']} coins")
        spos = shop_pos(view)
        out.append(_shop_line("Shop", spos, pos, eye, view.get("shop_in_reach")))
        shop_cells = {spos}

    # ---- blocks: visible now, changes, remembered
    visible: dict[Pos, dict] = {}
    for b in view.get("blocks", []):
        visible[tuple(b["pos"])] = b
    appeared: dict[Pos, tuple[str, str]] = {}
    for p, b in visible.items():
        k = memory.known.get(p)
        if not first and (k is None or k["type"] != b["type"] or k["placed_at"] != b.get("placed_at")):
            appeared[p] = (b["type"], _who(b.get("placed_by"), me))
    gone: dict[Pos, str] = {}
    candidates = {p: k for p, k in memory.known.items() if p not in visible}
    solids = ({p for p, b in visible.items() if is_opaque(b["type"])}
              | {p for p, k in candidates.items() if is_opaque(k["type"])} | shop_cells)
    changed = True
    while changed:  # a removed near block can uncover a removed far one
        changed = False
        for p in list(candidates):
            if _in_clear_view(eye, p, radius, solids - {p}, memory.view_occlusion):
                gone[p] = candidates.pop(p)["type"]
                solids.discard(p)
                changed = True
    for p in gone:
        memory.known.pop(p, None)
    for p, b in visible.items():
        memory.known[p] = {"type": b["type"], "placed_by": b.get("placed_by"),
                           "placed_at": b.get("placed_at"), "seen_t": t}

    out.append("")
    out.append(f"BLOCKS YOU CAN SEE NOW ({len(visible)}; within {radius:g} of your eyes and in line of sight)"
               + (":" if visible else ": none"))
    out.extend(format_cells({p: b["type"] for p, b in visible.items()},
                            owners={p: _who(b.get("placed_by"), me) for p, b in visible.items()}))

    remembered = {p: k for p, k in memory.known.items() if p not in visible}
    if remembered:
        ts = [k["seen_t"] for k in remembered.values()]
        out.append(f"REMEMBERED, NOT IN SIGHT NOW ({len(remembered)}; last seen t={min(ts):.1f}"
                   + (f"..{max(ts):.1f}" if max(ts) - min(ts) >= 0.05 else "") + "; may have changed):")
        out.extend(format_cells({p: k["type"] for p, k in remembered.items()},
                                owners={p: _who(k.get("placed_by"), me) for p, k in remembered.items()}))

    if not first:
        since = f"t={memory.last_t:.1f}" if memory.last_t is not None else "last observation"
        out.append("")
        out.append(f"CHANGES IN VIEW SINCE YOUR LAST OBSERVATION ({since}):")
        if not appeared and not gone:
            out.append("  none seen")
        last = memory.last_t if memory.last_t is not None else -1.0
        fresh = {p: v for p, v in appeared.items() if (visible[p].get("placed_at") or 0.0) >= last}
        older = {p: v for p, v in appeared.items() if p not in fresh}
        for who, cells in sorted(_group_by(fresh).items()):
            out.append(f"  placed by {who} since then:")
            out.extend(format_cells(cells, indent="    "))
        if older:
            out.append("  now in sight for the first time (placed before your last observation):")
            out.extend(format_cells({p: v[0] for p, v in older.items()}, indent="    ",
                                    owners={p: v[1] for p, v in older.items()}))
        if gone:
            out.append("  blocks no longer there (cell now empty):")
            out.extend(format_cells(gone, indent="    "))

    # ---- agents
    others = [a for a in agents if a != me]
    vis = {o["id"]: o for o in view.get("agents_visible", [])}
    if teams:
        groups = [(f"TEAMMATES ({my_team}, {len(teams.get(my_team, [])) - 1})",
                   [a for a in teams.get(my_team, []) if a != me])]
        groups += [(f"OTHER TEAM ({t}, {len(m)})", list(m)) for t, m in teams.items() if t != my_team]
    else:
        groups = [(f"TEAMMATES ({len(others)})", others)] if others else []
    for title, members in groups:
        out.append("")
        out.append(f"{title}:")
        for a in members:
            o = vis.get(a)
            mark = agent_mark(agents, a)
            if teams and a in memory.dead:
                by, kt = memory.dead[a]
                out.append(f"  {a}: killed" + (f" by {'you' if by == me else by}" if by else "")
                           + (f" at t={kt:.1f}" if kt is not None else "") + " (can no longer act)")
                memory.agents_seen.pop(a, None)
                continue
            if o is not None:
                p = tuple(o["pos"])
                d = min(math.dist(eye, (p[0] + 0.5, p[1] + 0.5, p[2] + dz + 0.5)) for dz in (0, 1))
                st = "walking" if o.get("walking") else ("airborne" if o.get("airborne") else "standing")
                out.append(f"  {a} [map {mark}]: feet at {fmt_pos(p)}, offset "
                           f"{_fmt_off(c - q for c, q in zip(p, pos))}, {d:.1f} from your eyes, {st}")
                memory.agents_seen[a] = (list(p), t)
            elif a in memory.agents_seen:
                p, st = memory.agents_seen[a]
                out.append(f"  {a} [map {mark}]: not in sight (last seen at {fmt_pos(p)}, t={st:.1f})")
            else:
                out.append(f"  {a} [map {mark}]: not in sight (not seen yet)")

    # ---- map
    out.append("")
    out.extend(_ascii_map(view, visible, vis, agents, me))

    # ---- task
    task = view.get("task") or {}
    brief = task.get("brief")
    if teams and (not brief or brief == DEFAULT_BRIEF):  # world built without the team brief
        from blockswarm.llm.prompts import TEAM_BRIEF
        brief = TEAM_BRIEF
    if first and brief:
        out.append("")
        out.append(f"BRIEF: {brief}")

    # ---- messages
    lines = []
    for m in view.get("messages", []):
        key = (m.get("event_id"), m.get("channel"), m.get("t"), m.get("text"))
        if key in memory._msg_keys:
            continue
        line = _team_message_line(m, me, my_team) if teams else _message_line(m, me)
        if line is not None:
            memory._msg_keys.add(key)
            lines.append(line)
    out.append("")
    out.append(("MESSAGES" if first else "NEW MESSAGES SINCE YOUR LAST OBSERVATION")
               + (":" if lines else ": none"))
    out.extend("  " + ln for ln in lines)

    # ---- votes
    votes = view.get("done_votes", [])
    out.append("")
    if teams:
        if isinstance(votes, dict):
            votes = votes.get(my_team) or []
        living = [a for a in teams.get(my_team, []) if a not in memory.dead]
        out.append(f"FINISHED VOTES (your team, {my_team}): {len(votes)} of {len(living)} living"
                   + (" (" + ", ".join("you" if v == me else v for v in votes) + ")" if votes else ""))
    else:
        out.append(f"FINISHED VOTES: {len(votes)} of {len(agents)}"
                   + (" (" + ", ".join("you" if v == me else v for v in votes) + ")" if votes else ""))

    # ---- notes
    notes = view.get("notes") or []
    if notes:
        shown, total = [], 0
        for i in range(len(notes) - 1, -1, -1):
            total += len(notes[i])
            if total > MAX_NOTES_CHARS and shown:
                break
            shown.append(f"  {i + 1}. {notes[i]}")
        hidden = len(notes) - len(shown)
        out.append("")
        out.append(f"YOUR NOTES ({len(notes)}" + (f"; oldest {hidden} not shown" if hidden else "") + "):")
        out.extend(reversed(shown))

    board = _leaderboard(view.get("scores"), agents)
    if board:
        out.append("")
        out.append("SCORES (blocks each agent placed that are still in the world): "
                   + ", ".join(f"{'you' if a == me else a} {v}" for a, v in board))

    memory.count += 1
    memory.last_t = t
    if view.get("last_event_id") is not None:
        memory.last_event_id = view["last_event_id"]
    return "\n".join(out)


def _ascii_map(view: dict, visible: dict[Pos, dict], vis_agents: dict[str, dict],
               agents: list[str], me: str) -> list[str]:
    """Top-down picture of each layer that has something visible. Approximate aid only.
    Item ids get letters in order of first appearance (reading the drawn layers bottom-up,
    each from the top row, left to right); the legend lists the ones drawn."""
    marks: dict[Pos, str] = {}
    teams = view.get("teams")
    shop_cells = [tuple(p) for p in (view.get("shops") or {}).values()] if teams else []
    pos = tuple(view["position"])
    radius = float(view["view_radius"])
    eye = tuple(view["eye"])
    for spos in shop_cells or [shop_pos(view)]:
        if math.dist(eye, (spos[0] + 0.5, spos[1] + 0.5, spos[2] + 0.5)) <= radius:
            marks[spos] = SHOP_MARK
    for a, o in vis_agents.items():
        marks[tuple(o["pos"])] = agent_mark(agents, a)
    marks[pos] = "@"
    cells = {p: b["type"] for p, b in visible.items()}
    g = view["grid"]
    every = set(marks) | set(cells)
    xs = [p[0] for p in every]
    ys = [p[1] for p in every]
    x0, x1 = max(0, min(xs) - 1), min(g["width"] - 1, max(xs) + 1)
    y0, y1 = max(0, min(ys) - 1), min(g["depth"] - 1, max(ys) + 1)
    layers = sorted({p[2] for p in every})
    clipped = len(layers) > MAX_MAP_LAYERS
    total = len(layers)
    layers = layers[:MAX_MAP_LAYERS]
    legend: dict[str, str] = {}
    for p in sorted((p for p in cells if p[2] in layers), key=lambda p: (p[2], -p[1], p[0])):
        t = cells[p]
        if t not in legend:
            legend[t] = ITEM_MARKS[len(legend)] if len(legend) < len(ITEM_MARKS) else OTHER_MARK
        marks.setdefault(p, legend[t])
    items = " ".join(f"{m}={t}" for t, m in legend.items() if m != OTHER_MARK)
    if OTHER_MARK in legend.values():
        items += f" {OTHER_MARK}=any other item id (see the lists above)"
    out = [f"MAP (approximate picture; exact coordinates are listed above). Top-down, one grid per "
           f"layer z; columns = x ({x0}..{x1}, increasing to the right), rows = y ({y1} at top down "
           f"to {y0}). Agents are drawn at their feet layer.",
           "Map legend (letters are assigned for this map only): "
           + (items + " " if items else "")
           + f"{SHOP_MARK}=shop @=you "
           + ("digits/letters=other agents (see TEAMMATES and OTHER TEAM) " if teams else
              "digits/letters=teammates (see TEAMMATES) " if len(agents) > 1 else "")
           + ".=no visible block"]
    if teams:
        dx = int(view.get("divider_x") if view.get("divider_x") is not None else g["width"] // 2)
        if x0 < dx <= x1:
            out.append(f"The line between the sides runs between columns x = {dx - 1} and x = {dx}.")
    if clipped:
        out.append(f"(showing the {MAX_MAP_LAYERS} lowest of {total} layers)")
    tens = "".join(str(x // 10) if x >= 10 else " " for x in range(x0, x1 + 1))
    units = "".join(str(x % 10) for x in range(x0, x1 + 1))
    for z in layers:
        out.append(f" z={z}")
        if tens.strip():
            out.append("      " + tens)
        out.append("   x= " + units)
        for y in range(y1, y0 - 1, -1):
            row = "".join(marks.get((x, y, z), ".") for x in range(x0, x1 + 1))
            out.append(f"y={y:>2}  {row}")
    return out


_FREE_REFUSALS = ("walking", "airborne", "cooldown", "action limit reached")


def render_tool_result(tool: str, args: dict, ok: bool, message: str, data: dict | None) -> str:
    """Short text for a tool-result message."""
    text = f"{tool}: {'OK' if ok else 'FAILED'}: {message}"
    if not ok and message.startswith(_FREE_REFUSALS):
        text += " (refused; not counted as an action)"
    if data and tool != "observe":
        s = json.dumps(data, separators=(",", ":"))
        text += " | " + (s if len(s) <= 400 else s[:400] + "...")
    return text
