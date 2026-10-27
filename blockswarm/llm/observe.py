"""Turn a `World.view()` dict into the text an LLM agent reads each turn.

Design (see also prompts.py):
- Exact coordinates first: every visible block is listed with copyable absolute coordinates,
  grouped by layer z and merged into rectangles ("(4..10, 5, 1) brick" = x 4 to 10, y 5, z 1).
  LLMs reconstruct 3D layouts poorly from raw lists, so we also give precomputed facts
  (offsets and distances to agents and the chest, what changed since the last look) and a
  small top-down ASCII map per layer, labelled as an approximate picture only.
- No invented facts: chest counts appear only when the view has them (within reach), else the
  last-seen counts and time. "Now empty" is reported only for a remembered block whose cell is
  within view radius with a clear line from the eyes (judged from blocks this agent knows of);
  otherwise the block stays in a "remembered, not in sight" list with its last-seen time.
- No strategy: no to-do list, no design, no suggestions. In blueprint mode the blueprint cells
  and which of them this agent has seen built are shown (the agents are given the plan there).
- The brief is in the system prompt and repeated only in the first observation.
- Scores are hidden unless SHOW_SCORES is True (open research question, SPEC §12).
- The agent's own tool results are not repeated here; the loop returns them as tool results
  (render_tool_result).
"""

from __future__ import annotations

import json
import math
from typing import Any, Iterable

from blockswarm.world import ray_cells

SHOW_SCORES = False   # show team/individual scores in observations (off by default)
MAX_NOTES_CHARS = 2000  # newest notes shown up to this many characters in total
MAX_MAP_LAYERS = 8

BLOCK_LETTERS = {"wood": "W", "stone": "S", "brick": "B", "glass": "G", "plank": "P",
                 "roof": "R", "door": "D"}
AGENT_MARKS = "123456789abcdefghijklmnopqrstuvwxyz"

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


def format_cells(cells: dict[Pos, str], indent: str = "  ") -> list[str]:
    """One line per layer: 'z=1 (12): (4..10, 5, 1) brick; (4, 6..8, 1) brick'."""
    by_z: dict[int, list[str]] = {}
    count: dict[int, int] = {}
    for x0, x1, y0, y1, z, t in rectangles(cells):
        n = (x1 - x0 + 1) * (y1 - y0 + 1)
        by_z.setdefault(z, []).append(f"({_span(x0, x1)}, {_span(y0, y1)}, {z}) {t}" + (f" x{n}" if n > 1 else ""))
        count[z] = count.get(z, 0) + n
    return [f"{indent}z={z} ({count[z]}): " + "; ".join(parts) for z, parts in sorted(by_z.items())]


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
    out.append(f"You: {me} | feet at {fmt_pos(pos)} (eyes at z={eye[2]:.2f})"
               + (" | " + "; ".join(status) if status else " | standing"))
    out.append(f"Actions used: {view['actions_used']} ({view['actions_remaining']} left)")
    inv = view.get("inventory") or {}
    held = sum(inv.values())
    out.append(f"Inventory ({held}/{view['inventory_cap']}): {_counts(inv)}")

    # ---- chest
    cpos = tuple(view["chest_pos"])
    cdist = math.dist(eye, (cpos[0] + 0.5, cpos[1] + 0.5, cpos[2] + 0.5))
    where = (f"Chest at {fmt_pos(cpos)}, offset {_fmt_off(c - p for c, p in zip(cpos, pos))}, "
             f"{cdist:.1f} from your eyes")
    if view.get("chest") is not None:
        out.append(f"{where}; within reach, contents now: {_counts(view['chest'], all_types=True)}")
    else:
        seen = view.get("chest_last_seen")
        if seen:
            out.append(f"{where}; not within reach, contents not visible. "
                       f"Last seen at t={seen['t']:.1f}: {_counts(seen['counts'], all_types=True)}")
        else:
            out.append(f"{where}; not within reach, contents not seen yet")

    # ---- blocks: visible now, changes, remembered
    visible: dict[Pos, dict] = {}
    for b in view.get("blocks", []):
        visible[tuple(b["pos"])] = b
    appeared: dict[Pos, tuple[str, str]] = {}
    for p, b in visible.items():
        k = memory.known.get(p)
        if not first and (k is None or k["type"] != b["type"] or k["placed_at"] != b.get("placed_at")):
            who = b.get("placed_by") or "unknown"
            appeared[p] = (b["type"], "you" if who == me else who)
    gone: dict[Pos, str] = {}
    candidates = {p: k for p, k in memory.known.items() if p not in visible}
    solids = set(visible) | set(candidates) | {cpos}
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
    out.extend(format_cells({p: b["type"] for p, b in visible.items()}))

    remembered = {p: k for p, k in memory.known.items() if p not in visible}
    if remembered:
        ts = [k["seen_t"] for k in remembered.values()]
        out.append(f"REMEMBERED, NOT IN SIGHT NOW ({len(remembered)}; last seen t={min(ts):.1f}"
                   + (f"..{max(ts):.1f}" if max(ts) - min(ts) >= 0.05 else "") + "; may have changed):")
        out.extend(format_cells({p: k["type"] for p, k in remembered.items()}))

    if not first:
        since = f"t={memory.last_t:.1f}" if memory.last_t is not None else "last observation"
        out.append("")
        out.append(f"CHANGES IN VIEW SINCE YOUR LAST OBSERVATION ({since}):")
        if not appeared and not gone:
            out.append("  none seen")
        last = memory.last_t if memory.last_t is not None else -1.0
        fresh = {p: v for p, v in appeared.items() if (visible[p].get("placed_at") or 0.0) >= last}
        older = {p: v[0] for p, v in appeared.items() if p not in fresh}
        for who, cells in sorted(_group_by(fresh).items()):
            out.append(f"  placed by {who} since then:")
            out.extend(format_cells(cells, indent="    "))
        if older:
            out.append("  now in sight for the first time (placed before your last observation):")
            out.extend(format_cells(older, indent="    "))
        if gone:
            out.append("  blocks no longer there (cell now empty):")
            out.extend(format_cells(gone, indent="    "))

    # ---- agents
    others = [a for a in agents if a != me]
    vis = {o["id"]: o for o in view.get("agents_visible", [])}
    if others:
        out.append("")
        out.append(f"TEAMMATES ({len(others)}):")
        for a in others:
            o = vis.get(a)
            mark = agent_mark(agents, a)
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
    if task.get("kind") == "blueprint" and task.get("blueprint"):
        out.append("")
        out.extend(_blueprint_status(task["blueprint"], memory, visible))
    if first and task.get("brief"):
        out.append("")
        out.append(f"BRIEF: {task['brief']}")

    # ---- messages
    lines = []
    for m in view.get("messages", []):
        key = (m.get("event_id"), m.get("channel"), m.get("t"), m.get("text"))
        if key in memory._msg_keys:
            continue
        line = _message_line(m, me)
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

    if SHOW_SCORES and view.get("scores"):
        s = view["scores"]
        out.append("")
        out.append(f"SCORES: team {s.get('team_score')}; individual {json.dumps(s.get('individual_scores'))}")

    memory.count += 1
    memory.last_t = t
    if view.get("last_event_id") is not None:
        memory.last_event_id = view["last_event_id"]
    return "\n".join(out)


def _ascii_map(view: dict, visible: dict[Pos, dict], vis_agents: dict[str, dict],
               agents: list[str], me: str) -> list[str]:
    """Top-down picture of each layer that has something visible. Approximate aid only."""
    marks: dict[Pos, str] = {}
    for p, b in visible.items():
        marks[p] = BLOCK_LETTERS.get(b["type"], "?")
    cpos = tuple(view["chest_pos"])
    pos = tuple(view["position"])
    radius = float(view["view_radius"])
    eye = tuple(view["eye"])
    if math.dist(eye, (cpos[0] + 0.5, cpos[1] + 0.5, cpos[2] + 0.5)) <= radius:
        marks[cpos] = "C"
    for a, o in vis_agents.items():
        marks[tuple(o["pos"])] = agent_mark(agents, a)
    marks[pos] = "@"
    g = view["grid"]
    xs = [p[0] for p in marks]
    ys = [p[1] for p in marks]
    x0, x1 = max(0, min(xs) - 1), min(g["width"] - 1, max(xs) + 1)
    y0, y1 = max(0, min(ys) - 1), min(g["depth"] - 1, max(ys) + 1)
    layers = sorted({p[2] for p in marks})
    out = [f"MAP (approximate picture; exact coordinates are listed above). Top-down, one grid per "
           f"layer z; columns = x ({x0}..{x1}, increasing to the right), rows = y ({y1} at top down "
           f"to {y0}). {' '.join(f'{v}={k}' for k, v in BLOCK_LETTERS.items())} C=chest @=you "
           f"{'digits/letters=teammates (see TEAMMATES), ' if len(agents) > 1 else ''}.=no visible block. Agents are drawn at their feet layer."]
    if len(layers) > MAX_MAP_LAYERS:
        out.append(f"(showing the {MAX_MAP_LAYERS} lowest of {len(layers)} layers)")
        layers = layers[:MAX_MAP_LAYERS]
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


def _blueprint_status(bp: dict, memory: ObservationMemory, visible: dict[Pos, dict]) -> list[str]:
    """Blueprint cells, split by what this agent has seen (now or remembered)."""
    cells = {(x, y, z): t for x, y, z, t in bp["cells"]}
    done, wrong, other = {}, {}, {}
    for p, want in cells.items():
        k = memory.known.get(p)
        if k is None:
            other[p] = want
        elif k["type"] == want:
            done[p] = want
        else:
            wrong[p] = want
    extra = {p: k["type"] for p, k in memory.known.items() if p not in cells}
    out = [f"BLUEPRINT '{bp.get('name', '')}' ({len(cells)} cells). From blocks you can see or remember: "
           f"{len(done)} built as planned, {len(wrong)} hold a different type, {len(other)} not seen built "
           f"(empty, or out of your sight), {len(extra)} blocks outside the blueprint."]
    if other:
        out.append(" not seen built (planned type shown):")
        out.extend(format_cells(other, indent="   "))
    if wrong:
        out.append(" holding a different type (planned type shown):")
        out.extend(format_cells(wrong, indent="   "))
    if extra:
        out.append(" blocks outside the blueprint:")
        out.extend(format_cells(extra, indent="   "))
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
