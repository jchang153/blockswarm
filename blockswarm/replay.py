"""Replay an episode log from its header by applying event diffs; check every snapshot."""

from __future__ import annotations

import json
from pathlib import Path

from blockswarm.pathing import path_length
from blockswarm.tasks import make_task
from blockswarm.world import AIRBORNE_MSG, COOLDOWN_MSG, LIMIT_MSG, WALKING_MSG


def load_log(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _add(counts: dict[str, int], delta: dict[str, int], what: str) -> None:
    for t, d in delta.items():
        counts[t] = counts.get(t, 0) + d
        if counts[t] < 0:
            raise AssertionError(f"{what}: count for {t} went negative during replay")


def _nonzero(counts: dict[str, int]) -> dict[str, int]:
    return {t: n for t, n in counts.items() if n}


def _counted(rec: dict) -> bool:
    """Whether an event used up one of the agent's actions."""
    if rec["cost"] != "action":
        return False
    return rec["ok"] or not rec["message"].startswith((LIMIT_MSG, COOLDOWN_MSG, WALKING_MSG, AIRBORNE_MSG))


def replay(records: list[dict]) -> dict:
    """Rebuild state from the header + event diffs, asserting it matches every snapshot.

    Checks blocks, chest, inventories, positions, walkers, actions used, votes and scores.

    Returns the final state: blocks, chest, inventories, actions_used, done_votes,
    snapshots_checked, end. Raises AssertionError on any mismatch.
    """
    if not records or records[0].get("type") != "header":
        raise AssertionError("first record must be the header")
    header = records[0]
    agents = header["agents"]
    cfg = header["config"]
    task = make_task(cfg["task"], brief=cfg["brief"], blueprint=cfg["blueprint"],
                     reference_design=cfg["reference_design"], target_interior=cfg["target_interior"],
                     extra_penalty=cfg["extra_penalty"])
    dims = (cfg["width"], cfg["depth"], cfg["height"])
    chest_pos = tuple(header["chest_pos"])
    positions = {a: list(p) for a, p in header["positions"].items()}
    walking_until = {a: 0.0 for a in agents}
    blocks: dict[tuple, list] = {}
    chest = dict(header["chest"])
    inventories = {a: dict(header["inventories"].get(a, {})) for a in agents}
    actions_used = {a: 0 for a in agents}
    done_votes: list[str] = []
    last_event_id = -1
    last_t = 0.0
    snapshots_checked = 0
    next_seq = 0
    end: dict | None = None

    def scores() -> dict:
        s = task.score({p: {"type": b[0], "placed_by": b[1], "placed_at": b[2]} for p, b in blocks.items()},
                       agents, dims, [chest_pos])
        return {k: s[k] for k in ("team_score", "individual_scores", "score_details")}

    def check_state(rec: dict, where: str) -> None:
        for key, val in json.loads(json.dumps(scores())).items():
            if rec[key] != val:
                raise AssertionError(f"{where}: {key} mismatch {rec[key]} vs {val}")
        if rec.get("actions_used", actions_used) != actions_used:
            raise AssertionError(f"{where}: actions_used mismatch {rec['actions_used']} vs {actions_used}")
        if "blocks" not in rec:
            return
        if [[x, y, z, *b] for (x, y, z), b in sorted(blocks.items())] != rec["blocks"]:
            raise AssertionError(f"{where}: blocks mismatch")
        if _nonzero(chest) != _nonzero(rec["chest"]):
            raise AssertionError(f"{where}: chest mismatch {chest} vs {rec['chest']}")
        for a in agents:
            if _nonzero(inventories[a]) != _nonzero(rec["inventories"].get(a, {})):
                raise AssertionError(f"{where}: inventory mismatch for {a}")
        if sorted(done_votes) != sorted(rec["done_votes"]):
            raise AssertionError(f"{where}: done_votes mismatch")
        if rec["positions"] != positions:
            raise AssertionError(f"{where}: positions mismatch {rec['positions']} vs {positions}")
        walkers = {a: w for a, w in walking_until.items() if w > rec["t"]}
        if rec["walking_until"] != walkers:
            raise AssertionError(f"{where}: walking_until mismatch {rec['walking_until']} vs {walkers}")

    for i, rec in enumerate(records[1:], start=1):
        kind = rec.get("type")
        if kind in ("event", "snapshot", "episode_end"):
            if end is not None:
                raise AssertionError(f"record {i} ({kind}) after episode_end")
            if rec["t"] < last_t:
                raise AssertionError(f"record {i}: t went backwards ({rec['t']} < {last_t})")
            last_t = rec["t"]
        if kind == "event":
            eid = rec["event_id"]
            if eid != last_event_id + 1:
                raise AssertionError(f"event_id {eid} does not follow {last_event_id}")
            last_event_id = eid
            diff = rec["diff"]
            if set(diff) != {"grid", "chest", "inventory", "agents"}:
                raise AssertionError(f"event {eid}: bad diff keys {sorted(diff)}")
            if not rec["ok"] and (diff["grid"] or diff["chest"] or diff["inventory"] or diff["agents"]):
                raise AssertionError(f"event {eid}: failed event has a non-empty diff")
            for g in diff["grid"]:
                pos = tuple(g["pos"])
                cur = blocks.get(pos)
                before = None if cur is None else {"type": cur[0], "placed_by": cur[1], "placed_at": cur[2]}
                if before != g["before"]:
                    raise AssertionError(f"event {eid}: grid before mismatch at {pos}")
                a = g["after"]
                if a is None:
                    blocks.pop(pos, None)
                else:
                    if a["placed_at"] != rec["t"] or a["placed_by"] != rec["agent"]:
                        raise AssertionError(f"event {eid}: placed block metadata mismatch")
                    blocks[pos] = [a["type"], a["placed_by"], a["placed_at"]]
            _add(chest, diff["chest"], f"event {eid} chest")
            for agent, delta in diff["inventory"].items():
                _add(inventories[agent], delta, f"event {eid} inventory {agent}")
            for agent, mv in diff["agents"].items():
                if mv["from"] != positions[agent] or mv["path"][0] != mv["from"] or mv["path"][-1] != mv["to"]:
                    raise AssertionError(f"event {eid}: bad move for {agent}")
                if mv["depart_t"] != rec["t"] or mv["arrive_t"] < mv["depart_t"]:
                    raise AssertionError(f"event {eid}: bad move times for {agent}")
                if rec["tool"] == "move_to" and agent == rec["agent"]:
                    want = round(rec["t"] + path_length(mv["path"]) / cfg["walk_speed"], 3)
                    if mv["arrive_t"] != want:
                        raise AssertionError(f"event {eid}: arrive_t {mv['arrive_t']} != {want}")
                if mv.get("jump"):
                    if mv["from"] != mv["to"] or len(mv["path"]) != 1:
                        raise AssertionError(f"event {eid}: a jump must not move {agent}")
                elif rec["tool"] == "place" and not mv.get("fall"):
                    pass  # lifted by a block placed under its feet mid-jump; lands at arrive_t
                else:
                    walking_until[agent] = mv["arrive_t"]  # walking or falling: busy until arrival
                positions[agent] = list(mv["to"])
            if _counted(rec):
                actions_used[rec["agent"]] += 1
            if rec["ok"] and rec["tool"] == "declare_done" and rec["agent"] not in done_votes:
                done_votes.append(rec["agent"])
        elif kind == "snapshot":
            if rec["seq"] != next_seq:
                raise AssertionError(f"snapshot seq {rec['seq']} expected {next_seq}")
            next_seq += 1
            if rec["after_event_id"] != last_event_id:
                raise AssertionError(f"snapshot {rec['seq']}: after_event_id {rec['after_event_id']} "
                                     f"but last event is {last_event_id}")
            check_state(rec, f"snapshot {rec['seq']}")
            snapshots_checked += 1
        elif kind == "episode_end":
            check_state(rec, "episode_end")
            end = rec
        # unknown record types are ignored

    return {
        "blocks": [[x, y, z, *b] for (x, y, z), b in sorted(blocks.items())],
        "chest": chest,
        "inventories": inventories,
        "actions_used": actions_used,
        "positions": positions,
        "done_votes": done_votes,
        "snapshots_checked": snapshots_checked,
        "end": end,
    }
