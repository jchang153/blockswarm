"""Replay an episode log from its header by applying event diffs; check every snapshot.

Reads v4 logs (shop and shared wallet) and v3 logs (physical chest; see docs/log-format.md).
"""

from __future__ import annotations

import json
from pathlib import Path

from blockswarm.pathing import path_length
from blockswarm.tasks import analysis, deployed, individual_scores, ranking
from blockswarm.world import AIRBORNE_MSG, COOLDOWN_MSG, DIFF_KEYS, LIMIT_MSG, WALKING_MSG

V3_DIFF_KEYS = {"grid", "chest", "inventory", "agents", "votes"}
LEGACY_DIFF_KEYS = {"grid", "chest", "inventory", "agents"}  # v3 logs written before diff.votes


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

    v4: checks blocks, wallet (every `buy` costs price x n from the header catalog; nothing else
    changes it), inventories, positions, walkers, actions used, votes, `individual_scores` and
    `analysis` (recomputed) in every snapshot and the `episode_end`, plus `ranking` and
    `deployed` at the end (recomputed from the config's incentive and deploy_top_k).
    v3: the same physical checks with the chest instead of the wallet; of the old scores only
    the open task's `score_details` are recomputed (with v3's `door` / `glass` types).
    Finished votes are recomputed from the rules (declare_done adds, cancel_done removes, any
    successful grid change clears all) and checked against every `diff.votes` and snapshot.
    v3 logs written before `diff.votes` existed (four diff keys) are replayed with the old rule
    (declare_done adds; nothing removes).

    Returns the final state: version, blocks, wallet (v4) or chest (v3), inventories,
    actions_used, positions, done_votes, snapshots_checked, end. Raises AssertionError on any
    mismatch.
    """
    if not records or records[0].get("type") != "header":
        raise AssertionError("first record must be the header")
    header = records[0]
    version = header.get("log_version")
    if version not in (3, 4):
        raise AssertionError(f"unsupported log_version {version!r} (this reader handles 3 and 4)")
    v4 = version == 4
    agents = header["agents"]
    cfg = header["config"]
    dims = (cfg["width"], cfg["depth"], cfg["height"])
    if v4:
        shop = header["shop"]
        solid_pos = tuple(shop["pos"])
        catalog = {i["id"]: i for i in shop["catalog"]}
        doors = {i for i, it in catalog.items() if it.get("door")}
        windows = {i for i, it in catalog.items() if it.get("transparent")}
        wallet = header["wallet"]
        if wallet != shop["starting_money"]:
            raise AssertionError(f"header wallet {wallet} != shop.starting_money {shop['starting_money']}")
        chest: dict[str, int] = {}
    else:
        solid_pos = tuple(header["chest_pos"])
        catalog, doors, windows = {}, {"door"}, {"glass"}
        wallet = 0
        chest = dict(header["chest"])
    positions = {a: list(p) for a, p in header["positions"].items()}
    walking_until = {a: 0.0 for a in agents}
    blocks: dict[tuple, list] = {}
    inventories = {a: dict(header["inventories"].get(a, {})) for a in agents}
    actions_used = {a: 0 for a in agents}
    done_votes: list[str] = []
    last_event_id = -1
    last_t = 0.0
    snapshots_checked = 0
    next_seq = 0
    end: dict | None = None
    legacy: bool | None = None  # v3 only: decided by the first event's diff keys

    def house() -> dict:
        return analysis({p: b[0] for p, b in blocks.items()}, dims, [solid_pos], doors=doors,
                        windows=windows, target_interior=cfg.get("target_interior"))

    def check_scores(rec: dict, where: str) -> None:
        if v4:
            want = {"individual_scores": individual_scores({p: {"placed_by": b[1]} for p, b in blocks.items()},
                                                           agents),
                    "analysis": house()}
            if rec.get("wallet") != wallet:
                raise AssertionError(f"{where}: wallet mismatch {rec.get('wallet')} vs {wallet}")
        elif cfg.get("task") == "open" and isinstance(rec.get("score_details"), dict):
            want = {"score_details": house()}
        else:
            return  # v3 blueprint task: its grader is gone; nothing to recompute
        for key, val in json.loads(json.dumps(want)).items():
            if rec.get(key) != val:
                raise AssertionError(f"{where}: {key} mismatch {rec.get(key)} vs {val}")

    def check_state(rec: dict, where: str) -> None:
        check_scores(rec, where)
        if rec.get("actions_used", actions_used) != actions_used:
            raise AssertionError(f"{where}: actions_used mismatch {rec['actions_used']} vs {actions_used}")
        if "blocks" not in rec:
            return
        if [[x, y, z, *b] for (x, y, z), b in sorted(blocks.items())] != rec["blocks"]:
            raise AssertionError(f"{where}: blocks mismatch")
        if not v4 and _nonzero(chest) != _nonzero(rec["chest"]):
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
            if v4:
                want_keys = set(DIFF_KEYS)
            else:
                if legacy is None:
                    legacy = set(diff) == LEGACY_DIFF_KEYS
                want_keys = LEGACY_DIFF_KEYS if legacy else V3_DIFF_KEYS
            if set(diff) != want_keys:
                raise AssertionError(f"event {eid}: bad diff keys {sorted(diff)}")
            if not rec["ok"] and any(diff[k] for k in diff):
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
                    if v4 and a["type"] not in catalog:
                        raise AssertionError(f"event {eid}: block type {a['type']!r} is not a catalog item")
                    blocks[pos] = [a["type"], a["placed_by"], a["placed_at"]]
            if v4:
                dw = diff["wallet"]
                if not isinstance(dw, int) or isinstance(dw, bool):
                    raise AssertionError(f"event {eid}: diff.wallet must be an int, got {dw!r}")
                if rec["ok"] and rec["tool"] == "buy":
                    item, n = rec["args"].get("item"), rec["args"].get("n")
                    if item not in catalog:
                        raise AssertionError(f"event {eid}: bought unknown item {item!r}")
                    if dw != -catalog[item]["price"] * n:
                        raise AssertionError(f"event {eid}: wallet delta {dw} != -{catalog[item]['price']} x {n}")
                    if diff["inventory"] != {rec["agent"]: {item: n}}:
                        raise AssertionError(f"event {eid}: buy inventory delta {diff['inventory']}")
                elif dw != 0:
                    raise AssertionError(f"event {eid}: only a successful buy changes the wallet")
                wallet += dw
                if wallet < 0:
                    raise AssertionError(f"event {eid}: wallet went negative ({wallet})")
            else:
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
            votes_before = sorted(done_votes)
            if rec["ok"] and rec["tool"] == "declare_done" and rec["agent"] not in done_votes:
                done_votes.append(rec["agent"])
            if not legacy:
                if rec["ok"] and rec["tool"] == "cancel_done" and rec["agent"] in done_votes:
                    done_votes.remove(rec["agent"])
                if rec["ok"] and diff["grid"]:
                    done_votes.clear()
                want = ({} if sorted(done_votes) == votes_before
                        else {"before": votes_before, "after": sorted(done_votes)})
                if diff["votes"] != want:
                    raise AssertionError(f"event {eid}: votes diff {diff['votes']} but the rules give {want}")
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
            if v4:
                scores = rec["individual_scores"]
                rank = ranking(scores, agents)
                if rec.get("ranking") != rank:
                    raise AssertionError(f"episode_end: ranking mismatch {rec.get('ranking')} vs {rank}")
                dep = deployed(rank, cfg["deploy_top_k"], cfg["incentive"])
                if rec.get("deployed") != dep:
                    raise AssertionError(f"episode_end: deployed mismatch {rec.get('deployed')} vs {dep}")
            end = rec
        # unknown record types are ignored

    out = {
        "version": version,
        "blocks": [[x, y, z, *b] for (x, y, z), b in sorted(blocks.items())],
        "inventories": inventories,
        "actions_used": actions_used,
        "positions": positions,
        "done_votes": done_votes,
        "snapshots_checked": snapshots_checked,
        "end": end,
    }
    if v4:
        out["wallet"] = wallet
    else:
        out["chest"] = chest
    return out
