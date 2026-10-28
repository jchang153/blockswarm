"""Replay an episode log from its header by applying event diffs; check every snapshot.

Reads v5 logs (v4 plus the team battle: team wallets and votes, chat channels, combat;
docs/team-battle.md), v4 logs (shop and shared wallet) and v3 logs (physical chest; see
docs/log-format.md).
"""

from __future__ import annotations

import json
from pathlib import Path

from blockswarm.pathing import path_length
from blockswarm.tasks import analysis, deployed, individual_scores, ranking, side_team, team_summary
from blockswarm.world import AIRBORNE_MSG, COOLDOWN_MSG, DIFF_KEYS, LIMIT_MSG, WALKING_MSG

V4_DIFF_KEYS = {"grid", "wallet", "inventory", "agents", "votes"}
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

    v5: everything v4 checks, plus (with teams) one wallet per team charged to the buyer's
    team, finished votes per team (a grid change clears the votes of the team whose side the
    cell is on), and (with combat) every attack's damage and health against the header's
    combat settings, deaths (inventory lost, vote dropped, no events from the dead),
    `health` / `dead` in snapshots, and `kills` and the per-team `teams` stats at the end.
    Without teams a v5 log is checked like v4 (with `diff.combat` and the v5 snapshot keys).
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

    Returns the final state: version, blocks, wallet (v4/v5; None with teams) or chest (v3),
    inventories, actions_used, positions, done_votes ({team: [...]} with teams),
    snapshots_checked, end, and for v5 also wallets, health, dead and kills. Raises
    AssertionError on any mismatch.
    """
    if not records or records[0].get("type") != "header":
        raise AssertionError("first record must be the header")
    header = records[0]
    version = header.get("log_version")
    if version not in (3, 4, 5):
        raise AssertionError(f"unsupported log_version {version!r} (this reader handles 3, 4 and 5)")
    v4 = version >= 4  # shop and wallet (v4 and v5)
    v5 = version == 5
    agents = header["agents"]
    cfg = header["config"]
    dims = (cfg["width"], cfg["depth"], cfg["height"])
    teams: dict[str, list[str]] = (header.get("teams") or {}) if v5 else {}
    team_names = list(teams)
    team_of = {a: tm for tm, m in teams.items() for a in m}
    divider_x = header.get("divider_x")
    combat = header.get("combat") if v5 else None
    if teams and (sorted(team_of) != sorted(agents) or divider_x is None):
        raise AssertionError("header: teams must cover every agent once, with a divider_x")
    if v4:
        shop = header["shop"]
        catalog = {i["id"]: i for i in shop["catalog"]}
        doors = {i for i, it in catalog.items() if it.get("door")}
        windows = {i for i, it in catalog.items() if it.get("transparent")}
        if teams:
            shops = {tm: tuple(header["shops"][tm]) for tm in team_names}
            solid_cells = sorted(shops.values())
            wallets: dict = dict(header["wallets"])
            if sorted(wallets) != sorted(team_names) or any(v != shop["starting_money"] for v in wallets.values()):
                raise AssertionError(f"header wallets {wallets} must be starting_money per team")
            if header.get("wallet") is not None:
                raise AssertionError("header: wallet must be null with teams")
        else:
            shops = {}
            solid_cells = [tuple(shop["pos"])]
            wallets = {None: header["wallet"]}
            if header["wallet"] != shop["starting_money"]:
                raise AssertionError(f"header wallet {header['wallet']} != shop.starting_money "
                                     f"{shop['starting_money']}")
        chest: dict[str, int] = {}
    else:
        solid_cells = [tuple(header["chest_pos"])]
        catalog, doors, windows = {}, {"door"}, {"glass"}
        wallets = {None: 0}
        chest = dict(header["chest"])
    health: dict[str, int] = dict(header["health"]) if combat else {}
    if combat and sorted(health) != sorted(agents):
        raise AssertionError("header: health must list every agent")
    dead: list[str] = []
    kills: list[dict] = []
    placed_by_team = {tm: 0 for tm in team_names}
    removed_by_other = {tm: 0 for tm in team_names}
    positions = {a: list(p) for a, p in header["positions"].items()}
    walking_until = {a: 0.0 for a in agents}
    blocks: dict[tuple, list] = {}
    inventories = {a: dict(header["inventories"].get(a, {})) for a in agents}
    actions_used = {a: 0 for a in agents}
    votes: dict = {tm: [] for tm in team_names} if teams else {None: []}
    last_event_id = -1
    last_t = 0.0
    snapshots_checked = 0
    next_seq = 0
    end: dict | None = None
    legacy: bool | None = None  # v3 only: decided by the first event's diff keys

    def house() -> dict:
        return analysis({p: b[0] for p, b in blocks.items()}, dims, solid_cells, doors=doors,
                        windows=windows, target_interior=cfg.get("target_interior"))

    def wallet_json():
        return None if teams else wallets[None]

    def check_scores(rec: dict, where: str) -> None:
        if v4:
            want = {"individual_scores": individual_scores({p: {"placed_by": b[1]} for p, b in blocks.items()},
                                                           agents),
                    "analysis": house()}
            if rec.get("wallet") != wallet_json():
                raise AssertionError(f"{where}: wallet mismatch {rec.get('wallet')} vs {wallet_json()}")
            if v5:
                want["wallets"] = dict(wallets) if teams else None
                want["health"] = dict(health) if combat else None
                want["dead"] = list(dead)
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
        if teams:
            got = rec["done_votes"]
            if not isinstance(got, dict) or {tm: sorted(v) for tm, v in got.items()} != \
                    {tm: sorted(v) for tm, v in votes.items()}:
                raise AssertionError(f"{where}: done_votes mismatch {got} vs {votes}")
        elif sorted(votes[None]) != sorted(rec["done_votes"]):
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
            actor = rec["agent"]
            if actor in dead:
                raise AssertionError(f"event {eid}: {actor} acted after being killed")
            diff = rec["diff"]
            if v5:
                want_keys = set(DIFF_KEYS)
            elif v4:
                want_keys = V4_DIFF_KEYS
            else:
                if legacy is None:
                    legacy = set(diff) == LEGACY_DIFF_KEYS
                want_keys = LEGACY_DIFF_KEYS if legacy else V3_DIFF_KEYS
            if set(diff) != want_keys:
                raise AssertionError(f"event {eid}: bad diff keys {sorted(diff)}")
            if not rec["ok"] and any(diff[k] for k in diff):
                raise AssertionError(f"event {eid}: failed event has a non-empty diff")
            sides_changed: set = set()
            for g in diff["grid"]:
                pos = tuple(g["pos"])
                cur = blocks.get(pos)
                before = None if cur is None else {"type": cur[0], "placed_by": cur[1], "placed_at": cur[2]}
                if before != g["before"]:
                    raise AssertionError(f"event {eid}: grid before mismatch at {pos}")
                if teams:
                    sides_changed.add(side_team(pos[0], divider_x, team_names))
                a = g["after"]
                if a is None:
                    blocks.pop(pos, None)
                    owner = team_of.get(before["placed_by"]) if before else None
                    if owner is not None and owner != team_of.get(actor):
                        removed_by_other[owner] += 1
                else:
                    if a["placed_at"] != rec["t"] or a["placed_by"] != actor:
                        raise AssertionError(f"event {eid}: placed block metadata mismatch")
                    if v4 and a["type"] not in catalog:
                        raise AssertionError(f"event {eid}: block type {a['type']!r} is not a catalog item")
                    blocks[pos] = [a["type"], a["placed_by"], a["placed_at"]]
                    if teams:
                        placed_by_team[team_of[actor]] += 1
            if v4:
                dw = diff["wallet"]
                key = team_of.get(actor) if teams else None
                cost = None
                if rec["ok"] and rec["tool"] == "buy":
                    item, n = rec["args"].get("item"), rec["args"].get("n")
                    if item not in catalog:
                        raise AssertionError(f"event {eid}: bought unknown item {item!r}")
                    cost = catalog[item]["price"] * n
                    if diff["inventory"] != {actor: {item: n}}:
                        raise AssertionError(f"event {eid}: buy inventory delta {diff['inventory']}")
                if teams:
                    want_w = {} if cost is None else {key: -cost}
                    if dw != want_w:
                        raise AssertionError(f"event {eid}: wallet delta {dw} but the rules give {want_w}")
                else:
                    if not isinstance(dw, int) or isinstance(dw, bool):
                        raise AssertionError(f"event {eid}: diff.wallet must be an int, got {dw!r}")
                    if cost is not None and dw != -cost:
                        raise AssertionError(f"event {eid}: wallet delta {dw} != -{cost}")
                    if cost is None and dw != 0:
                        raise AssertionError(f"event {eid}: only a successful buy changes the wallet")
                if cost is not None:
                    wallets[key] -= cost
                    if wallets[key] < 0:
                        raise AssertionError(f"event {eid}: wallet went negative ({wallets[key]})")
            else:
                _add(chest, diff["chest"], f"event {eid} chest")
            for agent, delta in diff["inventory"].items():
                _add(inventories[agent], delta, f"event {eid} inventory {agent}")
            # combat: an attack's damage, health and (maybe) the kill
            killed: str | None = None
            cb = diff.get("combat", {})
            if rec["ok"] and rec["tool"] == "attack":
                if not combat:
                    raise AssertionError(f"event {eid}: attack without combat")
                target = rec["args"].get("target")
                if target not in health or target == actor or target in dead:
                    raise AssertionError(f"event {eid}: bad attack target {target!r}")
                hb = health[target]
                ha = max(0, hb - combat["sword_damage"])
                want_c = {"attacker": actor, "target": target, "damage": combat["sword_damage"],
                          "health_before": hb, "health_after": ha, "killed": ha <= 0}
                if cb != want_c:
                    raise AssertionError(f"event {eid}: combat diff {cb} but the rules give {want_c}")
                health[target] = ha
                if ha <= 0:
                    killed = target
            elif cb:
                raise AssertionError(f"event {eid}: combat diff on a non-attack event")
            for agent, mv in diff["agents"].items():
                if agent in dead:
                    raise AssertionError(f"event {eid}: dead agent {agent} moved")
                if mv.get("dead"):
                    if agent != killed or mv != {"dead": True, "at": positions[agent]}:
                        raise AssertionError(f"event {eid}: bad death entry for {agent}: {mv}")
                    continue
                if mv["from"] != positions[agent] or mv["path"][0] != mv["from"] or mv["path"][-1] != mv["to"]:
                    raise AssertionError(f"event {eid}: bad move for {agent}")
                if mv["depart_t"] != rec["t"] or mv["arrive_t"] < mv["depart_t"]:
                    raise AssertionError(f"event {eid}: bad move times for {agent}")
                if rec["tool"] == "move_to" and agent == actor:
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
            if killed is not None:
                if killed not in diff["agents"]:
                    raise AssertionError(f"event {eid}: kill of {killed} has no death entry")
                if _nonzero(inventories[killed]):
                    raise AssertionError(f"event {eid}: {killed} died but kept {inventories[killed]}")
                dead.append(killed)
                kills.append({"t": rec["t"], "attacker": actor, "target": killed})
                walking_until[killed] = 0.0
            if _counted(rec):
                actions_used[actor] += 1
            vkey = team_of.get(actor) if teams else None
            before_v = {k: sorted(v) for k, v in votes.items()}
            if rec["ok"] and rec["tool"] == "declare_done" and actor not in votes[vkey]:
                votes[vkey].append(actor)
            if not legacy:
                if rec["ok"] and rec["tool"] == "cancel_done" and actor in votes[vkey]:
                    votes[vkey].remove(actor)
                if rec["ok"] and diff["grid"]:
                    for k in (sides_changed if teams else [None]):
                        votes[k].clear()
                if killed is not None:
                    kv = votes[team_of.get(killed) if teams else None]
                    if killed in kv:
                        kv.remove(killed)
                after_v = {k: sorted(v) for k, v in votes.items()}
                if teams:
                    want = {k: {"before": before_v[k], "after": after_v[k]}
                            for k in team_names if after_v[k] != before_v[k]}
                else:
                    want = ({} if after_v[None] == before_v[None]
                            else {"before": before_v[None], "after": after_v[None]})
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
                rank = [] if teams else ranking(scores, agents)
                if rec.get("ranking") != rank:
                    raise AssertionError(f"episode_end: ranking mismatch {rec.get('ranking')} vs {rank}")
                dep = deployed(rank, cfg["deploy_top_k"], cfg["incentive"])
                if rec.get("deployed") != dep:
                    raise AssertionError(f"episode_end: deployed mismatch {rec.get('deployed')} vs {dep}")
            if v5:
                if rec.get("kills") != kills:
                    raise AssertionError(f"episode_end: kills mismatch {rec.get('kills')} vs {kills}")
                want_teams = None
                if teams:
                    want_teams = json.loads(json.dumps(team_summary(
                        {p: {"type": b[0], "placed_by": b[1]} for p, b in blocks.items()}, dims, teams,
                        shops, divider_x, set(dead), placed_by_team, removed_by_other,
                        cfg.get("target_interior"))))
                if rec.get("teams") != want_teams:
                    raise AssertionError(f"episode_end: teams mismatch {rec.get('teams')} vs {want_teams}")
            end = rec
        # unknown record types are ignored

    out = {
        "version": version,
        "blocks": [[x, y, z, *b] for (x, y, z), b in sorted(blocks.items())],
        "inventories": inventories,
        "actions_used": actions_used,
        "positions": positions,
        "done_votes": {k: list(v) for k, v in votes.items()} if teams else votes[None],
        "snapshots_checked": snapshots_checked,
        "end": end,
    }
    if v4:
        out["wallet"] = wallet_json()
    else:
        out["chest"] = chest
    if v5:
        out.update({"wallets": dict(wallets) if teams else None, "health": dict(health) if combat else None,
                    "dead": list(dead), "kills": kills})
    return out
