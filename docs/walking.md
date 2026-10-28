# Continuous walking (design)

Decided 2026-10-28 by the researcher: a walking agent must actually walk, not jump to its
destination. Until now (`walking: "teleport"`, every log written before this change) an agent's
position became the destination the moment `move_to` succeeded, so a walking agent's body, eyes,
reach and visibility were already at the destination. This file is the contract for the engine,
the LLM layer and the viewer; update it if something has to change.

## 1. Config
`walking: "continuous" | "teleport"` in `blockswarm.world.Config`, default **`"continuous"`** for
every variant (shop and team battle). It is logged in the header config. **Readers treat a log
whose config has no `walking` key as `"teleport"`** (all v3/v4 logs and older v5 logs), so old runs
replay exactly as before.

## 2. Where a walking agent is
`move_to` plans the path at departure exactly as before (same Dijkstra, same costs: straight step
1, diagonal step √2, going up or down adds nothing, `pathing.path_length`). Let `path = [p0, …, pn]`
with `p0` the start, `C_i` the path length from `p0` to `p_i`, `depart_t` and speed `v`.
- The agent's **current cell** at time `t` is `p_i` for the largest `i` with
  `depart_t + C_i / v <= t` (so `p0` until the first step completes, `pn` from `arrive_t` on).
  Times are rounded to ms the same way `arrive_t` already is: step `i` is reached at
  `round(depart_t + C_i / v, 3)`.
- Everything that used `positions[agent]` uses the current cell at the time of the call: the
  body cells (feet and head) other agents collide with and can't place into, the eye for `view()`,
  reach and line-of-sight targets for `give` and `attack`, standing support and falls, visibility
  of the walker to others, `cut_off_if_solid`, start of a new path.
- **Destination reservation:** while walking, the agent's destination cell `pn` (and the cell above
  it) counts as occupied for *other agents' `move_to` destinations and placing* (so two agents can't
  plan to stop in the same cell, and nobody can wall in an arriving agent's spot). It does **not**
  block other agents' paths.
- **Walkers pass through each other.** Bodies of walking agents don't block other walks in
  progress (collisions are only checked at planning time against agents' current cells, as now).
- The walker's own actions stay refused until arrival (unchanged).

## 3. When the world changes during a walk
After every committed change to the grid (`place`, `remove`, a death removing a body), every
walk in progress is re-checked against the new grid, ignoring bodies:
- If the block under the walker's **current** cell is gone, it falls as today (`_apply_falls`), and
  the walk ends there.
- Otherwise, find the first **future** cell `p_j` (j > current index) that is no longer standable
  or no longer enterable from `p_{j-1}` with the step rules. If there is one, the walk is
  **truncated**: the walker continues to `p_{j-1}` and stops there (`arrive_t` becomes the time it
  reaches `p_{j-1}`, or now if it is already there).
- A walker whose destination reservation is placed into is impossible by §2; nothing else
  re-plans. Agents are never re-routed.

## 4. Log format
Still `log_version: 5`; the semantics are chosen by `config.walking`.
- `move_to` events: unchanged (`diff.agents[agent] = {from, to, path, depart_t, arrive_t}`).
- **Truncation:** the event that caused it carries, for each truncated walker,
  `diff.agents[walker] = {"from": current cell, "to": new end, "path": [current … new end],
  "depart_t": <the walk's original depart_t>, "arrive_t": new arrival, "stopped": true,
  "path_offset": i}` where `path_offset` is the index in the original path of the first listed cell,
  so readers keep timing from the original walk (cell `path[k]` of the entry is reached at
  `depart_t + C_{k+path_offset} / v`, `C` from the original path). Simplest for readers: keep the
  original walk and just cut it at the new end.
- **Falls** of a walker: as today (`fall: true` entry, which ends any walk).
- **Snapshots:** `positions` are **current cells** (not destinations). New key `walks`:
  `{agent: {"path": [...], "depart_t", "arrive_t"}}` for walks in progress at the snapshot time
  (after truncation), so a reader starting from a snapshot can continue. `walking_until` unchanged.
- `episode_end` positions are current cells.

## 5. What agents are told
Replace the sentence "Your position becomes the destination at once (for the rules on bodies and
reach), and you are walking until the arrival time." with, when `walking` is `"continuous"`:
*"While you walk you move along the path one cell at a time at that speed, and your body, your eyes
and what others can see or reach move with you. You are walking until you arrive. If a cell ahead
on your path becomes blocked or can no longer be stood on, you stop at the last cell before it.
Walking agents pass through each other; nobody can stop in, or place a block into, the cell
another agent is walking to."* With `"teleport"` the old sentence stays (v4 prompt byte-identical).
Observations: your own line shows your current cell and, while walking, the destination and
arrival time; other agents show their current cell and "walking".

## 6. Viewer
For `walking: "continuous"` logs, replayed positions (and the snapshot check) use the current cell
by §2 and truncation entries by §4; the smooth on-screen animation already interpolates along the
path and must stop at the truncated end. Teleport logs are unchanged.

## Implementation notes (engine, 2026-10-28)
Choices made where the contract above was open; `blockswarm/world.py`, `blockswarm/pathing.py`,
`blockswarm/replay.py` and `docs/log-format.md` ("Continuous walking") follow them.
- **Config / header:** `Config.walking` ("continuous" default, validated). In `"teleport"` mode
  the key is **left out of the header config**, so teleport logs are byte-identical to what the
  engine wrote before (checked against the previous engine on scripted shop and team runs);
  continuous logs carry `"walking": "continuous"`.
- **Walks:** `World.walks[agent] = {"path", "times", "depart_t", "arrive_t"}` for walks in
  progress (`times[i]` = when `path[i]` is reached, `pathing.walk_times`; `times[-1]` is pinned to
  the logged `arrive_t`, which equals it by construction). A zero-length `move_to` (already
  there) creates no walk. Teleport mode never creates walks.
- **Helpers:** `pathing.cumulative_lengths(path)` (`C_i`), `pathing.walk_times(path, depart_t,
  speed)`, `pathing.walk_index(times, t)` (current index), `Terrain.can_step(a, b)`;
  `World.current_cell(agent, t=None)`; `World._advance(t)` is the one place positions follow
  walks (it moves every walker to its current cell at `t` and drops finished walks), called at the
  start of every `act` (with the event's `t`, before the snapshot), `view`, `finish`, `snapshot`,
  `state`; `World.positions` is now a read property that advances walks to now first (it returns
  the engine's live dict of current cells; internal code uses `World._pos` at the event time).
  Public read helpers (`standable`, `walk_costs`, `find_path`, `reachable_stand_cells`,
  `cut_off_if_solid`, `shop_in_reach`) also advance to now first. After the episode ends,
  positions stay frozen at the end time.
- **Reservation:** `World.reserved_cells(exclude=None)` = destination feet and head cells of other
  walkers. `move_to` whose body would overlap one fails `"(x, y, z) is reserved: <walker> is
  walking to (x, y, z)"` (counted as an action, like any failed move); `place` into one fails the
  same way. `reachable_stand_cells` leaves reserved spots out (for the scripted bot and helpers).
- **Truncation (`World._recheck_walks`)** runs after every successful event with a grid change or
  a kill, after `_apply_falls` (so falling walkers have already ended their walk), walkers in
  `agents` order. Validity of a future cell = `Terrain.standable` and `Terrain.can_step` from the
  previous path cell on the grid alone (bodies ignored). **Deviation:** the new end backs off from
  `p_{j-1}` towards the current cell while it would overlap a *standing* (non-walking) agent's
  body or another walker's reserved destination, so the promise "nobody stops in a cell another
  agent is walking to" holds after truncation too; the current cell is the floor. Readers don't
  need to know this: the log entry gives the end.
- **Falls / deaths / jumps:** any non-walk position change (fall, jump, lift, death) ends the walk.
  A death's `at` is the current cell; dead walkers keep that cell.
- **view():** `position`, `eye`, visibility both ways and `side.current` use the current cell. New
  key (continuous only): **`destination`**: `[x, y, z]` of the walk's end while walking (after
  truncation, the new end), else `None`. Teleport views are unchanged (no `destination` key).
  Others' entries are unchanged (`pos` = current cell, `walking`).
- **Snapshots:** `walks` key only in continuous logs (`{}` when nobody walks); teleport snapshots
  have no `walks` (replay rejects one there).
- **Replay:** `continuous = log_version 5 and config.walking == "continuous"`; every record's `t`
  advances replayed walks first. Truncation entries are checked against the replayed walk
  (`path_offset` = current index, `from` = current cell, `path` = the original path slice,
  `depart_t` = original, `arrive_t = max(time of the new end, event t)`, only on an event with a
  grid change or a kill); `walks` in snapshots must match. Teleport, v3 and v4 replay unchanged:
  every log in `runs/pilot1-*`, `runs/pilot2-*`, `runs/demo` and the previous v5 fixture replays
  with byte-identical results under the old and the new `replay.py`.
- **Not done here (other owners):** `docs/team-battle.md` §3 still says "a walking agent is at
  its destination" for attack positions; with continuous walking it is the current cell.
- **Fixture:** `tests/data/v5_team_battle_sample.jsonl` is now continuous and adds a give to a
  walker mid-path (out of reach at its destination), a snapshot taken mid-walk and a walk cut by
  a block placed ahead of the walker. Tests: `tests/test_walking.py`.
