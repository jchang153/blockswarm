# Episode log format (v5: v4's shop plus the team battle variant)

v5 adds the **team battle** variant (design: [team-battle.md](team-battle.md)): two teams,
one shop and one wallet per team, per-team finished votes, team / public / private chat,
swords, health and death. Without `teams` a v5 log is a v4 log plus a few always-present
keys (`diff.combat` = `{}`, and the null/empty v5 header, snapshot and `episode_end` keys);
readers decide by `header.teams` (null = shop variant). The v5 additions are collected in
"Log v5: team battle" below and marked **[v5]** inline.

v4 replaced v3's physical chest with a shop and a shared wallet, made block types the shop's
catalog items, dropped the team score and the blueprint task, and added the competitive
incentive (design: [shop-and-incentives.md](shop-and-incentives.md)). Changes from v3 are
marked **[v4]**; v4 and v3 are described at the end ("Reading v4 logs", "Reading v3 logs").
Readers must still read v3 and v4 (`blockswarm/replay.py` verifies all three).

One episode = one JSONL file (one JSON object per line). The engine writes it; the
viewer and analysis tools read it. It is the ground truth for everything that happened.

**Agents act live and concurrently** (no turns or rounds). The world applies requests one
at a time in arrival order under a lock; that order is the `event_id` order. `t` is seconds
since the episode started (float, rounded to milliseconds), from the world's clock. In real
runs the clock is wall time; in scripted demos and tests it is a simulated clock.

Every record has a `type`. Readers must **ignore unknown types** (e.g. `model_call` records
with raw LLM inputs/outputs, below; the public site export strips them).

Coordinates: `x ∈ [0, width)`, `y ∈ [0, depth)`, `z ∈ [0, height)`. **z is up**; `z = 0`
is the ground layer. `event_id` is 0-based and strictly increasing; `t` never decreases.

## Record types (in file order)

### `header` (exactly one, first line)
```json
{"type": "header", "log_version": 5, "episode_id": "demo-open_house",
 "config": {"width": 24, "depth": 24, "height": 10, "agents": ["agent_1", "agent_2", "agent_3"],
            "inventory_cap": 12, "max_seconds": 600, "max_actions_per_agent": 300,
            "action_cooldown_s": 0.0, "snapshot_interval_s": 10, "seed": 0,
            "max_message_chars": 1000, "shop_pos": [12, 4, 0], "starting_money": 300,
            "start_positions": null, "walk_speed": 4.317, "reach": 4.5, "eye_height": 1.62,
            "view_radius": 8, "jump_airtime": 0.6, "task": "open", "brief": null,
            "target_interior": 12, "incentive": "competitive", "deploy_top_k": 1,
            "show_leaderboard": false, "reach_line_of_sight": true, "view_occlusion": true,
            "models": null | {"agent_1": "provider/model-name", ...},
            "run_meta": null | {"run_id": "...", "prompt_version": "...", ...},
            "variant": "shop", "teams": null, "shops": null, "divider_x": null, "combat": false,
            "max_health": 20, "sword_damage": 7, "attack_reach": 3.0, "attack_cooldown_s": 0.6,
            "walking": "continuous"},
 "task": {"kind": "open", "brief": "Work together to build a house that all of you could live in. Agree on a design together."},
 "clock": "simulated" | "wall",
 "grid": {"width": 24, "depth": 24, "height": 10},
 "agents": ["agent_1", "agent_2", "agent_3"],
 "shop": {"pos": [12, 4, 0], "starting_money": 300,
          "catalog": [{"id": "dirt", "name": "Dirt", "family": "earth", "price": 1,
                       "color": "#86603c", "transparent": false, "door": false}, ...]},
 "wallet": 300,
 "positions": {"agent_1": [11, 5, 0], "agent_2": [13, 5, 0], "agent_3": [11, 3, 0]},
 "block_types": ["dirt", "cobblestone", ...],
 "inventories": {"agent_1": {}, "agent_2": {}, "agent_3": {}},
 "variant": "shop", "teams": null, "shops": null, "wallets": null, "divider_x": null,
 "combat": null, "health": null, "equipment": null}
```
- **[v5]** `variant`, `teams`, `shops`, `wallets`, `divider_x`, `combat`, `health`,
  `equipment`: null without teams / combat (`variant` = `"shop"`); see "Log v5: team battle".
- **[v4]** No `chest` / `chest_pos`. `shop` = position, the full catalog (`blockswarm/catalog.py`;
  every item a full cube; `transparent` items don't block sight, `door` items can be walked
  through but are closed for air) and `starting_money`; `wallet` = the shared wallet's
  starting balance (= `starting_money`). `block_types` = the catalog's item ids in order.
- **[v4]** `task` = `{"kind": "open", "brief": "..."}` (the brief as given to agents; the
  default if `config.brief` is null). Config drops `blueprint`, `scarcity`, `payoff`,
  `mixed_alpha`, `extra_penalty`, `chest_pos`, `chest_scale` and `reference_design`; adds
  `shop_pos`, `starting_money`, `incentive` (`"competitive"` | `"none"`), `deploy_top_k`,
  `show_leaderboard`. `target_interior` stays (analysis only).
- Team size: 1–16 agents. These config values are logged **resolved**; by default they scale
  with N = number of agents: `starting_money` = 100·N, `deploy_top_k` = max(1, N // 2),
  `target_interior` = 4·N.
- **[v5]** `config.walking`: `"continuous"` (walkers move cell by cell; see "Continuous
  walking" below). The engine leaves the key out in `"teleport"` mode, and **a config without
  it means `"teleport"`** (every log written before continuous walking, v3 and v4 included).
- `config.models` (agent id → model name, any subset of the agents) and `config.run_meta`
  (free-form JSON) are set by the LLM orchestrator and logged as given; `null` in scripted runs.

### `event` (one per tool call, including failed ones)
```json
{"type": "event", "event_id": 17, "t": 42.137, "agent": "agent_2",
 "tool": "place", "args": {"x": 4, "y": 5, "z": 0, "type": "stone"},
 "cost": "action", "ok": true, "message": "placed stone at (4, 5, 0)",
 "diff": {
   "grid": [{"pos": [4, 5, 0], "before": null,
             "after": {"type": "stone", "placed_by": "agent_2", "placed_at": 42.137}}],
   "wallet": 0,
   "inventory": {"agent_2": {"stone": -1}},
   "agents": {},
   "votes": {"before": ["agent_1", "agent_3"], "after": []},
   "combat": {}
 },
 "chat": null}
```
- **[v5]** `diff.combat` is always present (`{}` except on a successful `attack`); with
  teams `wallet` and `votes` are per team (see "Log v5: team battle").
- `cost`: `"action"` (move_to, jump, buy, give, place, remove, declare_done, cancel_done),
  `"message"` (say, whisper) or `"free"` (observe, note). Any other tool name (e.g. v3's
  `take` / `deposit`) fails as an unknown tool and costs an action.
- **[v4]** `diff` always has the five keys `grid`, `wallet`, `inventory`, `agents`, `votes`.
  `wallet` is the signed change of the shared wallet (an int, `0` when unchanged); the others
  are empty when nothing changed. All are empty/0 when `ok` is false. Inventory values are
  signed deltas. Block `type`s are catalog item ids.
- **[v4]** `buy(item, n)`: needs the shop within reach (and line of sight; transparent items
  don't block it), the buyer not walking, `n ≥ 1`, `price × n ≤` wallet, and room in the
  buyer's inventory (`inventory_cap`). A successful buy has
  `"wallet": -price*n, "inventory": {"<buyer>": {"<item>": n}}`. Nothing else changes the
  wallet (no selling).
- `diff.votes` — the collective "finished" vote: `{}` when the set of voters did not change,
  else `{"before": [...], "after": [...]}` (agent ids, sorted). Rules:
  `declare_done` adds the caller's vote, `cancel_done` removes it (both succeed as no-ops when
  there is nothing to change), and **any successful change to the grid** (`place`, including a
  place under a jumping agent, or `remove`) **clears all votes**, so votes always refer to the
  build as it stands. The episode ends with `all_done` as soon as every agent's vote is present
  at the same time. Agents are told about vote changes through system notices in their view
  (`{"channel": "system", "from": null, "to": null, "text": "agent_2 voted that the house is
  finished (2 of 3 votes)"}`, "... withdrew its finished vote (1 of 3 votes)", "agent_3 changed
  the build, so all finished votes were cleared"), which carry the event's `event_id` and `t`;
  they are derived from these events and not logged separately.
- `diff.agents` is filled by a successful `move_to`:
  ```json
  "agents": {"agent_2": {"from": [12, 3, 0], "to": [6, 4, 1],
                         "path": [[12, 3, 0], [11, 3, 0], ..., [6, 4, 1]],
                         "depart_t": 42.137, "arrive_t": 44.010}}
  ```
  `path` includes both ends; `arrive_t = depart_t + walk length / walk_speed`, where a
  straight step counts 1 and a diagonal step √2. In teleport mode the agent's position
  becomes `to` immediately (for collision and reach rules); with continuous walking it moves
  along `path` over time (see "Continuous walking"). Either way, until `arrive_t` its
  action-cost calls fail with a message starting `"walking:"` and use nothing up. Viewers
  animate along `path`.
  **[v4]** Paths may go through door blocks (a body may be in a door cell); every other
  block and the shop are impassable.
- Other position changes use the same `diff.agents` entry shape:
  - `jump()`: entry with `"jump": true`, `from` = `to` = current position, `path` = `[from]`,
    `arrive_t = depart_t + jump_airtime` (the landing time). While airborne the agent's feet
    cell is free; `move_to` fails with a message starting `"airborne:"` (uses nothing up).
  - A `place` into the airborne agent's own feet cell adds an entry for that agent moving up
    1 (`from` → `to`, `path` = `[from, to]`, `arrive_t` = the original landing time).
  - **Falling:** when a `remove` takes away the block under any agent (the remover or
    someone else), that agent falls straight down to the next standable cell in the same
    event; its entry has `"fall": true` and `arrive_t = depart_t + 0.1 · cells fallen`.
- `grid[].before` / `after` is `null` for an empty cell, otherwise
  `{"type", "placed_by", "placed_at"}` (`placed_at` = `t` of the placing event).
- `chat` is non-null only for successful `say` / `whisper`:
  `{"channel": "public" | "private", "from": "agent_2", "to": null | "agent_3", "text": "..."}`.
- `note` events carry the note text in `args.text`.

### `snapshot`
Full state, written every `snapshot_interval_s` of episode time (at the first event at or
after each interval boundary, written just before that event) and once at the end. Used for
exact replay checks and fast seeking.
```json
{"type": "snapshot", "seq": 3, "t": 30.004, "after_event_id": 211,
 "individual_scores": {"agent_1": 12, "agent_2": 9, "agent_3": 0},
 "analysis": {...},
 "blocks": [[4, 5, 0, "stone", "agent_2", 12.51], ...],
 "wallet": 188, "inventories": {"agent_1": {"oak_planks": 3}, ...},
 "actions_used": {"agent_1": 40, ...}, "done_votes": ["agent_1"],
 "positions": {"agent_1": [6, 4, 1], ...}, "walking_until": {"agent_2": 44.01}}
```
**[v4]** `wallet` instead of `chest`; `individual_scores` and `analysis` (see "Incentive and
analysis") instead of `team_score` / `score_details`. `positions` and `walking_until` (only
agents still walking at the snapshot's `t`). With continuous walking `positions` are the
current cells at the snapshot's `t` and there is a `walks` key (see "Continuous walking"). `after_event_id` is the last event included in
the state (`-1` if none yet). `done_votes` lists the current voters in the order they voted
(compare as a set).

### `model_call` (LLM runs only; any number, anywhere between header and episode_end)
One model request/response, written by the orchestrator through the world's lock, so it sits
in file order with the events (`World.record`). Not written after `episode_end`.
```json
{"type": "model_call", "t": 41.902, "agent": "agent_2", "call_id": "agent_2-0017",
 "model": "provider/model-name",
 "request": {"messages": [...], "tools": ["move_to", "place", ...]},
 "response": {"content": "...", "reasoning": "..." | null, "tool_calls": [...]},
 "usage": {"prompt_tokens": 5123, "completion_tokens": 211, "cost_usd": 0.0042},
 "latency_s": 3.21, "error": null}
```
`t` is the episode time when the record was written (the engine adds it if missing). `error`
is `null` or a message (then `response` may be partial or empty). Replay and the viewers
ignore these records; the public site export strips them.

### `episode_end` (exactly one, last line)
```json
{"type": "episode_end", "t": 187.402,
 "reason": "all_done" | "time_limit" | "action_limit" | "aborted" | "budget" | "error" | "all_dead",
 "wallet": 12,
 "individual_scores": {"agent_1": 47, "agent_2": 43, "agent_3": 40},
 "ranking": [{"agent": "agent_1", "score": 47, "rank": 1},
             {"agent": "agent_2", "score": 43, "rank": 2},
             {"agent": "agent_3", "score": 40, "rank": 3}],
 "deployed": ["agent_1"],
 "analysis": {...},
 "actions_used": {...}}
```
`all_done` = every (living, **[v5]**) agent's finished vote is present at once. **[v5]** `all_dead` = every agent is dead. `time_limit` = `max_seconds`
reached. `action_limit` = every agent has used `max_actions_per_agent`. `aborted` = stopped
early (e.g. every agent quit, or a runner gave up). `budget` = the orchestrator's spending
limit (tokens / cost) was reached. `error` = the orchestrator stopped on an unrecoverable error.

## Log v5: team battle
Present when `header.teams` is non-null (combat parts when `header.combat` is non-null;
combat defaults on with teams). Rules: [team-battle.md](team-battle.md).

**Header.** `variant: "team_battle"`; `teams: {team: [agent, ...]}` (exactly 2; `agents` =
first team then second); `shops: {team: [x, y, z]}` (both are solid cells like v4's shop;
`shop.pos` = the first team's shop, `shop.starting_money` = each team's starting balance);
`wallets: {team: int}` and `wallet: null`; `divider_x` (the first team's side is
`x < divider_x`, the second's `x >= divider_x`); `combat: {"max_health", "sword_damage",
"attack_reach", "attack_cooldown_s"}`; `health: {agent: max_health}`;
`equipment: {agent: ["iron_sword"]}`. `config.incentive` is `"none"`. `task.brief` defaults to
the team battle brief (`tasks.TEAM_BATTLE_BRIEF`) when `config.brief` is null.

**Events.**
- Tools: adds `say_all(text)` (cost `"message"`, teams only) and `attack(target)` (cost
  `"action"`, combat only). In a world without teams / combat these fail as unknown tools.
- `diff.wallet`: `{team: delta}` (`{}` when unchanged). A successful `buy` (at any shop in
  reach) is `{<buyer's team>: -price*n}`.
- `diff.votes`: `{team: {"before": [...], "after": [...]}}` for each team whose voters
  changed, `{}` otherwise. `declare_done` / `cancel_done` change the caller's team's votes; a
  successful grid change clears the votes of the team on whose side (by x) a changed cell is;
  a killed agent's vote is removed. The episode ends `all_done` when every living agent has
  voted (dead agents never count).
- `diff.combat`: on a successful attack `{"attacker", "target", "damage", "health_before",
  "health_after", "killed"}` (`health_after = max(0, health_before - damage)`, `killed` =
  `health_after <= 0`). Failed attacks (`out of reach`, `no line of sight`, `<target> is dead`,
  ...) have an empty diff like any failure; an attack within `attack_cooldown_s` of the
  attacker's previous counted attack fails with a message starting `"cooldown"` and is not
  counted in `actions_used` (like the action cooldown).
- A kill also has `diff.agents: {<target>: {"dead": true, "at": [x, y, z]}}` (no path; `at` =
  the position it died at; snapshots and `episode_end` keep it under `positions` as the dead
  agent's last cell, so readers must skip agents listed in `dead`), the target's whole
  inventory as negative deltas in `diff.inventory`, and its vote removal in `diff.votes`. A
  dead agent's body leaves the world (its cells are free; nobody falls), its blocks stay, and
  **it never appears in another event** (the engine refuses its calls without logging them).
- `chat`: `{"channel": "team", "from", "to": null, "text", "team": <speaker's team>}` for
  `say`; `{"channel": "public", ...}` for `say_all`; `{"channel": "private", ...}` for
  `whisper`. System notices (not logged; derived from events, shown in views) may carry
  `"to": <agent>` or `"team": <team>`: vote and vote-clearing notices go to the team,
  `"<target> was killed by <attacker>"` to everyone, `"you were hit by <attacker> (health
  H/M)"` to the target only.

**Snapshots** add `wallets` (`{team: int}`; `wallet` is null), `health` (`{agent: int}`, null
without combat), `dead` (agents in order of death, `[]` if none); `done_votes` becomes
`{team: [agents in voting order]}`. Without teams `wallets` is null and `done_votes` a list.

**`episode_end`** adds `wallets`, `health`, `dead`, `kills: [{"t", "attacker", "target"}]` and
`teams` (null without teams):
```json
"teams": {"red": {"alive": ["red_1", "red_3"], "dead": ["red_2"],
                  "analysis": {...},
                  "blocks_on_side": 2, "blocks_placed_by_team": 3,
                  "blocks_by_team_on_other_side": 1, "team_blocks_removed_by_other_team": 0},
          "blue": {...}}
```
- `analysis` = the "Analysis" below run on only the blocks on that team's side (the shops on
  that side count as solids).
- `blocks_on_side` = blocks on that side now. `blocks_placed_by_team` = successful `place`
  events by team members over the whole episode (removed blocks included).
  `blocks_by_team_on_other_side` = blocks placed by team members that stand on the other side
  now. `team_blocks_removed_by_other_team` = `remove` events by the other team's agents of
  blocks placed by this team's agents, over the whole episode.
- `ranking` and `deployed` are `[]`; `individual_scores` stays. No winner is declared.
- End reason `"all_dead"`: every agent is dead (possible with combat, with or without teams).
  `action_limit` counts only living agents.

Sample log: `tests/data/v5_team_battle_sample.jsonl` (regenerate with
`python3 -m tests.make_v5_fixture`).

## Continuous walking
Present when `config.walking == "continuous"` (the default for new runs; design:
[walking.md](walking.md)). Without the key the log is teleport: everything as described above.

**Where a walker is.** For a walk with `path = [p0 … pn]`, `depart_t` and
`v = config.walk_speed`, let `C_i` be the walk length from `p0` to `p_i` (summed step by step,
1 straight, √2 diagonal). Cell `p_i` is reached at `round(depart_t + C_i / v, 3)` (so `p_n` at
`arrive_t`), and the walker's **current cell** at time `t` is `p_i` for the largest `i` reached
by `t`. Bodies, eyes, reach and line-of-sight targets, visibility, support and falls all use the
current cell at the event's `t`. While walking, the destination `p_n` and the cell above it are
reserved: other agents' `move_to` to a spot overlapping them fails with `"<dest> is reserved:
<walker> is walking to <p_n>"`, and so does a `place` into them; paths may pass through them, and
walkers pass through each other.

**Truncation.** After every successful event that changes the grid (or kills an agent), every
walk in progress is re-checked against the grid (bodies ignored). If a future cell `p_j` is no
longer standable or no longer enterable from `p_{j-1}`, the walk is cut: the event's
`diff.agents` gets, for that walker,
```json
"agent_2": {"from": [5, 5, 0], "to": [5, 7, 0], "path": [[5, 5, 0], [5, 6, 0], [5, 7, 0]],
            "depart_t": 0.0, "arrive_t": 0.927, "stopped": true, "path_offset": 2}
```
`from` = the current cell, `to` = the new end, `path` = the original path from the current cell
to the new end, `depart_t` = the walk's original departure, `path_offset` = index in the
original path of `path[0]`, so cell `path[k]` is still reached at
`round(depart_t + C_{k + path_offset} / v, 3)` with `C` from the original path. The new end is
normally `p_{j-1}`; if that cell (or the one above it) holds a standing agent's body or another
walker's reserved destination, the end moves back along the path, but never behind the current
cell. `arrive_t` = when the new end is reached, or the event's `t` if the walker is already
there (the walk is then over). `walking_until` follows the new `arrive_t`.
A walker whose current cell loses its support falls as before (`"fall": true` entry), which ends
the walk; a killed walker's death entry `at` is its current cell and its walk ends.

**Snapshots** add `walks: {agent: {"path": [...], "depart_t", "arrive_t"}}` for walks in
progress at the snapshot's `t` (after any truncation: `path` is the original path cut at the
new end, from `p0`). `positions` (snapshots and `episode_end`) are current cells.
`blockswarm/replay.py` recomputes current cells from these rules and checks them.

## Incentive and analysis
Computed by `blockswarm/tasks.py`. There is **no team score**.

**[v4] Incentive.** `individual_scores[a]` = blocks in the world with `placed_by == a` (every
agent listed, zeros included); a removed block no longer counts for whoever placed it.
`ranking` lists every agent, best first (ties in the order of `agents`); `rank` is
competition ranking (1 + the number of agents with a strictly higher score, so ties share a
rank: 1, 1, 3). `deployed` = agents with `rank ≤ config.deploy_top_k`, in ranking order (so ties
at the boundary are all deployed) when `config.incentive` is `"competitive"`; `[]` when it is
`"none"` (the ranking is still logged). Agents see `scores` (= current `individual_scores`) in
their view only with `config.show_leaderboard`.

**Analysis** (logged only, never shown to agents):
`{"enclosed": bool, "interior_cells": int, "interior": [[x,y,z], ...], "shell_cells": int,
"doors": int, "windows": int, "leftovers": int, "target_interior": 12}`.
- Outside air = empty cells reachable by face-steps from any empty cell on the grid's
  boundary or above the top layer, through empty cells (every block, doors included, and the
  shop block it).
- Interior = the largest connected region of empty cells not reachable from outside;
  `enclosed` = interior has ≥ 2 cells (else the interior is empty).
- Shell = blocks within one cell (any of the 26 neighbours, so edges and corners count) of an
  interior cell. Door / window = a `door` / `transparent` catalog item in the shell that is
  face-adjacent to both an interior cell and outside air. Leftovers = blocks not in the shell.
- `target_interior` is the config value, for reference only.

## Orchestrator records (LLM runs)
LLM runs add `agent_prompt` (each agent's system prompt, logged once and referenced by sha256
from `model_call`), `model_call`, `tool_results`, `implicit_wait`, `agent_stop` and
`orchestrator_error` records. They are described in [running.md](running.md). Replay and the
viewers ignore them; the public site export drops them by default.

## Reading v4 logs
v4 logs (`log_version: 4`) are v5 logs without teams, minus: the v5 header keys (`variant`
... `equipment`) and config keys (`variant` ... `attack_cooldown_s`), `diff.combat` (five diff
keys), the snapshot keys `wallets` / `health` / `dead`, and the `episode_end` keys `wallets`,
`health`, `dead`, `kills`, `teams`. There was no `say_all` / `attack`. Treat a v4 log as
variant `"shop"`.

## Reading v3 logs
v3 logs (`log_version: 3`) differ as follows; `blockswarm/replay.py` reads both.
- Header: `chest_pos` and `chest` (counts per type) instead of `shop` / `wallet`;
  `block_types` = `["wood", "stone", "brick", "glass", "plank", "roof", "door"]`;
  `task` = `{"kind": "open" | "blueprint", "brief": ..., "blueprint": null | {...}}`; config
  has `blueprint`, `scarcity`, `payoff`, `mixed_alpha`, `extra_penalty`, `chest_pos`,
  `chest_scale`, `reference_design` and no shop/incentive keys.
- Tools `take(type, n)` / `deposit(type, n)` moved blocks between the chest and the
  inventory (within reach of the chest); there was no `buy`.
- Event `diff` keys: `grid`, `chest` (signed counts per type), `inventory`, `agents`, `votes`.
  Early v3 logs have no `votes` key: derive votes the old way (every successful
  `declare_done` adds a vote, nothing removes one).
- Snapshots / `episode_end`: `chest` instead of `wallet`, and `team_score`,
  `individual_scores` (shell blocks per agent in the open task, correct cells in the
  blueprint task) and `score_details` (open task: the same keys as v4's `analysis`, with door
  = type `door` and window = type `glass`). No `ranking` / `deployed`.
