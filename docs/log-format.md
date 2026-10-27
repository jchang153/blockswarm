# Episode log format (v3: live, embodied)

v3 adds agent bodies: positions, walking and the physical chest (marked **[v3]** below).

One episode = one JSONL file (one JSON object per line). The engine writes it; the
viewer and analysis tools read it. It is the ground truth for everything that happened.

**Agents act live and concurrently** (no turns or rounds). The world applies requests one
at a time in arrival order under a lock; that order is the `event_id` order. `t` is seconds
since the episode started (float, rounded to milliseconds), from the world's clock. In real
runs the clock is wall time; in scripted demos and tests it is a simulated clock.

Every record has a `type`. Readers must **ignore unknown types** (later milestones add
e.g. `model_call` records with raw LLM inputs/outputs).

Coordinates: `x ∈ [0, width)`, `y ∈ [0, depth)`, `z ∈ [0, height)`. **z is up**; `z = 0`
is the ground layer. `event_id` is 0-based and strictly increasing; `t` never decreases.

## Record types (in file order)

### `header` (exactly one, first line)
```json
{"type": "header", "log_version": 3, "episode_id": "demo-house-exact",
 "config": {"width": 16, "depth": 16, "height": 8, "agents": ["agent_1", "agent_2", "agent_3"],
            "blueprint": "house", "scarcity": "exact", "inventory_cap": 12,
            "max_seconds": 600, "max_actions_per_agent": 300, "action_cooldown_s": 0.0,
            "snapshot_interval_s": 10, "payoff": "team", "mixed_alpha": 0.5,
            "extra_penalty": 0.1, "seed": 0, "max_message_chars": 1000,
            "chest_pos": [13, 2, 0], "start_positions": null, "walk_speed": 4.317,
            "reach": 4.5, "eye_height": 1.62, "view_radius": 8, "jump_airtime": 0.6,
            "task": "open", "brief": "Work together to build a house ...",
            "reference_design": "house", "target_interior": 12,
            "reach_line_of_sight": true, "view_occlusion": true},
 "task": {"kind": "open" | "blueprint", "brief": "..." | null,
          "blueprint": null | {"name": "house", "cells": [[4, 5, 0, "stone"], ...]}},
 "clock": "simulated" | "wall",
 "grid": {"width": 16, "depth": 16, "height": 8},
 "agents": ["agent_1", "agent_2", "agent_3"],
 "chest_pos": [13, 2, 0],
 "positions": {"agent_1": [12, 3, 0], "agent_2": [14, 3, 0], "agent_3": [12, 1, 0]},
 "block_types": ["wood", "stone", "brick", "glass", "plank", "roof", "door"],
 "chest": {"stone": 35, "brick": 52, ...},
 "inventories": {"agent_1": {}, "agent_2": {}, "agent_3": {}}}
```

### `event` (one per tool call, including failed ones)
```json
{"type": "event", "event_id": 17, "t": 42.137, "agent": "agent_2",
 "tool": "place", "args": {"x": 4, "y": 5, "z": 0, "type": "stone"},
 "cost": "action", "ok": true, "message": "placed stone at (4, 5, 0)",
 "diff": {
   "grid": [{"pos": [4, 5, 0], "before": null,
             "after": {"type": "stone", "placed_by": "agent_2", "placed_at": 42.137}}],
   "chest": {},
   "inventory": {"agent_2": {"stone": -1}},
   "agents": {}
 },
 "chat": null}
```
- `cost`: `"action"` (move_to, jump, take, deposit, give, place, remove, declare_done),
  `"message"` (say, whisper) or `"free"` (observe, note).
- `diff` always has the four keys `grid`, `chest`, `inventory`, `agents`; they are empty when
  nothing changed (always the case when `ok` is false). Chest and inventory values are
  signed deltas.
- **[v3]** `diff.agents` is filled by a successful `move_to`:
  ```json
  "agents": {"agent_2": {"from": [12, 3, 0], "to": [6, 4, 1],
                         "path": [[12, 3, 0], [11, 3, 0], ..., [6, 4, 1]],
                         "depart_t": 42.137, "arrive_t": 44.010}}
  ```
  `path` includes both ends; `arrive_t = depart_t + walk length / walk_speed`, where a
  straight step counts 1 and a diagonal step √2. The agent's position becomes `to`
  immediately (for collision and reach rules); until `arrive_t` its action-cost calls fail
  with a message starting `"walking:"` and use nothing up. Viewers animate along `path`.
- **[v3]** Other position changes use the same `diff.agents` entry shape:
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
 "team_score": 0.31, "individual_scores": {"agent_1": 12, ...},
 "score_details": {...},
 "blocks": [[4, 5, 0, "stone", "agent_2", 12.51], ...],
 "chest": {"stone": 0, ...}, "inventories": {"agent_1": {"brick": 3}, ...},
 "actions_used": {"agent_1": 40, ...}, "done_votes": ["agent_1"],
 "positions": {"agent_1": [6, 4, 1], ...}, "walking_until": {"agent_2": 44.01}}
```
**[v3]** `positions` and `walking_until` (only agents still walking at the snapshot's `t`).
`after_event_id` is the last event included in the state (`-1` if none yet).

### `episode_end` (exactly one, last line)
```json
{"type": "episode_end", "t": 187.402,
 "reason": "all_done" | "time_limit" | "action_limit" | "aborted",
 "team_score": 1.0, "individual_scores": {...}, "score_details": {...},
 "actions_used": {...}}
```
`action_limit` = every agent has used `max_actions_per_agent`.

## Scoring
Computed by the task (`blockswarm/tasks.py`). Every snapshot and the `episode_end` carry
`team_score` (0–1), `individual_scores` (agent → int) and `score_details` (task-specific;
viewers should show its keys generically).

**`open` task** — `score_details`:
`{"enclosed": bool, "interior_cells": int, "interior": [[x,y,z], ...], "shell_cells": int,
"doors": int, "windows": int, "leftovers": int, "target_interior": 12}`.
- Outside air = empty cells reachable by face-steps from any empty cell on the grid's
  boundary or above the top layer, through empty cells (doors and the chest block movement).
- Interior = the largest connected region of empty cells not reachable from outside;
  `enclosed` = interior has ≥ 2 cells.
- Shell = blocks within one cell (any of the 26 neighbours, so edges and corners count) of an
  interior cell. Door / window = door / glass block in the
  shell that is face-adjacent to both an interior cell and outside air. Leftovers = blocks not
  in the shell.
- `team_score` = 0 if not enclosed, else 0.5 + 0.2·[doors ≥ 1] + 0.15·min(windows, 2)/2
  + 0.15·min(1, interior_cells / target_interior) − 0.01·leftovers, clamped to [0, 1],
  rounded to 6 decimals.
- `individual_scores[a]` = shell blocks with `placed_by == a`.

**`blueprint` task** — `score_details`: `{"correct_cells", "blueprint_cells", "extra_cells"}`.
- Correct = blueprint position holds a block of the blueprint's type. Extra = block at a
  position not in the blueprint (wrong-type blocks on blueprint positions are neither).
- `team_score = correct / blueprint_cells − extra_penalty · extra / blueprint_cells`, floored at 0.
- `individual_scores[a]` = correct cells with `placed_by == a`.
