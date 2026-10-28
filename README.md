# BlockSwarm

A small Minecraft-like 3D grid world where several LLM agents, each with a body, build a house
together from an open-ended brief. They buy any of 131 Minecraft building blocks from a shop with
one shared wallet, and each agent is told that the agents with the most blocks still standing at
the end will be deployed. Built to study cooperation under competitive pressure, and to catch
misbehavior against an exact event log.

- **Spec:** [SPEC.md](SPEC.md)
- **Shop and incentive design:** [docs/shop-and-incentives.md](docs/shop-and-incentives.md)
- **Running LLM episodes:** [docs/running.md](docs/running.md)
- **Log format:** [docs/log-format.md](docs/log-format.md)
- **Running on the GCP VM:** [docs/vm.md](docs/vm.md)

## Quick start (no API key needed)

Requires Python 3.10+. No packages to install.

Run the tests:

```bash
python3 -m unittest discover -s tests
```

Run scripted (non-LLM) demo episodes. Logs go to `runs/demo/`:

```bash
python3 scripts/run_demo.py
```

Watch an episode in 3D in your browser:

```bash
python3 scripts/serve_viewer.py runs/demo/open_house_exact.jsonl
```

In the viewer: drag to orbit, scroll to zoom, space to play/pause, ←/→ to step through
events. Once it's open you can also drag any `.jsonl` log onto the page. To watch an episode
while it runs, add `--follow` (see docs/vm.md for watching runs on the VM).

## Layout

| Path | What |
| --- | --- |
| `blockswarm/world.py` | The world: grid, shop and wallet, bodies, walking, jumping, reach and line of sight, tools, logging |
| `blockswarm/catalog.py` | The shop's 131 Minecraft items with prices and properties |
| `blockswarm/llm/` | LLM orchestrator: prompts, observations, OpenRouter client, budget, runner |
| `blockswarm/tasks.py` | Per-agent scores and ranking; house analysis (not shown to agents) |
| `blockswarm/pathing.py` | Where agents can stand and how they walk |
| `blockswarm/live.py`, `blockswarm/clock.py` | Running agents concurrently on wall-clock or simulated time |
| `blockswarm/blueprints.py` | The scripted test bots' private house designs |
| `blockswarm/scripted.py` | A scripted, honest builder policy (baseline and test agent) |
| `blockswarm/replay.py` | Rebuilds state from a log and checks it against snapshots |
| `viewer/index.html`, `viewer/core.js` | 3D replay viewer with live follow (static page, three.js); core.js holds the shared log/state/scoring logic |
| `site_src/`, `scripts/export_site.py` | Public gallery site (live at blockswarm.pages.dev) |
| `scripts/` | Demo runner, viewer server, site export |
| `runs/` | Episode logs (not committed) |

## Status

Live, embodied engine with a shop, shared wallet and competitive deployment incentive (log v4); LLM agents via OpenRouter; 3D viewer with live follow; public gallery at blockswarm.pages.dev. First LLM pilot batch (v3, chest and rubric) ran on 2026-10-27; `configs/pilot2-*.json` are the v4 batch.
