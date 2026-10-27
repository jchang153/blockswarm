# BlockSwarm

A small Minecraft-like 3D grid world where several LLM agents build a shared structure
(e.g. a house) from a shared, limited chest of blocks. Built to study how agents split up
work and share resources, and to catch misbehavior against an exact event log.

- **Spec:** [SPEC.md](SPEC.md)
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
| `blockswarm/world.py` | The world: grid, chest, inventories, tools, rules, scoring, logging |
| `blockswarm/blueprints.py` | Target structures (`hut_small`, `house`) |
| `blockswarm/scripted.py` | A scripted, honest builder policy (baseline and test agent) |
| `blockswarm/replay.py` | Rebuilds state from a log and checks it against snapshots |
| `viewer/index.html`, `viewer/core.js` | 3D replay viewer with live follow (static page, three.js); core.js holds the shared log/state/scoring logic |
| `scripts/` | Demo runner and viewer server |
| `runs/` | Episode logs (not committed) |

## Status

Milestone 1 (engine, scripted demo, viewer). LLM agents via OpenRouter come next.
