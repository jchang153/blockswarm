"""Run scripted (non-LLM) demo episodes on simulated time, write their logs, and replay-check them."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blockswarm.clock import SimClock  # noqa: E402
from blockswarm.recorder import Recorder  # noqa: E402
from blockswarm.replay import load_log, replay  # noqa: E402
from blockswarm.scripted import run_simulated  # noqa: E402
from blockswarm.world import Config, World  # noqa: E402

# (log name, number of agents); the scripted bots build their shared private design (the house).
DEMOS = [("open_house", 3), ("open_house_5", 5)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(ROOT / "runs" / "demo"), help="where to write the logs")
    args = parser.parse_args()
    out = Path(args.out_dir)
    for name, n in DEMOS:
        path = out / f"{name}.jsonl"
        cfg = Config(agents=[f"agent_{i}" for i in range(1, n + 1)], max_seconds=900, seed=0)
        rec = Recorder(path)
        try:
            world = run_simulated(World(cfg, rec, episode_id=f"demo-{name}", clock=SimClock()))
        finally:
            rec.close()
        records = load_log(path)
        state = replay(records)
        end = state["end"]
        events = [r for r in records if r["type"] == "event"]
        failed = sum(not e["ok"] for e in events)
        details = {k: v for k, v in end["analysis"].items() if k != "interior"}
        ranking = ", ".join(f"{r['rank']}. {r['agent']} {r['score']}" for r in end["ranking"])
        print(f"{path.name}: t={end['t']:.1f}s reason={end['reason']} "
              f"wallet spent={cfg.starting_money - end['wallet']}/{cfg.starting_money} "
              f"ranking=[{ranking}] deployed={end['deployed']} analysis={details} events={len(events)} "
              f"failed={failed} replay_ok snapshots={state['snapshots_checked']}")
        assert world.done


if __name__ == "__main__":
    main()
