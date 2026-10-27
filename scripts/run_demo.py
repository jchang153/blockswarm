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

# (task, blueprint, scarcity); in open mode the scripted bots build their shared internal design.
DEMOS = [("open", "house", "exact"), ("open", "house", "short"),
         ("blueprint", "house", "exact"), ("blueprint", "hut_small", "exact")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(ROOT / "runs" / "demo"), help="where to write the logs")
    args = parser.parse_args()
    out = Path(args.out_dir)
    for task, blueprint, scarcity in DEMOS:
        name = f"{task}_{blueprint}_{scarcity}"
        path = out / f"{name}.jsonl"
        cfg = Config(task=task, blueprint=blueprint, scarcity=scarcity, seed=0)
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
        details = {k: v for k, v in end["score_details"].items() if k != "interior"}
        print(f"{path.name}: t={end['t']:.1f}s reason={end['reason']} team={end['team_score']:.3f} "
              f"individual={end['individual_scores']} details={details} events={len(events)} "
              f"failed={failed} replay_ok snapshots={state['snapshots_checked']}")
        assert world.done


if __name__ == "__main__":
    main()
