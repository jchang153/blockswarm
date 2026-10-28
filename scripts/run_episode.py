#!/usr/bin/env python3
"""Run one live LLM episode.

    python3 scripts/run_episode.py configs/pilot.json --dry-run     # validate + plan, no calls
    python3 scripts/run_episode.py configs/mock.json --mock         # offline, scripted mock model
    python3 scripts/run_episode.py configs/pilot.json               # real run (paid, OpenRouter)

Options: --run-dir DIR, --budget-usd X, --n-agents N, --allow-unknown-price.
Loads ./.env (repo root) without overriding variables already set. See docs/running.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from blockswarm.llm.runner import ConfigError, load_dotenv, run_episode  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run one live BlockSwarm LLM episode.")
    ap.add_argument("config", help="run config JSON (e.g. configs/pilot.json)")
    ap.add_argument("--dry-run", action="store_true", help="validate config, key, prices and budget; no calls")
    ap.add_argument("--mock", action="store_true", help="use the offline mock model (no network, no key)")
    ap.add_argument("--run-dir", help="output directory (default runs/<run_id>)")
    ap.add_argument("--budget-usd", type=float, help="override budget_usd (hard spending cap)")
    ap.add_argument("--n-agents", type=int, help="override the number of agents (ids agent_1..agent_N)")
    ap.add_argument("--allow-unknown-price", action="store_true",
                    help="allow models without a price in llm.prices (reserves at a high fallback price)")
    args = ap.parse_args(argv)
    if not args.mock:
        load_dotenv(REPO / ".env")
    try:
        res = run_episode(args.config, run_dir=args.run_dir, dry_run=args.dry_run, mock=args.mock,
                          budget_usd=args.budget_usd, n_agents=args.n_agents,
                          allow_unknown_price=args.allow_unknown_price)
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.dry_run:
        return 0 if res.get("ok") else 1
    return 130 if res.get("interrupted") else 0


if __name__ == "__main__":
    sys.exit(main())
