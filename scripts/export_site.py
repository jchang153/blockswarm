#!/usr/bin/env python3
"""Build the public BlockSwarm replay website (a plain static folder).

    python3 scripts/export_site.py runs/demo/*.jsonl --out site
    python3 -m http.server -d site 8000      # preview at http://localhost:8000/

What it does:
  * copies site_src/ into the output folder, plus viewer/core.js as js/core.js (shared replay logic),
    and writes site-config.json ({title, subtitle, data_base_url}; data_base_url defaults to ./data/);
  * for each episode log: checks it parses (header first), applies the public-export filter and
    writes data/<episode_id>.jsonl.gz;
  * writes data/episodes.json (one summary per episode incl. its final blocks for the gallery
    thumbnails, so the gallery never downloads full logs).

Split hosting (pages on one host, data elsewhere, e.g. a public Hugging Face dataset):
    python3 scripts/export_site.py runs/*.jsonl --data-only --out hf_data
    python3 scripts/export_site.py --site-only --data-base-url https://huggingface.co/datasets/<user>/<repo>/resolve/main/ --out site
See site_src/README-deploy.md.

Public-export filter (default): keeps header / event / snapshot / episode_end records and drops
everything else (`model_call` records with raw model inputs/outputs, and any record type this
script does not know). `--keep-model-calls` keeps `model_call` records. `--drop-notes` removes the
text of `note` events (agents' private scratchpads). Chat (public and private) is always kept.

Optional per-run metadata: a JSON sidecar next to the log, `<log>.meta.json` (e.g.
`runs/x/ep.jsonl.meta.json`) or `<log stem>.meta.json`, with any of
  {"title": str, "model": str | {agent: model}, "notes": str, "created": str, "hidden": bool}.

Standard library only (Python 3.10+). The output folder uses relative paths only, so it can be
hosted anywhere static (Cloudflare Pages, a Hugging Face static Space, GitHub Pages, ...).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE_SRC = ROOT / "site_src"
CORE_JS = ROOT / "viewer" / "core.js"
MARKER = ".blockswarm-site"  # marks a folder this script built (safe to wipe on rebuild)

KNOWN_TYPES = {"header", "event", "snapshot", "episode_end"}
OPEN_DETAIL_KEYS = ("enclosed", "interior_cells", "shell_cells", "doors", "windows", "leftovers", "target_interior")
BLUEPRINT_DETAIL_KEYS = ("correct_cells", "blueprint_cells", "extra_cells")


class LogError(Exception):
    pass


def read_records(path: Path) -> list[dict]:
    """Parse a JSONL log; the first record must be the header. Raises LogError."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise LogError(f"line {n}: not valid JSON ({e.msg})") from None
            if not isinstance(rec, dict):
                raise LogError(f"line {n}: expected a JSON object")
            if not records and rec.get("type") != "header":
                raise LogError(f"line {n}: first record must be the header (found type {rec.get('type')!r})")
            records.append(rec)
    if not records:
        raise LogError("empty file")
    return records


def public_filter(records: list[dict], keep_model_calls: bool, drop_notes: bool) -> tuple[list[dict], dict]:
    """Apply the public-export filter. Returns (kept records, counts of what was dropped)."""
    out, dropped = [], {}
    for rec in records:
        t = rec.get("type")
        if t in KNOWN_TYPES or (t == "model_call" and keep_model_calls):
            if drop_notes and t == "event" and rec.get("tool") == "note":
                rec = dict(rec)
                args = dict(rec.get("args") or {})
                if "text" in args:
                    args.pop("text")
                    args["redacted"] = True
                    dropped["note_text"] = dropped.get("note_text", 0) + 1
                rec["args"] = args
            out.append(rec)
        else:
            dropped[str(t)] = dropped.get(str(t), 0) + 1
    return out, dropped


def task_of(header: dict) -> tuple[str, str | None, str | None]:
    """(kind, brief, blueprint name) — mirrors taskOf() in viewer/core.js (log v1-v3)."""
    cfg = header.get("config") or {}
    task = header.get("task") if isinstance(header.get("task"), dict) else None
    bp = (task or {}).get("blueprint") or header.get("blueprint")
    kind = (task or {}).get("kind") or ("blueprint" if bp else (cfg.get("task") or "open"))
    if kind == "blueprint" and not bp:
        kind = "open"
    brief = (task or {}).get("brief") or cfg.get("brief")
    bp_name = (bp or {}).get("name") or cfg.get("blueprint") if kind == "blueprint" else None
    return kind, brief, bp_name


def agent_models(header: dict, agents: list[str], meta: dict) -> dict:
    """Best-effort model per agent from the header/config or the sidecar; None when unknown."""
    cfg = header.get("config") or {}
    out = {a: None for a in agents}
    for src in (cfg.get("model"), cfg.get("models"), header.get("model"), header.get("models"),
                header.get("agent_models"), cfg.get("agent_models"), meta.get("model")):
        if isinstance(src, str) and src:
            for a in agents:
                out[a] = out[a] or src
        elif isinstance(src, dict):
            for a in agents:
                v = src.get(a)
                if isinstance(v, dict):
                    v = v.get("model") or v.get("name")
                if isinstance(v, str) and v:
                    out[a] = out[a] or v
        elif isinstance(src, list) and len(src) == len(agents):
            for a, v in zip(agents, src):
                if isinstance(v, str) and v:
                    out[a] = out[a] or v
    # per-agent config blocks, e.g. config.agent_configs = {"agent_1": {"model": ...}}
    for k in ("agent_configs", "agents_config"):
        block = cfg.get(k) or header.get(k)
        if isinstance(block, dict):
            for a in agents:
                v = block.get(a)
                if isinstance(v, dict) and isinstance(v.get("model"), str):
                    out[a] = out[a] or v["model"]
    return out


def final_blocks(records: list[dict]) -> list[list]:
    """Replay grid diffs to get the final blocks as [x, y, z, type, placed_by]."""
    grid: dict[tuple, dict] = {}
    for rec in records:
        if rec.get("type") != "event":
            continue
        for g in ((rec.get("diff") or {}).get("grid") or []):
            pos = tuple(g.get("pos") or ())
            if len(pos) != 3:
                continue
            if g.get("after"):
                grid[pos] = g["after"]
            else:
                grid.pop(pos, None)
    return [[x, y, z, b.get("type"), b.get("placed_by")] for (x, y, z), b in sorted(grid.items())]


def summarize(records: list[dict], meta: dict) -> dict:
    header = records[0]
    cfg = header.get("config") or {}
    events = [r for r in records if r.get("type") == "event"]
    snaps = [r for r in records if r.get("type") == "snapshot"]
    end = next((r for r in reversed(records) if r.get("type") == "episode_end"), None)
    agents = list(header.get("agents") or cfg.get("agents") or [])
    kind, brief, bp_name = task_of(header)
    models = agent_models(header, agents, meta)

    # final scores: the engine's own numbers (episode_end, else the last snapshot)
    src = end or (snaps[-1] if snaps else None)
    details = {}
    team_score = None
    if src:
        team_score = src.get("team_score")
        det = src.get("score_details") if isinstance(src.get("score_details"), dict) else {}
        for k in OPEN_DETAIL_KEYS + BLUEPRINT_DETAIL_KEYS:
            if k in det:
                details[k] = det[k]
            elif k in src:  # log v2 put blueprint details at the top level
                details[k] = src[k]
    last_t = events[-1].get("t") if events else 0
    duration = end.get("t") if end and isinstance(end.get("t"), (int, float)) else last_t

    chats = [e for e in events if e.get("chat")]
    per_agent = {a: {"placed": 0, "removed": 0, "taken": 0, "messages": 0, "individual_score": None} for a in agents}
    for e in events:
        a = e.get("agent")
        if a not in per_agent or not e.get("ok"):
            continue
        tool = e.get("tool")
        if tool == "place":
            per_agent[a]["placed"] += 1
        elif tool == "remove":
            per_agent[a]["removed"] += 1
        elif tool == "take":
            per_agent[a]["taken"] += sum(-v for v in ((e.get("diff") or {}).get("chest") or {}).values() if v < 0)
        elif e.get("chat"):
            per_agent[a]["messages"] += 1
    if src and isinstance(src.get("individual_scores"), dict):
        for a, v in src["individual_scores"].items():
            if a in per_agent:
                per_agent[a]["individual_score"] = v

    grid = header.get("grid") or {k: cfg.get(k) for k in ("width", "depth", "height")}
    created = (meta.get("created") or header.get("created") or header.get("created_at")
               or header.get("started_at") or header.get("timestamp") or cfg.get("created_at"))
    return {
        "id": header.get("episode_id") or "episode",
        "title": meta.get("title"),
        "notes": meta.get("notes"),
        "log_version": header.get("log_version"),
        "task": kind,
        "brief": brief,
        "blueprint": bp_name,
        "agents": [{"id": a, "model": models.get(a)} for a in agents],
        "models": sorted({m for m in models.values() if m}),
        "scarcity": cfg.get("scarcity"),
        "payoff": cfg.get("payoff"),
        "clock": header.get("clock"),
        "duration_s": round(duration, 3) if isinstance(duration, (int, float)) else None,
        "end_reason": end.get("reason") if end else None,
        "complete": end is not None,
        "team_score": team_score,
        "score_details": details,
        "per_agent": per_agent,
        "event_count": len(events),
        "chat_count": len(chats),
        "created": created,
        "grid": {"width": grid.get("width"), "depth": grid.get("depth"), "height": grid.get("height")},
        "chest_pos": header.get("chest_pos") or cfg.get("chest_pos"),
        "final_blocks": [b[:4] for b in final_blocks(records)],
    }


def load_meta(log: Path) -> dict:
    for cand in (log.with_name(log.name + ".meta.json"), log.with_suffix(".meta.json")):
        if cand.is_file():
            try:
                meta = json.loads(cand.read_text(encoding="utf-8"))
                if isinstance(meta, dict):
                    return meta
                print(f"  warning: {cand} is not a JSON object; ignored", file=sys.stderr)
            except json.JSONDecodeError as e:
                print(f"  warning: {cand} is not valid JSON ({e.msg}); ignored", file=sys.stderr)
    return {}


def safe_id(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", str(s)).strip("-.") or "episode"
    return s[:120]


def prepare_out(out: Path, force: bool) -> None:
    """Empty `out` (only if this script built it, or with --force) and recreate it."""
    if out.exists():
        if not out.is_dir():
            sys.exit(f"error: {out} exists and is not a folder")
        if any(out.iterdir()) and not (out / MARKER).exists() and not force:
            sys.exit(f"error: {out} is not empty and was not built by this script; pass --force to overwrite it")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / MARKER).write_text("Built by scripts/export_site.py; this folder is wiped on rebuild.\n")


def write_pages(out: Path, config: dict) -> None:
    """Copy site_src/ (+ viewer/core.js as js/core.js) into out/ and write site-config.json."""
    if not SITE_SRC.is_dir():
        sys.exit(f"error: site source folder {SITE_SRC} not found")
    if not CORE_JS.is_file():
        sys.exit(f"error: {CORE_JS} not found (the site imports the shared replay logic from it)")
    shutil.copytree(SITE_SRC, out, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("README-deploy.md", ".DS_Store"))
    shutil.copy2(CORE_JS, out / "js" / "core.js")
    (out / "site-config.json").write_text(json.dumps(config, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def write_data(data_dir: Path, logs: list[Path], keep_model_calls: bool, drop_notes: bool) -> tuple[list[dict], list[str]]:
    """Write <id>.jsonl.gz per log and episodes.json into data_dir."""
    data_dir.mkdir(parents=True, exist_ok=True)
    episodes, failed, used_ids = [], [], set()
    for log in logs:
        try:
            records = read_records(log)
        except (OSError, UnicodeDecodeError, LogError) as e:
            print(f"  skip {log}: {e}", file=sys.stderr)
            failed.append(str(log))
            continue
        meta = load_meta(log)
        if meta.get("hidden"):
            print(f"  skip {log}: hidden in its .meta.json")
            continue
        kept, dropped = public_filter(records, keep_model_calls, drop_notes)
        summary = summarize(kept, meta)
        base = safe_id(summary["id"])
        ep_id, n = base, 2
        while ep_id in used_ids:
            ep_id, n = f"{base}-{n}", n + 1
        used_ids.add(ep_id)
        summary["id"] = ep_id
        summary["source_episode_id"] = records[0].get("episode_id")
        if not summary["created"]:
            summary["created"] = _dt.datetime.fromtimestamp(log.stat().st_mtime, _dt.timezone.utc).isoformat(timespec="seconds")
            summary["created_source"] = "file_mtime"
        fname = f"{ep_id}.jsonl.gz"
        body = "".join(json.dumps(r, separators=(",", ":"), ensure_ascii=False) + "\n" for r in kept).encode("utf-8")
        (data_dir / fname).write_bytes(gzip.compress(body, compresslevel=9, mtime=0))
        summary["file"] = fname  # relative to the data base URL
        summary["bytes"] = (data_dir / fname).stat().st_size
        episodes.append(summary)
        drop_txt = ", ".join(f"{v} {k}" for k, v in sorted(dropped.items())) or "nothing"
        score = summary["team_score"]
        print(f"  {ep_id:<32} {summary['task']:<9} events={summary['event_count']:<5} chat={summary['chat_count']:<4} "
              f"score={'-' if score is None else f'{score:.3f}'}  {summary['bytes'] / 1024:.1f} KiB  (dropped: {drop_txt})")
    (data_dir / "episodes.json").write_text(json.dumps(episodes, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return episodes, failed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Build the static BlockSwarm replay website.",
        epilog="Layouts: default = pages + data/ in one folder. For pages on one host and data in e.g. a "
               "Hugging Face dataset: --data-only --out hf_data (upload its contents), and "
               "--site-only --data-base-url https://huggingface.co/datasets/<user>/<repo>/resolve/main/ --out site.")
    ap.add_argument("logs", nargs="*", type=Path, help="episode logs (.jsonl)")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default: site/, or site_data/ with --data-only)")
    ap.add_argument("--title", default="BlockSwarm", help="site title")
    ap.add_argument("--subtitle", default="LLM agents with bodies build a house together in a shared block world.",
                    help="one-line description under the title")
    ap.add_argument("--data-base-url", default=None,
                    help="where the pages load episodes.json and the .jsonl.gz logs from (default ./data/)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--data-only", action="store_true",
                      help="write only the data (episodes.json + logs) at the root of --out, e.g. for a dataset repo")
    mode.add_argument("--site-only", action="store_true",
                      help="write only the pages + site-config.json (use with --data-base-url); no logs needed")
    ap.add_argument("--keep-model-calls", action="store_true", help="keep model_call records (raw model I/O) in the public logs")
    ap.add_argument("--drop-notes", action="store_true", help="remove the text of note events (agents' private scratchpads)")
    ap.add_argument("--force", action="store_true", help="overwrite a non-empty output folder not built by this script")
    args = ap.parse_args(argv)

    if not args.site_only and not args.logs:
        ap.error("give at least one log (or use --site-only)")
    if args.site_only and args.logs:
        print("note: --site-only ignores the logs given", file=sys.stderr)
    base_url = args.data_base_url or "./data/"
    if not base_url.endswith("/"):
        base_url += "/"
    if args.site_only and not args.data_base_url:
        print("warning: --site-only without --data-base-url; the pages will look for ./data/", file=sys.stderr)
    out: Path = args.out or (ROOT / ("site_data" if args.data_only else "site"))
    prepare_out(out, args.force)

    config = {"title": args.title, "subtitle": args.subtitle, "data_base_url": base_url,
              "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    episodes, failed = [], []
    if not args.site_only:
        data_dir = out if args.data_only else out / "data"
        episodes, failed = write_data(data_dir, args.logs, args.keep_model_calls, args.drop_notes)
    if not args.data_only:
        write_pages(out, config)

    what = "data" if args.data_only else ("pages" if args.site_only else "site")
    print(f"\nBuilt {what} in {out}" + (f" with {len(episodes)} episode(s)" if not args.site_only else "")
          + (f"; {len(failed)} log(s) failed" if failed else "") + ".")
    if args.data_only:
        print("Upload the folder's contents (episodes.json + *.jsonl.gz) to where --data-base-url will point.")
    else:
        if args.data_base_url:
            print(f"Pages load data from {base_url}" + (" (the local data/ copy is unused)" if not args.site_only else ""))
        print(f"Preview:  python3 -m http.server -d {out} 8000   then open http://localhost:8000/")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
