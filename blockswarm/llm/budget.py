"""Run budget: a hard USD ceiling with conservative pre-call reservations and a JSONL ledger.

Invariant: committed spend + open reservations <= limit_usd at every moment a reservation is
granted. Every HTTP attempt (including retries) reserves its own worst-case cost first:

    reserve = (estimated prompt tokens * input price + max_tokens * output price) * safety

with prices in USD per 1M tokens from the run config. After the attempt the reservation is
settled with the provider-reported cost (`usage.cost`) when there is one; otherwise with a
token-based cost if tokens are known, otherwise the full reservation is kept (outcome
"maybe_billed": timeouts, 5xx), or released (outcome "not_billed": 4xx/429, refused
connection). An actual cost above its reservation is recorded as an overrun and exhausts
the budget for further calls.

The ledger (one JSON object per line) has a row for every reserve / refuse / commit. It
contains no request content and no secrets.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from blockswarm.llm.client import BudgetRefused

CHARS_PER_TOKEN = 3.0  # conservative (real text is ~4 chars/token)
FALLBACK_PRICE = {"input_per_m": 15.0, "output_per_m": 75.0}  # used only with allow_unknown_price


class UnknownPrice(ValueError):
    pass


def price_for(prices: dict, model: str, allow_unknown: bool = False) -> dict:
    """{'input_per_m', 'output_per_m'} for `model`, or the fallback when allowed."""
    p = (prices or {}).get(model)
    if p is None:
        if not allow_unknown:
            raise UnknownPrice(f"no price for model {model!r} in llm.prices; add it "
                               "(USD per 1M input/output tokens) or pass --allow-unknown-price")
        return dict(FALLBACK_PRICE, unknown=True)
    try:
        inp, out = float(p["input_per_m"]), float(p["output_per_m"])
    except (KeyError, TypeError, ValueError):
        raise UnknownPrice(f"price for {model!r} must be {{'input_per_m': x, 'output_per_m': y}}, got {p!r}")
    if inp < 0 or out < 0:
        raise UnknownPrice(f"negative price for {model!r}")
    return {"input_per_m": inp, "output_per_m": out}


def estimate_prompt_tokens(messages: list, tools: list | None) -> int:
    n = len(json.dumps(messages)) + len(json.dumps(tools or []))
    return int(n / CHARS_PER_TOKEN) + 50


def estimate_cost(price: dict, prompt_tokens: int, max_tokens: int, safety: float = 1.25) -> float:
    return round((prompt_tokens * price["input_per_m"] + max_tokens * price["output_per_m"]) / 1e6 * safety, 8)


def token_cost(price: dict, usage: dict) -> float | None:
    pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if pt is None or ct is None:
        return None
    return round((pt * price["input_per_m"] + ct * price["output_per_m"]) / 1e6, 8)


class Budget:
    """Thread-safe USD budget shared by all agents of a run."""

    def __init__(self, limit_usd: float, ledger_path: str | Path | None = None):
        if not limit_usd > 0:
            raise ValueError(f"budget limit must be > 0, got {limit_usd!r}")
        self.limit_usd = float(limit_usd)
        self._lock = threading.Lock()
        self._spent = 0.0
        self._open: dict[int, dict] = {}
        self._next_id = 0
        self._exhausted = False
        self._exhausted_reason: str | None = None
        self.overruns = 0
        self.uncertain_usd = 0.0  # kept reservations whose real cost is unknown
        self._fh = None
        if ledger_path is not None:
            p = Path(ledger_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._fh = p.open("a", encoding="utf-8")
        self._log({"kind": "open", "limit_usd": self.limit_usd})

    # -------------------------------------------------------------- state

    @property
    def spent(self) -> float:
        with self._lock:
            return round(self._spent, 8)

    @property
    def reserved(self) -> float:
        with self._lock:
            return round(sum(r["amount"] for r in self._open.values()), 8)

    @property
    def remaining(self) -> float:
        with self._lock:
            return round(self.limit_usd - self._spent - sum(r["amount"] for r in self._open.values()), 8)

    @property
    def exhausted(self) -> bool:
        return self._exhausted

    @property
    def exhausted_reason(self) -> str | None:
        return self._exhausted_reason

    def summary(self) -> dict:
        with self._lock:
            return {"limit_usd": self.limit_usd, "spent_usd": round(self._spent, 8),
                    "open_reservations_usd": round(sum(r["amount"] for r in self._open.values()), 8),
                    "open_reservations": len(self._open), "uncertain_usd": round(self.uncertain_usd, 8),
                    "overruns": self.overruns, "exhausted": self._exhausted,
                    "exhausted_reason": self._exhausted_reason}

    # -------------------------------------------------------------- operations

    def reserve(self, estimate: float, meta: dict | None = None) -> int:
        """Reserve `estimate` USD; returns a reservation id. Raises BudgetRefused if it doesn't fit."""
        estimate = round(float(estimate), 8)
        if estimate < 0:
            raise ValueError("estimate must be >= 0")
        with self._lock:
            open_sum = sum(r["amount"] for r in self._open.values())
            if self._exhausted or self._spent + open_sum + estimate > self.limit_usd + 1e-12:
                if not self._exhausted:
                    self._exhausted = True
                    self._exhausted_reason = (f"reservation of ${estimate:.6f} does not fit: spent "
                                              f"${self._spent:.6f} + reserved ${open_sum:.6f} of ${self.limit_usd:.2f}")
                self._log_locked({"kind": "refuse", "amount": estimate, **(meta or {})})
                raise BudgetRefused(self._exhausted_reason)
            rid = self._next_id
            self._next_id += 1
            self._open[rid] = {"amount": estimate, "meta": dict(meta or {})}
            self._log_locked({"kind": "reserve", "id": rid, "amount": estimate, **(meta or {})})
            return rid

    def commit(self, rid: int, actual: float | None, outcome: str = "ok") -> float:
        """Settle reservation `rid`. `actual` None keeps the full reservation (uncertain)."""
        with self._lock:
            r = self._open.pop(rid, None)
            if r is None:
                raise KeyError(f"unknown or already settled reservation {rid}")
            uncertain = actual is None
            charged = r["amount"] if uncertain else round(float(actual), 8)
            if uncertain:
                self.uncertain_usd += charged
            self._spent += charged
            overrun = charged > r["amount"] + 1e-9
            if overrun:
                self.overruns += 1
                self._exhausted = True
                self._exhausted_reason = self._exhausted_reason or (
                    f"actual cost ${charged:.6f} exceeded its reservation ${r['amount']:.6f}")
            if self._spent >= self.limit_usd and not self._exhausted:
                self._exhausted = True
                self._exhausted_reason = f"spent ${self._spent:.6f} of ${self.limit_usd:.2f}"
            self._log_locked({"kind": "commit", "id": rid, "reserved": r["amount"], "charged": charged,
                              "outcome": outcome, "uncertain": uncertain, "overrun": overrun, **r["meta"]})
            return charged

    def guard(self, model: str, estimate: float, price: dict | None = None, meta: dict | None = None) -> "AttemptGuard":
        return AttemptGuard(self, model, estimate, price, meta or {})

    def close(self) -> None:
        with self._lock:
            self._log_locked({"kind": "close"})
            if self._fh:
                self._fh.close()
                self._fh = None

    # -------------------------------------------------------------- ledger

    def _log(self, row: dict) -> None:
        with self._lock:
            self._log_locked(row)

    def _log_locked(self, row: dict) -> None:
        if self._fh is None:
            return
        full = {"utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "mono": round(time.monotonic(), 3), **row,
                "spent": round(self._spent, 8),
                "reserved_open": round(sum(r["amount"] for r in self._open.values()), 8),
                "limit": self.limit_usd}
        self._fh.write(json.dumps(full, separators=(",", ":")) + "\n")
        self._fh.flush()


class AttemptGuard:
    """Spend guard for one model call: one reservation per HTTP attempt (see client.py)."""

    def __init__(self, budget: Budget, model: str, estimate: float, price: dict | None, meta: dict):
        self.budget = budget
        self.model = model
        self.estimate = estimate
        self.price = price
        self.meta = {"model": model, **meta}
        self.charged = 0.0
        self.attempts: list[dict] = []

    def before_attempt(self) -> Any:
        return self.budget.reserve(self.estimate, {**self.meta, "attempt": len(self.attempts) + 1})

    def after_attempt(self, token: Any, outcome: str, cost_usd: float | None, usage: dict | None = None) -> None:
        if outcome == "not_billed":
            actual: float | None = 0.0
        elif cost_usd is not None:
            actual = float(cost_usd)
        elif outcome == "ok" and self.price and usage and token_cost(self.price, usage) is not None:
            actual = token_cost(self.price, usage)
        else:
            actual = None  # keep the whole reservation
        charged = self.budget.commit(token, actual, outcome)
        self.charged += charged
        self.attempts.append({"outcome": outcome, "charged": charged, "uncertain": actual is None})
