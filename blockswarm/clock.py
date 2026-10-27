"""Episode clocks. `now()` returns seconds since the episode started."""

from __future__ import annotations

import threading
import time


class WallClock:
    """Real time: time.monotonic() relative to when the clock was created (or restarted)."""

    kind = "wall"

    def __init__(self) -> None:
        self._start = time.monotonic()

    def restart(self) -> None:
        self._start = time.monotonic()

    def now(self) -> float:
        return time.monotonic() - self._start


class SimClock:
    """Simulated time, advanced manually. Used by scripted demos and tests."""

    kind = "simulated"

    def __init__(self, start: float = 0.0) -> None:
        self._t = float(start)
        self._lock = threading.Lock()

    def now(self) -> float:
        with self._lock:
            return self._t

    def advance_to(self, t: float) -> None:
        with self._lock:
            if t < self._t:
                raise ValueError(f"cannot move clock back from {self._t} to {t}")
            self._t = float(t)

    def advance(self, dt: float) -> None:
        with self._lock:
            if dt < 0:
                raise ValueError("dt must be >= 0")
            self._t += dt
