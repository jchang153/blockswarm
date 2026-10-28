"""Model clients: an OpenRouter chat-completions client (urllib, stdlib only) and a mock.

The API key is held only in memory by `OpenRouterClient`. It is sent in the Authorization
header and never appears in a `ChatResult` (error texts are scrubbed of it), so results can
be logged as-is.

Spending control: `chat(..., guard=...)` takes an optional spend guard (see
blockswarm/llm/budget.py). The client calls `guard.before_attempt()` before EVERY HTTP
attempt, including retries, and `guard.after_attempt(token, outcome, cost_usd)` after it, so
each attempt that might be billed is covered by its own reservation. If `before_attempt`
raises, the call stops and returns a result with `budget_exhausted=True`.

Attempt outcomes passed to the guard:
- "ok": a response was parsed (cost_usd from `usage.cost` when present, else None)
- "not_billed": the request was refused before generation (HTTP 4xx incl. 429, connection
  refused / DNS failure)
- "maybe_billed": timeout, 5xx, connection dropped, or an unparseable body; the provider may
  have generated (and billed) tokens, so the guard keeps the whole reservation
"""

from __future__ import annotations

import json
import random
import socket
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional, Protocol

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
RETRY_STATUS = {408, 429, 500, 502, 503, 504, 520, 522, 524, 529}


class BudgetRefused(Exception):
    """Raised by a spend guard's `before_attempt` when the budget can't cover the attempt."""


class SpendGuard(Protocol):
    def before_attempt(self) -> Any: ...

    def after_attempt(self, token: Any, outcome: str, cost_usd: float | None,
                      usage: dict | None = None) -> None: ...


@dataclass
class ChatResult:
    content: str | None = None
    reasoning: str | None = None
    reasoning_details: list | None = None
    tool_calls: list[dict] = field(default_factory=list)  # {id, name, arguments(dict|None), raw_arguments}
    finish_reason: str | None = None
    usage: dict = field(default_factory=dict)  # prompt_tokens, completion_tokens, cached_tokens, reasoning_tokens, cost_usd
    latency_s: float = 0.0
    raw: dict | None = None
    error: str | None = None
    http_status: int | None = None
    attempts: int = 0
    budget_exhausted: bool = False
    response_id: str | None = None
    provider: str | None = None
    served_model: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_log(self) -> dict:
        d = asdict(self)
        d.pop("raw", None)
        return d


def parse_tool_calls(raw_calls: list | None) -> list[dict]:
    """OpenAI-style tool_calls -> [{id, name, arguments(dict or None), raw_arguments(str)}]."""
    out = []
    for i, tc in enumerate(raw_calls or []):
        fn = (tc or {}).get("function") or {}
        raw_args = fn.get("arguments")
        if raw_args is None:
            raw_args = ""
        if not isinstance(raw_args, str):  # some providers send an object
            raw_args = json.dumps(raw_args)
        args: dict | None
        if raw_args.strip() == "":
            args = {}
        else:
            try:
                parsed = json.loads(raw_args)
                args = parsed if isinstance(parsed, dict) else None
            except ValueError:
                args = None
        out.append({"id": tc.get("id") or f"call_{i}", "name": fn.get("name") or "",
                    "arguments": args, "raw_arguments": raw_args})
    return out


def parse_response(body: dict) -> ChatResult:
    """Parse a chat.completions response body (OpenRouter flavour)."""
    if body.get("error"):
        err = body["error"]
        msg = err.get("message") if isinstance(err, dict) else str(err)
        code = err.get("code") if isinstance(err, dict) else None
        return ChatResult(error=f"api error {code}: {msg}", raw=body,
                          http_status=code if isinstance(code, int) else None)
    choices = body.get("choices") or []
    if not choices:
        return ChatResult(error="response has no choices", raw=body)
    ch = choices[0]
    msg = ch.get("message") or {}
    u = body.get("usage") or {}
    pdet = u.get("prompt_tokens_details") or {}
    cdet = u.get("completion_tokens_details") or {}
    usage = {
        "prompt_tokens": u.get("prompt_tokens"),
        "completion_tokens": u.get("completion_tokens"),
        "cached_tokens": pdet.get("cached_tokens"),
        "cache_write_tokens": pdet.get("cache_write_tokens"),
        "reasoning_tokens": cdet.get("reasoning_tokens"),
        "cost_usd": u.get("cost"),
    }
    res = ChatResult(
        content=msg.get("content"),
        reasoning=msg.get("reasoning"),
        reasoning_details=msg.get("reasoning_details") or None,
        tool_calls=parse_tool_calls(msg.get("tool_calls")),
        finish_reason=ch.get("finish_reason"),
        usage=usage,
        raw=body,
        response_id=body.get("id"),
        provider=body.get("provider"),
        served_model=body.get("model"),
    )
    if ch.get("error"):  # mid-stream / provider error inside a choice
        e = ch["error"]
        res.error = f"choice error: {e.get('message') if isinstance(e, dict) else e}"
    return res


class OpenRouterClient:
    """POST {base_url}/chat/completions with retries. Thread-safe (no shared mutable state)."""

    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL, timeout_s: float = 120,
                 max_retries: int = 4, backoff_base_s: float = 2.0, backoff_max_s: float = 30.0,
                 app_title: str | None = "BlockSwarm", sleep: Callable[[float], None] = time.sleep,
                 urlopen: Callable[..., Any] | None = None):
        if not api_key:
            raise ValueError("api_key is empty")
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.backoff_max_s = backoff_max_s
        self.app_title = app_title
        self._sleep = sleep
        self._urlopen = urlopen or urllib.request.urlopen
        self._rng = random.Random()

    def __repr__(self) -> str:  # never show the key
        return f"OpenRouterClient(base_url={self.base_url!r}, timeout_s={self.timeout_s})"

    def _scrub(self, text: str) -> str:
        return text.replace(self._api_key, "[REDACTED]") if self._api_key else text

    def build_body(self, model: str, messages: list, tools: list | None, *, max_tokens: int,
                   temperature: float | None, reasoning: dict | None = None, provider: dict | None = None,
                   parallel_tool_calls: bool = True, extra_body: dict | None = None) -> dict:
        body: dict = {"model": model, "messages": messages, "max_tokens": int(max_tokens),
                      "usage": {"include": True}}
        if temperature is not None:
            body["temperature"] = temperature
        if tools:
            body["tools"] = tools
            body["parallel_tool_calls"] = bool(parallel_tool_calls)
        if reasoning is not None:
            body["reasoning"] = reasoning
        if provider is not None:
            body["provider"] = provider
        for k, v in (extra_body or {}).items():
            if k in ("model", "messages", "tools", "max_tokens", "usage"):
                raise ValueError(f"extra_body may not override {k!r}")
            body[k] = v
        return body

    def chat(self, model: str, messages: list, tools: list | None, *, max_tokens: int,
             temperature: float | None, reasoning: dict | None = None, provider: dict | None = None,
             parallel_tool_calls: bool = True, extra_body: dict | None = None,
             guard: SpendGuard | None = None, **_ignored: Any) -> ChatResult:
        body = self.build_body(model, messages, tools, max_tokens=max_tokens, temperature=temperature,
                               reasoning=reasoning, provider=provider,
                               parallel_tool_calls=parallel_tool_calls, extra_body=extra_body)
        data = json.dumps(body).encode("utf-8")
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        if self.app_title:
            headers["X-Title"] = self.app_title
        t0 = time.monotonic()
        last: ChatResult | None = None
        attempt = 0
        while True:
            attempt += 1
            token = None
            if guard is not None:
                try:
                    token = guard.before_attempt()
                except BudgetRefused as e:
                    res = last or ChatResult()
                    res.error = f"budget exhausted: {e}" + (f" (after: {last.error})" if last and last.error else "")
                    res.budget_exhausted = True
                    res.attempts = attempt - 1
                    res.latency_s = round(time.monotonic() - t0, 3)
                    return res
            res, outcome, retry = self._attempt(data, headers)
            if guard is not None:
                guard.after_attempt(token, outcome, res.usage.get("cost_usd") if outcome == "ok" else None,
                                    res.usage if outcome == "ok" else None)
            res.attempts = attempt
            last = res
            if res.error is None or not retry or attempt > self.max_retries:
                res.latency_s = round(time.monotonic() - t0, 3)
                return res
            delay = min(self.backoff_max_s, self.backoff_base_s * 2 ** (attempt - 1))
            self._sleep(delay * (0.5 + self._rng.random()))

    def _attempt(self, data: bytes, headers: dict) -> tuple[ChatResult, str, bool]:
        """One HTTP attempt -> (result, guard outcome, retryable)."""
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=data, headers=headers,
                                     method="POST")
        try:
            with self._urlopen(req, timeout=self.timeout_s) as resp:
                text = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode("utf-8", errors="replace")[:2000]
            except Exception:  # noqa: BLE001
                detail = ""
            status = e.code
            retry = status in RETRY_STATUS
            outcome = "maybe_billed" if status >= 500 else "not_billed"
            return ChatResult(error=self._scrub(f"HTTP {status}: {detail}"), http_status=status), outcome, retry
        except urllib.error.URLError as e:
            reason = e.reason
            if isinstance(reason, (socket.timeout, TimeoutError)):
                return ChatResult(error=f"timeout: {reason}"), "maybe_billed", True
            refused = isinstance(reason, (ConnectionRefusedError, socket.gaierror))
            return (ChatResult(error=self._scrub(f"network error: {reason}")),
                    "not_billed" if refused else "maybe_billed", True)
        except (socket.timeout, TimeoutError) as e:
            return ChatResult(error=f"timeout: {e}"), "maybe_billed", True
        except (ConnectionError, OSError) as e:
            return ChatResult(error=self._scrub(f"connection error: {e}")), "maybe_billed", True
        try:
            body = json.loads(text)
        except ValueError:
            return ChatResult(error=self._scrub(f"unparseable response: {text[:500]}")), "maybe_billed", True
        res = parse_response(body)
        if res.error:
            res.error = self._scrub(res.error)
            # OpenRouter may return 200 with an error object (e.g. upstream 429/502)
            retry = res.http_status in RETRY_STATUS or res.http_status is None and not res.usage.get("cost_usd")
            return res, ("ok" if res.usage.get("cost_usd") else "maybe_billed"), bool(retry)
        return res, "ok", False


# ---------------------------------------------------------------------------------- mock


def tool_call(name: str, _id: str | None = None, **args: Any) -> dict:
    """Build a parsed tool call for scripted mock results."""
    raw = json.dumps(args)
    return {"id": _id or f"call_{name}_{abs(hash(raw)) % 10**8}", "name": name, "arguments": args,
            "raw_arguments": raw}


def mock_result(*calls: dict, content: str | None = None, cost_usd: float | None = 0.0,
                prompt_tokens: int = 100, completion_tokens: int = 20, **kw: Any) -> ChatResult:
    """A ChatResult with the given tool calls (ids made unique per result)."""
    tcs = []
    for i, c in enumerate(calls):
        c = dict(c)
        c["id"] = f"{c['id']}_{i}"
        tcs.append(c)
    return ChatResult(content=content, tool_calls=tcs, finish_reason="tool_calls" if tcs else "stop",
                      usage={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                             "cached_tokens": 0, "reasoning_tokens": 0, "cost_usd": cost_usd}, **kw)


MockFn = Callable[[dict, list], ChatResult]


class MockClient:
    """Deterministic stand-in for OpenRouterClient. Never touches the network.

    `script` is one of:
    - a list of ChatResults, consumed in order by whichever agent calls next;
    - a dict agent_id -> list of ChatResults (per-agent order);
    - a function (state, messages) -> ChatResult, where `state` is a per-agent dict that
      persists across calls with keys agent, model, call_index, tools (names) and anything
      the function stores.
    When a list runs out, `on_exhausted` (default: a `wait(1)` tool call) is returned.
    Results are copied, so a scripted result can be reused. `latency_s` sleeps per call;
    `guard` is honoured like the real client (one attempt per call).
    """

    def __init__(self, script: list | dict | MockFn, latency_s: float | tuple[float, float] = 0.0,
                 on_exhausted: Callable[[], ChatResult] | None = None, seed: int = 0):
        self.script = script
        self.latency_s = latency_s
        self.on_exhausted = on_exhausted or (lambda: mock_result(tool_call("wait", seconds=1)))
        self._lock = threading.Lock()
        self._states: dict[str, dict] = {}
        self._queue_pos: dict[str, int] = {}
        self._rng = random.Random(seed)
        self.calls: list[dict] = []  # {agent, model, n_messages, tools} for tests

    def chat(self, model: str, messages: list, tools: list | None, *, max_tokens: int,
             temperature: float | None, reasoning: dict | None = None, provider: dict | None = None,
             parallel_tool_calls: bool = True, extra_body: dict | None = None,
             guard: SpendGuard | None = None, agent: str | None = None, **_ignored: Any) -> ChatResult:
        t0 = time.monotonic()
        token = None
        if guard is not None:
            try:
                token = guard.before_attempt()
            except BudgetRefused as e:
                return ChatResult(error=f"budget exhausted: {e}", budget_exhausted=True, attempts=0)
        key = agent or "_"
        tool_names = [t.get("function", {}).get("name") for t in (tools or [])]
        with self._lock:
            state = self._states.setdefault(key, {"agent": agent, "model": model, "call_index": 0})
            state["tools"] = tool_names
            self.calls.append({"agent": agent, "model": model, "n_messages": len(messages),
                               "messages": messages, "tools": tool_names})
            lat = self.latency_s
            if isinstance(lat, (tuple, list)):
                lat = self._rng.uniform(lat[0], lat[1])
            if callable(self.script):
                fn = self.script
                res = None
            else:
                seq = self.script.get(key, []) if isinstance(self.script, dict) else self.script
                pos_key = key if isinstance(self.script, dict) else "_"
                i = self._queue_pos.get(pos_key, 0)
                self._queue_pos[pos_key] = i + 1
                res = _copy(seq[i]) if i < len(seq) else self.on_exhausted()
                fn = None
        if lat:
            time.sleep(lat)
        if fn is not None:
            res = fn(state, messages)
        with self._lock:
            state["call_index"] += 1
        assert res is not None
        res.attempts = res.attempts or 1
        res.latency_s = round(time.monotonic() - t0, 3)
        if guard is not None:
            outcome = "ok" if res.error is None else "not_billed"
            guard.after_attempt(token, outcome, res.usage.get("cost_usd") if res.error is None else None,
                                    res.usage if res.error is None else None)
        return res


def _copy(res: ChatResult) -> ChatResult:
    return ChatResult(**json.loads(json.dumps(asdict(res))))
