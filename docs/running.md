# Running LLM episodes

One episode = N identical LLM agents acting live in one world, each in its own thread, each
calling its model through OpenRouter. Code: `blockswarm/llm/` (client, budget, agent loop,
runner); CLI: `scripts/run_episode.py`; configs: `configs/`.

Always go in this order: **mock run → dry run → real run.**

## 1. Mock run (offline, free)

```bash
python3 scripts/run_episode.py configs/mock.json --mock
```

`--mock` replaces the model with a scripted random-but-valid policy (walk to the shop,
buy, walk near the site, place, sometimes jump+place or say something). No network, no key.
It exercises everything else: threads, tool execution, logging, the budget (with the
config's fake prices) and the outputs below. `--mock` works with any config (unknown
prices are then allowed, at a high fallback price).

## 2. Dry run (no calls)

```bash
python3 scripts/run_episode.py configs/pilot.json --dry-run
```

Validates the config (agents, models, world settings), that every model has a price, that
`OPENROUTER_API_KEY` is present (never printed), and prints the spending plan:

- per-call reservation per agent ≈ (≈8000 prompt tokens × input price + `max_tokens` ×
  output price) × `price_safety`;
- **max spend without a budget ≈ N × `max_model_calls_per_agent` × per-call reservation**;
- whether `budget_usd` binds, and how many calls it covers.

`configs/pilot2-*.json` are ready-to-run 5-agent configs for four models. `configs/pilot.json` ships with `"model": "CHOOSE_MODEL"`; the dry run refuses it until you
put in a real OpenRouter model id and its prices (`llm.prices`, USD per 1M input/output
tokens, from the model's OpenRouter page). Candidates are listed in `model_candidates`.

## 3. Real run (paid)

```bash
cp .env.example .env && chmod 600 .env    # then put the key after OPENROUTER_API_KEY=
python3 scripts/run_episode.py configs/pilot.json
```

The key is read only from `.env` (repo root; existing environment variables win) or the
environment. It is sent only in the Authorization header and never logged; error texts are
scrubbed of it.

Options: `--run-dir DIR`, `--budget-usd X` (override the cap), `--n-agents N`,
`--allow-unknown-price` (reserve at a high fallback price for models without a price).
Ctrl-C ends the episode as `aborted`, waits up to 30 s for in-flight calls to settle their
cost, and writes the manifest.

## Number of agents

Set `"n_agents": N` (ids `agent_1..agent_N`, 1–16) in the config, or give an explicit
`"agents": [...]` list (if both are given they must agree), or pass `--n-agents N` on the
command line (overrides both). Models: `"model": "id"` for all agents, or `"models"` as a
dict id → model or a list with one entry per agent (a list must match N; prefer `"model"`
when you change N on the command line). The world scales the shared wallet (`starting_money`, default
100 × N coins), `deploy_top_k` (default N // 2) and the analysis-only `target_interior` (4·N)
unless the config sets them. Cost
scales roughly linearly with N; check the dry run.

## Team battle (v5)

Two teams build on opposite halves of one platform; design and rules in docs/team-battle.md.
Same three steps:

```bash
python3 scripts/run_episode.py configs/teams-mock.json --mock          # offline, 2 x 5 mock agents, 120 s
python3 scripts/run_episode.py configs/teams-glm53flash.json --dry-run  # validate + spending plan
python3 scripts/run_episode.py configs/teams-glm53flash.json            # real run (paid)
```

`configs/teams-{dsv4flash,glm53flash,gpt6luna,haiku55}.json` are 2 x 5 configs (40 x 20 x 10
platform, line at `divider_x` 20, shops red `[6, 10, 0]` / blue `[33, 10, 0]`, 1800 s,
`budget_usd` 20; LLM settings and prices as the pilot2 configs). A team battle config sets
`"variant": "team_battle"` and `"teams": {"red": [...], "blue": [...]}`; the agent list is the
teams' members in order (`agents` / `n_agents` may be omitted, `--n-agents` is refused), plus
`shops`, `divider_x` and the combat keys (`combat`, `max_health`, `sword_damage`,
`attack_reach`, `attack_cooldown_s`). To pit two models against each other, replace `"model"`
with `"team_models": {"red": "<model a>", "blue": "<model b>"}` and put both models' prices in
`llm.prices`. Budget and cost control are unchanged (one shared `budget_usd` for all 10 agents,
so expect about twice the spend of a 5-agent pilot).

Agents get a separate prompt (prompts.py, `PROMPT_VERSION` v5.0): the brief, their team, the
other team, the sides and the line, both shops, the team wallet, the channels (`say` = own team,
`say_all` = everyone, `whisper` = one agent) and the sword/health rules as plain facts, with no
hint either way about attacking or crossing. An agent that is killed stops its loop
(`agent_stop` reason `dead`); its blocks stay. The `--mock` policy for team battle builds on
its own side and now and then uses team say, `say_all`, raids across the line and `attack`, so
a mock log shows every new mechanic in the viewer. The manifest adds `wallets`, `health`,
`dead`, `kills` and per-team `teams` from `episode_end`.

## Outputs: `runs/<run_id>/` (git-ignored)

| File | Contents |
|---|---|
| `episode.jsonl` | The episode log (docs/log-format.md) plus orchestrator records, below |
| `config.json` | The config as given (CLI overrides under `_overrides`) |
| `manifest.json` | git commit/dirty, prompt version, models, prices, start/end, end reason, scores, cost, per-agent stats (calls, tokens, errors, stop reason), python version |
| `ledger.jsonl` | Every budget `reserve` / `refuse` / `commit` (no prompts, no secrets) |
| `late_records.jsonl` | Only if needed: model calls that returned after the episode ended (the episode log must end with `episode_end`) |

Orchestrator record types in `episode.jsonl` (viewers ignore them):
- `agent_prompt` — once per agent: full system prompt, tool schemas, their sha256, prompt version.
- `model_call` — every call, written when the response arrives: `agent`, `call_id`, `model`,
  `t_request`, `request` (full messages; the system message is a reference to
  `agent_prompt`; tool names; sampling settings), `response` (content, reasoning,
  reasoning_details, tool calls with raw argument strings, finish reason, provider, served
  model), `usage` (tokens, cached tokens, reasoning tokens, provider cost), `charged_usd`,
  `attempts`, `latency_s`, `error`.
- `tool_results` — the text returned to the model for each tool call of that response.
- `implicit_wait`, `agent_stop` (reason + stats, only for agents that stop before the episode ends; manifest.json has every agent's), `orchestrator_error` (traceback).

## Watching

Live, while it runs (the runner prints this command):

```bash
python3 scripts/serve_viewer.py runs/<run_id>/episode.jsonl --follow
```

Afterwards: the same without `--follow`. On the VM, see docs/vm.md for the SSH tunnel and
for copying `runs/` back.

## Budget and cost control

- `budget_usd` is a hard cap for the run, shared by all agents. Before **every HTTP attempt**
  (retries included) the agent reserves the attempt's worst-case cost; a reservation that
  doesn't fit is refused and the episode ends with reason `budget`. So committed spend +
  open reservations never exceed the cap, as long as the prices in the config are not below
  the real ones and outputs respect `max_tokens`.
- After each attempt the reservation is settled with the cost OpenRouter reports
  (`usage.cost`), or from token counts × prices, or kept in full when the outcome is
  unknown (timeouts, 5xx: counted as `uncertain_usd`). 4xx/429 refusals cost nothing. An
  actual cost above its reservation is recorded as an overrun and stops further calls.
- Other stops: `max_model_calls_per_agent`, `max_consecutive_errors` API failures in a row
  (episode ends with reason `error`; each call already has `max_retries` retries with
  exponential backoff and jitter), and the world's own limits (`max_seconds`, actions).
- Prompt caching: the system prompt and tools are fixed per agent, so providers with
  automatic caching cache that prefix. The history is a sliding window (`history_steps`);
  `history_chunk` > 1 drops old steps in chunks, which keeps the prefix stable longer.
  Anthropic models on OpenRouter need explicit caching, e.g.
  `"extra_body": {"cache_control": {"type": "ephemeral"}}` in `llm`.

## LLM settings (`llm` in the config)

`max_tokens`, `temperature`, `reasoning` (passed through to OpenRouter, e.g.
`{"effort": "low"}`), `provider` (OpenRouter provider routing, e.g.
`{"require_parameters": true, "order": [...], "allow_fallbacks": false}`), `extra_body`,
`parallel_tool_calls`, `history_steps` (6), `history_chunk` (1), `max_model_calls_per_agent`,
`max_consecutive_errors` (3), `error_pause_s`, `empty_steps_before_wait` (2),
`implicit_wait_s` (5), `replay_reasoning` (send `reasoning_details` back), `prices`,
`price_safety` (1.25), `stagger_s` (random start delay up to this, seconds).

## On the VM

Follow docs/vm.md to start the VM, copy the repo and open the viewer tunnel. The VM has
Python 3.10 and needs nothing else (stdlib only). Put the key in `~/blockswarm/.env`
(chmod 600) on the VM, never in the repo. Run the mock and dry run there first, then the real
run inside `tmux` so it survives a dropped SSH session:

```bash
cd ~/blockswarm
python3 scripts/run_episode.py configs/mock.json --mock
python3 scripts/run_episode.py configs/pilot.json --dry-run
tmux new -s pilot 'python3 scripts/run_episode.py configs/pilot.json'
```

Copy `runs/` back as in docs/vm.md §5 and stop the VM when done.
