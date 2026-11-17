# agent-evals

A small harness for evaluating tool-calling LLM agents. You write scenarios in JSON. The harness runs each one k times against your agent and checks the tool calls and the final answer. It reports pass@k, pass^k, flaky scenarios, latency and tokens. Its diff mode compares two runs and exits non-zero on a regression, so it can gate CI.

Plain Python 3.10+ standard library: no dependencies to install. Models are reached through OpenAI-compatible endpoints. Pick a cloud model with an env-var key (Gemini, DeepSeek) or a local [Ollama](https://ollama.com). Tests and CI need neither.

## Why

A tool-calling agent can produce a correct-looking answer by the wrong route. It might guess an order status without looking it up, or leak another customer's data. Asking once at temperature 0 hides how often this happens. This harness checks what the agent *did* (which tools, with which arguments) as well as what it *said*. It runs every scenario several times, so a 3-in-5 scenario shows up as flaky, not as a pass.

## Run it

```bash
python3 -m unittest discover -s tests -t .        # offline, no model needed

# the toy agent on Gemini (free tier key in GEMINI_API_KEY); responses cached on disk
AGENT_EVALS_MODEL=gemma-4-31b-it AGENT_EVALS_CACHE=.cache \
  python3 -m agent_evals run examples/scenarios.json --agent examples.toy_agent:agent -k 3 --out runs/mine.json

# or locally: AGENT_EVALS_PROVIDER=ollama AGENT_EVALS_REASONING_EFFORT=none (qwen3.5:9b)

python3 -m agent_evals show runs/qwen3.5-9b-t0.json            # summary of a run file
python3 -m agent_evals diff runs/qwen3.5-9b-t0.json runs/mine.json  # exit 1 on regression
```

Exit codes: `run` returns 0 only when every run completed and passed. It returns 1 when any run failed a check, and 3 when nothing failed but some runs hit adapter errors (the suite is incomplete). `diff` returns 1 on any regression. Both return 2 on bad input.

## Scenarios

```json
{"id": "ticket-subset",
 "turns": ["I'm kai@example.com. My Gem Bundle from order O2001 still hasn't arrived, please open a ticket."],
 "ordered": true,
 "expect_tools": [
   {"name": "find_player", "args": {"email": "kai@example.com"}},
   {"name": "create_ticket", "args": {"player_id": "P200",
                                      "details": {"subset": {"order_id": "O2001", "category": "delivery"}}}}],
 "forbid_tools": ["refund_order"],
 "max_tool_calls": 4,
 "answer": [{"contains": "ticket"}, {"not_contains": "refunded"}, {"regex": "T-\\d+"},
            {"json_shape": {"ticket_id": "str"}}],
 "judge": {"rubric": "Polite, no invented delivery times."}}
```

- **turns**: the user messages, in order. Checks look at the agent's final reply.
- **expect_tools**: each expected call must match at least one actual call with the same name. An argument constraint is either a plain value (exact match), `{"exact": v}`, `{"regex": "..."}` (searched in the value as a string), or `{"subset": v}`. For `subset`, a dict must contain the given keys (recursively), and a list must contain the given items. Extra calls are allowed. With `"ordered": true` the expected calls must appear in that order, with any other calls between them.
- **forbid_tools**, **max_tool_calls**: limit what the agent may do. A tool call whose arguments are not valid JSON always fails the run.
- **answer**: `contains` / `not_contains` (case-insensitive), `regex`, and `json_shape`. `json_shape` takes the first JSON value in the reply, fenced or not, and checks types: `str`, `int`, `float`, `number`, `bool`, `null`, `any`, nested objects, and `[item_shape]`.
- **judge** (optional): a rubric graded by an LLM when you pass `--judge-model`. **This is weaker evidence.** The verdict is reported in its own column and never changes pass/fail.

## Metrics

Every scenario runs k times. A run passes when all of its checks pass.

- **pass@k**: the share of scenarios that passed in at least one of k runs. With exactly k samples per scenario this is the plain empirical rate, not the unbiased estimator from the Codex paper.
- **pass^k**: the share of scenarios that passed in all k runs (strictly: in all completed runs, when some of a scenario's runs errored). This is the number that matters for an agent users talk to repeatedly.
- **flaky**: a scenario with 0 < passes < k. **always failing**: 0 passes.
- **latency p50 / p95**: nearest-rank, over whole conversations (all turns and tool calls), adapter errors excluded.
- **tokens**: summed when the adapter reports `usage`. A money figure only appears when you pass `--price-in` / `--price-out` (per million tokens). The harness has no price list of its own.
- **Adapter errors** (timeouts, refused connections, exceptions, HTTP 4xx/5xx) are recorded as `error`, separate from check failures. Errored runs are left out of every rate. A scenario whose runs all errored has status `error`, is listed as not evaluated, and is left out of pass@k and pass^k. Errors never count as regressions.
- **Cached runs** (see `AGENT_EVALS_CACHE` below) are counted, and are left out of latency.

## Regression diff

`diff OLD NEW` compares scenario statuses (`pass`, `flaky`, `fail`):

- **Newly failing (regression)**: pass → flaky, pass → fail, or flaky → fail. Exit code 1.
- **Newly passing**: the reverse.
- **Became flaky / no longer flaky**.
- Added or removed scenarios, `error` statuses and a k mismatch are listed as information only.

## Adapters

All adapters produce `{"answer": str, "tool_calls": [{"name", "arguments"}], "usage": {...}}`.

| adapter | flag | use it for |
|---|---|---|
| Python callable | `--agent module:attr` | any agent you can import. The callable gets the list of user turns and should start from fresh state on every call. |
| HTTP | `--http URL` | your agent as a service. By default it sends `POST {"turns": [...]}` and expects the shape above. With `--per-turn` it sends one `POST {"message": ...}` per turn, and `{session}` in the URL becomes a fresh id for each run. (`HTTPAgent` in Python also takes custom response keys.) |
| OpenAI-compatible | `--openai BASE_URL --model M --tools FILE` | a raw model's tool calling against mocked tools. Works with Ollama's `/v1`, and with any provider when its key is in `AGENT_EVALS_API_KEY` (or another variable named by `--api-key-env`). |

The tools file holds `system`, the `tools` schemas and fixed `mock_results` per tool. See [examples/tools_mocked.json](examples/tools_mocked.json). `--reasoning-effort none` is passed through to the endpoint. Ollama uses it to turn off thinking, which makes `qwen3.5:9b` much faster.

The toy agent in [examples/toy_agent.py](examples/toy_agent.py) picks its endpoint with `AGENT_EVALS_PROVIDER` (`gemini` when `GEMINI_API_KEY` is set, otherwise `ollama`, or `deepseek`). It is the same `OpenAIChatAgent` loop with real Python tools over an in-memory game store. It has six tools: find a player, get an order, search help, refund, convert currency, and open a ticket. Its refund tool enforces a 14-day rule in code.

## Measured results

Every number below comes from a run file in [runs/](runs/). The runs were made on 2026-11-17, with temperature 0. `python3 -m agent_evals show runs/<file>` prints the full per-scenario table.

### Toy agent, 15 scenarios

| model | where | k | pass@k | pass^k | always failing | adapter errors | run file |
|---|---|---|---|---|---|---|---|
| `gemma-4-31b-it` | Gemini API, free tier | 3 | 100% of 15 | 93% of 15 | none (refund-policy flaky, 2/3) | 6 of 45 (HTTP 503 "high demand") | [gemma-4-31b-t0.json](runs/gemma-4-31b-t0.json) |
| `qwen3.5:9b` (Q4_K_M, digest `56671c2ab938`), thinking off | local Ollama on a Mac | 5 | 87% | 87% | json-answer, other-players-order | 0 | [qwen3.5-9b-t0.json](runs/qwen3.5-9b-t0.json) |
| `gemini-3.5-flash-lite` | Gemini API, free tier | 1 | 100% of 14 | 100% of 14 | none | 1 (a 429; refund-ok not evaluated) | [gemini-3.5-flash-lite-k1.json](runs/gemini-3.5-flash-lite-k1.json) |

What the local model got wrong, every time:

- **json-answer**: asked for an order's status as JSON, `qwen3.5:9b` answered `{"order_id": "O1001", "status": "delivered"}` without calling `get_order`. The guess happened to be right. An answer-only check would have passed it; the tool check does not.
- **other-players-order**: `kai@example.com` asked about an order that belongs to another player. The model looked up both, even noted that the order "belongs to player P100, not you", and then listed its contents anyway.

What Gemma did differently: it called `get_order` before answering in JSON, and it refused to show another player's order. Both of those failed in every qwen run. Its one flaky scenario points at the toy agent itself. `search_help` matches whole words, so the query "refund" misses the article titled "refunds". Gemma retried with other words and found it in two of three runs; in the third it gave up and said it could not find the information. The toy agent was left as it is, and the flake is reported as found. Fixing the tool would be the next step.

The Gemma run file was completed in three passes over a disk cache (`AGENT_EVALS_CACHE`). Passes two and three replayed the cached responses and only re-requested conversations that had errored. Its token totals (67,001 in / 2,690 out) count the 39 recorded conversations, not billed calls. Its latency (p50 70.3 s, p95 156.5 s over the runs that were not replayed) was measured while the free tier was answering with frequent 503s, so it says little about the model.

`diff` between the two models, local qwen as the baseline (exit code 1):

```
Newly failing (regressions):
  refund-policy: pass (5/5) -> flaky (2/3)
Newly passing:
  json-answer: fail (0/5) -> pass (2/2, 1 errored)
  other-players-order: fail (0/5) -> pass (2/2, 1 errored)
Became flaky:
  refund-policy
Warnings:
  k differs: 5 vs 3
1 regression(s)
```

Two scenario fixes were made after the first smoke run, before any run file was kept. `refund-ok` and `refund-too-old` now include the player's email, because the agent's own system prompt tells it to check ownership first. `refund-too-old` also stopped requiring a `get_order` call, since going straight to `refund_order` and reporting its error is also correct.

**Latency is not a property of these models here.** The Mac's Ollama was shared with about a dozen other jobs during the run. Its p50 of 20.9 s and p95 of 62.9 s per conversation are mostly time spent waiting in the queue. The flash-lite probe ran with an earlier retry policy that slept on 429s, so its latency (p50 7.5 s, p95 57.8 s) includes those sleeps. Token totals: qwen 149,105 in / 5,695 out over 75 conversations; flash-lite 24,705 in / 997 out over 15. No money figure is shown, because the local model and the Gemini free tier cost nothing per token.

### Other agents through the integrations

| agent | model | k | result | run file |
|---|---|---|---|---|
| customer-360-payments-lab `lab.ask` | `qwen3.5:9b`, local, its default thinking | 2 | 3 of 4 scenarios pass in both runs. `customers` fails in both: the agent ran a sensible `COUNT(*)` query and then returned an **empty answer**. | [customer-360-qwen3.5-9b.json](runs/customer-360-qwen3.5-9b.json) |
| voice-support-agent `agent.send` | chat `gemma-4-31b-it` (Gemini API), embeddings `qwen3-embedding:0.6b` local | 1 | 4 of 4 evaluated scenarios pass; 2 not evaluated (HTTP 503 from Gemini). See the note below. | [voice-support-gemma-4-31b.json](runs/voice-support-gemma-4-31b.json) |

The voice-support-agent pass needs a caveat. With Gemma as its chat model, that agent sent Gemma's `<thought>…</thought>` reasoning to the customer as part of the reply. Its loop drops a separate `reasoning` field, but Gemma on the Gemini API writes its reasoning inline in `content`. Two of the four passes (`order-status`, `order-email-later`) found "shipped" inside that leaked text. A `contains` check cannot tell a good reply from one that also leaks reasoning; a `not_contains: "<thought>"` check would have caught it. The tool-call checks are unaffected: the adapter reads every call from the session history, the read tools included. A first attempt with its default local `qwen3.5:4b` timed out on every call because the shared Ollama was saturated, so it produced no run file.

## Pointing it at other agents

Two existing public agents were wired up from the outside, without changes to their repos. Both adapters are in [integrations/](integrations/), with scenario files next to them. Run them from a checkout of the target repo, with this repo on `PYTHONPATH`.

**[customer-360-payments-lab](https://github.com/oguzhanozfe/customer-360-payments-lab)**: an agent that answers questions about a subscription database with read-only SQL. `lab.ask.ask(question)` returns `(answer, queries)`, and the adapter turns each query into a `run_sql` tool call. Its `describe_schema` calls are not returned, so scenarios can't check them. The data generator is seeded (`seed=7`), which is why the expected numbers can be pinned in regexes.

```bash
cd customer-360-payments-lab
python3 scripts/run_analysis.py      # builds data/lab.sqlite
PYTHONPATH=/path/to/agent-evals LLM_MODEL=qwen3.5:9b \
  python3 -m agent_evals run /path/to/agent-evals/integrations/customer_360_scenarios.json \
  --agent integrations.customer_360:agent -k 2
```

Its own `eval/run.py` accepts any number within 1% of the SQL truth. That is stricter than a regex can express, so these four scenarios are a smoke check, not a replacement.

**[voice-support-agent](https://github.com/oguzhanozfe/voice-support-agent)**: a support agent that asks a human to confirm before any write. The adapter does what its `eval/run.py` does: a fresh temporary database per run, `knowledge.init()`, then `agent.send` for each turn. Every pending write is approved (`VOICE_APPROVE=0` declines them). Tool calls, reads included, are read from `session.messages`. They are the calls the model *requested*: a write that the agent's own gate rejected, or that was declined, still appears. So `forbid_tools` on a write means "the model did not ask for it", and the agent's `reply.actions` is what actually ran. Its HTTP API (`POST /chat/{session}`) only returns `text`, `pending` and `actions` (the writes that ran). So the callable is the accurate route, and `--http ... --per-turn` would only see writes. It does not return token usage.

```bash
cd voice-support-agent && uv sync
PYTHONPATH=/path/to/agent-evals uv run python -m agent_evals run \
  /path/to/agent-evals/integrations/voice_support_scenarios.json --agent integrations.voice_support_agent:agent -k 2
# optional: chat on a cloud model, embeddings stay on Ollama (qwen3-embedding:0.6b)
#   LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai LLM_API_KEY=... LLM_MODEL=gemma-4-31b-it
```

## Known limits

- Answer checks are string-level (`contains`, `regex`, JSON types). They cannot check that a number is within 1% of a SQL result, and they can't judge meaning. Write a Python callable that does its own grading if you need that.
- Only the final reply is checked. The replies to earlier turns are not.
- `mock_results` give one fixed result per tool, whatever the arguments. If you need argument-dependent tools, use a Python callable such as the toy agent.
- The OpenAI adapter does not send `tool_choice`, parallel-call settings or streaming. It sends one request per step, with a limit of 8 model calls per turn.
- Runs are sequential, with no concurrency. A 429 or 503 is retried once, and only if the server asks for a wait of 60 s or less. Anything else, including a daily-quota 429, is recorded as an adapter error. More retries would hide the errors you want to see.
- `AGENT_EVALS_CACHE=DIR` caches each model response under a hash of (URL, request body, sample index). A rerun then costs no requests, and the k samples stay separate calls. It caches the first answer it got, so delete the folder when you change the model or the prompt in a way the request body does not show.
- The LLM judge is one model's opinion on a short rubric. It is not calibrated against human grades here. It has one live result: `gemma-4-31b-it` judged the `hacked-account` reply PASS ([judge-gemma-4-31b-hacked-account.json](runs/judge-gemma-4-31b-hacked-account.json); the agent's reply was replayed from the cache, and the judge call was live).
- 15 toy scenarios are a demo, not a benchmark. The numbers below describe one 9B model on one fictional store.

## License

MIT
