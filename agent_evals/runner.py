"""Run every scenario k times, check each run, and summarize."""
import math
import re
import time

from . import adapters
from .adapters import normalize
from .checks import check_run

JUDGE_PROMPT = """You grade one reply of an AI support agent against a rubric.
Conversation (user turns):
{turns}

Agent's final reply:
{answer}

Rubric: {rubric}

Reply with PASS or FAIL on the first line, then one sentence why."""


def run_once(scenario, agent, judge=None):
    started, hits = time.perf_counter(), adapters.CACHE_HITS
    try:
        run = normalize(agent(scenario["turns"]))
    except Exception as e:  # noqa: BLE001 - an adapter error is recorded, never fatal to the suite
        return {"error": f"{type(e).__name__}: {e}"[:300], "passed": False,
                "latency_s": round(time.perf_counter() - started, 3)}
    run["latency_s"] = round(time.perf_counter() - started, 3)
    if adapters.CACHE_HITS > hits:
        run["cache_hits"] = adapters.CACHE_HITS - hits
    run["failures"] = check_run(scenario, run)
    run["passed"] = not run["failures"]
    if judge and scenario.get("judge"):
        run["judge"] = grade(judge, scenario, run["answer"])
    return run


def grade(chat, scenario, answer):
    """LLM judge: a second opinion only. It never changes "passed"."""
    prompt = JUDGE_PROMPT.format(turns="\n".join(scenario["turns"]), answer=answer,
                                 rubric=scenario["judge"]["rubric"])
    try:
        text = adapters.THOUGHT.sub("", chat([{"role": "user", "content": prompt}])[0].get("content") or "")
    except Exception as e:  # noqa: BLE001
        return {"verdict": None, "reason": f"judge error: {type(e).__name__}"}
    first = text.strip().splitlines()[0] if text.strip() else ""
    verdict = True if re.search(r"\bPASS\b", first) else False if re.search(r"\bFAIL\b", first) else None
    return {"verdict": verdict, "reason": text.strip()[:300]}


def status(runs):
    done = [r for r in runs if "error" not in r]
    if not done:
        return "error"
    passes = sum(r["passed"] for r in done)
    return "pass" if passes == len(done) else "fail" if passes == 0 else "flaky"


def percentile(values, p):
    """Nearest-rank percentile; None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


def run_suite(scenarios, agent, k=1, judge=None, meta=None, progress=None):
    results = {}
    for scenario in scenarios:
        runs = []
        for i in range(k):
            adapters.SAMPLE = i
            run = run_once(scenario, agent, judge)
            runs.append(run)
            if progress:
                progress(scenario["id"], i + 1, k, run)
        results[scenario["id"]] = {"runs": runs, "passes": sum(r["passed"] for r in runs),
                                   "errors": sum("error" in r for r in runs), "status": status(runs)}
    return {"meta": {**(meta or {}), "k": k}, "scenarios": results, "summary": summarize(results, meta or {})}


def summarize(results, meta):
    evaluated = [s for s in results.values() if s["status"] != "error"]  # at least one completed run
    n = len(evaluated)
    runs = [r for s in results.values() for r in s["runs"]]
    done = [r for r in runs if "error" not in r]
    # Replayed (cached) runs and errors say nothing about latency.
    latencies = [r["latency_s"] for r in runs if "error" not in r and not r.get("cache_hits")]
    prompt = sum(r.get("usage", {}).get("prompt_tokens", 0) for r in runs)
    completion = sum(r.get("usage", {}).get("completion_tokens", 0) for r in runs)
    judged = [r["judge"]["verdict"] for r in runs if r.get("judge", {}).get("verdict") is not None]
    summary = {
        "scenarios": len(results),
        "scenarios_evaluated": n,
        "runs": len(runs),
        "errors": sum("error" in r for r in runs),
        "cached_runs": sum(bool(r.get("cache_hits")) for r in runs),
        # With exactly k samples per scenario: pass@k = at least one pass, pass^k = all k pass.
        # Over scenarios with at least one completed run; errored runs are left out of every rate.
        "pass_at_k": round(sum(s["passes"] > 0 for s in evaluated) / n, 4) if n else None,
        "pass_all_k": round(sum(s["status"] == "pass" for s in evaluated) / n, 4) if n else None,
        "run_pass_rate": round(sum(r["passed"] for r in done) / len(done), 4) if done else None,
        "errored": sorted(i for i, s in results.items() if s["status"] == "error"),
        "flaky": sorted(i for i, s in results.items() if s["status"] == "flaky"),
        "failing": sorted(i for i, s in results.items() if s["status"] == "fail"),
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
        "tokens": {"prompt": prompt, "completion": completion} if prompt or completion else None,
        "judge_pass_rate": round(sum(judged) / len(judged), 4) if judged else None,
    }
    if summary["tokens"] and meta.get("price_in_per_mtok") is not None:
        summary["cost"] = round(prompt / 1e6 * meta["price_in_per_mtok"]
                                + completion / 1e6 * meta.get("price_out_per_mtok", 0), 6)
    return summary


def markdown(report):
    s, k = report["summary"], report["meta"]["k"]
    pct = lambda v: "n/a" if v is None else f"{v:.0%}"  # noqa: E731
    lines = [
        f"| metric | value |", "|---|---|",
        f"| scenarios x k | {s['scenarios']} x {k} = {s['runs']} runs ({s['errors']} adapter errors, "
        f"{s.get('cached_runs', 0)} replayed from cache) |",
        f"| scenarios evaluated (>= 1 completed run) | {s.get('scenarios_evaluated', s['scenarios'])} |",
        f"| pass@{k} (any of {k} passes) | {pct(s['pass_at_k'])} |",
        f"| pass^{k} (all {k} pass) | {pct(s['pass_all_k'])} |",
        f"| per-run pass rate | {pct(s['run_pass_rate'])} |",
        f"| flaky scenarios | {', '.join(s['flaky']) or 'none'} |",
        f"| always failing | {', '.join(s['failing']) or 'none'} |",
        f"| latency p50 / p95 (live runs only) | "
        + (f"{s['latency_p50_s']} s / {s['latency_p95_s']} s |" if s["latency_p50_s"] is not None else "n/a |"),
    ]
    if s["tokens"]:
        lines.append(f"| tokens in / out | {s['tokens']['prompt']:,} / {s['tokens']['completion']:,} |")
    if "cost" in s:
        lines.append(f"| cost (from given prices) | {s['cost']} |")
    if s["judge_pass_rate"] is not None:
        lines.append(f"| LLM judge pass rate (weaker evidence) | {pct(s['judge_pass_rate'])} |")
    lines += ["", "| scenario | passes / completed | errors | status | first failure |", "|---|---|---|---|---|"]
    for sid, r in report["scenarios"].items():
        first = next((f for run in r["runs"] for f in run.get("failures") or [] if f), "")
        done = len(r["runs"]) - r["errors"]
        lines.append(f"| {sid} | {r['passes']}/{done} | {r['errors']} | {r['status']} | {first[:90].replace('|', '/')} |")
    return "\n".join(lines)
