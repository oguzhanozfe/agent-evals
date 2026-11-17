"""`python -m agent_evals demo`: the toy agent's 15 scenarios, no key needed.

Model resolution: GEMINI_API_KEY set -> live Gemini; else a local Ollama with qwen3.5:9b -> live local;
else replay of the committed responses in examples/recorded (gemma-4-31b-it), with no network at all.
Then the pass^k table and the regression diff between two committed runs.
"""
import html
import json
import os
import sys
import urllib.request

from . import adapters
from .diff import diff, render
from .runner import markdown, run_suite

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORDED = os.path.join(HERE, "examples", "recorded")
RECORDED_RUN = os.path.join(HERE, "runs", "gemma-4-31b-t0.json")
DIFF_OLD, DIFF_NEW = (os.path.join(HERE, "runs", f) for f in ("qwen3.5-9b-t0.json", "gemma-4-31b-t0.json"))


def ollama_has(model, url="http://127.0.0.1:11434/api/tags"):
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            return any(m["name"] == model for m in json.load(r).get("models", []))
    except (OSError, ValueError):
        return False


def resolve(mode):
    """Set the toy agent's env for the chosen mode; returns (mode, description, k)."""
    if mode == "auto":
        mode = "gemini" if os.environ.get("GEMINI_API_KEY") else "ollama" if ollama_has("qwen3.5:9b") else "replay"
    if mode == "gemini":
        os.environ["AGENT_EVALS_PROVIDER"] = "gemini"
        return mode, f"live Gemini ({os.environ.get('AGENT_EVALS_MODEL', 'gemini-3.5-flash-lite')})", 1
    if mode == "ollama":
        os.environ.update(AGENT_EVALS_PROVIDER="ollama", AGENT_EVALS_REASONING_EFFORT="none")
        return mode, "live local Ollama (qwen3.5:9b, thinking off)", 1
    meta = json.load(open(RECORDED_RUN, encoding="utf-8"))["meta"]
    # The exact request the recording was made with; the key is never part of a recorded request.
    os.environ.update(AGENT_EVALS_PROVIDER="gemini", AGENT_EVALS_MODEL="gemma-4-31b-it", AGENT_EVALS_API_KEY="replay")
    os.environ.pop("AGENT_EVALS_TEMPERATURE", None)
    os.environ.pop("AGENT_EVALS_REASONING_EFFORT", None)
    adapters.CACHE_DIR, adapters.REPLAY_ONLY = RECORDED, True
    return "replay", f"replay (recorded gemma-4-31b-it, {meta['started'][:10]})", meta["k"]


def run_demo(mode="auto", k=None, html_path=None, out=print):
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    from examples import toy_agent  # imported after resolve() so it reads the env it set

    mode, described, default_k = resolve(mode)
    out(f"mode: {described}")
    scenarios = json.load(open(os.path.join(HERE, "examples", "scenarios.json"), encoding="utf-8"))["scenarios"]
    report = run_suite(scenarios, toy_agent.agent, k or default_k, meta={"label": described, "mode": mode})
    table = markdown(report)
    changes = diff(json.load(open(DIFF_OLD, encoding="utf-8")), json.load(open(DIFF_NEW, encoding="utf-8")))
    diff_text = render(changes)
    out(table)
    out("\nRegression diff of two committed runs: runs/qwen3.5-9b-t0.json -> runs/gemma-4-31b-t0.json")
    out(diff_text)
    if html_path:
        os.makedirs(os.path.dirname(html_path) or ".", exist_ok=True)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(page(report, described, diff_text))
        out(f"\nHTML report: {html_path}")
    return report


def page(report, described, diff_text):
    s, k = report["summary"], report["meta"]["k"]
    pct = lambda v: "n/a" if v is None else f"{v:.0%}"  # noqa: E731
    cards = [(f"pass^{k}", pct(s["pass_all_k"]), f"every completed run of {k} passes"),
             (f"pass@{k}", pct(s["pass_at_k"]), f"at least 1 of {k} passes"),
             ("per-run", pct(s["run_pass_rate"]), "completed runs passing"),
             ("adapter errors", str(s["errors"]), f"of {s['runs']} runs, not counted")]
    rows = []
    for sid, r in report["scenarios"].items():
        done = len(r["runs"]) - r["errors"]
        dots = "".join('<span class="dot e" title="adapter error"></span>' if "error" in run else
                       f'<span class="dot {"p" if run["passed"] else "f"}" title="{"pass" if run["passed"] else "fail"}"></span>'
                       for run in r["runs"])
        why = next((f for run in r["runs"] for f in run.get("failures") or [] if f), "")
        rows.append(f'<tr><td><code>{html.escape(sid)}</code></td><td>{dots}</td><td>{r["passes"]}/{done}</td>'
                    f'<td><span class="tag {r["status"]}">{r["status"]}</span></td><td>{html.escape(why)}</td></tr>')
    card_html = "".join(f'<div class="card"><div class="v">{v}</div><div class="l">{html.escape(t)}</div>'
                        f'<div class="s">{html.escape(sub)}</div></div>' for t, v, sub in cards)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>agent-evals demo</title>
<style>
:root {{ --bg:#fafaf9; --fg:#1c1917; --muted:#57534e; --line:#e7e5e4; --card:#fff;
  --pass:#15803d; --fail:#b91c1c; --flaky:#b45309; --err:#78716c; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#141210; --fg:#f5f5f4; --muted:#a8a29e; --line:#2e2a27;
  --card:#1c1917; --pass:#4ade80; --fail:#f87171; --flaky:#fbbf24; --err:#a8a29e; }} }}
* {{ box-sizing:border-box }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 ui-sans-serif,system-ui,sans-serif }}
main {{ max-width:960px; margin:0 auto; padding:32px 16px 48px }}
h1 {{ font-size:28px; margin:0 0 4px }} h2 {{ font-size:18px; margin:32px 0 8px }}
p.lead {{ color:var(--muted); margin:0 0 20px }}
.mode {{ display:inline-block; font:13px ui-monospace,monospace; padding:4px 10px; border:1px solid var(--line);
  border-radius:999px; background:var(--card) }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin-top:20px }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px }}
.card .v {{ font-size:30px; font-weight:650 }} .card .l {{ font-weight:600 }} .card .s {{ color:var(--muted); font-size:13px }}
.wrap {{ overflow-x:auto; border:1px solid var(--line); border-radius:10px; background:var(--card) }}
table {{ border-collapse:collapse; width:100%; font-size:14px }}
th,td {{ text-align:left; padding:8px 12px; border-bottom:1px solid var(--line); vertical-align:top }}
th {{ color:var(--muted); font-weight:600 }} tr:last-child td {{ border-bottom:0 }}
.dot {{ display:inline-block; width:12px; height:12px; border-radius:3px; margin-right:3px }}
.dot.p {{ background:var(--pass) }} .dot.f {{ background:var(--fail) }}
.dot.e {{ background:transparent; border:2px dashed var(--err) }}
.tag {{ font-size:12px; font-weight:600; padding:2px 8px; border-radius:999px; border:1px solid currentColor }}
.tag.pass {{ color:var(--pass) }} .tag.fail {{ color:var(--fail) }} .tag.flaky {{ color:var(--flaky) }} .tag.error {{ color:var(--err) }}
pre {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px; overflow-x:auto; font-size:13px }}
footer {{ color:var(--muted); font-size:13px; margin-top:32px }} a {{ color:inherit }}
</style></head><body><main>
<h1>agent-evals</h1>
<p class="lead">A tool-calling support agent for a fictional game store, run against 15 JSON scenarios. Each scenario checks
the tool calls (names, argument constraints) and the final answer, k times per scenario.</p>
<span class="mode">mode: {html.escape(described)}</span>
<div class="cards">{card_html}</div>
<h2>Scenarios</h2>
<div class="wrap"><table><thead><tr><th>scenario</th><th>runs</th><th>passes</th><th>status</th><th>first failure</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>
<p style="color:var(--muted);font-size:13px">Filled square: pass. Red: a check failed. Dashed: adapter error (in the recording:
HTTP 503 from the Gemini free tier); errored runs are left out of every rate.</p>
<h2>Regression diff: local qwen3.5:9b (k=5) &rarr; gemma-4-31b-it (k=3)</h2>
<pre>{html.escape(diff_text)}</pre>
<p style="color:var(--muted);font-size:13px">In CI, <code>python -m agent_evals diff OLD NEW</code> exits 1 on a regression.</p>
<footer>Generated by <code>python -m agent_evals demo --html docs/index.html</code>.
Source and measured results: <a href="https://github.com/oguzhanozfe/agent-evals">github.com/oguzhanozfe/agent-evals</a>.</footer>
</main></body></html>
"""
