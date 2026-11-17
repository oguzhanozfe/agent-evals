"""Command line: python -m agent_evals run|diff|show ...

Exit codes: run 0 = every run completed and passed, 1 = some run failed a check, 3 = no check failed but
some runs hit adapter errors (timeouts, HTTP errors), so the suite is incomplete; diff 0 = no regression,
1 = regression; 2 = bad input for either.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

from . import __version__
from .adapters import HTTPAgent, OpenAIChatAgent, load_callable, mocked_tools
from .diff import diff, render
from .runner import markdown, run_suite


def load_scenarios(path, only=None):
    data = json.load(open(path, encoding="utf-8"))
    scenarios = data["scenarios"] if isinstance(data, dict) else data
    ids = [s["id"] for s in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError("scenario ids must be unique")
    for s in scenarios:
        if not s.get("turns"):
            raise ValueError(f"scenario {s['id']} has no turns")
    return [s for s in scenarios if not only or s["id"].startswith(only)]


def build_agent(args):
    if args.agent:
        return load_callable(args.agent), {"adapter": "callable", "target": args.agent}
    if args.http:
        return HTTPAgent(args.http, per_turn=args.per_turn), {"adapter": "http", "per_turn": args.per_turn}
    spec = json.load(open(args.tools, encoding="utf-8"))
    agent = OpenAIChatAgent(args.openai, args.model, spec["tools"], mocked_tools(spec), spec.get("system"),
                            api_key=os.environ.get(args.api_key_env), temperature=args.temperature,
                            reasoning_effort=args.reasoning_effort)
    return agent, {"adapter": "openai", "model": args.model, "temperature": args.temperature,
                   "reasoning_effort": args.reasoning_effort, "tools_file": os.path.basename(args.tools)}


def cmd_run(args):
    scenarios = load_scenarios(args.scenarios, args.only)
    agent, meta = build_agent(args)
    judge = None
    if args.judge_model:
        judge = OpenAIChatAgent(args.judge_base_url, args.judge_model, None, {},
                                api_key=os.environ.get(args.judge_api_key_env or args.api_key_env),
                                reasoning_effort=args.reasoning_effort).chat
        meta["judge_model"] = args.judge_model
    meta.update({"scenarios_file": os.path.basename(args.scenarios), "label": args.label,
                 "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "agent_evals": __version__})
    if args.price_in is not None:
        meta.update(price_in_per_mtok=args.price_in, price_out_per_mtok=args.price_out or 0)

    def progress(sid, i, k, run):
        mark = "ERR " if "error" in run else "pass" if run["passed"] else "FAIL"
        why = run.get("error") or "; ".join(run.get("failures", []))
        print(f"{mark} {sid} [{i}/{k}] {run['latency_s']:.1f}s {why[:100]}", file=sys.stderr, flush=True)

    report = run_suite(scenarios, agent, args.k, judge, meta, progress)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1, ensure_ascii=False)
    print(markdown(report))
    runs = [r for sc in report["scenarios"].values() for r in sc["runs"]]
    if any("error" not in r and not r["passed"] for r in runs):
        return 1
    return 3 if any("error" in r for r in runs) else 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="agent_evals", description="Evaluate tool-calling LLM agents.")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run scenarios k times against an agent")
    r.add_argument("scenarios")
    target = r.add_mutually_exclusive_group(required=True)
    target.add_argument("--agent", help="Python callable, module:attr, taking a list of user turns")
    target.add_argument("--http", help="URL of your agent service")
    target.add_argument("--openai", metavar="BASE_URL", help="OpenAI-compatible base URL, e.g. http://127.0.0.1:11434/v1")
    r.add_argument("--per-turn", action="store_true", help="http: one POST per turn ({session} in the URL)")
    r.add_argument("--model", help="openai: model name")
    r.add_argument("--tools", help="openai: JSON file with system, tools and mock_results")
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--reasoning-effort", help='passed through, e.g. "none" to turn thinking off on Ollama')
    r.add_argument("--api-key-env", default="AGENT_EVALS_API_KEY", help="env var holding the API key, if any")
    r.add_argument("-k", type=int, default=1, help="runs per scenario")
    r.add_argument("--only", help="run scenarios whose id starts with this")
    r.add_argument("--out", help="write the run file here (JSON)")
    r.add_argument("--label", default="")
    r.add_argument("--judge-model", help="optional LLM judge for scenarios with a rubric; reported, never gating")
    r.add_argument("--judge-base-url", default="http://127.0.0.1:11434/v1")
    r.add_argument("--judge-api-key-env", help="env var holding the judge's API key (default: --api-key-env)")
    r.add_argument("--price-in", type=float, help="price per million prompt tokens, for a cost figure")
    r.add_argument("--price-out", type=float, help="price per million completion tokens")
    d = sub.add_parser("diff", help="compare two run files; exit 1 on regression")
    d.add_argument("old")
    d.add_argument("new")
    s = sub.add_parser("show", help="print the summary of a run file")
    s.add_argument("run")
    args = p.parse_args(argv)
    try:
        if args.cmd == "run":
            if args.openai and not (args.model and args.tools):
                p.error("--openai needs --model and --tools")
            return cmd_run(args)
        if args.cmd == "show":
            print(markdown(json.load(open(args.run, encoding="utf-8"))))
            return 0
        result = diff(json.load(open(args.old, encoding="utf-8")), json.load(open(args.new, encoding="utf-8")))
        print(render(result))
        return 1 if result["regressions"] else 0
    except (OSError, ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
