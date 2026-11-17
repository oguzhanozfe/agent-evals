import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from agent_evals.__main__ import main
from agent_evals.diff import diff
from agent_evals.runner import percentile, run_suite

SCENARIOS = [
    {"id": "always", "turns": ["hi"], "answer": [{"contains": "hello"}]},
    {"id": "never", "turns": ["hi"], "answer": [{"contains": "nope"}]},
    {"id": "sometimes", "turns": ["flip"], "answer": [{"contains": "heads"}]},
    {"id": "judged", "turns": ["hi"], "judge": {"rubric": "polite"}},
]


def scripted_agent():
    flips = iter(["heads", "tails"] * 10)

    def agent(turns):
        if turns == ["flip"]:
            return {"answer": next(flips), "usage": {"prompt_tokens": 10, "completion_tokens": 2}}
        return {"answer": "hello", "tool_calls": ["greet"]}
    return agent


def broken_agent(turns):
    raise TimeoutError("model took too long")


class Runner(unittest.TestCase):
    def test_pass_at_k_pass_all_k_and_flaky(self):
        judge = lambda messages: ({"content": "PASS\npolite enough"}, {})  # noqa: E731
        report = run_suite(SCENARIOS, scripted_agent(), k=2, judge=judge, meta={"price_in_per_mtok": 1.0})
        s = report["summary"]
        self.assertEqual(report["scenarios"]["sometimes"]["status"], "flaky")
        self.assertEqual(s["flaky"], ["sometimes"])
        self.assertEqual(s["failing"], ["never"])
        self.assertEqual(s["pass_at_k"], 0.75)
        self.assertEqual(s["pass_all_k"], 0.5)
        self.assertEqual(s["tokens"], {"prompt": 20, "completion": 4})
        self.assertAlmostEqual(s["cost"], 20 / 1e6)
        self.assertEqual(s["judge_pass_rate"], 1.0)
        # The judge is reported but never decides pass/fail: "judged" has no checks, so it passes.
        self.assertTrue(all(r["passed"] for r in report["scenarios"]["judged"]["runs"]))

    def test_adapter_errors_are_recorded_not_raised(self):
        report = run_suite(SCENARIOS[:1], broken_agent, k=2)
        r = report["scenarios"]["always"]
        self.assertEqual((r["status"], r["errors"]), ("error", 2))
        self.assertIn("TimeoutError", r["runs"][0]["error"])
        self.assertIsNone(report["summary"]["latency_p50_s"])
        self.assertEqual((report["summary"]["errored"], report["summary"]["pass_all_k"]), (["always"], None))

    def test_errored_runs_are_left_out_of_rates(self):
        calls = iter([TimeoutError("x"), {"answer": "hello"}, {"answer": "hello"}, {"answer": "nope"}])

        def agent(turns):
            item = next(calls)
            if isinstance(item, Exception):
                raise item
            return item
        report = run_suite(SCENARIOS[:2], agent, k=2)
        self.assertEqual(report["scenarios"]["always"]["status"], "pass")  # 1 error + 1 pass
        s = report["summary"]
        self.assertEqual((s["pass_all_k"], s["run_pass_rate"], s["errors"]), (0.5, round(2 / 3, 4), 1))

    def test_percentile_nearest_rank(self):
        values = list(range(1, 21))
        self.assertEqual(percentile(values, 50), 10)
        self.assertEqual(percentile(values, 95), 19)
        self.assertEqual(percentile([3.0], 95), 3.0)


def report_with(statuses, k=3):
    passes = {"pass": k, "fail": 0, "flaky": 1, "error": 0}
    return {"meta": {"k": k}, "scenarios": {
        sid: {"status": st, "passes": passes[st], "runs": [{}] * k} for sid, st in statuses.items()}}


class Diff(unittest.TestCase):
    def test_categories(self):
        old = report_with({"a": "pass", "b": "flaky", "c": "fail", "d": "pass", "e": "pass", "gone": "pass"})
        new = report_with({"a": "fail", "b": "pass", "c": "flaky", "d": "flaky", "e": "error", "new": "fail"})
        d = diff(old, new)
        self.assertEqual([line.split(":")[0] for line in d["regressions"]], ["a", "d"])
        self.assertEqual([line.split(":")[0] for line in d["improvements"]], ["b", "c"])
        self.assertEqual(d["flaky_new"], ["c", "d"])
        self.assertEqual(d["flaky_fixed"], ["b"])
        self.assertEqual((d["added"], d["removed"]), (["new"], ["gone"]))
        self.assertEqual(len(d["errored"]), 1)

    def test_k_mismatch_warns(self):
        self.assertTrue(diff(report_with({"a": "pass"}, 3), report_with({"a": "pass"}, 5))["warnings"])


class Cli(unittest.TestCase):
    def test_diff_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = {}
            for name, st in [("good", "pass"), ("bad", "fail")]:
                paths[name] = os.path.join(tmp, name + ".json")
                with open(paths[name], "w") as f:
                    json.dump(report_with({"a": st}), f)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main(["diff", paths["good"], paths["good"]]), 0)
                self.assertEqual(main(["diff", paths["good"], paths["bad"]]), 1)
                self.assertEqual(main(["diff", paths["good"], os.path.join(tmp, "missing.json")]), 2)

    def test_run_with_callable_writes_a_run_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            scenarios, out = os.path.join(tmp, "s.json"), os.path.join(tmp, "out", "run.json")
            with open(scenarios, "w") as f:
                json.dump({"scenarios": SCENARIOS[:1]}, f)
            stdout = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
                code = main(["run", scenarios, "--agent", "tests.test_runner_diff:echo_agent", "-k", "2", "--out", out])
            self.assertEqual(code, 0)
            self.assertIn("pass^2", stdout.getvalue())
            self.assertEqual(json.load(open(out))["summary"]["pass_all_k"], 1.0)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = main(["run", scenarios, "--agent", "tests.test_runner_diff:flaky_network_agent"])
            self.assertEqual(code, 3)  # nothing failed a check, but the suite is incomplete


def flaky_network_agent(turns):
    raise ConnectionError("refused")


def echo_agent(turns):
    return {"answer": "hello " + turns[-1]}


if __name__ == "__main__":
    unittest.main()
