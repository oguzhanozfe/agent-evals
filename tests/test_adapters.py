"""The HTTP and OpenAI-compatible adapters against an in-process stub server. No network beyond localhost."""
import json
import tempfile
import threading
from unittest import mock

from agent_evals import adapters
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from agent_evals.adapters import HTTPAgent, OpenAIChatAgent, mocked_tools, normalize
from examples import toy_agent


class Stub(BaseHTTPRequestHandler):
    replies = []  # scripted JSON responses, popped in order
    seen = []  # (path, body) of every request

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Stub.seen.append((self.path, body, self.headers.get("Authorization")))
        reply = Stub.replies.pop(0)
        code, reply = reply if isinstance(reply, tuple) else (200, reply)
        data = json.dumps(reply).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def chat_reply(content="", tool_calls=None, prompt=5, completion=3):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = [{"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": a}}
                                 for i, (n, a) in enumerate(tool_calls)]
    return {"choices": [{"message": message}], "usage": {"prompt_tokens": prompt, "completion_tokens": completion}}


class StubServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        Stub.replies, Stub.seen = [], []


class Adapters(StubServer):
    def test_openai_tool_round_trip(self):
        Stub.replies = [chat_reply(tool_calls=[("get_order", '{"order_id": "O2001"}')]),
                        chat_reply("<thought>check status</thought>It is pending.")]
        agent = OpenAIChatAgent(self.base + "/v1", "m", toy_agent.TOOLS, toy_agent.make_tools(), "sys",
                                api_key="test-key", reasoning_effort="none")
        result = agent(["Where is O2001?"])
        self.assertEqual(result["answer"], "It is pending.")
        self.assertEqual(result["tool_calls"], [{"name": "get_order", "arguments": {"order_id": "O2001"}}])
        self.assertEqual(result["usage"], {"prompt_tokens": 10, "completion_tokens": 6})
        path, second, auth = Stub.seen[1]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(auth, "Bearer test-key")
        self.assertEqual(second["reasoning_effort"], "none")
        tool_message = second["messages"][-1]
        self.assertEqual((tool_message["role"], tool_message["tool_call_id"]), ("tool", "c0"))
        self.assertIn('"status": "pending"', tool_message["content"])

    def test_openai_bad_arguments_and_tool_errors_go_back_to_the_model(self):
        Stub.replies = [chat_reply(tool_calls=[("get_order", "{oops"), ("get_order", '{"order_id": "O9"}'),
                                               ("nope", "{}")]), chat_reply("sorry")]
        agent = OpenAIChatAgent(self.base, "m", toy_agent.TOOLS, toy_agent.make_tools())
        result = agent(["x"])
        self.assertTrue(result["tool_calls"][0]["arguments"]["_invalid_json"])
        contents = [m["content"] for m in Stub.seen[1][1]["messages"] if m["role"] == "tool"]
        self.assertEqual(contents[0], "Error: arguments were not valid JSON")
        self.assertIn("not found", contents[1])
        self.assertEqual(contents[2], "Error: unknown tool nope")

    def test_openai_step_limit(self):
        Stub.replies = [chat_reply(tool_calls=[("search_help", '{"query": "x"}')])] * 2
        result = OpenAIChatAgent(self.base, "m", toy_agent.TOOLS, toy_agent.make_tools(), max_steps=2)(["x"])
        self.assertIn("stopped after 2", result["answer"])

    def test_http_whole_conversation(self):
        Stub.replies = [{"answer": "done", "tool_calls": [{"name": "t", "arguments": '{"a": 1}'}]}]
        result = normalize(HTTPAgent(self.base + "/agent")(["a", "b"]))
        self.assertEqual(Stub.seen[0][1], {"turns": ["a", "b"]})
        self.assertEqual(result["tool_calls"], [{"name": "t", "arguments": {"a": 1}}])

    def test_http_per_turn_with_session_and_custom_keys(self):
        Stub.replies = [{"text": "first", "actions": ["get_customer"], "usage": {"prompt_tokens": 1}},
                        {"text": "second", "actions": [], "usage": {"prompt_tokens": 2}}]
        agent = HTTPAgent(self.base + "/chat/{session}", per_turn=True, answer_key="text", tool_calls_key="actions")
        result = normalize(agent(["hi", "there"]))
        self.assertEqual(result["answer"], "second")
        self.assertEqual(result["tool_calls"], [{"name": "get_customer", "arguments": {}}])
        self.assertEqual(result["usage"], {"prompt_tokens": 3})
        self.assertEqual(Stub.seen[0][0], Stub.seen[1][0])  # same session within a run
        self.assertNotIn("{session}", Stub.seen[0][0])
        self.assertEqual(Stub.seen[0][1], {"message": "hi"})

    def test_mocked_tools(self):
        impls = mocked_tools({"tools": toy_agent.TOOLS[:2], "mock_results": {"find_player": {"player_id": "P1"}}})
        self.assertEqual(impls["find_player"](email="x"), {"player_id": "P1"})
        self.assertEqual(impls["get_order"](order_id="x"), "ok")


class JudgeThroughCli(StubServer):
    def test_judge_verdict_is_reported_and_thoughts_are_stripped(self):
        import io
        import os
        from contextlib import redirect_stderr, redirect_stdout
        from agent_evals.__main__ import main
        Stub.replies = [{"choices": [{"message": {"content": "<thought>hmm</thought>FAIL\nNo steps given."}}]}]
        with tempfile.TemporaryDirectory() as tmp:
            scenarios, out = os.path.join(tmp, "s.json"), os.path.join(tmp, "run.json")
            with open(scenarios, "w") as f:
                json.dump([{"id": "j", "turns": ["help"], "judge": {"rubric": "gives steps"}}], f)
            with redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()):
                code = main(["run", scenarios, "--agent", "tests.test_runner_diff:echo_agent", "--out", out,
                             "--judge-model", "judge", "--judge-base-url", self.base])
            run = json.load(open(out))["scenarios"]["j"]["runs"][0]
        self.assertEqual(code, 0)  # the judge never gates
        self.assertTrue(run["passed"])
        self.assertIs(run["judge"]["verdict"], False)
        self.assertIn("rubric: gives steps", Stub.seen[0][1]["messages"][0]["content"].lower())
        self.assertIn("weaker evidence", stdout.getvalue())


class RateLimitsAndCache(StubServer):
    def test_short_429_is_retried_once(self):
        Stub.replies = [(429, {"error": "slow down, retry in 0.01s please"}), {"ok": 1}]
        self.assertEqual(adapters.post_json(self.base, {}), {"ok": 1})

    def test_daily_quota_is_not_looped_on(self):
        Stub.replies = [(429, {"error": "quota exceeded, retry in 23549.3s"})]
        with self.assertRaises(adapters.urllib.error.HTTPError):
            adapters.post_json(self.base, {})
        self.assertEqual(len(Stub.seen), 1)

    def test_cache_is_per_sample_and_skips_the_network(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(adapters, "CACHE_DIR", tmp):
            Stub.replies = [{"n": 1}, {"n": 2}]
            first = adapters.post_json(self.base, {"q": 1})
            self.assertEqual(adapters.post_json(self.base, {"q": 1}), first)  # cached
            with mock.patch.object(adapters, "SAMPLE", 1):
                self.assertEqual(adapters.post_json(self.base, {"q": 1}), {"n": 2})  # next sample: new call
            self.assertEqual(len(Stub.seen), 2)


class ToyAgentTools(unittest.TestCase):
    def test_refund_rules_and_fresh_state(self):
        tools = toy_agent.make_tools()
        self.assertEqual(tools["refund_order"]("O1001", "mistake")["status"], "refunded")
        with self.assertRaises(ValueError):
            tools["refund_order"]("O1001", "again")
        with self.assertRaises(ValueError):
            tools["refund_order"]("O1002", "too old")
        self.assertEqual(toy_agent.make_tools()["get_order"]("O1001")["status"], "delivered")
        self.assertEqual(tools["convert_price"](24.99, "TRY"), {"amount": 874.65, "currency": "TRY"})


if __name__ == "__main__":
    unittest.main()
