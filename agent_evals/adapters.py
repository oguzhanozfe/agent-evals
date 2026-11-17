"""Adapters turn "run this conversation" into {"answer", "tool_calls", "usage"}.

Every adapter is a callable: agent(turns: list[str]) -> dict with
  answer      final reply text
  tool_calls  [{"name": str, "arguments": dict}] in call order (a bare name string is accepted too)
  usage       optional {"prompt_tokens": int, "completion_tokens": int}
"""
import hashlib
import importlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid


CACHE_DIR = os.environ.get("AGENT_EVALS_CACHE")  # set to a folder to cache model responses on disk
SAMPLE = 0  # set by the runner: run i of k gets its own cache entries, so k samples stay independent
CACHE_HITS = 0  # counted so the runner can keep replayed runs out of latency figures
REPLAY_ONLY = False  # demo replay mode: a cache miss is an error, the network is never used


def post_json(url, body, headers=None, timeout=120):
    """POST JSON, with an optional disk cache keyed by (url, body, sample index).

    A rerun of the same suite then costs no requests. Keys and headers are not part of the key or the file.
    A 429 or 503 is retried once, and only when the server asks for a wait of 60 s or less; a daily quota
    error is raised at once and recorded as an adapter error, never looped on.
    """
    path = None
    if CACHE_DIR:
        key = hashlib.sha256(json.dumps([url, body, SAMPLE], sort_keys=True).encode()).hexdigest()
        path = os.path.join(CACHE_DIR, key + ".json")
        if os.path.exists(path):
            global CACHE_HITS
            CACHE_HITS += 1
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    if REPLAY_ONLY:
        raise LookupError("no recorded response for this request (replay mode)")
    request = urllib.request.Request(
        url, json.dumps(body).encode(),
        # Some providers' CDNs (Groq's, for one) reject urllib's default User-Agent with a 403.
        {"Content-Type": "application/json", "User-Agent": "agent-evals/0.1", **(headers or {})},
    )
    for attempt in (0, 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.load(response)
            break
        except urllib.error.HTTPError as e:
            wait = retry_after(e)
            if e.code not in (429, 503) or attempt or wait is None or wait > 60:
                raise
            time.sleep(wait)
    if path:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    return data


def retry_after(error):
    """Seconds the server asks us to wait: Retry-After, or Gemini's "retry in 12.3s". 503 without a hint: 5 s."""
    header = error.headers.get("Retry-After") if error.headers else None
    if header and header.replace(".", "", 1).isdigit():
        return float(header)
    try:
        found = re.search(r"(?:retry|try again) in ([\d.]+)s\b", error.read().decode("utf-8", "replace"))
    except OSError:
        found = None
    return float(found.group(1)) if found else (5.0 if error.code == 503 else None)


def normalize(result):
    """Bring any adapter's output to the one shape the checks read."""
    calls = []
    for c in result.get("tool_calls") or []:
        if isinstance(c, str):
            calls.append({"name": c, "arguments": {}})
        else:
            args = c.get("arguments") or {}
            if isinstance(args, str):
                args = parse_arguments(args)
            calls.append({"name": c["name"], "arguments": args})
    usage = result.get("usage") or {}
    return {
        "answer": result.get("answer") or "",
        "tool_calls": calls,
        "usage": {k: usage[k] for k in ("prompt_tokens", "completion_tokens") if isinstance(usage.get(k), int)},
    }


def parse_arguments(raw):
    try:
        args = json.loads(raw or "{}")
        return args if isinstance(args, dict) else {"_value": args}
    except ValueError:
        return {"_raw": raw, "_invalid_json": True}


def load_callable(spec):
    """"package.module:attr" -> the callable. The current directory is importable."""
    module, _, attr = spec.partition(":")
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    return getattr(importlib.import_module(module), attr or "agent")


class HTTPAgent:
    """POST to your own agent service.

    Whole conversation (default): POST url {"turns": [...]} -> {"answer", "tool_calls", "usage"}.
    per_turn=True: one POST per turn with {message_key: text}; "{session}" in the url becomes a fresh id
    per run. Tool calls and usage are summed over turns, the answer is the last reply.
    """

    def __init__(self, url, per_turn=False, message_key="message", answer_key="answer",
                 tool_calls_key="tool_calls", usage_key="usage", headers=None, timeout=120):
        self.url, self.per_turn, self.message_key = url, per_turn, message_key
        self.keys = (answer_key, tool_calls_key, usage_key)
        self.headers, self.timeout = headers or {}, timeout

    def __call__(self, turns):
        url = self.url.replace("{session}", uuid.uuid4().hex[:12])
        bodies = [{self.message_key: t} for t in turns] if self.per_turn else [{"turns": turns}]
        answer, calls, usage = "", [], {}
        for body in bodies:
            data = post_json(url, body, self.headers, self.timeout)
            answer_key, calls_key, usage_key = self.keys
            answer = data.get(answer_key) or ""
            calls += data.get(calls_key) or []
            for k, v in (data.get(usage_key) or {}).items():
                if isinstance(v, int):
                    usage[k] = usage.get(k, 0) + v
        return {"answer": answer, "tool_calls": calls, "usage": usage}


THOUGHT = re.compile(r"<thought>.*?</thought>|</?thought>", re.S)  # Gemma on the Gemini API puts these in content


class OpenAIChatAgent:
    """A minimal tool-calling loop over any OpenAI-compatible /chat/completions endpoint.

    tool_impls maps a tool name to a Python callable taking the arguments as keywords; its return
    value (str, or anything JSON-serializable) goes back to the model. Works with Ollama's /v1.
    """

    def __init__(self, base_url, model, tools, tool_impls, system=None, api_key=None, temperature=0.0,
                 reasoning_effort=None, max_steps=8, timeout=120):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.tools, self.tool_impls, self.system = model, tools, tool_impls, system
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.temperature, self.reasoning_effort = temperature, reasoning_effort
        self.max_steps, self.timeout = max_steps, timeout

    def chat(self, messages, tools=None):
        body = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if tools:
            body["tools"] = tools
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        data = post_json(self.url, body, self.headers, self.timeout)
        return data["choices"][0]["message"], data.get("usage") or {}

    def __call__(self, turns):
        messages = [{"role": "system", "content": self.system}] if self.system else []
        calls, usage, answer = [], {"prompt_tokens": 0, "completion_tokens": 0}, ""
        for turn in turns:
            messages.append({"role": "user", "content": turn})
            for _ in range(self.max_steps):
                message, used = self.chat(messages, self.tools)
                for k in usage:
                    usage[k] += used.get(k) or 0
                tool_calls = message.get("tool_calls") or []
                messages.append({"role": "assistant", "content": message.get("content") or "",
                                 **({"tool_calls": tool_calls} if tool_calls else {})})
                if not tool_calls:
                    answer = THOUGHT.sub("", message.get("content") or "").strip()
                    break
                for tc in tool_calls:
                    name = tc["function"]["name"]
                    args = parse_arguments(tc["function"].get("arguments"))
                    calls.append({"name": name, "arguments": args})
                    messages.append({"role": "tool", "tool_call_id": tc.get("id", name),
                                     "content": self.run_tool(name, args)})
            else:
                answer = f"[stopped after {self.max_steps} model calls]"
        return {"answer": answer, "tool_calls": calls, "usage": usage}

    def run_tool(self, name, args):
        if args.get("_invalid_json"):
            return "Error: arguments were not valid JSON"
        impl = self.tool_impls.get(name)
        if impl is None:
            return f"Error: unknown tool {name}"
        try:
            out = impl(**args)
        except (TypeError, ValueError, LookupError) as e:  # the model sees the error and can recover
            return f"Error: {e}"
        return out if isinstance(out, str) else json.dumps(out)


def mocked_tools(spec):
    """From a tools file {"system", "tools": [schemas], "mock_results": {name: result}}: fixed results."""
    results = spec.get("mock_results", {})
    return {
        t["function"]["name"]: (lambda result: (lambda **_: result))(results.get(t["function"]["name"], "ok"))
        for t in spec["tools"]
    }
