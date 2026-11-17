"""Deterministic checks on one agent run: tool calls and the final answer.

A run is a dict: {"answer": str, "tool_calls": [{"name": str, "arguments": dict}], ...}.
Every check returns a list of failure strings; an empty list means it passed.
"""
import json
import re

TYPES = {
    "str": str, "int": int, "float": float, "number": (int, float),
    "bool": bool, "null": type(None), "list": list, "dict": dict,
}


def arg_matches(constraint, value):
    """A constraint is a plain value (exact match) or one of {"exact"}, {"regex"}, {"subset"}."""
    if isinstance(constraint, dict) and len(constraint) == 1:
        (kind, expected), = constraint.items()
        if kind == "exact":
            return value == expected
        if kind == "regex":
            return value is not None and re.search(expected, str(value)) is not None
        if kind == "subset":
            return is_subset(expected, value)
    return value == constraint


def is_subset(expected, actual):
    """Dicts: every expected key present with a subset value. Lists: every expected item found. Else equal."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(k in actual and is_subset(v, actual[k]) for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and all(any(is_subset(e, a) for a in actual) for e in expected)
    return expected == actual


def call_matches(expected, call):
    if call["name"] != expected["name"]:
        return False
    args = call.get("arguments") or {}
    return all(arg_matches(c, args.get(name)) for name, c in expected.get("args", {}).items())


def check_tools(scenario, calls):
    failures = []
    # Ordered: expected calls must appear as a subsequence. Unordered: each needs some matching call.
    # Extra calls are allowed either way; use forbid_tools / max_tool_calls to limit them.
    position = 0
    for expected in scenario.get("expect_tools", []):
        pool = calls[position:] if scenario.get("ordered") else calls
        hit = next((i for i, c in enumerate(pool) if call_matches(expected, c)), None)
        if hit is None:
            seen = [c for c in calls if c["name"] == expected["name"]]
            why = f"called with {[c.get('arguments') for c in seen]}" if seen else "never called"
            failures.append(f"expected tool {expected['name']} {expected.get('args', {})}: {why}")
        elif scenario.get("ordered"):
            position += hit + 1
    for name in scenario.get("forbid_tools", []):
        if any(c["name"] == name for c in calls):
            failures.append(f"forbidden tool called: {name}")
    limit = scenario.get("max_tool_calls")
    if limit is not None and len(calls) > limit:
        failures.append(f"{len(calls)} tool calls, limit {limit}")
    for c in calls:
        if (c.get("arguments") or {}).get("_invalid_json"):
            failures.append(f"tool {c['name']} got arguments that are not valid JSON")
    return failures


def extract_json(text):
    """The first JSON object or array in the text, with or without a ```json fence."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1)
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                return decoder.raw_decode(text[i:])[0]
            except ValueError:
                continue
    raise ValueError("no JSON found in the answer")


def shape_errors(shape, value, path="$"):
    """shape: a type name ("str", "number", "any", ...), a dict of key -> shape, or [item_shape]."""
    if isinstance(shape, dict):
        if not isinstance(value, dict):
            return [f"{path}: expected object"]
        errors = []
        for key, sub in shape.items():
            if key not in value:
                errors.append(f"{path}.{key}: missing")
            else:
                errors += shape_errors(sub, value[key], f"{path}.{key}")
        return errors
    if isinstance(shape, list):
        if not isinstance(value, list):
            return [f"{path}: expected array"]
        return [e for i, v in enumerate(value) for e in shape_errors(shape[0], v, f"{path}[{i}]")] if shape else []
    if shape == "any":
        return []
    wanted = TYPES[shape]
    if isinstance(value, bool) and shape in ("int", "float", "number"):
        return [f"{path}: expected {shape}, got bool"]
    return [] if isinstance(value, wanted) else [f"{path}: expected {shape}, got {type(value).__name__}"]


def check_answer(scenario, answer):
    failures = []
    for check in scenario.get("answer", []):
        (kind, expected), = check.items()
        if kind == "contains" and expected.lower() not in answer.lower():
            failures.append(f"answer lacks {expected!r}")
        elif kind == "not_contains" and expected.lower() in answer.lower():
            failures.append(f"answer contains {expected!r}")
        elif kind == "regex" and not re.search(expected, answer):
            failures.append(f"answer does not match /{expected}/")
        elif kind == "json_shape":
            try:
                failures += [f"json {e}" for e in shape_errors(expected, extract_json(answer))]
            except ValueError as e:
                failures.append(str(e))
        elif kind not in ("contains", "not_contains", "regex", "json_shape"):
            raise ValueError(f"unknown answer check {kind!r} in scenario {scenario['id']}")
    return failures


def check_run(scenario, run):
    return check_tools(scenario, run["tool_calls"]) + check_answer(scenario, run["answer"])
