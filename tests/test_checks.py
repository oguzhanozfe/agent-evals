import unittest

from agent_evals.checks import arg_matches, check_answer, check_tools, extract_json, is_subset


def call(name, **args):
    return {"name": name, "arguments": args}


class ArgConstraints(unittest.TestCase):
    def test_exact_regex_subset(self):
        self.assertTrue(arg_matches("O1", "O1"))
        self.assertTrue(arg_matches({"exact": "O1"}, "O1"))
        self.assertFalse(arg_matches({"exact": "O1"}, "o1"))
        self.assertTrue(arg_matches({"regex": "^O\\d+$"}, "O12"))
        self.assertFalse(arg_matches({"regex": "x"}, None))
        self.assertTrue(arg_matches({"subset": {"a": 1}}, {"a": 1, "b": 2}))
        self.assertFalse(arg_matches({"subset": {"a": 1}}, {"a": 2}))

    def test_subset_nested_and_lists(self):
        self.assertTrue(is_subset({"x": {"y": [1]}}, {"x": {"y": [3, 1], "z": 0}}))
        self.assertFalse(is_subset([{"id": 2}], [{"id": 1}]))
        self.assertFalse(is_subset({"a": 1}, "a"))

    def test_a_dict_with_other_keys_is_an_exact_value(self):
        self.assertTrue(arg_matches({"exact": 1, "other": 2}, {"exact": 1, "other": 2}))


class ToolChecks(unittest.TestCase):
    calls = [call("find", email="a@x"), call("get", id="O1"), call("get", id="O2")]

    def test_unordered_any_match(self):
        s = {"expect_tools": [{"name": "get", "args": {"id": "O2"}}, {"name": "find"}]}
        self.assertEqual(check_tools(s, self.calls), [])

    def test_missing_and_wrong_args_are_explained(self):
        s = {"expect_tools": [{"name": "get", "args": {"id": "O9"}}, {"name": "refund"}]}
        failures = check_tools(s, self.calls)
        self.assertIn("called with", failures[0])
        self.assertIn("never called", failures[1])

    def test_ordered_is_a_subsequence(self):
        ok = {"ordered": True, "expect_tools": [{"name": "find"}, {"name": "get", "args": {"id": "O2"}}]}
        bad = {"ordered": True, "expect_tools": [{"name": "get"}, {"name": "find"}]}
        self.assertEqual(check_tools(ok, self.calls), [])
        self.assertEqual(len(check_tools(bad, self.calls)), 1)

    def test_forbid_limit_and_invalid_json(self):
        s = {"forbid_tools": ["get"], "max_tool_calls": 1}
        self.assertEqual(len(check_tools(s, self.calls)), 2)
        bad = [{"name": "get", "arguments": {"_raw": "{", "_invalid_json": True}}]
        self.assertIn("not valid JSON", check_tools({}, bad)[0])


class AnswerChecks(unittest.TestCase):
    def test_contains_is_case_insensitive(self):
        self.assertEqual(check_answer({"answer": [{"contains": "pending"}]}, "Status: PENDING"), [])
        self.assertEqual(len(check_answer({"answer": [{"not_contains": "x"}]}, "X marks")), 1)

    def test_regex(self):
        self.assertEqual(check_answer({"answer": [{"regex": "874[.,]65"}]}, "874,65 TRY"), [])

    def test_json_shape(self):
        s = {"answer": [{"json_shape": {"id": "str", "n": "number", "tags": ["str"], "x": "any"}}]}
        self.assertEqual(check_answer(s, 'Here: ```json\n{"id": "a", "n": 2, "tags": ["t"], "x": null}\n```'), [])
        failures = check_answer(s, '{"id": 1, "n": true, "tags": [2]}')
        self.assertEqual(len(failures), 4)  # id type, bool is not a number, tags[0], x missing
        self.assertIn("no JSON", check_answer(s, "no json here")[0])

    def test_extract_json_skips_braces_that_are_not_json(self):
        self.assertEqual(extract_json('use {curly} then {"a": 1}'), {"a": 1})

    def test_unknown_check_is_an_error(self):
        with self.assertRaises(ValueError):
            check_answer({"id": "s", "answer": [{"startswith": "x"}]}, "x")


if __name__ == "__main__":
    unittest.main()
