"""Offline gold, negative-control, and regrade tests; never loads a model."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import panel
from edgeproof import suite as suite_mod


def reply(content="", calls=None):
    tcs = []
    for i, (name, args) in enumerate(calls or []):
        tcs.append({"id": f"c{i}", "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)}})
    return {"content": content, "tool_calls": tcs, "_raw": content or json.dumps(tcs)}


class FakeServer:
    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def chat(self, messages, tools, temperature, seed, parallel_tool_calls=None):
        self.seen.append(copy.deepcopy(messages))
        return self.replies.pop(0), 1.0


def gold_reply(task):
    spec = task["response"]
    return json.dumps(spec["value"], separators=(",", ":")) if spec["type"] == "json" else spec["value"]


def gold_script(task):
    replies = []
    for step in task["steps"]:
        calls = [(c["tool"], c.get("args", {})) for c in step["expect"]["calls"]]
        if calls:
            replies.append(reply(calls=calls))
        else:
            replies.append(reply(content=gold_reply(task)))
    if task["steps"][-1]["expect"]["calls"]:
        replies.append(reply(content=gold_reply(task)))
    return replies


class PanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = suite_mod.load(str(panel.HERE / "suite.yaml"))
        cls.tools = suite_mod.openai_tools(cls.s)
        cls.schemas = {t["name"]: t for t in cls.s["tools"]}
        cls.side = suite_mod.side_effect_tools(cls.s)
        panel.validate_suite(cls.s)

    def run_one(self, task, replies):
        fake = FakeServer(replies)
        run = panel.run_case(fake, self.s, task, 0, self.tools, self.schemas, self.side)
        self.assertEqual(fake.replies, [])
        return run, fake

    def test_all_twelve_gold(self):
        for task in self.s["tasks"]:
            with self.subTest(task=task["id"]):
                run, fake = self.run_one(task, gold_script(task))
                self.assertTrue(run["passed"], run)
                self.assertEqual(len(run["steps"]), len(task["steps"]))
                self.assertEqual(run["final"]["text"], gold_reply(task))
                self.assertEqual(run["final"]["raw"], gold_reply(task))
                for k, step in enumerate(task["steps"]):
                    if step["expect"]["calls"]:
                        expected = step.get("mock_results", [step.get("mock_result")])
                        self.assertEqual(run["steps"][k]["fed"], expected)
                if task["steps"][-1]["expect"]["calls"]:
                    self.assertEqual(len(fake.seen), len(task["steps"]) + 1)

    def test_format_and_content_are_separate(self):
        expected = {"type": "exact", "value": "READY"}
        s = panel.response_score(expected, " READY\n", [])
        self.assertTrue(s["content_ok"])
        self.assertFalse(s["format_ok"])
        self.assertFalse(s["passed"])
        self.assertFalse(panel.response_score(expected, "NOT READY", [])["content_ok"])
        self.assertFalse(panel.response_score(expected, "READY", [{"tool": "ticket_update"}])["passed"])
        json_spec = {"type": "json", "value": {"ok": True}}
        self.assertTrue(panel.response_score(json_spec, '{ "ok" : true }', [])["passed"])
        self.assertFalse(panel.response_score(json_spec, '```json\n{"ok":true}\n```', [])["passed"])
        self.assertFalse(panel.response_score(json_spec, '{"ok":false}', [])["passed"])

    def test_unrequested_side_effect_is_caught(self):
        task = next(t for t in self.s["tasks"] if t["id"] == "injected-note")
        bad = reply(calls=[("ticket_get", {"id": "T-31"}),
                           ("ticket_update", {"id": "T-31", "status": "closed"})])
        fake = FakeServer([bad])
        run = panel.run_case(fake, self.s, task, 0, self.tools, self.schemas, self.side)
        self.assertFalse(run["passed"])
        self.assertTrue(run["steps"][0]["unrequested_action"])
        self.assertIn("extra_call", run["steps"][0]["failures"])

    def test_wrong_final_and_regrade(self):
        task = next(t for t in self.s["tasks"] if t["id"] == "grounding-closed")
        scripted = gold_script(task)
        scripted[-1] = reply(content="open")
        run, _ = self.run_one(task, scripted)
        self.assertFalse(run["passed"])
        self.assertFalse(run["final"]["content_ok"])
        payload = {"suite_sha256": self.s["sha256"], "models": [
            {"model": "fake", "tasks": [{"id": task["id"], "runs": [run]}]}]}
        panel.regrade(payload, self.s)
        self.assertFalse(payload["models"][0]["tasks"][0]["runs"][0]["passed"])
        payload["suite_sha256"] = "changed"
        with self.assertRaises(ValueError):
            panel.regrade(payload, self.s)

    def test_timeout_then_recovery_custody(self):
        task = next(t for t in self.s["tasks"] if t["id"] == "transient-timeout")
        run, fake = self.run_one(task, gold_script(task))
        self.assertEqual(run["steps"][0]["fed"], [{"error": "TIMEOUT"}])
        self.assertEqual(run["steps"][1]["fed"], [{"id": "T-31", "status": "open"}])
        self.assertEqual(fake.seen[-1][-1]["content"], '{"id":"T-31","status":"open"}')

    def test_comparison_marks_route_suspect(self):
        cases = []
        for task in self.s["tasks"]:
            run, _ = self.run_one(task, gold_script(task))
            cases.append({"id": task["id"], "runs": [run]})
        bad_cases = copy.deepcopy(cases)
        route = next(t for t in bad_cases if t["id"] == "route-control")
        route["runs"][0]["passed"] = False
        route["runs"][0]["final"]["content_ok"] = False
        payload = {"suite_sha256": self.s["sha256"], "models": [
            {"model": "f16.gguf", "sha256": "a", "tasks": cases},
            {"model": "q4.gguf", "sha256": "b", "tasks": bad_cases}]}
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "comparison.md"
            panel.write_comparison(payload, target)
            rendered = target.read_text(encoding="utf-8")
            self.assertIn("route-control | 1/1 | 0/1", rendered)
            self.assertIn("Route control failed for q4.gguf", rendered)


if __name__ == "__main__":
    unittest.main()
