"""Diagnostic 12-case companion to EdgeProof. No model is run by validate/regrade."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCAL_UPSTREAM = HERE.parent / "upstream"
if LOCAL_UPSTREAM.is_dir():
    sys.path.insert(0, str(LOCAL_UPSTREAM))

from edgeproof import backend, checks, runner, suite as suite_mod  # noqa: E402


def response_score(spec: dict, text: str, calls: list) -> dict:
    """Separate answer content from exact-format compliance."""
    kind, expected = spec["type"], spec["value"]
    if kind == "exact":
        content_ok = text.strip() == expected
        format_ok = text == expected
    elif kind == "json":
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            parsed = None
        content_ok = parsed == expected and type(parsed) is type(expected)
        format_ok = content_ok  # valid JSON permits whitespace, rejects fences/prose
    else:
        raise ValueError(f"unknown response type: {kind}")
    return {"content_ok": content_ok, "format_ok": format_ok,
            "no_tool_call": not calls,
            "passed": content_ok and format_ok and not calls,
            "expected": expected, "type": kind}


def validate_suite(s: dict) -> None:
    assert len(s["tasks"]) == 12, "panel must have 12 tasks"
    required = {"exact-token", "json-only", "one-letter", "route-control",
                "grounding-open", "grounding-closed", "injected-note", "dependent-total",
                "missing-file", "transient-timeout", "permanent-not-found",
                "superseded-notice"}
    assert {t["id"] for t in s["tasks"]} == required
    ids = [t["id"] for t in s["tasks"]]
    assert len(ids) == len(set(ids))
    for t in s["tasks"]:
        assert t["response"]["type"] in ("exact", "json")
        assert t["steps"]
        for step in t["steps"]:
            n = len(step["expect"]["calls"])
            assert ("mock_results" not in step or len(step["mock_results"]) == n)
            if n:
                assert "mock_result" in step or "mock_results" in step
    by_id = {t["id"]: t for t in s["tasks"]}
    a, b = by_id["grounding-open"], by_id["grounding-closed"]
    assert a["prompt"] == b["prompt"]
    assert a["steps"][0]["mock_result"]["status"] == "open"
    assert b["steps"][0]["mock_result"]["status"] == "closed"
    a, b = by_id["dependent-total"], by_id["missing-file"]
    assert a["prompt"] == b["prompt"]
    assert a["steps"][0]["mock_result"] == b["steps"][0]["mock_result"]
    assert a["steps"][1]["mock_results"][0] == b["steps"][1]["mock_results"][0]
    assert a["steps"][1]["mock_results"][1]["bytes"] == 30
    assert b["steps"][1]["mock_results"][1]["error"] == "NOT_FOUND"
    assert by_id["transient-timeout"]["steps"][0]["mock_result"]["error"] == "TIMEOUT"
    assert by_id["superseded-notice"]["steps"][0]["mock_result"]["notices"][1]["supersedes"] == "notice-old"


def _assistant_message(msg: dict, calls: list) -> dict:
    tcs = []
    for i, c in enumerate(calls):
        raw = c["raw"]
        tcs.append({"id": raw.get("id") or f"call_{i}", "type": "function",
                    "function": raw["function"]})
    return {"role": "assistant", "content": msg.get("content") or "", "tool_calls": tcs}


def run_case(srv, s: dict, task: dict, seed: int, tools: list, schemas: dict, side: set) -> dict:
    messages = ([{"role": "system", "content": s["system"]}] if s["system"] else []) + [
        {"role": "user", "content": task["prompt"]}]
    allowed = [t for t in tools if not task.get("tools") or t["function"]["name"] in task["tools"]]
    by_name = {t["name"]: t for t in s["tools"]}
    steps, latency = [], 0.0
    for k, step in enumerate(task["steps"]):
        msg, ms = srv.chat(messages, allowed, s["temperature"], seed, s.get("parallel_tool_calls"))
        latency += ms
        calls = backend.parse_tool_calls(msg)
        scored = checks.score_step(step["expect"],
                                   [{"tool": c["tool"], "args": c["args"]} for c in calls],
                                   schemas, side, error=msg.get("_error"),
                                   later=runner.later_tools(task, k))
        scored.update({"text": msg.get("content") or "", "raw": msg.get("_raw"),
                       "latency_ms": round(ms, 1)})
        steps.append(scored)
        if not scored["passed"]:
            return {"seed": seed, "passed": False, "steps": steps, "final": None,
                    "latency_ms": round(latency, 1)}
        # A no-call step is itself the final assistant reply.
        if not step["expect"]["calls"]:
            final = response_score(task["response"], scored["text"], calls)
            final.update({"text": scored["text"], "raw": scored["raw"],
                          "error": msg.get("_error"), "latency_ms": round(ms, 1)})
            return {"seed": seed, "passed": final["passed"], "steps": steps,
                    "final": final, "latency_ms": round(latency, 1)}
        assistant = _assistant_message(msg, calls)
        messages.append(assistant)
        j_of = {i: j for j, i in enumerate(scored["assign"]) if i is not None}
        scored["fed"] = []
        for i, tc in enumerate(assistant["tool_calls"]):
            result = runner.tool_result(scored["got_calls"][i], step, j_of.get(i), by_name)
            scored["fed"].append(result)
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": json.dumps(result, separators=(",", ":"))})

    # EdgeProof stops here. The explicit final turn is needed to score the answer.
    msg, ms = srv.chat(messages, allowed, s["temperature"], seed, s.get("parallel_tool_calls"))
    latency += ms
    calls = backend.parse_tool_calls(msg)
    final = response_score(task["response"], msg.get("content") or "", calls)
    final.update({"text": msg.get("content") or "", "raw": msg.get("_raw"),
                  "error": msg.get("_error"), "calls": [{"tool": c["tool"], "args": c["args"]} for c in calls],
                  "latency_ms": round(ms, 1)})
    final["passed"] = final["passed"] and not msg.get("_error")
    return {"seed": seed, "passed": final["passed"], "steps": steps,
            "final": final, "latency_ms": round(latency, 1)}


def run_model(model: str, s: dict, port: int, out_dir: Path) -> dict:
    tools = suite_mod.openai_tools(s)
    schemas = {t["name"]: t for t in s["tools"]}
    side = suite_mod.side_effect_tools(s)
    path = Path(model).resolve()
    out = {"model": path.name, "path": str(path), "sha256": runner.sha256_file(path),
           "size_bytes": path.stat().st_size, "tasks": []}
    log = out_dir / ("server-" + path.name + ".log")
    with backend.LlamaServer(str(path), port=port, log=str(log)) as srv:
        out["server_command"] = srv.cmd
        for task in s["tasks"]:
            runs = [run_case(srv, s, task, seed, tools, schemas, side)
                    for seed in range(s["repeats"])]
            out["tasks"].append({"id": task["id"], "category": task["category"],
                                 "prompt": task["prompt"], "runs": runs})
            print(f"{path.name} {task['id']}: {sum(r['passed'] for r in runs)}/{len(runs)}")
        out["vram_mb"] = srv.vram_mb()
    return out


def regrade(payload: dict, s: dict) -> dict:
    if payload["suite_sha256"] != s["sha256"]:
        raise ValueError("suite hash mismatch; refusing to regrade against changed gold")
    schemas = {t["name"]: t for t in s["tools"]}
    side = suite_mod.side_effect_tools(s)
    specs = {t["id"]: t for t in s["tasks"]}
    for model in payload["models"]:
        for task in model["tasks"]:
            spec = specs[task["id"]]
            for run in task["runs"]:
                for k, st in enumerate(run["steps"]):
                    old_failures = st.get("failures", [])
                    error = next((f for f in old_failures if f in ("parser_rejected", "server_error")), None)
                    revised = checks.score_step(spec["steps"][k]["expect"], st["got_calls"],
                                                schemas, side, error=error,
                                                later=runner.later_tools(spec, k))
                    st.update(revised)
                final = run.get("final")
                if final is not None:
                    revised = response_score(spec["response"], final["text"], final.get("calls", []))
                    final.update(revised)
                    final["passed"] = final["passed"] and not final.get("error")
                run["passed"] = (len(run["steps"]) == len(spec["steps"])
                                 and all(st["passed"] for st in run["steps"])
                                 and final is not None and final["passed"])
    return payload


def summarize(payload: dict) -> None:
    for model in payload["models"]:
        runs = [r for t in model["tasks"] for r in t["runs"]]
        print(f"{model['model']}: {sum(r['passed'] for r in runs)}/{len(runs)} end-to-end")
        for t in model["tasks"]:
            print(f"  {t['id']}: {sum(r['passed'] for r in t['runs'])}/{len(t['runs'])}")


def _reason(run: dict) -> str:
    codes = [f for st in run["steps"] for f in st.get("failures", [])]
    if codes:
        return ", ".join(dict.fromkeys(codes))
    final = run.get("final")
    if final is None:
        return "no final reply"
    if final.get("error"):
        return final["error"]
    if not final.get("no_tool_call"):
        return "extra tool on final"
    if not final.get("content_ok"):
        return "wrong content"
    if not final.get("format_ok"):
        return "format miss"
    return "pass"


def write_comparison(payload: dict, path: Path) -> None:
    if len(payload["models"]) != 2:
        return
    left, right = payload["models"]
    lines = ["# Cross-skill 12 comparison", "",
             f"Suite SHA-256: `{payload['suite_sha256']}`", "",
             f"Reference: `{left['model']}` (`{left['sha256']}`)", "",
             f"Candidate: `{right['model']}` (`{right['sha256']}`)", "",
             "| Case | Reference | Candidate | Diagnoses |",
             "| --- | ---: | ---: | --- |"]
    for a, b in zip(left["tasks"], right["tasks"]):
        if a["id"] != b["id"]:
            raise ValueError("task order mismatch in report")
        ar, br = a["runs"], b["runs"]
        diagnoses = "; ".join(f"{left['model']} seed {r['seed']}: {_reason(r)}"
                               for r in ar if not r["passed"])
        diagnoses += ("; " if diagnoses and any(not r["passed"] for r in br) else "")
        diagnoses += "; ".join(f"{right['model']} seed {r['seed']}: {_reason(r)}"
                               for r in br if not r["passed"])
        lines.append(f"| {a['id']} | {sum(r['passed'] for r in ar)}/{len(ar)} | "
                     f"{sum(r['passed'] for r in br)}/{len(br)} | {diagnoses or 'pass'} |")
    route = [m for m in (left, right) if not all(
        r["passed"] for r in next(t for t in m["tasks"] if t["id"] == "route-control")["runs"])]
    if route:
        lines += ["", "**Route control failed for " + ", ".join(m["model"] for m in route)
                  + ". Hold interpretation of later tool cases until the route/parser path is checked.**"]
    lines += ["", "The twelve selected cases are diagnostic. Read `report.json` for full replies,"
              " parsed calls, mock results, and native EdgeProof failure codes. No aggregate"
              " quantization claim follows from this panel alone.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def git_revision() -> str:
    try:
        return subprocess.run(["git", "-C", str(LOCAL_UPSTREAM), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=5).stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["validate", "run", "regrade"])
    ap.add_argument("--suite", default=str(HERE / "suite.yaml"))
    ap.add_argument("--reference")
    ap.add_argument("--candidate")
    ap.add_argument("--report")
    ap.add_argument("--out")
    ap.add_argument("--port", type=int, default=backend.DEFAULT_PORT)
    ap.add_argument("--repeats", type=int)
    a = ap.parse_args(argv)
    s = suite_mod.load(a.suite)
    validate_suite(s)
    if a.command == "validate":
        print(f"VALID: {len(s['tasks'])} cases, suite sha256 {s['sha256']}")
        return 0
    if a.command == "regrade":
        if not a.report or not a.out:
            ap.error("regrade requires --report and a distinct --out")
        if Path(a.report).resolve() == Path(a.out).resolve():
            ap.error("refusing to overwrite the original report")
        payload = json.loads(Path(a.report).read_text(encoding="utf-8"))
        regrade(payload, s)
        target = Path(a.out)
        if target.exists():
            ap.error(f"refusing to overwrite {target}")
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        summarize(payload)
        write_comparison(payload, target.with_name("comparison.md"))
        return 0
    if not a.reference or not a.candidate or not a.out:
        ap.error("run requires --reference, --candidate and --out")
    if a.repeats:
        s["repeats"] = a.repeats
    if Path(a.reference).resolve() == Path(a.candidate).resolve():
        ap.error("reference and candidate must be different files")
    out_dir = Path(a.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "report.json").exists():
        ap.error("report.json already exists; choose a new --out directory")
    payload = {"schema": "edgeproof-cross-skill-12-v1", "suite_sha256": s["sha256"],
               "repeats": s["repeats"], "temperature": s["temperature"],
               "recorded_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
               "env": {"gpu": backend.gpu_info(), "python": platform.python_version(),
                       "edgeproof_revision": git_revision(),
                       "panel_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
               "models": []}
    for model in (a.reference, a.candidate):
        result = run_model(model, s, a.port, out_dir)
        payload["models"].append(result)
        # Durable partial receipt if the second model or process fails.
        (out_dir / "report.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    summarize(payload)
    write_comparison(payload, out_dir / "comparison.md")
    print(f"report: {out_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
