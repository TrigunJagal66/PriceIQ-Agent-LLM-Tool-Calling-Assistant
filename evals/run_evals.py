"""Run the eval suite against a real LLM and report per-case and overall scores.

    python -m evals.run_evals --prompt-version v1
    python -m evals.run_evals --prompt-version v2 --live-priceiq   # use the real PriceIQ server

Scoring (each case, 0/1 per check, case passes only if all checks pass):
  tools     every expected tool was called, no forbidden tool was called
  args      the first call of each expected tool contains the expected arguments
  grounded  numbers/values from tool results that the case marks as key appear in the answer
  contains  required strings appear in the answer (e.g. the forecast-horizon date)
  steps     the agent finished without hitting the step limit
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from agent.agent import AgentResult, PricingAgent
from agent.config import Settings
from agent.llm import OpenAICompatClient
from agent.prompts import SYSTEM_PROMPTS
from agent.tools import PricingTools, build_registry

from .fake_priceiq import fake_priceiq_client

CASES_PATH = Path(__file__).with_name("cases.json")


def _num_in_text(value, text: str) -> bool:
    if isinstance(value, (int, float)):
        forms = {str(value), f"{value:,}", f"{value:.2f}", f"{value:g}"}
        if float(value).is_integer():
            forms |= {str(int(value)), f"{int(value):,}"}
        return any(f in text for f in forms)
    return str(value) in text


def score_case(case: dict, res: AgentResult) -> dict:
    called = [c.name for c in res.tool_calls]
    tools_ok = all(t in called for t in case.get("expected_tools", [])) and not any(
        t in called for t in case.get("forbidden_tools", [])
    )
    if not case.get("expected_tools") and called and case.get("forbidden_tools") is None:
        tools_ok = False

    args_ok = True
    for tool, expected in case.get("expected_args", {}).items():
        first = next((c for c in res.tool_calls if c.name == tool), None)
        if first is None:
            args_ok = False
            continue
        for k, v in expected.items():
            got = first.arguments.get(k)
            if isinstance(v, (int, float)) and isinstance(got, (int, float)):
                args_ok &= float(got) == float(v)
            else:
                args_ok &= got == v

    grounded_ok = True
    for tool, field in case.get("grounded_fields", []):
        ok_calls = [c for c in res.tool_calls if c.name == tool and "error" not in c.result]
        grounded_ok &= bool(ok_calls) and _num_in_text(ok_calls[-1].result.get(field), res.answer)

    contains_ok = all(s in res.answer for s in case.get("answer_contains", []))
    steps_ok = not res.hit_step_limit
    checks = {"tools": tools_ok, "args": args_ok, "grounded": grounded_ok, "contains": contains_ok, "steps": steps_ok}
    return {"id": case["id"], "passed": all(checks.values()), "checks": checks,
            "tools_called": called, "steps": res.steps, "answer": res.answer}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prompt-version", default="v2", choices=sorted(SYSTEM_PROMPTS))
    p.add_argument("--live-priceiq", action="store_true", help="call the real PriceIQ server instead of the stub")
    p.add_argument("--out-dir", default="evals/results")
    args = p.parse_args()

    s = Settings.from_env()
    llm = OpenAICompatClient(s.llm_base_url, s.llm_api_key, s.llm_model)
    http = None if args.live_priceiq else fake_priceiq_client()
    registry = build_registry(PricingTools(s.priceiq_url, s.demand_csv, http=http))
    agent = PricingAgent(llm, registry, args.prompt_version, s.max_steps)

    results = []
    for case in json.loads(CASES_PATH.read_text()):
        r = score_case(case, agent.run(case["question"]))
        results.append(r)
        print(f"{'PASS' if r['passed'] else 'FAIL'}  {r['id']:<16} steps={r['steps']} "
              f"failed={[k for k, v in r['checks'].items() if not v]}")

    passed = sum(r["passed"] for r in results)
    print(f"\n{llm.model} | prompt {args.prompt_version} | {passed}/{len(results)} cases passed")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    safe_model = re.sub(r"[^A-Za-z0-9_.-]", "_", llm.model)
    (out / f"{safe_model}__{args.prompt_version}.json").write_text(
        json.dumps({"model": llm.model, "prompt_version": args.prompt_version,
                    "passed": passed, "total": len(results), "cases": results}, indent=2))


if __name__ == "__main__":
    main()
