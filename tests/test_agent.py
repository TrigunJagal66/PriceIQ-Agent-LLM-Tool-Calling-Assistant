import json

from agent.agent import PricingAgent
from agent.llm import ScriptedLLM
from agent.tools import PricingTools, build_registry
from evals.fake_priceiq import fake_priceiq_client
from evals.run_evals import score_case


def registry():
    return build_registry(PricingTools("http://priceiq", "data/daily_demand.csv", http=fake_priceiq_client()))


def tool_msg(call_id, name, args):
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": call_id, "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)}}]}


def test_agent_calls_tool_then_answers():
    llm = ScriptedLLM([
        tool_msg("1", "get_price_recommendation", {"target_date": "2026-05-15", "base_price": 150, "capacity": 200}),
        {"role": "assistant", "content": "Set the price to 150.0."},
    ])
    res = PricingAgent(llm, registry()).run("price?")
    assert res.answer == "Set the price to 150.0." and res.steps == 2
    assert res.tool_calls[0].result["predicted_demand"] == 145
    # the tool result was fed back to the model on the second call
    assert llm.calls[1][-1]["role"] == "tool"


def test_agent_recovers_from_tool_error():
    llm = ScriptedLLM([
        tool_msg("1", "get_price_recommendation", {"target_date": "2027-01-10", "base_price": 150, "capacity": 200}),
        {"role": "assistant", "content": "Forecasts only go up to 2026-10-07."},
    ])
    res = PricingAgent(llm, registry()).run("price for 2027?")
    assert "error" in res.tool_calls[0].result and "2026-10-07" in res.answer


def test_agent_stops_at_step_limit():
    loop = tool_msg("1", "get_demand_stats", {"group_by": "month"})
    res = PricingAgent(ScriptedLLM([loop] * 3), registry(), max_steps=3).run("loop")
    assert res.hit_step_limit and res.steps == 3


def test_trace_is_written(tmp_path):
    llm = ScriptedLLM([{"role": "assistant", "content": "hi"}])
    PricingAgent(llm, registry(), trace_dir=tmp_path).run("hello")
    rec = json.loads((tmp_path / "traces.jsonl").read_text())
    assert rec["prompt_version"] == "v2" and rec["answer"] == "hi"


def test_eval_scoring_pass_and_fail():
    case = {"id": "x", "expected_tools": ["get_price_recommendation"],
            "expected_args": {"get_price_recommendation": {"target_date": "2026-05-15", "base_price": 150, "capacity": 200}},
            "grounded_fields": [["get_price_recommendation", "recommended_price"]]}
    good = PricingAgent(ScriptedLLM([
        tool_msg("1", "get_price_recommendation", {"target_date": "2026-05-15", "base_price": 150, "capacity": 200}),
        {"role": "assistant", "content": "Charge 150.0 (demand 145)."}]), registry()).run("q")
    assert score_case(case, good)["passed"]
    hallucinated = PricingAgent(ScriptedLLM([
        tool_msg("1", "get_price_recommendation", {"target_date": "2026-05-15", "base_price": 150, "capacity": 200}),
        {"role": "assistant", "content": "Charge 999."}]), registry()).run("q")
    assert not score_case(case, hallucinated)["checks"]["grounded"]
    no_tool = PricingAgent(ScriptedLLM([{"role": "assistant", "content": "150"}]), registry()).run("q")
    assert not score_case(case, no_tool)["checks"]["tools"]
