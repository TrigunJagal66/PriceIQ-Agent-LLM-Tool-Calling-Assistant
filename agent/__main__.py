"""CLI:  python -m agent "What price should I set for 2026-05-15 with 200 rooms and base 150?" """
from __future__ import annotations

import argparse

from .agent import PricingAgent
from .config import Settings
from .llm import OpenAICompatClient
from .prompts import DEFAULT_PROMPT_VERSION, SYSTEM_PROMPTS
from .tools import PricingTools, build_registry


def main() -> None:
    p = argparse.ArgumentParser(description="PriceIQ pricing agent")
    p.add_argument("question")
    p.add_argument("--prompt-version", default=DEFAULT_PROMPT_VERSION, choices=sorted(SYSTEM_PROMPTS))
    p.add_argument("--trace-dir", default="runs", help="where to append JSONL traces")
    p.add_argument("-v", "--verbose", action="store_true", help="print each tool call")
    args = p.parse_args()

    s = Settings.from_env()
    llm = OpenAICompatClient(s.llm_base_url, s.llm_api_key, s.llm_model)
    registry = build_registry(PricingTools(s.priceiq_url, s.demand_csv))
    agent = PricingAgent(llm, registry, args.prompt_version, s.max_steps, args.trace_dir)

    result = agent.run(args.question)
    if args.verbose:
        for c in result.tool_calls:
            print(f"[tool] {c.name}({c.arguments}) -> {c.result}")
    print(result.answer)
    print(f"\n({result.steps} steps, {len(result.tool_calls)} tool calls, {result.seconds}s)")


if __name__ == "__main__":
    main()
