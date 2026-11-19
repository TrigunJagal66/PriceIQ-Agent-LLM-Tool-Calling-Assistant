"""The agent loop: call the LLM, execute requested tools, feed results back, repeat."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .llm import LLMClient, Message
from .prompts import DEFAULT_PROMPT_VERSION, get_system_prompt
from .tools import ToolRegistry


@dataclass
class ToolCall:
    name: str
    arguments: dict
    result: dict


@dataclass
class AgentResult:
    question: str
    answer: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    steps: int = 0
    hit_step_limit: bool = False
    seconds: float = 0.0


class PricingAgent:
    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
        max_steps: int = 6,
        trace_dir: str | Path | None = None,
    ):
        self.llm = llm
        self.registry = registry
        self.prompt_version = prompt_version
        self.max_steps = max_steps
        self.trace_dir = Path(trace_dir) if trace_dir else None

    def run(self, question: str) -> AgentResult:
        start = time.time()
        messages: list[Message] = [
            {"role": "system", "content": get_system_prompt(self.prompt_version)},
            {"role": "user", "content": question},
        ]
        result = AgentResult(question=question, answer="")
        specs = self.registry.specs()

        for step in range(1, self.max_steps + 1):
            result.steps = step
            reply = self.llm.chat(messages, specs)
            messages.append(reply)
            calls = reply.get("tool_calls") or []
            if not calls:
                result.answer = (reply.get("content") or "").strip()
                break
            for call in calls:
                fn = call["function"]
                raw = self.registry.execute(fn["name"], fn.get("arguments") or "{}")
                parsed_args = _safe_json(fn.get("arguments"))
                result.tool_calls.append(ToolCall(fn["name"], parsed_args, json.loads(raw)))
                messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": raw})
        else:
            result.hit_step_limit = True
            result.answer = "I could not finish within the step limit."

        result.seconds = round(time.time() - start, 2)
        self._write_trace(result, messages)
        return result

    def _write_trace(self, result: AgentResult, messages: list[Message]) -> None:
        if not self.trace_dir:
            return
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "model": self.llm.model,
            "prompt_version": self.prompt_version,
            "question": result.question,
            "answer": result.answer,
            "steps": result.steps,
            "seconds": result.seconds,
            "messages": messages,
        }
        with (self.trace_dir / "traces.jsonl").open("a") as fh:
            fh.write(json.dumps(record, default=str) + "\n")


def _safe_json(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw or "{}")
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}
