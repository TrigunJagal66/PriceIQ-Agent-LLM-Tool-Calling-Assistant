"""LLM clients. The agent only depends on the small `LLMClient` interface."""
from __future__ import annotations

from typing import Any, Protocol

import httpx

Message = dict[str, Any]


class LLMClient(Protocol):
    model: str

    def chat(self, messages: list[Message], tools: list[dict]) -> Message:
        """Return the assistant message (may contain `tool_calls`)."""


class OpenAICompatClient:
    """Minimal client for any OpenAI-compatible /chat/completions endpoint."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60.0):
        if not api_key:
            raise ValueError("LLM_API_KEY is not set (see .env.example).")
        self.model = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._timeout = timeout

    def chat(self, messages: list[Message], tools: list[dict]) -> Message:
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0,
        }
        resp = httpx.post(self._url, json=payload, headers=self._headers, timeout=self._timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]


class ScriptedLLM:
    """Deterministic fake LLM that replays a list of assistant messages.

    Used for unit tests so the agent loop can be verified without network or API keys.
    """

    model = "scripted"

    def __init__(self, script: list[Message]):
        self._script = list(script)
        self.calls: list[list[Message]] = []

    def chat(self, messages: list[Message], tools: list[dict]) -> Message:
        self.calls.append([dict(m) for m in messages])
        if not self._script:
            raise RuntimeError("ScriptedLLM ran out of scripted responses")
        return self._script.pop(0)
