from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: str = ""
    llm_model: str = "llama-3.3-70b-versatile"
    priceiq_url: str = "http://127.0.0.1:8000"
    demand_csv: str = "data/daily_demand.csv"
    max_steps: int = 6

    @classmethod
    def from_env(cls) -> "Settings":
        d = cls()
        return cls(
            llm_base_url=os.getenv("LLM_BASE_URL", d.llm_base_url),
            llm_api_key=os.getenv("LLM_API_KEY", d.llm_api_key),
            llm_model=os.getenv("LLM_MODEL", d.llm_model),
            priceiq_url=os.getenv("PRICEIQ_URL", d.priceiq_url),
            demand_csv=os.getenv("DEMAND_CSV", d.demand_csv),
            max_steps=int(os.getenv("MAX_STEPS", d.max_steps)),
        )
