"""Tools the agent can call, plus a registry that validates arguments and never raises.

Errors are returned to the model as JSON so it can recover (retry with fixed arguments,
pick another tool, or explain the limitation) instead of crashing the loop.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

import httpx
import pandas as pd


class ToolError(Exception):
    """A problem the model should be told about (bad args, upstream failure)."""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema
    func: Callable[..., Any]

    def spec(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _parse_date(value: str, field: str) -> date:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        raise ToolError(f"{field} must be an ISO date (YYYY-MM-DD), got {value!r}")


class PricingTools:
    """Tool implementations backed by the PriceIQ API and the historical demand CSV."""

    def __init__(self, priceiq_url: str, demand_csv: str, http: httpx.Client | None = None):
        self._base = priceiq_url.rstrip("/")
        self._http = http or httpx.Client(timeout=30.0)
        self._csv = demand_csv
        self._df: pd.DataFrame | None = None

    # ---- data access -------------------------------------------------
    def _demand(self) -> pd.DataFrame:
        if self._df is None:
            df = pd.read_csv(self._csv, parse_dates=["arrival_date"])
            self._df = df.sort_values("arrival_date").reset_index(drop=True)
        return self._df

    # ---- tools -------------------------------------------------------
    def get_price_recommendation(
        self, target_date: str, base_price: float, capacity: int, is_holiday: bool = False
    ) -> dict:
        d = _parse_date(target_date, "target_date")
        if base_price <= 0 or capacity <= 0:
            raise ToolError("base_price and capacity must be positive")
        body = {
            "target_date": d.isoformat(),
            "base_price": base_price,
            "capacity": capacity,
            "is_holiday": is_holiday,
        }
        try:
            resp = self._http.post(f"{self._base}/api/v1/recommend-price", json=body)
        except httpx.HTTPError as exc:
            raise ToolError(f"PriceIQ API unreachable: {exc}")
        if resp.status_code != 200:
            detail = resp.json().get("detail", resp.text) if resp.content else resp.text
            raise ToolError(f"PriceIQ API error {resp.status_code}: {detail}")
        return resp.json()

    def compare_price_scenarios(
        self, target_date: str, capacity: int, base_prices: list[float]
    ) -> dict:
        if not base_prices or len(base_prices) > 5:
            raise ToolError("base_prices must contain between 1 and 5 values")
        scenarios = []
        for bp in base_prices:
            rec = self.get_price_recommendation(target_date, bp, capacity)
            scenarios.append(
                {
                    "base_price": bp,
                    "recommended_price": rec["recommended_price"],
                    "predicted_demand": rec["predicted_demand"],
                    "occupancy_ratio": rec["occupancy_ratio"],
                    "estimated_revenue": round(rec["recommended_price"] * min(rec["predicted_demand"], capacity), 2),
                }
            )
        best = max(scenarios, key=lambda s: s["estimated_revenue"])
        return {"target_date": target_date, "scenarios": scenarios, "best_base_price": best["base_price"]}

    def get_historical_demand(self, start_date: str, end_date: str) -> dict:
        start, end = _parse_date(start_date, "start_date"), _parse_date(end_date, "end_date")
        if start > end:
            raise ToolError("start_date must not be after end_date")
        df = self._demand()
        window = df[(df["arrival_date"].dt.date >= start) & (df["arrival_date"].dt.date <= end)]
        if window.empty:
            lo, hi = df["arrival_date"].min().date(), df["arrival_date"].max().date()
            raise ToolError(f"No data in that range. Available data covers {lo} to {hi}.")
        return {
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "days": int(len(window)),
            "total_bookings": int(window["bookings"].sum()),
            "mean_daily_bookings": round(float(window["bookings"].mean()), 1),
            "peak_day": window.loc[window["bookings"].idxmax(), "arrival_date"].date().isoformat(),
            "peak_bookings": int(window["bookings"].max()),
            "lowest_day": window.loc[window["bookings"].idxmin(), "arrival_date"].date().isoformat(),
            "lowest_bookings": int(window["bookings"].min()),
        }

    _DOW = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    def get_demand_stats(self, group_by: str) -> dict:
        df = self._demand()
        if group_by == "day_of_week":
            g = df.groupby("day_of_week")["bookings"].mean().round(1)
            rows = {self._DOW[int(k)]: float(v) for k, v in g.items()}
        elif group_by == "month":
            g = df.groupby("month")["bookings"].mean().round(1)
            rows = {int(k): float(v) for k, v in g.items()}
        elif group_by == "holiday":
            g = df.groupby("is_holiday")["bookings"].mean().round(1)
            rows = {("holiday" if k else "non_holiday"): float(v) for k, v in g.items()}
        else:
            raise ToolError("group_by must be one of: day_of_week, month, holiday")
        return {"group_by": group_by, "mean_daily_bookings": rows}


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self._tools = {t.name: t for t in tools}

    def specs(self) -> list[dict]:
        return [t.spec() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def execute(self, name: str, raw_arguments: str | dict) -> str:
        """Run a tool and return a JSON string. Never raises."""
        tool = self._tools.get(name)
        if tool is None:
            return json.dumps({"error": f"Unknown tool {name!r}. Available: {self.names()}"})
        try:
            args = json.loads(raw_arguments) if isinstance(raw_arguments, str) else dict(raw_arguments)
        except json.JSONDecodeError:
            return json.dumps({"error": "Tool arguments were not valid JSON"})
        required = tool.parameters.get("required", [])
        missing = [r for r in required if r not in args]
        if missing:
            return json.dumps({"error": f"Missing required arguments: {missing}"})
        unknown = [a for a in args if a not in tool.parameters["properties"]]
        if unknown:
            return json.dumps({"error": f"Unexpected arguments: {unknown}"})
        try:
            return json.dumps(tool.func(**args), default=str)
        except ToolError as exc:
            return json.dumps({"error": str(exc)})
        except TypeError as exc:
            return json.dumps({"error": f"Bad argument types: {exc}"})


def build_registry(pt: PricingTools) -> ToolRegistry:
    date_prop = {"type": "string", "description": "ISO date, YYYY-MM-DD"}
    return ToolRegistry(
        [
            Tool(
                "get_price_recommendation",
                "Forecast hotel demand for one date and recommend a room price. Returns predicted "
                "demand, occupancy ratio, recommended price, confidence and an explanation. "
                "The model can only forecast up to 2026-10-07.",
                {
                    "type": "object",
                    "properties": {
                        "target_date": date_prop,
                        "base_price": {"type": "number", "description": "Normal room price"},
                        "capacity": {"type": "integer", "description": "Rooms available"},
                        "is_holiday": {"type": "boolean", "description": "Whether the date is a holiday"},
                    },
                    "required": ["target_date", "base_price", "capacity"],
                },
                pt.get_price_recommendation,
            ),
            Tool(
                "compare_price_scenarios",
                "Compare several candidate base prices for one date and report estimated revenue "
                "for each (recommended price x min(predicted demand, capacity)).",
                {
                    "type": "object",
                    "properties": {
                        "target_date": date_prop,
                        "capacity": {"type": "integer"},
                        "base_prices": {"type": "array", "items": {"type": "number"}, "description": "1-5 base prices"},
                    },
                    "required": ["target_date", "capacity", "base_prices"],
                },
                pt.compare_price_scenarios,
            ),
            Tool(
                "get_historical_demand",
                "Summarise actual daily bookings between two dates: total, mean, peak and lowest day.",
                {
                    "type": "object",
                    "properties": {"start_date": date_prop, "end_date": date_prop},
                    "required": ["start_date", "end_date"],
                },
                pt.get_historical_demand,
            ),
            Tool(
                "get_demand_stats",
                "Average historical daily bookings grouped by day_of_week, month or holiday.",
                {
                    "type": "object",
                    "properties": {"group_by": {"type": "string", "enum": ["day_of_week", "month", "holiday"]}},
                    "required": ["group_by"],
                },
                pt.get_demand_stats,
            ),
        ]
    )
