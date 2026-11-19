"""Deterministic stand-in for the PriceIQ API so tests/evals run offline and reproducibly."""
from __future__ import annotations

import json
from datetime import date

import httpx

MAX_DATE = date(2026, 10, 7)


def _handler(request: httpx.Request) -> httpx.Response:
    if not request.url.path.endswith("/recommend-price"):
        return httpx.Response(404, json={"detail": "not found"})
    body = json.loads(request.content)
    d = date.fromisoformat(body["target_date"])
    if d > MAX_DATE:
        return httpx.Response(400, json={"detail": f"Forecast horizon exceeded. Maximum allowed date is {MAX_DATE}."})
    demand = 100 + (45 if d.weekday() >= 4 else 0) + (30 if body.get("is_holiday") else 0)
    occ = round(demand / body["capacity"], 2)
    base = body["base_price"]
    price = round(base * (1.3 if occ > 0.9 else 0.9 if occ < 0.5 else 1.0), 2)
    return httpx.Response(
        200,
        json={
            "target_date": body["target_date"],
            "predicted_demand": demand,
            "occupancy_ratio": occ,
            "recommended_price": price,
            "confidence": "medium",
            "explanation": [f"Predicted demand: {demand}", f"Occupancy ratio {occ}"],
        },
    )


def fake_priceiq_client() -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(_handler))
