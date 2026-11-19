import json

import pytest

from agent.tools import PricingTools, build_registry
from evals.fake_priceiq import fake_priceiq_client

CSV = "data/daily_demand.csv"


@pytest.fixture
def registry():
    return build_registry(PricingTools("http://priceiq", CSV, http=fake_priceiq_client()))


def run(reg, name, args):
    return json.loads(reg.execute(name, json.dumps(args)))


def test_price_recommendation_ok(registry):
    out = run(registry, "get_price_recommendation",
              {"target_date": "2026-05-15", "base_price": 150, "capacity": 200})
    assert out["recommended_price"] > 0 and out["predicted_demand"] == 145


def test_forecast_horizon_error_is_returned_not_raised(registry):
    out = run(registry, "get_price_recommendation",
              {"target_date": "2027-01-10", "base_price": 150, "capacity": 200})
    assert "2026-10-07" in out["error"]


def test_unknown_tool_and_bad_args(registry):
    assert "Unknown tool" in run(registry, "nope", {})["error"]
    assert "Missing required" in run(registry, "get_price_recommendation", {"target_date": "2026-05-15"})["error"]
    assert "Unexpected arguments" in run(registry, "get_demand_stats", {"group_by": "month", "x": 1})["error"]
    assert "ISO date" in run(registry, "get_historical_demand",
                             {"start_date": "tomorrow", "end_date": "2025-01-02"})["error"]
    assert "not valid JSON" in json.loads(registry.execute("get_demand_stats", "{oops"))["error"]


def test_historical_demand_matches_pandas(registry):
    import pandas as pd
    df = pd.read_csv(CSV, parse_dates=["arrival_date"])
    w = df[(df.arrival_date >= "2025-07-01") & (df.arrival_date <= "2025-07-31")]
    out = run(registry, "get_historical_demand", {"start_date": "2025-07-01", "end_date": "2025-07-31"})
    assert out["total_bookings"] == int(w.bookings.sum()) and out["days"] == len(w)


def test_historical_demand_out_of_range(registry):
    out = run(registry, "get_historical_demand", {"start_date": "2010-01-01", "end_date": "2010-01-31"})
    assert "Available data covers" in out["error"]


def test_demand_stats_groups(registry):
    dow = run(registry, "get_demand_stats", {"group_by": "day_of_week"})["mean_daily_bookings"]
    assert set(dow) == {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
    assert "error" in run(registry, "get_demand_stats", {"group_by": "weather"})


def test_compare_scenarios_picks_best_revenue(registry):
    out = run(registry, "compare_price_scenarios",
              {"target_date": "2026-06-12", "capacity": 180, "base_prices": [120, 150, 180]})
    best = max(out["scenarios"], key=lambda s: s["estimated_revenue"])
    assert out["best_base_price"] == best["base_price"] == 180
