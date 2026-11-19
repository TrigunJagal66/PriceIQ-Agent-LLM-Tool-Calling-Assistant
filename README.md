# PriceIQ Agent

A tool-calling LLM agent that answers hotel revenue-management questions by calling the
[PriceIQ forecasting API](https://github.com/TrigunJagal66/hotel-demand-pricing) and analysing historical
booking data with Pandas. It is written in plain Python with no agent framework, so every step of
the loop is visible.

> "What price should I set for 2026-06-12 with 150 rooms and a base price of 140, and how does that
> Friday compare with a typical weekday?"

The agent plans, calls `get_price_recommendation` and `get_demand_stats`, reads the results, and writes an
answer that quotes only the numbers the tools returned.

## Architecture

```
question ──► PricingAgent (agent/agent.py)
                │  loop (max N steps)
                ├─► LLM  (agent/llm.py: any OpenAI-compatible chat API, temperature 0)
                │      └─ returns text  → final answer
                │      └─ returns tool_calls ─┐
                │                             ▼
                └─◄ ToolRegistry (agent/tools.py): validate args → run → JSON result / JSON error
                         ├─ get_price_recommendation  → PriceIQ REST API (/api/v1/recommend-price)
                         ├─ compare_price_scenarios   → several PriceIQ calls, picks best revenue
                         ├─ get_historical_demand     → Pandas over data/daily_demand.csv
                         └─ get_demand_stats          → Pandas group-by (day of week / month / holiday)
```

Design choices

- **Errors go back to the model.** Tools never raise. Bad arguments, an unreachable API, or
  a date past the forecast horizon (2026-10-07) come back as `{"error": ...}` so the agent can fix its call or
  explain the limit.
- **Provider-agnostic.** `OpenAICompatClient` works with Groq, Gemini (OpenAI-compatible endpoint), OpenAI or a local
  Ollama server. Change three environment variables.
- **Testable without an LLM.** `ScriptedLLM` replays fixed assistant messages, so the loop, error recovery,
  step limit, tracing and eval scoring are unit tested offline.
- **Every run is traceable.** The CLI appends the full message history to `runs/traces.jsonl`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # fill in LLM_API_KEY, then: export $(grep -v '^#' .env | xargs)

# start PriceIQ (separate repo) so the forecast tools work:
#   uvicorn app:app --port 8000
python -m agent -v "What price should I set for 2026-05-15 with 200 rooms and base price 150?"
```

Tests (no key or network needed): `python -m pytest -q`

## Prompt versions (`agent/prompts.py`)

| Version | What it contains |
|---|---|
| v1 | One-line role and "use the tools". Baseline. |
| v2 | Role, explicit tool policy (which tool for which question), grounding rule (no numbers without a tool result), error-recovery rule, ask-don't-guess rule for missing inputs, fixed answer format. |

## Evaluation (`evals/`)

`evals/cases.json` holds 8 cases: basic price, scenario comparison, history range, day-of-week stats,
holiday effect, a multi-tool question, an out-of-horizon date (error recovery), and a missing-information
question (the agent should ask, not call a tool). Each case is scored on:

`tools` (right tools called, forbidden ones not) · `args` (arguments extracted correctly) ·
`grounded` (key tool values appear in the answer) · `contains` (e.g. the horizon date is mentioned) ·
`steps` (finished within the step limit). A case passes only if all checks pass.

```bash
python -m evals.run_evals --prompt-version v1
python -m evals.run_evals --prompt-version v2
python -m evals.run_evals --prompt-version v2 --live-priceiq   # real PriceIQ instead of the deterministic stub
```

### Experiment log

Record each real run here (results are written to `evals/results/<model>__<prompt>.json`).

| Date | Model | Prompt | Passed | Failed cases and what I changed |
|---|---|---|---|---|
| _fill in after running_ | | v1 | | |
| _fill in after running_ | | v2 | | |

## Limitations

- Forecasts come from PriceIQ, which is trained on a public hotel-bookings dataset and only forecasts to 2026-10-07.
- `compare_price_scenarios` estimates revenue as `recommended_price × min(predicted_demand, capacity)`, a simple proxy.
- The eval stub in `evals/fake_priceiq.py` is deterministic and does not model real demand. Use `--live-priceiq` for end-to-end checks.
- Single-turn only: no conversation memory yet.
