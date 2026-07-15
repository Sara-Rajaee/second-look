#!/usr/bin/env bash

# Model: current model behind the legacy `deepseek-reasoner` alias (V4 Flash).
# Pricing/docs:
# https://api-docs.deepseek.com/guides/reasoning_model
# https://api-docs.deepseek.com/quick_start/pricing-details-usd
# Current cache-miss pricing used here: $0.14 / 1M input, $0.28 / 1M output.
# Recheck the alias and price before rerunning because the alias is being retired.

.venv/bin/python scripts/cost_estimation/estimate_cost.py \
  --quick --sample-n 300 --sample-seed 42 --jobs 5 \
  --price-in 0.14 --price-out 0.28 \
  --reasoning-budgets 1000 5000 20000 --visible-output-tokens 1000 \
  --allow-partial-languages --model-id deepseek-v4-flash-via-reasoner-alias \
  --summary-json outputs/cost_deepseek_reasoning_scenarios.json \
  --summary-markdown outputs/cost_deepseek_reasoning_scenarios.md
