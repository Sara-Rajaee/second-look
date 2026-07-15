#!/usr/bin/env bash

# Model: OpenAI GPT-5 (`gpt-5`)
# Pricing/docs:
# https://developers.openai.com/api/docs/models/gpt-5
# https://developers.openai.com/api/docs/guides/reasoning
# API cost used here: $1.25 / 1M input tokens, $10 / 1M output tokens.
# The shared 1k/5k/20k reasoning scenarios are project sensitivities; they do
# not claim to map exactly to an API reasoning-effort setting.

.venv/bin/python scripts/cost_estimation/estimate_cost.py \
  --quick --sample-n 300 --sample-seed 42 --jobs 5 \
  --price-in 1.25 --price-out 10 \
  --reasoning-budgets 1000 5000 20000 --visible-output-tokens 1000 \
  --allow-partial-languages --model-id gpt-5 \
  --summary-json outputs/cost_gpt5_reasoning_scenarios.json \
  --summary-markdown outputs/cost_gpt5_reasoning_scenarios.md
