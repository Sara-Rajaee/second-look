#!/usr/bin/env bash

# Model: OpenAI GPT-5 (`gpt-5`)
# Pricing/docs:
# https://developers.openai.com/api/docs/models/gpt-5
# https://developers.openai.com/api/docs/guides/reasoning
# API cost used here: $1.25 / 1M input tokens, $10 / 1M output tokens.
# We only model `reasoning.effort=high` for now.
# OpenAI documents configurable reasoning effort, but not a separate hard cap on reasoning tokens;
# the documented hard cap is 128k max output tokens, which includes reasoning and visible output.

.venv/bin/python scripts/cost_estimation/estimate_cost.py --quick --sample-n 1 --jobs 5 --price-in 1.25 --price-out 10 --out-upper 128000 --summary-json outputs/cost_gpt5_high_128k.json
