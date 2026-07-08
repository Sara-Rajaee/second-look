#!/usr/bin/env bash

# Model: DeepSeek reasoner endpoint for the DeepSeek-V3-era family (`deepseek-reasoner`)
# Pricing/docs:
# https://api-docs.deepseek.com/guides/reasoning_model
# https://api-docs.deepseek.com/quick_start/pricing-details-usd
# API cost used here: $0.55 / 1M input tokens (cache miss), $2.19 / 1M output tokens.
# The pricing details page lists MAX COT TOKENS = 32K and MAX OUTPUT TOKENS = 8K.
# We therefore model a 32K thinking budget with a 40K total billed-output upper bound.

.venv/bin/python scripts/cost_estimation/estimate_cost.py --quick --sample-n 1 --jobs 5 --price-in 0.55 --price-out 2.19 --out-upper 40000 --summary-json outputs/cost_deepseek_v3_reasoner_32kthinking_40kcap.json
