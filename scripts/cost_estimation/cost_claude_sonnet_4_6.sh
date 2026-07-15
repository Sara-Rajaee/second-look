#!/usr/bin/env bash

# Model: Anthropic Claude Sonnet 4.6 (`claude-sonnet-4-6`)
# Pricing/docs:
# https://platform.claude.com/docs/en/about-claude/models/overview
# https://platform.claude.com/docs/en/build-with-claude/effort
# https://platform.claude.com/docs/en/build-with-claude/extended-thinking
# API cost used here: $3 / 1M input tokens, $15 / 1M output tokens.
# These are planning sensitivities, not claimed percentiles: 1k, 5k, and 20k
# reasoning tokens, each plus a fixed 1k visible-report allowance.

.venv/bin/python scripts/cost_estimation/estimate_cost.py \
  --quick --sample-n 300 --sample-seed 42 --jobs 5 \
  --price-in 3 --price-out 15 \
  --reasoning-budgets 1000 5000 20000 --visible-output-tokens 1000 \
  --allow-partial-languages --model-id claude-sonnet-4-6 \
  --summary-json outputs/cost_claude_sonnet_4_6_reasoning_scenarios.json \
  --summary-markdown outputs/cost_claude_sonnet_4_6_reasoning_scenarios.md
