#!/usr/bin/env bash

# Model: Anthropic Claude Sonnet 4.6 (`claude-sonnet-4-6`)
# Pricing/docs:
# https://platform.claude.com/docs/en/about-claude/models/overview
# https://platform.claude.com/docs/en/build-with-claude/effort
# https://platform.claude.com/docs/en/build-with-claude/extended-thinking
# API cost used here: $3 / 1M input tokens, $15 / 1M output tokens.
# We only model `effort=high` for now.
# Sonnet 4.6 still accepts deprecated manual thinking budgets, so we choose a concrete high-thinking setup:
# `thinking: {type: "enabled", budget_tokens: 32000}` with `max_tokens` capped at 40000 total output tokens.

.venv/bin/python scripts/cost_estimation/estimate_cost.py --quick --sample-n 1 --jobs 5 --price-in 3 --price-out 15 --out-upper 40000 --summary-json outputs/cost_claude_sonnet_4_6_high_32kthinking_40kcap.json
