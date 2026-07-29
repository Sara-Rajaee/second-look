import argparse
import sys
from pathlib import Path
import unittest


SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "cost_estimation"
sys.path.insert(0, str(SCRIPT_DIR))

import estimate_cost as ec


class FakeEncoding:
    def encode(self, text, disallowed_special=()):
        return list(text)


def config(**overrides):
    values = {
        "price_in": 3.0,
        "price_out": 15.0,
        "reasoning_budgets": (1000, 5000, 20000),
        "visible_output_tokens": 1000,
        "sample_n": 3,
        "sample_seed": 42,
        "instruction": "Check translation.",
        "model_id": "test-model",
        "tokenizer_name": "fake",
        "enc": FakeEncoding(),
    }
    values.update(overrides)
    return ec.RunConfig(**values)


class EstimateCostTests(unittest.TestCase):
    def test_all_dataset_loads_have_immutable_revisions(self):
        for spec in ec.SPECS.values():
            self.assertRegex(spec.revision, r"^[0-9a-f]{40}$")
            if spec.external_english:
                self.assertRegex(
                    spec.external_english["revision"], r"^[0-9a-f]{40}$"
                )

    def test_mmmlu_external_english_is_normalized(self):
        row = {
            "question": "Q",
            "choices": ["a", "b", "c", "d"],
            "answer": 2,
            "subject": "math",
        }
        normalized = ec.normalize_mmmlu_english_row(row)
        self.assertEqual(normalized["Question"], "Q")
        self.assertEqual(normalized["C"], "c")
        self.assertEqual(normalized["Answer"], "C")

    def test_mmmlu_reports_missing_ru_and_vi(self):
        spec = ec.SPECS["mmmlu"]
        resolved, missing = ec.resolve_language_configs(
            spec, list(ec.TARGET_LANGUAGE_SET), spec.configs
        )
        self.assertEqual(missing, ["ru", "vi"])
        self.assertEqual(dict(resolved)["pt"], "PT_BR")

    def test_sampling_is_repeatable_and_changes_with_seed(self):
        first = ec.deterministic_sample_positions(100, 10, "42:test")
        self.assertEqual(first, ec.deterministic_sample_positions(100, 10, "42:test"))
        self.assertNotEqual(first, ec.deterministic_sample_positions(100, 10, "43:test"))

    def test_reservoir_sampling_handles_unknown_stream_length(self):
        rows = ({"value": i} for i in range(50))
        sampled, total = ec.deterministic_reservoir_sample(rows, 5, "42:mmmlu")
        repeated, repeated_total = ec.deterministic_reservoir_sample(
            ({"value": i} for i in range(50)), 5, "42:mmmlu"
        )
        self.assertEqual(total, 50)
        self.assertEqual(repeated_total, 50)
        self.assertEqual(sampled, repeated)
        self.assertEqual(len(sampled), 5)
        self.assertEqual([index for index, _ in sampled], sorted(index for index, _ in sampled))

    def test_serialization_counts_real_template_and_rejects_schema_drift(self):
        spec = ec.SPECS["global_mmlu"]
        src = {name: f"src-{name}" for name in spec.prompt_fields}
        tgt = {name: f"tgt-{name}" for name in spec.prompt_fields}
        src["answer"] = "A"
        tgt["answer"] = "A"
        body = ec.serialized_input(src, tgt, spec, config(), "fr")
        self.assertIn('"target_language":"fr"', body)
        self.assertEqual(ec.tuple_tokens(src, tgt, spec, config(), "fr"), len(body))
        del tgt["question"]
        with self.assertRaises(KeyError):
            ec.serialized_input(src, tgt, spec, config(), "fr")

    def test_invariant_answer_mismatch_is_included_and_flagged(self):
        spec = ec.SPECS["global_mmlu"]
        src = {name: f"src-{name}" for name in spec.prompt_fields}
        tgt = {name: f"tgt-{name}" for name in spec.prompt_fields}
        src["answer"], tgt["answer"] = "A", "B"
        body = ec.serialized_input(src, tgt, spec, config(), "de")
        self.assertIn('"answer":{"source":"A","target":"B"}', body)
        self.assertTrue(ec.has_invariant_gold_mismatch(src, tgt, spec))

    def test_pair_cost_uses_each_language_input_and_item_count(self):
        rows = [
            {"iso_language": "fr", "pairs": 2, "in_tokens": 100},
            {"iso_language": "de", "pairs": 1, "in_tokens": 1000},
        ]
        ec.add_pair_costs(rows, config(reasoning_budgets=(1000,)))
        self.assertEqual(rows[0]["cost_reasoning_1000_usd"], 0.06)
        self.assertEqual(rows[1]["cost_reasoning_1000_usd"], 0.03)

    def test_markdown_keeps_unavailable_pairs(self):
        args = argparse.Namespace(
            model_id="test-model",
            tokenizer="fake",
            quick=True,
            languages=["fr", "ru"],
        )
        results = [{
            "dataset": "mmmlu",
            "missing_isos": ["ru"],
            "per_lang": [{
                "iso_language": "fr",
                "dataset_config": "FR_FR",
                "pairs": 2,
                "gold_mismatches_observed": 1,
                "rows_observed": 2,
                "gold_mismatch_examples": ["('test', 'example/1')"],
                "cost_reasoning_1000_usd": 0.06,
                "cost_reasoning_5000_usd": 0.18,
                "cost_reasoning_20000_usd": 0.63,
            }],
        }]
        rendered = ec.render_markdown(results, config(), args)
        self.assertIn("| mmmlu | en–ru | — | — | — | — | — | — | unavailable |", rendered)
        self.assertIn("| mmmlu | en–fr | `FR_FR` | 2 |", rendered)
        self.assertIn("| 1/2 | complete |", rendered)
        self.assertIn("## Potential dataset mistakes", rendered)
        self.assertIn("example/1", rendered)


if __name__ == "__main__":
    unittest.main()
