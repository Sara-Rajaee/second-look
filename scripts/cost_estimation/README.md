# Cost Estimation Language and Locale Policy

## Scope

This policy applies to the six dataset keys exported by
`outputs/cost_claude_sonnet_4_6_high_32kthinking_40kcap.json`:

- `global_mmlu`
- `mmmlu`
- `mmlu_prox`
- `mmlu_prox_lite`
- `polymath`
- `global_piqa_parallel`

The canonical language set comes from the supplied 13-language list:

`ar`, `zh`, `en`, `fr`, `de`, `id`, `it`, `ja`, `ko`, `pt`, `ru`, `es`, `vi`

English is the source language. Consequently, the cost filter contains 12
English-to-target pairs rather than an `en-en` pair.

```yaml
source_language: en
target_languages: [ar, zh, fr, de, id, it, ja, ko, pt, ru, es, vi]
require_all_targets: true
```

## Tentative standard locale choices

ISO 639-1 codes identify languages, but several datasets identify a specific
locale, script, or variety. Until the locale policy is reviewed, use these
tentative defaults:

| ISO | Tentative interpretation | Global-MMLU | MMMLU | MMLU-ProX | ProX-Lite | PolyMath | Global-PIQA-Parallel |
|---|---|---|---|---|---|---|---|
| `en` | English, Latin script (source) | `en` | external `cais/mmlu:all` | `en` | `en` | `en` | embedded `eng_latn` |
| `ar` | Modern Standard Arabic, Arabic script | `ar` | `AR_XY` | `ar` | `ar` | `ar` | `arb_arab` |
| `zh` | Mandarin Chinese, Simplified script | `zh` | `ZH_CN` | `zh` | `zh` | `zh` | `cmn_hans` |
| `fr` | France French, Latin script | `fr` | `FR_FR` | `fr` | `fr` | `fr` | `fra_latn_fran` |
| `de` | Standard German, Latin script | `de` | `DE_DE` | `de` | `de` | `de` | `deu_latn` |
| `id` | Standard Indonesian, Latin script | `id` | `ID_ID` | `id` | `id` | `id` | `ind_latn` |
| `it` | Standard Italian, Latin script | `it` | `IT_IT` | `it` | `it` | `it` | `ita_latn` |
| `ja` | Standard Japanese, Japanese script | `ja` | `JA_JP` | `ja` | `ja` | `ja` | `jpn_jpan` |
| `ko` | Standard Korean, Hangul script | `ko` | `KO_KR` | `ko` | `ko` | `ko` | `kor_hang` |
| `pt` | Brazilian Portuguese, Latin script | `pt` | `PT_BR` | `pt` | `pt` | `pt` | `por_latn_braz` |
| `ru` | Standard Russian, Cyrillic script | `ru` | — | `ru` | `ru` | `ru` | `rus_cyrl` |
| `es` | Spain Spanish, Latin script | `es` | `ES_LA` | `es` | `es` | `es` | `spa_latn_spai` |
| `vi` | Standard Vietnamese, Latin script | `vi` | — | `vi` | `vi` | `vi` | `vie_latn` |

`MMMLU` has no Russian or Vietnamese config. With strict validation enabled,
selecting all six datasets and all 12 targets must fail for MMMLU and report
the two unavailable pairs. It must not silently remove them.

## Findings that still require review

### Arabic

Global-PIQA offers several Arabic varieties. `arb_arab` is the best available
default for Modern Standard Arabic, but `AR_XY` in MMMLU does not communicate a
specific country or variety. The datasets should not be described as
locale-matched without further provenance checks.

### Chinese

`zh`, `ZH_CN`, and `cmn_hans` are mapped tentatively to Simplified Mandarin.
Global-PIQA also offers Traditional Mandarin (`cmn_hant`) and Cantonese
(`yue_hant`); these are deliberately excluded from the generic `zh` filter.

### French

Global-PIQA distinguishes France and Canadian French. `fra_latn_fran` matches
MMMLU's `FR_FR` most closely and is the tentative default.

### Portuguese

Global-PIQA distinguishes Brazil and Portugal. Brazilian Portuguese is the
tentative default because MMMLU only supplies `PT_BR` and Brazil represents the
larger speaker population. Generic `pt` configs in the other datasets do not
prove that their variety is Brazilian, so this is not yet a fully matched
locale comparison.

### Spanish

Global-PIQA offers Mexico, Peru, and Spain variants. Spain Spanish is used as
the tentative generic-language default, but MMMLU only supplies Latin-American
Spanish (`ES_LA`). If cross-dataset locale consistency is more important than a
generic default, switch Global-PIQA to `spa_latn_mexi` and document Mexico as a
proxy for Latin-American Spanish.

## Implemented validation and export

1. ISO codes are the public interface; dataset config names remain internal.
2. Every ISO code must resolve to at most one config for a dataset.
3. Missing or ambiguous resolutions fail before dataset rows are loaded.
4. `--allow-partial-languages` downgrades missing pairs to Markdown
   rows marked `unavailable`.
5. The export records the ISO code, resolved config, script/locale policy, and
   missing pairs so results remain auditable.

The estimator now computes each language pair from that pair's measured input
tokens and item count. It does not divide an aggregate dataset total by the
number of languages. Its three default scenarios are 1k, 5k, and 20k reasoning
tokens plus a fixed 1k visible-report allowance.

The request instruction and tuple are serialized to stable JSON and tokenized
as a whole. This replaces the old synthetic 600-token instruction overhead.
Quick mode uses a fixed-seed random sample rather than the first row(s), and the
sample parameters are included in report metadata.
When Hugging Face omits a config's split-size metadata, quick mode uses uniform
reservoir sampling without replacement. This scans the stream once, retains
only the configured sample size, and preserves original row indices for joins.

Answer keys and other nominally invariant gold fields are included from both
source and target rows. If they differ, the exporter records the observed count
and up to five example IDs instead of aborting. The Markdown report labels these
as potential dataset mistakes requiring review; it does not automatically decide
whether the translation, key, source data, or pairing is at fault. Stable-ID,
duplicate-key, and full-run cardinality checks remain strict alignment guards.

Example Claude report:

```bash
./scripts/cost_estimation/cost_claude_sonnet_4_6.sh
```

The runner writes CSV and JSON summaries, per-dataset language CSV files, and a
Markdown table at `outputs/cost_claude_sonnet_4_6_reasoning_scenarios.md`.
MMMLU's `en–ru` and `en–vi` rows remain visible as `unavailable`.

## Pinned dataset revisions

The implementation pins every load and metadata lookup to these commits,
resolved on 15 July 2026:

| Dataset | Hugging Face revision |
|---|---|
| Global-MMLU | `0e619dbeb34206cd48705a1a0ea7fb21cae09993` |
| MMMLU | `325a01dc3e173cac1578df94120499aaca2e2504` |
| MMMLU English source (`cais/mmlu`) | `c30699e8356da336a370243923dbaf21066bb9fe` |
| MMLU-ProX | `8e6106a6c6ce1c5027e66cc338143cf997b2aa09` |
| MMLU-ProX-Lite | `e82aafb9460529687d3c7e51b401d8dd1dd309dd` |
| PolyMath | `71e0db902e0ece05e208dfd8b1695bd3d95cf130` |
| Global-PIQA-Parallel | `b0b18516a8bc2cb1106bce3dd4db32848ca715ea` |

ProX and ProX-Lite remain separate rows in the Markdown table. Because Lite
overlaps the full dataset, their costs are not automatically summed.
