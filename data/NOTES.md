# Data Notes — MSMARCO-XI

Single source of truth for all project data. **No other datasets are used.**

## Source

- `ai4bharat/MSMARCO-XI` (HuggingFace) — multilingual MS MARCO translation.
- Repo layout: per-language parquet per split: `train/{prefix}train.parquet`,
  `validation/{prefix}val.parquet`.
- Auto-converted Hub config only exposes `default` (all languages, ~55.6 GB);
  per-language configs from the original loading script are **not** available.
  We therefore load the raw per-language parquet files directly (see
  `app/ingestion/dataset.py`), which lets us cache only the languages we use.

## Cached (local, `data/raw/`)

| lang | split | file | rows |
|------|-------|------|------|
| hi   | validation | `validation/hinval.parquet` | 97,941 |

To add more: `python -m app.ingestion -l <lang> download --split train validation`

## Schema (per example)

| field | type | notes |
|-------|------|-------|
| `query_id` | int | unique query id |
| `query` | str | translated (Hindi) query |
| `Eng_Query` | str | original English query |
| `Answer` | str | translated gold answer |
| `Eng_Answer` | str | original English answer |
| `query_type` | str | DESCRIPTION / ENTITY / NUMERIC / LOCATION / PERSON |
| `source_lang` | str | `eng_Latn` |
| `target_lang` | str | e.g. `hin_Deva` |
| `meta` | dict | translation model sampling metadata |
| `passages.Translated_passages` | list[str] | candidate passages (translated) |
| `passages.English_passages` | list[str] | same passages (English) |
| `passages.is_selected` | list[int] | 1 = passage answers the query (gold label) |

## Observed stats (hi/validation, 97,941 rows)

- Passages per query: mean 10.0 (min 1, max 27) — corpus ≈ 1.0M passages.
- Selected passages per query: mean 0.59; **53,895 rows (55%) have ≥1 selected
  passage** → usable as the gold-labeled retrieval eval set.
- Translated passage length: mean 324 chars, median 292 — short, self-contained
  snippets → chunking should *combine* passages, not aggressively split
  (see CHUNKING_STRATEGY.md).
- Query length: mean 43 chars.
- `query_type` distribution: DESCRIPTION 52,912 · NUMERIC 24,741 · ENTITY
  8,427 · PERSON 6,206 · LOCATION 5,655.

## Usage constraints

- Retrieval corpus = `passages.Translated_passages` (indexed once, offline).
- Eval = `query` → retrieve → must return the `is_selected == 1` passage.
- Answer grounding target = `Answer` / selected passage text.
- Only rows with ≥1 selected passage are used for accuracy eval; all rows can
  be used for latency benchmarking.
