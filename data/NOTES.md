# Data Notes

## Current Index

| Field | Value |
|-------|-------|
| Lang | `hi` (Hindi) |
| Strategy | `metadata` (passage-level metadata indexing) |
| Chunks | 15,008 |
| Queries | 1,500 |
| Embedding model | `intfloat/multilingual-e5-small` |
| Dense index | `IndexFlatIP` (FAISS inner-product) |
| Sparse index | `BM25Okapi` (rank_bm25) |
| Split | `validation` (MSMARCO-XI `hi`) |
| File | `data/raw/validation/hinval.parquet` (462 MB cached) |

## Latency Benchmarks (2026-08-17)

Measured via `scripts/e2e_voice.py` on 3 eval_gold queries:
- **Retrieval P50**: ~34 ms (hybrid dense+sparse)
- **Generation**: <1 ms (sentence-level extractive, pure Python)
- **STT (WS path)**: ~2-3 s (Sarvam realtime via sarvam.ws)
- **End-to-end total**: ~3-3.5 s (dominated by STT)
- Pipeline loads in ~13 s (first call: SentenceTransformer cold load + FAISS mmap)

## Corpus Expansion (not done — too large for in-session work)

The train split (`hintrain.parquet`) is not cached and is very large (MSMARCO-XI
Hindi train is hundreds of MB of parquet, with millions of passages). Steps to
expand the index offline:

1. **Download train split**: `python -c "from datasets import load_dataset; ds=load_dataset('parquet',data_files='data/raw/validation/hinval.parquet',split='train'); ds.to_parquet('data/raw/train/hintrain.parquet')"` — or download from HuggingFace `kharvid/msmarco-xi-hi` directly.

2. **Ingest**: `python -m app.ingestion.loader --corpus data/raw/train/hintrain.parquet --lang hi --strategy metadata`

3. **Embed**: `python -m app.ingestion.embed --corpus data/index/hi`

4. **Rebuild sparse index**: Built automatically by the embed step.

**Warning**: CPU embedding of millions of passages (multilingual-e5-small) will take
hours. Consider using a GPU runner or pre-computed embeddings if available.

## Sarvam Batch API Notes

Verified live (2026-08-17). Useful for audio >30 s (sync endpoints cap at 30 s):

- Init: `POST /speech-to-text/job/v1` (JSON `job_parameters`) -> 202, `job_id`
- Upload: `POST /speech-to-text/job/v1/upload-files` -> presigned blob URL
- Start: `POST /speech-to-text/job/v1/{job_id}/start`
- Poll: `GET /speech-to-text/job/v1/{job_id}/status` -> `Completed`/`Failed`
- Download: `POST /speech-to-text/job/v1/download-files` -> `GET` each URL -> JSON `{transcript: ...}`
- Latency: ~5-8 s for a 48 s WAV (including polling overhead)

OpenAPI spec saved locally: `C:\Users\aryan\AppData\Local\Temp\opencode\sarvam_job_api.yml`
