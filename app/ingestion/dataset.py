"""MSMARCO-XI dataset loading and caching.

Loads the ai4bharat/MSMARCO-XI dataset per language from the raw parquet
files on the Hub, caches them under ``data/raw``, and exposes typed accessors.
This is the only dataset used by the project.
"""

from __future__ import annotations

from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

REPO_ID = "ai4bharat/MSMARCO-XI"

LANGUAGES = {
    "as": "asm",
    "bn": "ben",
    "gu": "guj",
    "hi": "hin",
    "kn": "kan",
    "ml": "mal",
    "mr": "mar",
    "ne": "nep",
    "or": "ori",
    "pa": "pan",
    "sa": "san",
    "ta": "tam",
    "te": "tel",
    "ur": "urd",
}

# Note: MSMARCO-XI ships Telugu (te) only in the validation split.
SPLITS = ("train", "validation")

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def _lang_prefix(lang: str) -> str:
    if lang not in LANGUAGES:
        raise ValueError(
            f"Unsupported language '{lang}'. Choose from: {sorted(LANGUAGES)}"
        )
    return LANGUAGES[lang]


def parquet_file(lang: str, split: str) -> str:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    suffix = "train" if split == "train" else "val"
    return f"{split}/{_lang_prefix(lang)}{suffix}.parquet"


def download(lang: str, split: str) -> Path:
    """Download a language split's parquet file to ``data/raw``.

    Returns the local path. Uses HF's resumable download; already-present
    files are left untouched.
    """
    repo_file = parquet_file(lang, split)
    return Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=repo_file,
            repo_type="dataset",
            local_dir=RAW_DIR,
        )
    )


def load(lang: str, split: str):
    """Load a language split from local cache, downloading it if needed.

    Returns a HuggingFace ``Dataset`` (not streamed) with the full schema.
    """
    local_file = download(lang, split)
    ds = load_dataset("parquet", data_files=str(local_file), split="train")
    return ds
