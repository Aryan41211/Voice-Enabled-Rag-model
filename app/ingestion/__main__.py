"""CLI for dataset download/exploration.

Usage:
    python -m app.ingestion download --lang hi --split validation
    python -m app.ingestion explore --lang hi --limit 3
"""

from __future__ import annotations

import argparse
import json

from app.ingestion.dataset import SPLITS, LANGUAGES, load


def cmd_download(args: argparse.Namespace) -> None:
    from app.ingestion.dataset import download

    for split in args.split:
        path = download(args.lang, split)
        print(f"[dataset] {args.lang}/{split} -> {path}")


def cmd_explore(args: argparse.Namespace) -> None:
    ds = load(args.lang, args.split)
    print(f"[dataset] {args.lang}/{args.split}: {len(ds)} rows, "
          f"{ds.num_columns} columns")
    print(f"[dataset] features: {ds.features}")
    for i, ex in enumerate(ds.select(range(min(args.limit, len(ds))))):
        print(f"\n--- row {i} (query_id={ex['query_id']}, "
              f"type={ex['query_type']}, target_lang={ex['target_lang']}) ---")
        print(f"query: {ex['query'][:200]}")
        print(f"Answer: {ex['Answer'][:200]}")
        passages = ex["passages"]
        n = len(passages["Translated_passages"])
        print(f"passages: {n} total, selected="
              f"{sum(passages['is_selected'])}")
        for j, (p, sel) in enumerate(
            zip(passages["Translated_passages"], passages["is_selected"])
        ):
            marker = "[SELECTED]" if sel else ""
            print(f"  {j}: {marker} {p[:150]}")
    if args.json:
        print("\n[json] first example:")
        print(json.dumps(ds[0], ensure_ascii=False, indent=2)[:4000])


def build_parser() -> argparse.ArgumentParser:
    lang_help = f"language code, one of {sorted(LANGUAGES)}"
    sub = argparse.ArgumentParser(prog="python -m app.ingestion")
    sub.add_argument("-l", "--lang", default="hi", help=lang_help)
    subcommands = sub.add_subparsers(dest="command", required=True)

    d = subcommands.add_parser("download",
                               help="download dataset split(s) to data/raw")
    d.add_argument("--split", nargs="+", default=["validation"],
                   choices=SPLITS)
    d.set_defaults(func=cmd_download)

    e = subcommands.add_parser("explore", help="inspect schema and print examples")
    e.add_argument("--split", default="validation", choices=SPLITS)
    e.add_argument("--limit", type=int, default=3)
    e.add_argument("--json", action="store_true",
                   help="also dump first example as JSON")
    e.set_defaults(func=cmd_explore)

    return sub


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
