"""Create/push the Hugging Face Space for the live demo.

Prereq: authenticate once with ``huggingface-cli login`` (paste a Write-scope
token from https://huggingface.co/settings/tokens), then:

    python scripts/deploy_space.py                # default: voice-rag-demo
    python scripts/deploy_space.py --name demo2   # different name
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import HfApi


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="voice-rag-demo")
    parser.add_argument("--folder", default="space")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    api = HfApi()
    who = api.whoami()
    namespace = who.get("name")
    if not namespace:
        print("[deploy] ERROR: could not determine HF namespace", file=sys.stderr)
        return 1
    repo_id = f"{namespace}/{args.name}"
    print(f"[deploy] namespace: {namespace}")

    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        private=args.private,
        exist_ok=True,
    )
    print(f"[deploy] space ready: {repo_id}")

    folder = Path(args.folder)
    if not (folder / "Dockerfile").exists():
        print(f"[deploy] ERROR: {folder}/Dockerfile not found", file=sys.stderr)
        return 1

    api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="space",
        commit_message="deploy voice-rag demo",
    )
    print(f"[deploy] pushed; live in a few minutes:")
    print(f"[deploy]   https://huggingface.co/spaces/{repo_id}")
    print(f"[deploy]   https://{namespace}-{args.name}.hf.space")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
