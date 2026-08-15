"""Container entrypoint: ensure the index exists (first boot), then serve."""

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    lang = os.environ.get("DATA_LANG", "hi")
    strategy = os.environ.get("DATA_STRATEGY", "metadata")
    index_dir = Path(os.environ.get("INDEX_DIR", "./data/index"))
    marker = index_dir / lang / strategy / "dense.faiss"

    if not marker.exists():
        print("[entrypoint] index missing — building (first boot only)...", flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "app.ingestion.build_index",
                "--lang",
                lang,
                "--strategies",
                strategy,
            ],
            check=True,
        )

    from uvicorn import run

    port = int(os.environ.get("PORT", "8000"))
    run("app.api.server:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
