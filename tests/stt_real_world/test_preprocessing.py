"""Verify audio preprocessing does not hurt clean audio WER."""
import asyncio
import json
from pathlib import Path


def test_preprocessing_preserves_clean_audio():
    """On clean TTS clips, preprocessing should not increase WER."""
    manifest = json.loads(
        (Path(__file__).parent.parent / "stt_ground_truth/manifest.json").read_text(encoding="utf-8")
    )
    from app.stt.client import SarvamSTT
    stt = SarvamSTT()
    clean = [c for c in manifest["clips"] if c["language"] == "hi" and c["category"] == "clear"][:3]
    for clip in clean:
        result = asyncio.run(stt.transcribe(clip["path"]))
        assert result.text.strip().rstrip("?!।") == clip["expected"]
