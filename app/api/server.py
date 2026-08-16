"""FastAPI server exposing the voice-RAG pipeline.

Endpoints:
* ``GET  /``         — browser voice UI (self-contained static page).
* ``GET  /health``  — pipeline + model readiness.
* ``POST /query``   — text query → ``QueryResponse`` (JSON).
* ``POST /v1/voice``— audio upload → STT → ``QueryResponse``.

The pipeline is loaded once at startup (including model warm-up) so the first
live request doesn't pay cold-start. Generation defaults to the extractive
stage; configure ``LLM_PROVIDER``/keys to enable a hosted LLM.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.harness.pipeline import Pipeline
from app.harness.schemas import Transcript
from app.stt.client import FakeSTT, SarvamSTT

logger = logging.getLogger("voice-rag")
settings = get_settings()

APP_VERSION = "0.1.0"

STATIC_DIR = Path(__file__).resolve().parent / "static"


class QueryRequest(BaseModel):
    text: str = Field(min_length=1)
    language: str = "hi"


class _Health(BaseModel):
    status: str
    version: str
    strategy: str
    generation: str
    stt_provider: str


class Server:
    """Holds app state so the FastAPI ``app`` object stays module-level."""

    def __init__(self) -> None:
        self.pipeline: Pipeline | None = None
        self.ready = False
        self.health: _Health | None = None

    def load(self) -> None:
        self.pipeline = Pipeline.from_index(
            lang=settings.data_lang,
            strategy=settings.data_strategy,
        )
        self.pipeline.stt = make_stt()
        self.pipeline.warmup()
        self.ready = True


state = Server()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    try:
        state.load()
        logger.info("pipeline loaded and warmed up")
    except Exception:
        logger.exception("pipeline failed to load — /query will 503")
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Voice-Enabled RAG", version=APP_VERSION, lifespan=_lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health", response_model=_Health)
    async def health() -> _Health:
        if not state.ready or state.pipeline is None:
            raise HTTPException(status_code=503, detail="pipeline not loaded")
        return _Health(
            status="ok",
            version=APP_VERSION,
            strategy=settings.data_strategy,
            generation=state.pipeline.generator.name,
            stt_provider=settings.stt_provider,
        )

    @app.post("/query")
    async def query(req: QueryRequest) -> dict:
        if state.pipeline is None:
            raise HTTPException(status_code=503, detail="pipeline not loaded")
        transcript = Transcript(text=req.text, language=req.language)
        return (await state.pipeline.query_async(transcript)).model_dump()

    @app.post("/v1/voice")
    async def voice(audio: UploadFile = File(...)) -> dict:
        if state.pipeline is None:
            raise HTTPException(status_code=503, detail="pipeline not loaded")
        if state.pipeline.stt is None:
            raise HTTPException(
                status_code=503,
                detail="STT is not configured (set SARVAM_API_KEY or STT_PROVIDER=fake)",
            )
        tmp = Path(settings.index_dir) / "tmp" / audio.filename or "upload.wav"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        data = await audio.read()
        tmp.write_bytes(data)
        try:
            return (await state.pipeline.process_audio(tmp)).model_dump()
        finally:
            tmp.unlink(missing_ok=True)

    return app


def make_stt():
    """Build the STT client selected by config (sarvam default, fake for dev)."""
    if settings.stt_provider == "fake":
        return FakeSTT()
    return SarvamSTT()


app = create_app()
