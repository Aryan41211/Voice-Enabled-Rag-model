import asyncio
from app.harness.schemas import Transcript, STTError


class WorkingSTT:
    async def transcribe(self, audio_path):
        return Transcript(text="fallback result", language="hi-IN", confidence=0.9)


class FailingSTT:
    async def transcribe(self, audio_path):
        raise STTError("provider down", retryable=True)


def test_fallback_uses_primary_when_working():
    from app.stt.fallback import FallbackSTTChain
    chain = FallbackSTTChain(providers=[WorkingSTT(), FailingSTT()])
    result = asyncio.run(chain.transcribe("test.wav"))
    assert result.text == "fallback result"


def test_fallback_uses_secondary_on_primary_failure():
    from app.stt.fallback import FallbackSTTChain
    chain = FallbackSTTChain(providers=[FailingSTT(), WorkingSTT()])
    result = asyncio.run(chain.transcribe("test.wav"))
    assert result.text == "fallback result"


def test_fallback_raises_when_all_fail():
    from app.stt.fallback import FallbackSTTChain
    chain = FallbackSTTChain(providers=[FailingSTT(), FailingSTT()])
    try:
        asyncio.run(chain.transcribe("test.wav"))
        assert False, "should have raised"
    except STTError:
        pass


def test_fallback_skips_broken_circuit():
    from app.stt.fallback import FallbackSTTChain
    from app.harness.pipeline import CircuitBreaker
    cb = CircuitBreaker(threshold=1)
    cb.record_failure()
    primary = FailingSTT()
    chain = FallbackSTTChain(providers=[primary, WorkingSTT()], circuit_breakers=[cb, None])
    result = asyncio.run(chain.transcribe("test.wav"))
    assert result.text == "fallback result"
