import numpy as np

from app.stt.audio_processing import preprocess_audio, compute_rms, normalize_gain, trim_silence


def test_compute_rms_silence():
    pcm = np.zeros(16000, dtype=np.int16).tobytes()
    assert compute_rms(pcm) == 0.0


def test_compute_rms_signal():
    pcm = (np.ones(16000, dtype=np.int16) * 1000).tobytes()
    rms = compute_rms(pcm)
    assert 900 < rms < 1100


def test_normalize_gain_soft_clip():
    quiet = (np.ones(16000, dtype=np.int16) * 100).tobytes()
    normalized = normalize_gain(quiet, target_rms=1000)
    assert compute_rms(normalized) > 500


def test_normalize_gain_already_loud():
    loud = (np.ones(16000, dtype=np.int16) * 15000).tobytes()
    normalized = normalize_gain(loud, target_rms=1000)
    samples = np.frombuffer(normalized, dtype=np.int16)
    assert np.max(np.abs(samples)) <= 32767


def test_trim_silence_removes_leading():
    silence = np.zeros(8000, dtype=np.int16)
    speech = np.ones(8000, dtype=np.int16) * 5000
    pcm = np.concatenate([silence, speech]).tobytes()
    trimmed = trim_silence(pcm, threshold=200)
    assert len(trimmed) < len(pcm)


def test_preprocess_audio_runs():
    pcm = np.random.randint(-1000, 1000, 16000, dtype=np.int16).tobytes()
    result = preprocess_audio(pcm, sample_rate=16000)
    assert len(result) == len(pcm)
    assert isinstance(result, bytes)
