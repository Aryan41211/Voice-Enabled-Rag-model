"""Client-side audio preprocessing: gain normalization, noise gate, silence trimming."""
from __future__ import annotations

import numpy as np


def compute_rms(pcm_bytes: bytes) -> float:
    if len(pcm_bytes) < 2:
        return 0.0
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


def normalize_gain(pcm_bytes: bytes, target_rms: float = 1000.0) -> bytes:
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float64)
    current_rms = float(np.sqrt(np.mean(samples ** 2)))
    if current_rms < 1.0:
        return pcm_bytes
    gain = target_rms / current_rms
    max_sample = np.max(np.abs(samples))
    max_gain = 32767.0 / max_sample if max_sample > 0 else 1.0
    gain = min(gain, max_gain * 0.95)
    normalized = samples * gain
    return np.clip(normalized, -32768, 32767).astype(np.int16).tobytes()


def trim_silence(pcm_bytes: bytes, sample_rate: int = 16000, threshold: float = 200.0) -> bytes:
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float64)
    frame_size = int(sample_rate * 0.02)
    if len(samples) < frame_size:
        return pcm_bytes
    start = 0
    for i in range(0, len(samples) - frame_size, frame_size):
        frame_rms = np.sqrt(np.mean(samples[i:i + frame_size] ** 2))
        if frame_rms >= threshold:
            start = i
            break
    else:
        return pcm_bytes
    end = len(samples)
    for i in range(len(samples) - frame_size, start, -frame_size):
        frame_rms = np.sqrt(np.mean(samples[i:i + frame_size] ** 2))
        if frame_rms >= threshold:
            end = i + frame_size
            break
    trimmed = samples[start:end].astype(np.int16).tobytes()
    return trimmed if len(trimmed) >= 2 else pcm_bytes


def preprocess_audio(pcm_bytes: bytes, sample_rate: int = 16000, target_rms: float = 1000.0, noise_gate_rms: float = 100.0) -> bytes:
    pcm = trim_silence(pcm_bytes, sample_rate, threshold=noise_gate_rms)
    pcm = normalize_gain(pcm, target_rms=target_rms)
    return pcm
