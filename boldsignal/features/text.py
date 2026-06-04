"""
Generate text embeddings from video transcript at 2Hz.
Pipeline: video → Whisper (speech-to-text + word timings) → BGE (semantic embeddings)
Output: (T, 768) array saved as .npy

Usage:
    python -m boldsignal.features.text \
        --video path/to/video.mp4 \
        --output features/cache/sub-01_friends_s01e01_text.npy
"""
import argparse
import numpy as np
from pathlib import Path

try:
    import torch
    from transformers import pipeline
except (ImportError, RuntimeError):
    torch = None
    pipeline = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

WHISPER_MODEL = "openai/whisper-base"
BGE_MODEL = "BAAI/bge-base-en-v1.5"
SAMPLING_HZ = 2.0
BIN_DURATION = 1.0 / SAMPLING_HZ  # 0.5s per bin
EMBED_DIM = 768


def transcribe_with_timings(video_path: str, device: str = "cuda") -> list:
    """Run Whisper on video audio. Returns list of (word, start_sec, end_sec)."""
    asr = pipeline(
        "automatic-speech-recognition",
        model=WHISPER_MODEL,
        return_timestamps="word",
        device=0 if device == "cuda" else -1
    )
    result = asr(video_path)
    words = []
    for chunk in result.get("chunks", []):
        word = chunk["text"].strip()
        start, end = chunk["timestamp"]
        if word and start is not None:
            words.append((word, float(start), float(end or start + 0.2)))
    return words


def align_text_to_bins(words: list, n_bins: int, bin_duration: float = BIN_DURATION) -> list:
    """
    Assign words to time bins. Returns list of strings (one per bin).
    words: list of (word, start_sec, end_sec)
    """
    bins = [""] * n_bins
    for word, start, end in words:
        bin_idx = int(start / bin_duration)
        if bin_idx < n_bins:
            bins[bin_idx] = (bins[bin_idx] + " " + word).strip()
    for i in range(1, n_bins):
        if not bins[i]:
            bins[i] = bins[i - 1]
    return bins


def extract_text_features(video_path: str, output_path: str = None, device: str = "cuda",
                           duration_seconds: float = None) -> np.ndarray:
    """
    Extract BGE text embeddings at 2Hz from video transcript.
    Returns (T, 768) numpy array. Saves to output_path if provided.
    """
    bge = SentenceTransformer(BGE_MODEL)
    words = transcribe_with_timings(video_path, device)

    if duration_seconds is None:
        duration_seconds = words[-1][2] if words else 60.0
    n_bins = int(duration_seconds * SAMPLING_HZ)

    bins = align_text_to_bins(words, n_bins, BIN_DURATION)
    result = bge.encode(bins, batch_size=64, show_progress_bar=False)  # (T, 768)

    if output_path:
        np.save(output_path, result)
        print(f"Saved text features: {result.shape} → {output_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    extract_text_features(args.video, args.output, args.device)
