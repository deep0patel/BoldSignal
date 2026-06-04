"""
Extract WavLM-base-plus audio embeddings at 2Hz.
WavLM captures speech, music, ambient sound.
Output: (T, 768) array saved as .npy

Usage:
    python -m boldsignal.features.audio \
        --video path/to/video.mp4 \
        --output features/cache/sub-01_friends_s01e01_audio.npy
"""
import argparse
import numpy as np
from pathlib import Path

try:
    import torch
    import librosa
    from transformers import AutoProcessor, AutoModel
    import av
except (ImportError, RuntimeError):
    torch = None
    librosa = None
    AutoProcessor = AutoModel = None
    av = None

MODEL_ID = "microsoft/wavlm-base-plus"
SAMPLING_HZ = 2.0
AUDIO_SR = 16000
CHUNK_SECONDS = 0.5
EMBED_DIM = 768


def extract_audio_from_video(video_path: str) -> np.ndarray:
    """Extract raw audio waveform from video at 16kHz."""
    container = av.open(video_path)
    audio_frames = []
    for frame in container.decode(audio=0):
        audio_frames.append(frame.to_ndarray())
    if not audio_frames:
        raise ValueError(f"No audio stream found in {video_path}")
    waveform = np.concatenate(audio_frames, axis=-1).mean(axis=0)
    orig_sr = container.streams.audio[0].rate
    waveform = librosa.resample(waveform.astype(np.float32), orig_sr=orig_sr, target_sr=AUDIO_SR)
    return waveform


def extract_audio_features(video_path: str, output_path: str = None, device: str = "cuda") -> np.ndarray:
    """
    Extract WavLM embeddings at 2Hz from video audio.
    Returns (T, 768) numpy array. Saves to output_path if provided.
    """
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device)
    model.eval()

    waveform = extract_audio_from_video(video_path)
    chunk_size = int(AUDIO_SR * CHUNK_SECONDS)
    embeddings = []

    with torch.no_grad():
        for start in range(0, len(waveform), chunk_size):
            chunk = waveform[start:start + chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            inputs = processor(chunk, sampling_rate=AUDIO_SR, return_tensors="pt").to(device)
            outputs = model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(emb)

    result = np.vstack(embeddings)  # (T, 768)
    if output_path:
        np.save(output_path, result)
        print(f"Saved audio features: {result.shape} → {output_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    extract_audio_features(args.video, args.output, args.device)
