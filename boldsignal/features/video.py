"""
Extract X-CLIP video embeddings at 2Hz.
X-CLIP processes 8 frames per clip with temporal attention.
Output: (T, 512) array saved as .npy

Usage:
    python -m boldsignal.features.video \
        --video path/to/video.mp4 \
        --output features/cache/sub-01_friends_s01e01_video.npy
"""
import argparse
import numpy as np
from pathlib import Path

try:
    import torch
    from transformers import AutoProcessor, AutoModel
    import av
except (ImportError, RuntimeError):
    torch = None
    AutoProcessor = AutoModel = None
    av = None

MODEL_ID = "microsoft/xclip-base-patch32"
SAMPLING_HZ = 2.0
FRAMES_PER_CLIP = 8
EMBED_DIM = 512


def load_model(device="cuda"):
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device)
    model.eval()
    return processor, model


def extract_frames(video_path: str, target_hz: float = SAMPLING_HZ):
    """Extract frames at target_hz from video. Returns list of PIL Images."""
    container = av.open(video_path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate)
    frame_interval = max(1, int(fps / target_hz))
    frames = []
    for i, frame in enumerate(container.decode(stream)):
        if i % frame_interval == 0:
            frames.append(frame.to_image())
    return frames


def extract_video_features(video_path: str, output_path: str = None, device: str = "cuda") -> np.ndarray:
    """
    Extract X-CLIP embeddings at 2Hz.
    Returns (T, 512) numpy array. Saves to output_path if provided.
    """
    processor, model = load_model(device)
    frames = extract_frames(video_path, SAMPLING_HZ)
    embeddings = []

    with torch.no_grad():
        for i in range(0, len(frames), FRAMES_PER_CLIP):
            clip_frames = frames[i:i + FRAMES_PER_CLIP]
            if len(clip_frames) < FRAMES_PER_CLIP:
                clip_frames += [clip_frames[-1]] * (FRAMES_PER_CLIP - len(clip_frames))
            inputs = processor(videos=[clip_frames], return_tensors="pt").to(device)
            feats = model.get_video_features(**inputs)
            embeddings.append(feats.cpu().numpy())

    result = np.vstack(embeddings)  # (T, 512)
    if output_path:
        np.save(output_path, result)
        print(f"Saved video features: {result.shape} → {output_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    extract_video_features(args.video, args.output, args.device)
