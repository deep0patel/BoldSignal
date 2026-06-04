import numpy as np
import torch
import pytest
from boldsignal.config import Config
from boldsignal.model.encoder import CrossModalTransformer
from boldsignal.model.head import SubjectHead
from boldsignal.data.dataset import BrainDataset
from boldsignal.score import score_video
from torch.utils.data import DataLoader


def test_full_forward_pass():
    cfg = Config()
    cfg.n_subjects = 2
    cfg.window_seconds = 20

    encoder = CrossModalTransformer(
        video_dim=cfg.video_dim, audio_dim=cfg.audio_dim, text_dim=cfg.text_dim,
        proj_dim=cfg.proj_dim, d_model=cfg.d_model, n_heads=cfg.n_heads,
        n_layers=cfg.n_layers, dropout=0.0, max_seq_len=cfg.max_seq_len,
    )
    head = SubjectHead(cfg.d_model, cfg.n_rois, cfg.n_subjects)

    T = 200
    data = {
        "video": np.random.randn(T, 512).astype(np.float32),
        "audio": np.random.randn(T, 768).astype(np.float32),
        "text":  np.random.randn(T, 768).astype(np.float32),
        "fmri":  np.random.randn(T, 360).astype(np.float32),
    }
    ds = BrainDataset(data, window=cfg.window_seconds, subject_id=0)
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    encoder.eval(); head.eval()
    batch = next(iter(loader))
    with torch.no_grad():
        latent = encoder(batch["video"], batch["audio"], batch["text"],
                         output_timesteps=cfg.window_seconds)
        pred = head(latent, batch["subject_id"])

    assert pred.shape == (4, cfg.window_seconds, 360)


def test_score_video_runs():
    preds = np.random.randn(100, 360).astype(np.float32)
    roi_map = {
        "attention": list(range(0, 20)),
        "emotion":   list(range(20, 50)),
        "memory":    list(range(50, 70)),
        "hook":      list(range(70, 90)),
        "social":    list(range(90, 110)),
        "cognitive": list(range(110, 120)),
    }
    result = score_video(preds, roi_map)
    assert "neural_score" in result
    assert "timeline" in result
    assert "drop_offs" in result
    assert "peak_moments" in result
    assert 0 <= result["neural_score"] <= 100
