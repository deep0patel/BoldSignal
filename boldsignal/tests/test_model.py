import torch
import pytest
from boldsignal.model.encoder import CrossModalTransformer

def test_encoder_output_shape():
    model = CrossModalTransformer(
        video_dim=512, audio_dim=768, text_dim=768,
        proj_dim=128, d_model=384, n_heads=8, n_layers=6,
        dropout=0.1, max_seq_len=1024
    )
    B, T = 2, 200
    video = torch.randn(B, T, 512)
    audio = torch.randn(B, T, 768)
    text  = torch.randn(B, T, 768)
    out = model(video, audio, text, output_timesteps=100)
    assert out.shape == (B, 100, 384)

def test_encoder_no_crash_with_different_batch_sizes():
    model = CrossModalTransformer(
        video_dim=512, audio_dim=768, text_dim=768,
        proj_dim=128, d_model=384, n_heads=8, n_layers=6,
        dropout=0.1, max_seq_len=1024
    )
    for B in [1, 4, 8]:
        video = torch.randn(B, 100, 512)
        audio = torch.randn(B, 100, 768)
        text  = torch.randn(B, 100, 768)
        out = model(video, audio, text, output_timesteps=50)
        assert out.shape == (B, 50, 384)
