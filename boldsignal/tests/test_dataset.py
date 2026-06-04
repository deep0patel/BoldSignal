import numpy as np
import pytest
from boldsignal.data.dataset import BrainDataset

def make_fake_data(T=500, n_rois=360, video_dim=512, audio_dim=768, text_dim=768):
    return {
        "video": np.random.randn(T, video_dim).astype(np.float32),
        "audio": np.random.randn(T, audio_dim).astype(np.float32),
        "text":  np.random.randn(T, text_dim).astype(np.float32),
        "fmri":  np.random.randn(T, n_rois).astype(np.float32),
    }

def test_dataset_length():
    data = make_fake_data(T=500)
    ds = BrainDataset(data, window=100, subject_id=0)
    assert len(ds) == 401  # 500 - 100 + 1

def test_dataset_item_shapes():
    data = make_fake_data(T=500)
    ds = BrainDataset(data, window=100, subject_id=0)
    item = ds[0]
    assert item["video"].shape == (100, 512)
    assert item["audio"].shape == (100, 768)
    assert item["text"].shape  == (100, 768)
    assert item["fmri"].shape  == (100, 360)
    assert item["subject_id"]  == 0

def test_dataset_no_overlap_across_boundary():
    data = make_fake_data(T=200)
    ds = BrainDataset(data, window=100, subject_id=1)
    first  = ds[0]["fmri"]
    second = ds[100]["fmri"]
    assert not np.allclose(first[-1].numpy(), second[0].numpy())
