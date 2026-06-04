import numpy as np
from boldsignal.score import compute_scores, build_timeline, find_drop_offs, find_peak_moments

def fake_preds(T=100, n_rois=360):
    return np.random.randn(T, n_rois).astype(np.float32)

def fake_roi_map():
    return {
        "attention": list(range(0, 20)),
        "emotion":   list(range(20, 50)),
        "memory":    list(range(50, 70)),
        "hook":      list(range(70, 90)),
        "social":    list(range(90, 110)),
        "cognitive": list(range(110, 120)),
    }

def test_compute_scores_keys():
    result = compute_scores(fake_preds(), fake_roi_map())
    assert set(result.keys()) == {"neural_score", "attention", "emotion", "memory", "hook"}

def test_compute_scores_range():
    result = compute_scores(fake_preds(), fake_roi_map())
    for key in ["neural_score", "attention", "emotion", "memory", "hook"]:
        assert 0 <= result[key] <= 100, f"{key} out of range"

def test_build_timeline_length():
    timeline = build_timeline(fake_preds(T=100), fake_roi_map())
    assert len(timeline) == 100
    assert all(isinstance(v, float) for _, v in timeline)

def test_find_drop_offs_returns_list():
    timeline = build_timeline(fake_preds(T=100), fake_roi_map())
    drop_offs = find_drop_offs(timeline)
    assert isinstance(drop_offs, list)

def test_find_peak_moments_returns_list():
    timeline = build_timeline(fake_preds(T=100), fake_roi_map())
    peaks = find_peak_moments(timeline)
    assert isinstance(peaks, list)
