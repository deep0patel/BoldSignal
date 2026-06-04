"""
Maps predicted fMRI activation (T, 360 ROIs) to marketing metrics.
All scores normalized 0-100.
"""
import numpy as np


def _normalize(values: np.ndarray) -> float:
    """Normalize mean of values to 0-100 using 5th/95th percentile range."""
    v = float(np.mean(values))
    lo = float(np.percentile(values, 5))
    hi = float(np.percentile(values, 95))
    if hi == lo:
        return 50.0
    return float(np.clip((v - lo) / (hi - lo) * 100, 0, 100))


def compute_scores(preds: np.ndarray, roi_map: dict) -> dict:
    """
    preds: (T, n_rois) predicted fMRI activation
    roi_map: dict mapping metric names to lists of ROI indices
    returns: dict with neural_score and per-dimension scores (0-100)
    """
    T = preds.shape[0]
    hook_window = min(int(T * 0.15), 15)

    attention = _normalize(preds[:, roi_map["attention"]])
    emotion   = _normalize(preds[:, roi_map["emotion"]])
    memory    = _normalize(preds[:, roi_map["memory"]])
    hook      = _normalize(preds[:hook_window, roi_map["hook"]])

    neural_score = min(round(0.40 * attention + 0.35 * emotion + 0.25 * memory, 1), 100.0)

    return {
        "neural_score": neural_score,
        "attention": round(attention, 1),
        "emotion":   round(emotion, 1),
        "memory":    round(memory, 1),
        "hook":      round(hook, 1),
    }


def build_timeline(preds: np.ndarray, roi_map: dict) -> list:
    """Per-second engagement score. Returns list of (second, score) tuples."""
    attention_ts = preds[:, roi_map["attention"]].mean(axis=1)
    emotion_ts   = preds[:, roi_map["emotion"]].mean(axis=1)
    cognitive_ts = preds[:, roi_map["cognitive"]].mean(axis=1)

    engagement = (attention_ts + emotion_ts) / 2 - cognitive_ts

    lo, hi = engagement.min(), engagement.max()
    if hi > lo:
        engagement = (engagement - lo) / (hi - lo) * 100
    else:
        engagement = np.full_like(engagement, 50.0)

    return [(int(t), round(float(v), 2)) for t, v in enumerate(engagement)]


def find_drop_offs(timeline: list, threshold_std: float = 1.0) -> list:
    """Return timestamps where engagement drops > 1 std below mean."""
    scores = [v for _, v in timeline]
    mean, std = np.mean(scores), np.std(scores)
    return [t for t, v in timeline if v < mean - threshold_std * std]


def find_peak_moments(timeline: list, threshold_std: float = 1.0) -> list:
    """Return timestamps where engagement is > 1 std above mean."""
    scores = [v for _, v in timeline]
    mean, std = np.mean(scores), np.std(scores)
    return [t for t, v in timeline if v > mean + threshold_std * std]


def score_video(preds: np.ndarray, roi_map: dict) -> dict:
    """Full scoring pipeline. Returns complete output JSON."""
    scores = compute_scores(preds, roi_map)
    timeline = build_timeline(preds, roi_map)
    return {
        **scores,
        "drop_offs":    find_drop_offs(timeline),
        "peak_moments": find_peak_moments(timeline),
        "timeline":     timeline,
    }
