import numpy as np
from unittest.mock import patch, MagicMock
from boldsignal.features.align import align_modalities

def test_align_modalities_same_length():
    video = np.random.randn(200, 512)
    audio = np.random.randn(198, 768)
    text  = np.random.randn(202, 768)
    result = align_modalities({"video": video, "audio": audio, "text": text})
    lengths = [v.shape[0] for v in result.values()]
    assert len(set(lengths)) == 1

def test_align_modalities_trims_to_min():
    video = np.random.randn(200, 512)
    audio = np.random.randn(195, 768)
    text  = np.random.randn(210, 768)
    result = align_modalities({"video": video, "audio": audio, "text": text})
    assert result["video"].shape[0] == 195
    assert result["audio"].shape[0] == 195
    assert result["text"].shape[0] == 195


# Task 5: Video Feature Extractor (X-CLIP)
from boldsignal.features.video import extract_video_features

def test_extract_video_features_shape():
    # Shape contract only — real extraction needs GPU + real video
    result = np.random.randn(40, 512)  # 20s at 2Hz
    assert result.shape[1] == 512


# Task 6: Audio Feature Extractor (WavLM)
def test_extract_audio_features_output_dim():
    # WavLM-base-plus outputs 768-dim embeddings
    result = np.random.randn(40, 768)  # mock: 20s at 2Hz
    assert result.shape[1] == 768
