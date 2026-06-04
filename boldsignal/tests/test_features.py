import numpy as np
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
