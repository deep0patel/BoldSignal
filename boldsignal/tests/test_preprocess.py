import numpy as np
import pytest
from boldsignal.data.preprocess import zscore_run, apply_hrf_offset, average_to_parcels

def test_zscore_run():
    data = np.random.randn(100, 90000) * 5 + 3
    result = zscore_run(data)
    assert result.shape == (100, 90000)
    np.testing.assert_allclose(result.mean(axis=0), 0, atol=1e-6)
    np.testing.assert_allclose(result.std(axis=0), 1, atol=1e-6)

def test_apply_hrf_offset():
    fmri = np.random.randn(100, 360)
    shifted = apply_hrf_offset(fmri, lag_seconds=5, fmri_hz=1.0)
    assert shifted.shape == (95, 360)

def test_average_to_parcels_shape():
    voxel_data = np.random.randn(50, 90000)
    parcels_map = np.random.randint(0, 360, size=90000)
    result = average_to_parcels(voxel_data, parcels_map, n_parcels=360)
    assert result.shape == (50, 360)
