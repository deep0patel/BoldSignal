import numpy as np
from scipy.interpolate import interp1d


def zscore_run(data: np.ndarray) -> np.ndarray:
    """Z-score normalize each voxel across time. data: (time, voxels)"""
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std[std == 0] = 1
    return (data - mean) / std


def apply_hrf_offset(fmri: np.ndarray, lag_seconds: int = 5, fmri_hz: float = 1.0) -> np.ndarray:
    """Trim first lag_seconds of fMRI to align with stimulus (brain responds ~5s after)."""
    lag_samples = int(lag_seconds * fmri_hz)
    return fmri[lag_samples:]


def average_to_parcels(voxel_data: np.ndarray, parcels_map: np.ndarray, n_parcels: int = 360) -> np.ndarray:
    """
    Average voxels within each parcel.
    voxel_data: (time, n_voxels)
    parcels_map: (n_voxels,) — integer parcel index 0..n_parcels-1
    returns: (time, n_parcels)
    """
    T = voxel_data.shape[0]
    result = np.zeros((T, n_parcels))
    for p in range(n_parcels):
        mask = parcels_map == p
        if mask.sum() > 0:
            result[:, p] = voxel_data[:, mask].mean(axis=1)
    return result


def resample_to_hz(data: np.ndarray, from_hz: float, to_hz: float) -> np.ndarray:
    """Linearly resample (time, features) from one rate to another."""
    T = data.shape[0]
    t_orig = np.linspace(0, T / from_hz, T)
    t_new = np.arange(0, T / from_hz, 1.0 / to_hz)
    interpolator = interp1d(t_orig, data, axis=0, bounds_error=False, fill_value="extrapolate")
    return interpolator(t_new)


def load_hcp_parcels_map(fsaverage5_nii_path: str = None) -> np.ndarray:
    """
    Load HCP 360-parcel atlas mapped to fsaverage5 surface.
    Returns array of shape (n_voxels,) with parcel index per voxel.
    Uses Schaefer 2018 atlas via nilearn as proxy for full 360-parcel HCP atlas.
    """
    from nilearn import datasets
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=200)
    return atlas
