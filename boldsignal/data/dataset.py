import numpy as np
import torch
from torch.utils.data import Dataset


class BrainDataset(Dataset):
    """
    Sliding window dataset over aligned (features, fMRI) arrays.
    Returns windows of shape (window, dim) for each modality.
    """

    def __init__(self, data: dict, window: int, subject_id: int):
        """
        data: dict with keys "video", "audio", "text", "fmri"
              each value is (T, D) numpy float32 array, all same T
        window: number of timepoints per sample
        subject_id: integer index of subject (used by SubjectHead)
        """
        self.data = data
        self.window = window
        self.subject_id = subject_id
        self.T = data["fmri"].shape[0]
        assert all(v.shape[0] == self.T for v in data.values()), \
            "All modalities must have same time length"

    def __len__(self):
        return self.T - self.window + 1

    def __getitem__(self, idx):
        s, e = idx, idx + self.window
        return {
            "video":      torch.from_numpy(self.data["video"][s:e]),
            "audio":      torch.from_numpy(self.data["audio"][s:e]),
            "text":       torch.from_numpy(self.data["text"][s:e]),
            "fmri":       torch.from_numpy(self.data["fmri"][s:e]),
            "subject_id": torch.tensor(self.subject_id, dtype=torch.long),
        }
