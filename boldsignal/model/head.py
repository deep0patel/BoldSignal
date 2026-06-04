import torch
import torch.nn as nn


class SubjectHead(nn.Module):
    """
    Per-subject linear projection: (B, T, d_model) → (B, T, n_rois).
    Each subject has its own weight matrix — transformer is shared,
    output mapping is individual to each brain.
    """

    def __init__(self, d_model: int, n_rois: int, n_subjects: int):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(n_subjects, d_model, n_rois) * 0.01)
        self.biases  = nn.Parameter(torch.zeros(n_subjects, n_rois))

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        """
        x:           (B, T, d_model)
        subject_ids: (B,) integer subject indices
        returns:     (B, T, n_rois)
        """
        W = self.weights[subject_ids]      # (B, d_model, n_rois)
        b = self.biases[subject_ids]       # (B, n_rois)
        return torch.bmm(x, W) + b.unsqueeze(1)  # (B, T, n_rois)
