from dataclasses import dataclass, field


@dataclass
class Config:
    # Data
    fmri_dir: str = "data/processed"
    features_dir: str = "features/cache"
    n_subjects: int = 4
    subject_ids: list = field(default_factory=lambda: ["sub-01", "sub-02", "sub-03", "sub-05"])
    n_rois: int = 360
    hrf_lag_seconds: int = 5
    fmri_hz: float = 1.0
    stimulus_hz: float = 2.0

    # Model
    video_dim: int = 512    # X-CLIP output
    audio_dim: int = 768    # WavLM output
    text_dim: int = 768     # BGE output
    proj_dim: int = 128     # each modality projected to this
    d_model: int = 384      # proj_dim * 3
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1
    max_seq_len: int = 1024

    # Training
    window_seconds: int = 100
    batch_size: int = 16
    lr: float = 1e-4
    warmup_ratio: float = 0.1
    max_epochs: int = 20
    patience: int = 3
    checkpoint_dir: str = "checkpoints"
    val_split: float = 0.1

    # Score mapping ROI indices (filled by preprocess.py after atlas loading)
    attention_rois: list = field(default_factory=list)
    emotion_rois: list = field(default_factory=list)
    memory_rois: list = field(default_factory=list)
    hook_rois: list = field(default_factory=list)
    social_rois: list = field(default_factory=list)
    cognitive_rois: list = field(default_factory=list)
