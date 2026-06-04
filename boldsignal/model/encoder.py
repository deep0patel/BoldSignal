import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalTransformer(nn.Module):
    """
    Projects video, audio, text to shared dim → concatenates → transformer → pool.
    Input:  (B, T, D_video), (B, T, D_audio), (B, T, D_text)
    Output: (B, T_out, d_model) where T_out = output_timesteps
    """

    def __init__(self, video_dim: int, audio_dim: int, text_dim: int,
                 proj_dim: int, d_model: int, n_heads: int, n_layers: int,
                 dropout: float, max_seq_len: int):
        super().__init__()
        assert d_model == proj_dim * 3, "d_model must equal proj_dim * 3"

        self.video_proj = nn.Linear(video_dim, proj_dim)
        self.audio_proj = nn.Linear(audio_dim, proj_dim)
        self.text_proj  = nn.Linear(text_dim,  proj_dim)

        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, video: torch.Tensor, audio: torch.Tensor,
                text: torch.Tensor, output_timesteps: int) -> torch.Tensor:
        """
        video: (B, T, video_dim)
        audio: (B, T, audio_dim)
        text:  (B, T, text_dim)
        output_timesteps: T_out — pool T → T_out
        returns: (B, T_out, d_model)
        """
        _, T, _ = video.shape

        v = self.video_proj(video)
        a = self.audio_proj(audio)
        t = self.text_proj(text)

        x = torch.cat([v, a, t], dim=-1)       # (B, T, d_model)
        x = x + self.pos_embed[:, :T, :]
        x = self.transformer(x)                # (B, T, d_model)

        x = x.transpose(1, 2)                  # (B, d_model, T)
        x = F.adaptive_avg_pool1d(x, output_timesteps)
        return x.transpose(1, 2)               # (B, T_out, d_model)
