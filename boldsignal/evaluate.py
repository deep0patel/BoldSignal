"""
Load a trained checkpoint and compute Pearson r on held-out data.

Usage:
    python -m boldsignal.evaluate \
        --checkpoint checkpoints/best.pt \
        --features_dir features/cache \
        --fmri_dir data/processed
"""
import argparse
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from boldsignal.config import Config
from boldsignal.model.encoder import CrossModalTransformer
from boldsignal.model.head import SubjectHead
from boldsignal.train import load_subject_data, pearson_score


def evaluate(checkpoint_path: str, features_dir: str, fmri_dir: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg: Config = ckpt["cfg"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = CrossModalTransformer(
        video_dim=cfg.video_dim, audio_dim=cfg.audio_dim, text_dim=cfg.text_dim,
        proj_dim=cfg.proj_dim, d_model=cfg.d_model, n_heads=cfg.n_heads,
        n_layers=cfg.n_layers, dropout=0.0, max_seq_len=cfg.max_seq_len,
    ).to(device)
    head = SubjectHead(cfg.d_model, cfg.n_rois, cfg.n_subjects).to(device)

    encoder.load_state_dict(ckpt["encoder"])
    head.load_state_dict(ckpt["head"])
    encoder.eval(); head.eval()

    all_pred, all_target = [], []
    for idx, sub_id in enumerate(cfg.subject_ids):
        ds = load_subject_data(features_dir, fmri_dir, sub_id, idx, cfg)
        n = len(ds)
        split = int(n * (1 - cfg.val_split))
        val_ds = torch.utils.data.Subset(ds, range(split, n))
        loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2)

        with torch.no_grad():
            for batch in loader:
                video   = batch["video"].to(device)
                audio   = batch["audio"].to(device)
                text    = batch["text"].to(device)
                fmri    = batch["fmri"]
                sub_ids = batch["subject_id"].to(device)
                latent  = encoder(video, audio, text, output_timesteps=cfg.window_seconds)
                pred    = head(latent, sub_ids).cpu().numpy().reshape(-1, cfg.n_rois)
                all_pred.append(pred)
                all_target.append(fmri.numpy().reshape(-1, cfg.n_rois))

    pred_all   = np.vstack(all_pred)
    target_all = np.vstack(all_target)
    mean_r = pearson_score(pred_all, target_all)

    print(f"\nEvaluation Results")
    print(f"==================")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Subjects:   {cfg.subject_ids}")
    print(f"Mean Pearson r across all ROIs: {mean_r:.4f}")
    print()
    print(f"Benchmark:")
    print(f"  r > 0.00 — beating chance")
    print(f"  r > 0.10 — real signal (better than linear baseline)")
    print(f"  r > 0.15 — strong MVP")
    return mean_r


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--features_dir", default="features/cache")
    parser.add_argument("--fmri_dir", default="data/processed")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.features_dir, args.fmri_dir)
