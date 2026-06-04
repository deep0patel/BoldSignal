"""
Training loop for the BoldSignal brain encoder.

Usage (on RunPod GPU):
    python -m boldsignal.train \
        --features_dir features/cache \
        --fmri_dir data/processed \
        --checkpoint_dir checkpoints

Expected output per epoch:
    Epoch 1/20 | train_loss=0.8432 | val_pearson=0.0312 | time=142s
"""
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset
from scipy.stats import pearsonr

from boldsignal.config import Config
from boldsignal.data.dataset import BrainDataset
from boldsignal.model.encoder import CrossModalTransformer
from boldsignal.model.head import SubjectHead


def load_subject_data(features_dir: str, fmri_dir: str, subject_id: str, subject_idx: int,
                      cfg: Config) -> BrainDataset:
    feat_path = Path(features_dir) / subject_id
    fmri_path = Path(fmri_dir) / subject_id

    video = np.load(feat_path / "video.npy").astype(np.float32)
    audio = np.load(feat_path / "audio.npy").astype(np.float32)
    text  = np.load(feat_path / "text.npy").astype(np.float32)
    fmri  = np.load(fmri_path / "fmri_parcels.npy").astype(np.float32)

    T = min(video.shape[0], audio.shape[0], text.shape[0], fmri.shape[0])
    data = {"video": video[:T], "audio": audio[:T], "text": text[:T], "fmri": fmri[:T]}
    return BrainDataset(data, window=cfg.window_seconds, subject_id=subject_idx)


def pearson_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean Pearson r across all ROIs. pred, target: (T, n_rois)"""
    scores = []
    for roi in range(pred.shape[1]):
        r, _ = pearsonr(pred[:, roi], target[:, roi])
        if not np.isnan(r):
            scores.append(r)
    return float(np.mean(scores)) if scores else 0.0


def train(cfg: Config, features_dir: str, fmri_dir: str, checkpoint_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    all_datasets = []
    for idx, sub_id in enumerate(cfg.subject_ids):
        ds = load_subject_data(features_dir, fmri_dir, sub_id, idx, cfg)
        all_datasets.append(ds)

    train_ds, val_ds = [], []
    for ds in all_datasets:
        n = len(ds)
        split = int(n * (1 - cfg.val_split))
        train_ds.append(torch.utils.data.Subset(ds, range(0, split)))
        val_ds.append(torch.utils.data.Subset(ds, range(split, n)))

    train_loader = DataLoader(ConcatDataset(train_ds), batch_size=cfg.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(ConcatDataset(val_ds), batch_size=cfg.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    encoder = CrossModalTransformer(
        video_dim=cfg.video_dim, audio_dim=cfg.audio_dim, text_dim=cfg.text_dim,
        proj_dim=cfg.proj_dim, d_model=cfg.d_model, n_heads=cfg.n_heads,
        n_layers=cfg.n_layers, dropout=cfg.dropout, max_seq_len=cfg.max_seq_len,
    ).to(device)
    head = SubjectHead(cfg.d_model, cfg.n_rois, cfg.n_subjects).to(device)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()), lr=cfg.lr
    )
    total_steps = len(train_loader) * cfg.max_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, total_steps=total_steps,
        pct_start=cfg.warmup_ratio, anneal_strategy="cos"
    )
    criterion = nn.MSELoss()

    best_val_pearson = -999.0
    patience_counter = 0

    for epoch in range(1, cfg.max_epochs + 1):
        t0 = time.time()
        encoder.train(); head.train()
        train_loss = 0.0

        for batch in train_loader:
            video   = batch["video"].to(device)
            audio   = batch["audio"].to(device)
            text    = batch["text"].to(device)
            fmri    = batch["fmri"].to(device)
            sub_ids = batch["subject_id"].to(device)

            latent = encoder(video, audio, text, output_timesteps=cfg.window_seconds)
            pred   = head(latent, sub_ids)
            loss   = criterion(pred, fmri)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(head.parameters()), max_norm=1.0
            )
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        encoder.eval(); head.eval()
        all_pred, all_target = [], []
        with torch.no_grad():
            for batch in val_loader:
                video   = batch["video"].to(device)
                audio   = batch["audio"].to(device)
                text    = batch["text"].to(device)
                fmri    = batch["fmri"].to(device)
                sub_ids = batch["subject_id"].to(device)
                latent  = encoder(video, audio, text, output_timesteps=cfg.window_seconds)
                pred    = head(latent, sub_ids)
                all_pred.append(pred.cpu().numpy().reshape(-1, cfg.n_rois))
                all_target.append(fmri.cpu().numpy().reshape(-1, cfg.n_rois))

        val_pearson = pearson_score(np.vstack(all_pred), np.vstack(all_target))
        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{cfg.max_epochs} | "
              f"train_loss={train_loss:.4f} | "
              f"val_pearson={val_pearson:.4f} | "
              f"time={elapsed:.0f}s")

        if val_pearson > best_val_pearson:
            best_val_pearson = val_pearson
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "encoder": encoder.state_dict(),
                "head": head.state_dict(),
                "val_pearson": val_pearson,
                "cfg": cfg,
            }, f"{checkpoint_dir}/best.pt")
            print(f"  Saved best checkpoint (val_pearson={val_pearson:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nTraining complete. Best val Pearson r = {best_val_pearson:.4f}")
    print(f"Target: r > 0.10 (real signal). r > 0.15 = strong MVP.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", default="features/cache")
    parser.add_argument("--fmri_dir", default="data/processed")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    args = parser.parse_args()
    train(Config(), args.features_dir, args.fmri_dir, args.checkpoint_dir)
