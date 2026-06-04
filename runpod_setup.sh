#!/bin/bash
# Run once on a fresh RunPod instance (PyTorch template recommended)
# Tested on: RTX 3090, PyTorch 2.2, CUDA 12.1

set -e

echo "=== BoldSignal RunPod Setup ==="

# 1. Clone repo (replace with your GitHub username)
# git clone https://github.com/<your-username>/BoldSignal.git
# cd BoldSignal

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download CNeuroMod — start with sub-01 to test pipeline
python -m boldsignal.data.download --output data/raw --subject sub-01

# 4. Preprocess fMRI for sub-01
# (runs after BIDS fMRI .nii.gz are downloaded)
mkdir -p data/processed/sub-01

# 5. Extract features (run each independently — GPU-bound, ~1-2h per modality)
mkdir -p features/cache/sub-01

python -m boldsignal.features.video \
    --video data/raw/sub-01/video/friends_s01e01.mp4 \
    --output features/cache/sub-01/video.npy \
    --device cuda

python -m boldsignal.features.audio \
    --video data/raw/sub-01/video/friends_s01e01.mp4 \
    --output features/cache/sub-01/audio.npy \
    --device cuda

python -m boldsignal.features.text \
    --video data/raw/sub-01/video/friends_s01e01.mp4 \
    --output features/cache/sub-01/text.npy \
    --device cuda

# 6. Train
python -m boldsignal.train \
    --features_dir features/cache \
    --fmri_dir data/processed \
    --checkpoint_dir checkpoints

# 7. Evaluate
python -m boldsignal.evaluate \
    --checkpoint checkpoints/best.pt \
    --features_dir features/cache \
    --fmri_dir data/processed

echo "=== Done. Download checkpoints/ back to your Mac. ==="
