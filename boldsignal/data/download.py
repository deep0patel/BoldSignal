"""
CNeuroMod download via CONP / datalad.

Usage:
    python -m boldsignal.data.download --output data/raw

What it downloads (open access, CC0, no DTA):
    - 4 subjects: sub-01, sub-02, sub-03, sub-05
    - Friends seasons 1-6 fMRI (already fMRIPrep preprocessed)
    - Corresponding video stimuli timings

Full download is ~500GB. Start with one subject to test pipeline:
    python -m boldsignal.data.download --output data/raw --subject sub-01
"""

import argparse
import subprocess
from pathlib import Path

CONP_URL = "https://github.com/CONP-PCNO/conp-dataset"
CNEUROMOD_FRIENDS = "projects/cneuromod/friends"

def download(output_dir: str, subject: str = None):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Installing CONP dataset via datalad...")
    subprocess.run(["datalad", "install", "-r", CONP_URL, str(out / "conp-dataset")], check=True)

    dataset_path = out / "conp-dataset" / CNEUROMOD_FRIENDS
    subjects = [subject] if subject else ["sub-01", "sub-02", "sub-03", "sub-05"]

    for sub in subjects:
        print(f"Getting fMRI for {sub}...")
        subprocess.run([
            "datalad", "get",
            str(dataset_path / sub / "ses-*" / "func" / "*.nii.gz")
        ], check=True)

    print(f"Download complete. Data at {dataset_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw")
    parser.add_argument("--subject", default=None, help="e.g. sub-01. Omit for all 4.")
    args = parser.parse_args()
    download(args.output, args.subject)
