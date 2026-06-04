---
title: BoldSignal
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# BoldSignal — Brain Score for Video Content

An experiment built on top of Meta's TRIBE v2 fMRI brain encoding model. Upload a video, get predicted brain activation across 20,000 cortical voxels, and see a Brain Score (0-100) across four neuroscience-grounded dimensions.

The scoring layer is my own work — dimension weights and normalisation are judgment calls based on the literature, not validated parameters. Take the numbers as directional signals.

---

## What It Does

TRIBE v2 takes a video and predicts BOLD (Blood-Oxygen-Level-Dependent) signal responses across ~20,000 cortical voxels per 2-second timestep on the fsaverage5 surface. Those predictions represent how an average brain might respond to the content — not any specific person.

BoldSignal aggregates those voxel predictions into four dimensions using anatomically defined ROIs from the Schaefer 2018 parcellation and Destrieux atlas, then applies an Inter-Subject Correlation (ISC) proxy as a final multiplier.

It also outputs a second-by-second timeline of predicted attention and emotional activation — which is probably more informative than the aggregate score.

Who might find this useful:
- **Researchers** prototyping stimulus screening pipelines without running participants in a scanner
- **ML practitioners** curious about what TRIBE v2 outputs look like on real-world content
- **Anyone curious** about how a brain encoding model responds to different video formats

---

## How It Works

### The Model: TRIBE v2

Powered by [`facebook/tribev2`](https://huggingface.co/facebook/tribev2), Meta FAIR's brain encoding model trained on large-scale fMRI data from participants watching naturalistic video. It maps video features to predicted cortical BOLD responses — no scanner required.

### Parcellation

Voxel predictions are aggregated onto the cortical surface using:
- **Schaefer 2018 parcellation** (400 parcels, 7-network variant) for network-level ROIs
- **Destrieux atlas** for subcortical region extraction
- **fsaverage5 surface** as the common reference space

### ISC Multiplier

A variance-based proxy for Inter-Subject Correlation — rough measure of how consistently the predicted responses hold across the cortex. This is an approximation of real ISC, which requires multiple subjects in a scanner.

---

## Brain Score Dimensions

The weights (30/25/25/20) and the specific region choices are mine. The neuroscience behind each region is well-established; the aggregation scheme is not peer-reviewed.

| Dimension | Max Points | Brain Regions | What I'm Trying to Capture |
|---|---|---|---|
| **Sustained Attention** | 30 | PCC, mPFC, Angular Gyrus (Default Mode Network) | DMN suppression during externally-directed attention; DMN synchrony during narrative comprehension |
| **Emotional Memory Encoding** | 25 | Amygdala, Parahippocampal Gyrus, Hippocampus | Limbic circuit co-activation linked in the literature to emotionally-driven episodic encoding |
| **Social Processing** | 25 | pSTS, Fusiform Gyrus, rTPJ | Circuits involved in biological motion, face processing, and mentalising |
| **Sensory Salience / Hook** | 20 | V1-V4, A1, Insula | Early visual/auditory + interoceptive activation — computed on the **first 5-15 seconds only** |
| **ISC Multiplier** | — | Whole cortex | Variance-based proxy for cross-viewer response consistency |

---

## Scientific Basis

The dimension design draws on these papers. The scoring approach is inspired by the neuroscience, not validated against it.

- Hasson et al. (2004). Intersubject synchronization of cortical activity during natural vision. *Science, 303*(5664), 1634-1640.
- Nastase et al. (2019). Measuring shared responses across subjects using intersubject correlation. *Social Cognitive and Affective Neuroscience, 14*(6), 667-685.
- Schaefer et al. (2018). Local-Global Parcellation of the Human Cerebral Cortex. *Cerebral Cortex, 28*(9), 3095-3114.
- TRIBE v2: Meta AI Research — see citation below.

---

## Hardware Requirements

TRIBE v2 requires a CUDA GPU. The free Colab T4 works.

| Environment | Status | Notes |
|---|---|---|
| NVIDIA T4 (Colab free) | Tested | ~20 min per 60s video |
| NVIDIA A100 | Should work | Haven't benchmarked |
| Apple Silicon / CPU | Not supported | No CUDA, no MPS backend |

---

## Quickstart

### Run on this Space

Upload a video using the interface. Processing time depends on video length and the Space's GPU tier.

### Run on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deep0patel/boldsignal/blob/main/boldsignal_batch.ipynb)

The notebook handles everything — install, auth, inference, scoring, and visualisation. Edit the video URL list in Cell 8 and run.

Expected output:

```json
{
  "brain_score": 59.8,
  "isc_multiplier": 1.15,
  "content_mode": "non_narrative",
  "dimensions": {
    "sustained_attention": 19.62,
    "emotional_memory": 11.43,
    "social_processing": 12.5,
    "sensory_salience": 8.43
  }
}
```

---

## Caveats

**Population average, not individual.** TRIBE v2 was trained on group-averaged fMRI data. The score reflects a hypothetical average adult brain — individual variation in real fMRI is large.

**Encoding model, not ground truth.** These are predicted activations, not measured neural data. Don't treat a score as equivalent to running an actual fMRI study.

**The scoring weights are mine.** Dimension weights, normalisation ranges, and the ISC proxy formula are choices I made. Someone with more fMRI experience might make very different ones.

**Scores cluster more than expected.** Across five YouTube Shorts, scores ranged 59-66. Whether that's the format, my normalisation, or something else — I don't know yet.

**No semantics or culture.** TRIBE v2 predicts from visual and auditory features. It doesn't model meaning, cultural context, or audience expectations.

---

## Citation

```bibtex
@inproceedings{tribe2023,
  title     = {TRIBE: A Large-Scale Benchmark for Brain Encoding of Video},
  author    = {Wehbe, Leila and others},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2023}
}
```

```bibtex
@article{hasson2004intersubject,
  title   = {Intersubject synchronization of cortical activity during natural vision},
  author  = {Hasson, Uri and Nir, Yuval and Levy, Ifat and Fuhrmann, Galit and Malach, Rafael},
  journal = {Science},
  volume  = {303},
  pages   = {1634--1640},
  year    = {2004}
}
```

---

*Built with PyTorch, nilearn, and Gradio. TRIBE v2 is by Meta FAIR — BoldSignal wouldn't exist without it.*
