# BoldSignal Brain Encoder — Design Spec
**Date:** 2026-06-03  
**Version:** 1.0  
**Status:** Approved — ready for implementation planning

---

## What We Are Building

A multimodal brain encoding model that takes a video as input and predicts how the human brain responds to it — without any human subjects. The model outputs predicted fMRI activation across 360 brain regions, which map directly to marketing-relevant metrics: attention, emotion, memory, and hook strength.

This is Phase 1 of the BoldSignal product. The model is built from scratch using open-source encoders and public fMRI data. No Meta code, weights, or licenses are used.

---

## Constraints

- **No Meta IP:** No TRIBE v2 code or weights. Architecture is inspired by published neuroscience research, not copied.
- **Commercial-safe licenses only:** All encoders must be MIT or Apache 2.0.
- **Limited data start:** 4-subject CNeuroMod open access (CC0, no DTA required).
- **Single developer, learning while building:** Ship mode — working code first, deep theory second.
- **Mac for development, no CUDA:** All training runs on RunPod cloud GPU.

---

## Data

### Source
**Courtois NeuroMod (CNeuroMod)** — 4 subjects, open access, CC0 license.  
Download via CONP: [portal.conp.ca](https://portal.conp.ca)

Subjects watched:
- Friends TV show, seasons 1–6 (~60h fMRI per subject)
- 4 movies: Bourne Supremacy, Hidden Figures, Wolf of Wall Street, Life

**Total:** ~240 hours of fMRI across 4 subjects. fMRI is pre-preprocessed by the CNeuroMod team using fMRIPrep — no additional preprocessing pipeline required.

### fMRI Processing Steps
```
Raw fMRI (provided, ~90k voxels)
    │
    ▼
Z-score normalize per voxel per run
    │
    ▼
Average into 360 parcels — HCP atlas (nilearn, 3 lines of code)
    │
    ▼
Apply +5 second HRF lag offset (brain responds 5s after stimulus)
    │
    ▼
Resample to 1Hz
    │
    ▼
Save as (time, 360) numpy array — ~2.5GB total, fits in RAM
```

### Train / Validation Split
- Train: earlier seasons/episodes
- Validate: held-out episodes never seen during training
- Never split mid-episode (temporal leakage risk)

---

## Encoders (Feature Extraction)

All frozen — no training. Run once, cache to disk. Never re-run.

| Modality | Model | License | Output dim | Sampling rate |
|----------|-------|---------|-----------|---------------|
| Video | `microsoft/xclip-base-patch32` | MIT | 512 | 2Hz |
| Audio | `microsoft/wavlm-base-plus` | MIT | 768 | 2Hz |
| Text transcript | `BAAI/bge-base-en-v1.5` | MIT | 768 | 2Hz (word-aligned) |
| Transcript generation | `openai/whisper-base` | MIT | — | Used to generate transcript only |

**Why these over the original plan:**
- X-CLIP processes 8 frames with temporal attention — understands motion, not just appearance
- WavLM captures music, sound effects, ambient noise — Whisper is speech-only
- BGE-base consistently top of semantic embedding benchmarks (MTEB leaderboard)
- Whisper retained for speech-to-text only (transcript generation)

**Planned later additions (Phase 2+):**
- Optical flow via RAFT (MIT) — targets MT/V5 motion cortex directly
- Face presence + expression via deepface (MIT) — social circuit signal
- CLAP audio-language (`laion/larger_clap_general`, Apache 2.0) — music valence/arousal
- Luminance + contrast (numpy only) — V1 primary visual cortex

---

## Model Architecture

### Overview
```
Frozen encoders → Project to shared dim → Concat → Transformer → Pool → Subject head → 360 ROI predictions
```

### Trainable Components Only

**Step 1 — Modality projection:**
```python
video_proj: Linear(512 → 128)   # X-CLIP
audio_proj: Linear(768 → 128)   # WavLM
text_proj:  Linear(768 → 128)   # BGE
# Concat → (T, 384)
```

**Step 2 — Cross-modal transformer:**
```
6 layers, 8 attention heads, d_model=384
Learns which combinations of video+audio+text drive brain responses
```

**Step 3 — Temporal pooling:**
```
AdaptiveAvgPool: 2Hz → 1Hz (match fMRI sampling rate)
```

**Step 4 — Subject-conditioned output head:**
```python
SubjectLinear: (T, 384) → (T, 360)
# One weight matrix per subject
# Transformer learns universal patterns
# Head learns each subject's individual brain layout
```

**Total trainable parameters:** ~8M — trains in hours on a single GPU.

### Training Configuration
```
Loss:        MSE (predicted vs real fMRI)
Eval metric: Pearson r per ROI, averaged across ROIs and subjects
Optimizer:   AdamW, lr=1e-4
Schedule:    Linear warmup (10% of steps) → cosine decay
Batch size:  16 windows of 100 seconds
Epochs:      20 max, early stop if val Pearson flat for 3 epochs
Window size: 100s stimulus → ~200 timepoints at 2Hz input, ~100 at 1Hz output
```

### Success Milestones
| Pearson r | Meaning |
|-----------|---------|
| r > 0.0 | Model is learning (beats chance) |
| r > 0.10 | Real signal — better than linear baseline |
| r > 0.15 | Strong MVP — ready to build product layer |

---

## Output: Brain → Marketing Metrics

### ROI Groups (HCP Atlas parcels, mapped via nilearn)
```python
ATTENTION = [V1, V2, V3, MT_V5]           # visual + motion cortex
EMOTION   = [amygdala, PHG, hippocampus]  # emotional memory circuit
MEMORY    = [hippocampus, PHG]            # encoding circuit
HOOK      = [V1, A1, insula]              # early sensory (first 5s only)
SOCIAL    = [pSTS, FFA, rTPJ]            # social cognition
COGNITIVE = [dlPFC]                       # load/confusion (inverse scored)
```

### Scoring Formula (v1)
```python
attention    = normalize(mean(ATTENTION_rois), 0, 100)
emotion      = normalize(mean(EMOTION_rois), 0, 100)
memory       = normalize(mean(MEMORY_rois), 0, 100)
hook         = normalize(mean(HOOK_rois, first_5s_only), 0, 100)
neural_score = (0.40 × attention) + (0.35 × emotion) + (0.25 × memory)
```

### Per-Second Engagement Timeline
```python
# For every 1-second window
engagement[t] = mean(ATTENTION[t] + EMOTION[t]) - COGNITIVE_LOAD[t]
# Drop-offs: timestamps where engagement drops > 1 std dev
# Peak moments: timestamps where engagement is in top 10%
```

### Full Output JSON
```json
{
  "neural_score": 73,
  "attention": 81,
  "emotion": 68,
  "memory": 61,
  "hook": 79,
  "drop_offs": [12, 34],
  "peak_moments": [8, 41],
  "timeline": [[0, 71], [1, 74], [2, 69], "..."]
}
```

---

## Project Structure
```
boldsignal/
├── data/
│   ├── download.py       # pulls CNeuroMod from CONP
│   ├── preprocess.py     # z-score, HRF offset, parcel averaging
│   └── dataset.py        # PyTorch Dataset — 100s windows of stimulus + fMRI
├── features/
│   ├── video.py          # X-CLIP embeddings at 2Hz → cached .npy
│   ├── audio.py          # WavLM embeddings at 2Hz → cached .npy
│   ├── text.py           # Whisper transcript → BGE embeddings → cached .npy
│   └── align.py          # aligns all modalities to fMRI TR
├── model/
│   ├── encoder.py        # cross-modal transformer
│   └── head.py           # subject-conditioned linear → 360 ROI outputs
├── train.py              # training loop + validation
├── evaluate.py           # Pearson r per ROI, averaged
├── score.py              # ROI activations → marketing metrics JSON
├── config.py             # all hyperparameters in one place
└── requirements.txt
```

---

## Infrastructure

| Phase | Compute | Use |
|-------|---------|-----|
| Development | Google Colab free (T4) | Write + debug code, test on small data samples |
| Training | RunPod RTX 3090 (~$0.44/hr) | Feature extraction + full training run |
| After training | Mac | Evaluate results, iterate on score.py |

**Workflow:**
```
Mac (write code) → GitHub → RunPod (train) → download weights → Mac (evaluate)
```

Feature extraction runs once and is cached. After that, re-training costs < $5.

---

## Phase 2 — Suggestions Module (Not in MVP)

After the model is validated:
- Extract scene descriptions from X-CLIP zero-shot classification (no extra model — reuse existing embeddings)
- Combine with: transcript (Whisper), brain scores, weak regions, drop-off timestamps
- Send structured prompt to **DeepSeek API**
- Output: natural language creative suggestions keyed to specific timestamps and brain regions

---

## What This Is NOT

- Not using Meta's TRIBE v2 code or weights
- Not predicting individual brain responses (population average only)
- Not a clinical brain scanner — a content analysis tool
- Not real-time inference (batch job, minutes not milliseconds)

---

## Next Step

Implementation plan — break this into ordered, executable tasks with clear done criteria per task.
