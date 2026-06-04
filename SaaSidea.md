# Neural Ad Scorer — Product Specification
**Version:** 0.1 (Founder Brief)  
**Date:** June 2026  
**Purpose:** Developer handoff document covering product vision, architecture, data strategy, and build roadmap.

---

## 1. What We're Building

A SaaS platform that predicts **how the human brain responds to video/image ads** — without recruiting a single human subject. Brands upload an ad, the model returns a neural engagement scorecard in minutes.

This replaces traditional neuromarketing studies that cost $50k–$200k and take 6–8 weeks. We deliver the same quality of insight in under 5 minutes at software margins.

**Core insight:** Meta FAIR published TRIBE v2 (March 2026), a trimodal (video + audio + text) foundation model that accurately predicts fMRI brain responses to any stimulus. We build a proprietary brain encoding model inspired by this architecture — trained on public fMRI datasets, productized into a developer-friendly API and a marketer-friendly dashboard.

> We are NOT using Meta's TRIBE v2 code or weights (ToS restriction). We build our own model from scratch using public data and open architectures.

---

## 2. The Problem We Solve

### Current neuromarketing workflow:
```
Brand wants to test an ad
  → Hire agency ($50k–$200k per study)
    → Recruit 30–50 subjects
      → EEG caps or fMRI scanners in a lab
        → 6–8 weeks of study + analysis
          → Report delivered AFTER campaign launches
            → Insight is too late to act on
```

### Our workflow:
```
Brand uploads video ad
  → Model runs (< 5 minutes)
    → Neural scorecard delivered
      → Brand edits ad before spend
        → Better ROI
```

---

## 3. Product: Neural Ad Scorer

### 3.1 Core Output — The Neural Scorecard

For every uploaded video ad, the platform returns:

| Output | Description |
|---|---|
| **Attention Curve** | Frame-by-frame predicted visual attention score (0–100) |
| **Emotional Engagement** | Second-by-second emotional activation (positive/negative/neutral) |
| **Memory Encoding Score** | Predicted recall likelihood at end of ad |
| **Drop-off Risk Flags** | Timestamps where neural engagement drops sharply |
| **Peak Moments** | Frames with highest brain activation (use for thumbnails) |
| **Overall Neural Score** | Single 0–100 composite score for the ad |
| **Brain Region Map** | Which cortical areas activate and when (visual for premium tier) |

### 3.2 Secondary Use Case — Package & Image Testing (Product 3 from roadmap)

Same model, same API. Users upload static images (product packaging, logos, landing page screenshots). Returns:
- Attention heatmap overlay
- Predicted visual cortex activation strength
- Comparison scoring across multiple design variants

This requires zero additional model work — it's a subset of the video pipeline.

---

## 4. The Model Architecture

> **Developer Note:** This is the core IP. Build this first before anything else.

### 4.1 What the Model Does

Takes multimodal content (video frames + audio + text transcript) as input and predicts fMRI brain activation patterns as output. The predicted activation patterns map to marketing-relevant metrics (attention, emotion, memory).

### 4.2 High-Level Architecture

```
INPUT
  ├── Video frames (sampled at fMRI TR rate, ~1–2 fps)
  ├── Audio waveform
  └── Text transcript (optional)
         │
         ▼
FEATURE EXTRACTION (pretrained, frozen encoders)
  ├── Vision: ViT or VideoMAE (extract frame embeddings)
  ├── Audio: Whisper encoder (extract audio embeddings)
  └── Text:  LLaMA / sentence-transformers (extract semantic embeddings)
         │
         ▼
TEMPORAL ALIGNMENT
  └── Align all modality embeddings to fMRI TR (repetition time ~1s)
      Upsample/downsample to match fMRI sampling rate
         │
         ▼
CROSS-MODAL TRANSFORMER (this is the core trainable module)
  └── Multi-head attention across modalities and time
      Learns which combinations of video+audio+text matter for brain response
      Architecture: ~6–12 transformer blocks, d_model=512
         │
         ▼
VOXEL PROJECTION HEAD
  └── Linear layer: maps transformer output → brain voxel activations
      Output dim = number of brain ROIs (use HCP 360-parcel atlas)
         │
         ▼
OUTPUT: Predicted activation per brain region per timepoint
  └── Map to marketing metrics:
      - Visual cortex activation → Attention
      - Reward/limbic activation → Emotional engagement  
      - Hippocampal activation  → Memory encoding
      - Prefrontal activation   → Cognitive load
```

### 4.3 Key Technical Details

**Pretrained encoders to use (all open source):**
- Vision: `facebook/videomae-base` or `openai/clip-vit-base-patch32`
- Audio: `openai/whisper-base` (encoder only, no decoder needed)
- Text: `sentence-transformers/all-MiniLM-L6-v2` or LLaMA 3 8B

**The trainable part:**  
Only the cross-modal transformer and voxel projection head are trained. Pretrained encoders are frozen. This dramatically reduces compute requirements.

**fMRI temporal alignment:**
- fMRI TR = ~1–2 seconds
- Video is 25fps → downsample embeddings to 1fps for alignment
- Audio is continuous → chunk into 1s windows, extract embeddings per window
- Apply HRF (hemodynamic response function) convolution to model the ~5s brain delay

**Loss function:**
```python
def pearson_loss(pred, target):
    pred_z = pred - pred.mean(dim=-1, keepdim=True)
    tgt_z  = target - target.mean(dim=-1, keepdim=True)
    return -(pred_z * tgt_z).sum(-1) / (pred_z.norm(-1) * tgt_z.norm(-1) + 1e-8).mean()
```

**Evaluation metric:** Pearson correlation per brain ROI, averaged across held-out subjects.

---

## 5. Training Data

> **No proprietary data needed. All datasets are public.**

### 5.1 Primary Training Datasets

| Dataset | Subjects | Hours fMRI | Modalities | Access |
|---|---|---|---|---|
| CNeuroMod | 6 (deep) | ~268h | Video + Audio + Text | Public, DUA required |
| NSD (Natural Scenes Dataset) | 8 (deep) | ~30h | Images | Public |
| Narratives | 345 | ~147h | Audio + Text | Public (OpenNeuro) |
| BoldMoments | 10 | ~62h | Video + Audio | Public |
| HCP (Human Connectome Project) | 1,200 | ~178h | Multi | Public, DUA required |

**Start with CNeuroMod + Narratives.** They cover video+audio+text and have enough depth per subject to train a meaningful model.

### 5.2 Data Access Process

1. Register at OpenNeuro.org — most datasets available immediately
2. HCP requires signing a Data Use Agreement at db.humanconnectome.org
3. CNeuroMod: courtois.neuro.polymtl.ca — requires DUA
4. Budget ~2–4 weeks for approvals

### 5.3 Preprocessing Pipeline

Use **fMRIPrep** (standard tool, Docker-based) for all datasets:
```bash
docker run -it --rm \
  -v /data/raw:/data:ro \
  -v /data/output:/out \
  nipreps/fmriprep:latest \
  /data /out participant
```

Steps fMRIPrep handles automatically:
- Motion correction
- Slice timing correction  
- Distortion correction
- Registration to MNI space
- Surface projection (for HCP parcellation)

After fMRIPrep, apply:
- Z-score normalization per voxel per run
- HCP 360-parcel atlas averaging (reduces voxel space from ~90k → 360 ROIs)

---

## 6. Mapping Brain Regions to Marketing Metrics

This is the translation layer from neuroscience to business value:

| Brain Region (HCP Atlas) | What it signals | Marketing Metric |
|---|---|---|
| V1, V2, V3 (early visual) | Basic visual processing | Is the visual being seen? |
| MT/V5 (motion area) | Motion detection | Is motion engaging? |
| FFA (fusiform face area) | Face processing | Face in ad activating? |
| Reward circuit (NAcc, vmPFC) | Desire, positive valence | Emotional pull |
| Amygdala | Fear, arousal, salience | Emotional intensity |
| Hippocampus | Memory encoding | Will they remember this? |
| dlPFC (prefrontal) | Cognitive load, working memory | Is the ad confusing? |
| Broca's area (IFG) | Language processing | Is the message landing? |
| TPJ | Social cognition, narrative | Story resonance |

**The scoring formula (v1, can be refined with customer feedback):**

```
Attention Score    = mean(V1 + V2 + MT activation) normalized 0–100
Emotional Score    = mean(reward circuit + amygdala) normalized 0–100  
Memory Score       = hippocampal activation normalized 0–100
Cognitive Load     = dlPFC activation (high = confusing, inverse scored)
Neural Score       = 0.4×Attention + 0.35×Emotional + 0.25×Memory
```

---

## 7. Product Tiers & Pricing

| Tier | Price | Features |
|---|---|---|
| **Starter** | $199/mo | 10 video analyses/mo, scorecard only, no brain maps |
| **Growth** | $799/mo | 50 analyses/mo, full scorecard + brain region breakdown |
| **Pro** | $2,499/mo | Unlimited analyses, API access, A/B variant comparison, team seats |
| **Enterprise** | Custom | White-label, dedicated model, SLA, custom metrics |

**Per-analysis pricing (for agencies):** $49 per video analysis, no subscription.

---

## 8. Technical Stack Recommendation

### ML / Model
- **Language:** Python 3.11+
- **Framework:** PyTorch 2.x
- **Transformer:** HuggingFace Transformers + custom cross-modal layers
- **fMRI tools:** nibabel, nilearn, fMRIPrep
- **Experiment tracking:** Weights & Biases
- **Model serving:** TorchServe or Triton Inference Server

### Backend / API
- **Framework:** FastAPI (Python)
- **Queue:** Celery + Redis (video processing is async, not instant)
- **Storage:** S3 or GCS for video uploads + model outputs
- **Database:** PostgreSQL (users, jobs, results)
- **Auth:** Clerk or Auth0

### Frontend / Dashboard
- **Framework:** Next.js 14+ (App Router)
- **Charts:** Recharts or D3.js (for attention curves, brain maps)
- **Brain visualization:** Nilearn's JavaScript export OR a custom Three.js brain renderer
- **Styling:** Tailwind CSS

### Infrastructure
- **GPU training:** A100 or H100 on Lambda Labs / RunPod (cheaper than AWS for training)
- **GPU inference:** T4 or A10G on AWS/GCP (1 analysis ~ 10–30 seconds on T4)
- **Deployment:** Docker + Kubernetes or Railway for MVP

---

## 9. Build Phases

### Phase 0 — Learning (Month 1–2)
**Owner: Founder**
- Learn PyTorch fundamentals
- Learn transformer architecture from scratch
- Build toy multimodal fusion model on NSD dataset (images → fMRI)
- Read: Karpathy's nanoGPT, Sebastian Raschka's LLMs from Scratch
- **Milestone:** Predict fMRI from images with above-chance Pearson correlation

### Phase 1 — Core Model (Month 3–4)
**Owner: ML Engineer**
- Download + preprocess CNeuroMod and Narratives datasets
- Run fMRIPrep preprocessing pipeline
- Build feature extraction pipeline (ViT + Whisper + sentence-transformers)
- Build temporal alignment module (embeddings → fMRI TR)
- Train cross-modal transformer on CNeuroMod
- **Milestone:** Pearson r > 0.15 on held-out subjects (beats linear baseline)

### Phase 2 — Marketing Metric Layer (Month 4–5)
**Owner: ML Engineer + Founder**
- Define ROI → metric mapping (section 6 above)
- Build scoring formula on top of model outputs
- Validate against known neuromarketing results (faces activate FFA, etc.)
- Build internal demo: upload video → get attention curve
- **Milestone:** Demo that produces interpretable, directionally correct scorecards

### Phase 3 — API + Dashboard MVP (Month 5–6)
**Owner: Full-Stack Developer**
- FastAPI backend: upload endpoint, async job queue, results endpoint
- Simple Next.js dashboard: upload UI, scorecard display, attention curve chart
- Auth + billing (Stripe)
- Deploy to cloud with GPU inference
- **Milestone:** 3 beta customers uploading real ads

### Phase 4 — Product Polish + GTM (Month 6–8)
- Brain map visualization (3D cortex renderer)
- A/B variant comparison (upload 2 ads, compare scores)
- PDF report export
- Agency API with bulk pricing
- **Milestone:** $10k MRR

---

## 10. Go-To-Market

### Who to sell to first
**Tier 1 targets (easiest yes):**
- Digital advertising agencies (they test many ads, high volume)
- YouTube creators with 100k+ subscribers (obsessed with retention)
- E-commerce brands running video ads on Meta/TikTok

**Tier 2 targets (bigger contracts):**
- CPG brands (Unilever, P&G, Nestlé) — they spend millions on ad testing
- TV/streaming content studios
- Political campaign ad teams

### Sales motion
- Cold outreach to agency creative directors with a free analysis of one of their live ads
- Show them the attention curve and say "here's where your ad loses the viewer"
- The demo IS the pitch — no deck needed

### Positioning
> "Test your ad on a digital brain before spending a dollar on media."

Competitive differentiators vs Neurons Inc, Nielsen Neuro:
- 10 minutes vs 6–8 weeks
- $49 per analysis vs $50k per study
- No human subjects needed
- API-first for programmatic integration

---

## 11. Key Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Model accuracy not good enough initially | Start with directional insights, not clinical precision. Marketers don't need r=0.9, they need "scene 3 is weaker than scene 7" |
| fMRI data access takes too long | Start NSD application immediately (fastest approval), begin building pipeline while waiting |
| Competitors copy the model | Speed to market + proprietary training data collected from early customers (with consent) |
| Regulatory concerns about "reading minds" | Frame as content analysis, not personal brain scanning. No individual user data ever collected |
| GPU inference costs too high | Optimize model to run on T4 (~$0.35/hr). One analysis = ~30s = < $0.01 compute cost |

---

## 12. Key Papers to Read (for Developer Context)

1. **TRIBE v2** — d'Ascoli et al., Meta FAIR, March 2026. The direct inspiration.  
   `https://github.com/facebookresearch/tribev2` (reference only, not for use)
2. **Attention Is All You Need** — Vaswani et al., 2017. The transformer paper.
3. **A shared neural encoding model for the human brain** — Huth et al., 2016. How language maps to brain regions.
4. **Natural Scenes Dataset** — Allen et al., 2022. The training dataset you'll start with.
5. **Algonauts 2025 Challenge** — Gifford et al. Best practices from the brain encoding competition.

---

## 13. Glossary (for Non-Neuroscientist Developers)

| Term | Plain English |
|---|---|
| fMRI | Brain scanner that measures blood flow as a proxy for neural activity |
| Voxel | 3D pixel of brain data (~2mm³ cube) |
| BOLD signal | The blood-oxygen signal fMRI actually measures |
| HRF | Hemodynamic Response Function — the ~5s delay between neural event and fMRI signal |
| TR | Repetition Time — how often the fMRI takes a brain snapshot (typically 1–2 seconds) |
| ROI | Region of Interest — a named brain area |
| Encoding model | A model that predicts brain activity FROM a stimulus (our direction) |
| Decoding model | A model that reconstructs a stimulus FROM brain activity (opposite direction, not what we build) |
| HCP parcellation | A standard atlas that divides the brain into 360 named regions |
| Pearson r | Correlation coefficient — our main accuracy metric (0 = chance, 1 = perfect) |
| Noise ceiling | The theoretical maximum Pearson r achievable given measurement noise |

---

*Document prepared by founder. For questions contact the founding team.*  
*Next review: after Phase 1 milestone.*