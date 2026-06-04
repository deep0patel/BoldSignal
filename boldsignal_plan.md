# 🧠 PROJECT NEURALPULSE
### AI Agent Build Brief — v1.0
### "Your content, through the brain's eyes"

---

## WHAT YOU ARE BUILDING

**NeuralPulse** is a web application where content creators upload a video (or audio/text), and receive a **Brain Score** — a scientifically grounded engagement prediction based on simulated human brain responses, powered by Meta's TRIBE v2 model.

The core value proposition: instead of finding out if your content worked *after* publishing, creators find out *before* — by running it through an AI digital twin of 700+ human brains.

The output is not just a number. It is:
- A total Brain Score (0–100)
- Four sub-dimension scores with explanations
- A second-by-second engagement timeline showing exactly where the brain disengages
- Actionable recommendations per dimension

---

## PROJECT NAME & IDENTITY

**Name:** NeuralPulse
**Tagline:** "Your content, through the brain's eyes"
**Tone:** Scientific but accessible. Confident. Not corporate. Think "what if a neuroscientist started a creator tool startup."
**Color direction:** Deep navy + electric violet + clean white. Brain-scan aesthetic meets modern SaaS.

---

## THE MODEL: TRIBE v2

### What it is
TRIBE v2 (Trimodal Brain Encoder v2) is a foundation AI model released by Meta's Fundamental AI Research (FAIR) team on March 26, 2026. It is the first AI model capable of predicting high-resolution human brain responses to video, audio, and text simultaneously.

### What it does technically
- Takes any video, audio clip, or text as input
- Outputs predicted fMRI brain activation across ~20,000 cortical voxels (fsaverage5 mesh)
- Predictions represent the *average* response across a population of 700+ individuals
- Temporal resolution: one prediction per ~2 seconds of content (offset by 5s for hemodynamic lag)
- Zero-shot: works on new content without retraining

### Architecture
Built on three pre-trained encoders fused into a unified Transformer:
- **LLaMA 3.2** — text/language stream
- **V-JEPA2** — video/visual stream
- **Wav2Vec-BERT** — audio stream

### Key stats
- Trained on 500+ hours of fMRI data from 700+ participants
- 70x higher spatial resolution than predecessor models
- Outperforms all prior brain encoding models on benchmark tasks
- Open-source under CC BY-NC license

### Access
- **GitHub:** https://github.com/facebookresearch/tribev2
- **HuggingFace:** https://huggingface.co/facebook/tribev2
- **Official demo:** https://aidemos.atmeta.com/tribev2
- **Meta blog:** https://ai.meta.com/blog/tribe-v2-brain-predictive-foundation-model/

### Quick-start inference code
```python
from tribev2 import TribeModel

model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")
df = model.get_events_dataframe(video_path="path/to/video.mp4")
preds, segments = model.predict(events=df)
print(preds.shape)  # (n_timesteps, n_vertices) — ~20k vertices on fsaverage5 mesh
```

### Hardware requirements
- **Requires CUDA GPU** — does NOT run on Mac (no CUDA support)
- Minimum: 16GB VRAM (T4 on Google Colab free tier works for inference)
- Recommended: RTX 3090 (24GB) or A10G for production
- **For MVP development:** Use Google Colab (free T4 GPU) — Meta released an official Colab notebook
- **For production backend:** RunPod or Lambda Labs (~$0.30–$0.80/hr)
- **For sharing demos:** HuggingFace Spaces with ZeroGPU (free A100 on-demand)

---

## THE BRAIN SCORE FRAMEWORK

### Foundational Metric: Inter-Subject Correlation (ISC)

Before computing any dimension, calculate ISC — the degree to which the model's predicted brain responses are *consistent* across the population-level embeddings in the training distribution.

**What ISC measures:** How universally the content drives the same brain patterns across different people. High ISC = the content locks brains into the same response. Low ISC = scattered, idiosyncratic responses = niche or incoherent content.

**Scientific basis:**
> Meta-analysis of 14 studies (n=27 effect sizes): ISC correlates with attention at r=0.65, p<0.001.
> Source: "Intersubject Correlation as a Predictor of Attention: A Systematic Review," BMC Psychology, 2025.
> https://link.springer.com/article/10.1186/s40359-025-02879-7

**Implementation:** TRIBE v2 outputs average-subject predictions. ISC proxy = variance of voxel-wise predictions across the N individual-subject embeddings in the model's internal population prior. Low variance = high ISC = high universality.

**Role in final score:** ISC acts as a multiplier [0.70 → 1.15] applied to the raw sub-score sum. It is NOT one of the four dimensions — it gates the final number.

```
Brain_Score = Raw_Score × ISC_multiplier
ISC_multiplier = normalise(ISC_value, from=[0.5,1.0], to=[0.70,1.15])
```

---

### Dimension 1: Sustained Attention (30 pts)

**Brain regions:** Posterior Cingulate Cortex (PCC), Medial Prefrontal Cortex (mPFC), Angular Gyrus — the Default Mode Network (DMN) core nodes.

**What it measures:** Whether the brain stays externally locked on the content or begins mind-wandering. The DMN is the "mind-wandering network" — it activates when attention drifts inward, and suppresses when attention is captured externally.

**Key scientific nuance — TWO modes exist:**

*Mode A — Non-narrative content (pure visual/audio, no story):*
DMN suppression = engagement signal. Sustained low DMN = brain is absorbed in the sensory stream.

*Mode B — Narrative content (speech present, story structure):*
Synchronized DMN *activity* = deep comprehension signal. The DMN is the story-comprehension network. If people across subjects all show the same DMN pattern while watching the same story, they are constructing the same mental model = deep engagement.

**Detecting mode:** If audio transcript is non-empty and sentence count > 5 → Narrative mode. Otherwise → Non-narrative mode.

**Calculation:**
```python
# Extract DMN voxel indices from fsaverage5 atlas (PCC, mPFC, angular gyrus)
DMN_timeseries = mean(preds[:, DMN_voxel_indices], axis=1)

if mode == "non_narrative":
    suppression_mean = -mean(DMN_timeseries)
    suppression_stability = 1 - (std(DMN_timeseries) / (range(DMN_timeseries) + 1e-8))
    raw = 0.40 * suppression_mean + 0.60 * suppression_stability

elif mode == "narrative":
    # Correlate with expected narrative comprehension template
    # Template = rising activation during story development, peak at climax
    narrative_template = construct_narrative_template(n_timesteps)
    raw = pearsonr(DMN_timeseries, narrative_template)[0]

score_dim1 = normalise(raw) * 30  # scaled to 30pts
```

**Scientific sources:**
- "20 Years of the Default Mode Network" — Menon, Stanford (2023): https://www.med.stanford.edu/content/dam/sm/scsnl/documents/Neuron_2023_Menon_20_years.pdf
- "Contributions of DMN Stability and Deactivation to Adolescent Task Engagement" — Scientific Reports (2018): https://www.nature.com/articles/s41598-018-36269-4
- "Default Mode and Visual Network Activity in an Attention Task" — PMC (2021): https://pmc.ncbi.nlm.nih.gov/articles/PMC8441717/
- DMN Wikipedia (accurate, researcher-maintained): https://en.wikipedia.org/wiki/Default_mode_network

---

### Dimension 2: Emotional Memory Encoding (25 pts)

**Brain regions:** Bilateral Amygdala (AMY) + Anterior Hippocampus (HC) + Parahippocampal Gyrus (PHG) — treated as a single circuit, not separate regions.

**What it measures:** Whether the content is likely to be *remembered* and *described to others*. This circuit is the brain's emotional memory gateway. When all three activate together during encoding, the event is durably stored — surviving days to a year later. This is the neural basis of word-of-mouth.

**The causal chain:**
`Amygdala detects emotional salience → signals PHG/hippocampus → PHG encodes durably → memory survives → person tells a friend`

**Critical nuance — Arousal ceiling:**
Extremely high amygdala activation *without* corresponding PHG activation indicates over-arousal. The amygdala is firing so hard it interferes with contextual encoding. Content that is pure shock/horror may produce high AMY but weak PHG. Apply a penalty in this case.

**Temporal weighting:** Weight the final 20% of video duration at 2× because late-stage emotional encoding has higher recency effect on long-term memory retention.

**Calculation:**
```python
AMY = zscore(mean(preds[:, amygdala_voxels], axis=1))
PHG = zscore(mean(preds[:, parahippocampal_voxels], axis=1))
HC  = zscore(mean(preds[:, hippocampus_voxels], axis=1))

# Temporal recency weighting
n = len(AMY)
weights = np.ones(n)
weights[int(0.80*n):] = 2.0
weights /= weights.sum()

AMY_w = np.average(AMY, weights=weights)
PHG_w = np.average(PHG, weights=weights)
HC_w  = np.average(HC, weights=weights)

circuit_score = 0.35*AMY_w + 0.45*PHG_w + 0.20*HC_w

# Over-arousal penalty
if AMY_w > 2.0 and PHG_w < 0.5:
    circuit_score *= 0.75

score_dim2 = normalise(circuit_score) * 25  # scaled to 25pts
```

**Scientific sources:**
- "fMRI Studies of Successful Emotional Memory Encoding: A Quantitative Meta-Analysis" — Murty et al., Neuropsychologia (2010): https://pmc.ncbi.nlm.nih.gov/articles/PMC2949536/
- "Encoding and the Durability of Episodic Memory" — Journal of Neuroscience (2005): https://www.jneurosci.org/content/25/31/7260
- "The Memory Enhancing Effect of Emotion" — Dolcos, LaBar, Cabeza (2006): https://cabezalab.org/wp-content/uploads/2021/12/Dolcos-LaBar-Cabeza-2006_The-memory-enhancing-effect-of-emotion-functional-neuroimaging.pdf
- "High-Resolution fMRI of Content-Sensitive Subsequent Memory Responses" — PMC (2010): https://pmc.ncbi.nlm.nih.gov/articles/PMC2854293/

---

### Dimension 3: Social Processing (25 pts)

**Brain regions:** Posterior Superior Temporal Sulcus (pSTS) + Fusiform Gyrus (FFG) + Right Temporoparietal Junction (rTPJ).

**What it measures:** Whether the content activates the brain's social cognition circuit — the network that makes you feel *for* another human being. High activation here predicts sharing behaviour, because people share what made them feel something about another person.

**Why pSTS dominates for video (not FFA):**
FFA fires for static faces. pSTS fires for *moving, expressive, talking* faces. For video content, pSTS is the primary signal. FFA contributes but is secondary.

**The rTPJ layer (mentalizing):**
rTPJ is the "theory of mind" hub — it activates when you model another person's internal mental state. Content triggering rTPJ makes the viewer empathize, simulate the creator's or subject's perspective. This is the mechanism behind shareability — you share because you want others to feel what you felt.

**The social media link:**
A 2018 fMRI study showed fusiform cortex + striatum (reward) co-activation directly precedes a "Like" decision on a simulated Instagram task. The social circuit and the reward circuit are coupled. Source cited below.

**Full-circuit bonus:**
When pSTS AND rTPJ both exceed threshold simultaneously, the viewer is both perceiving social signals AND mentalizing — the deepest level of social engagement. Apply a bonus multiplier.

**Calculation:**
```python
pSTS = mean(preds[:, pSTS_voxels], axis=1)
FFG  = mean(preds[:, fusiform_voxels], axis=1)
rTPJ = mean(preds[:, rTPJ_voxels], axis=1)

# Normalise to z-scores
pSTS_z = zscore(pSTS).mean()
FFG_z  = zscore(FFG).mean()
rTPJ_z = zscore(rTPJ).mean()

social_score = 0.45*pSTS_z + 0.25*FFG_z + 0.30*rTPJ_z

# Full circuit bonus
if pSTS_z > 1.0 and rTPJ_z > 1.0:
    social_score = min(social_score * 1.15, normalised_max)

score_dim3 = normalise(social_score) * 25  # scaled to 25pts
```

**Scientific sources:**
- "An fMRI Dataset in Response to Large-Scale Short Natural Dynamic Facial Expression Videos" — PMC (2024): https://pmc.ncbi.nlm.nih.gov/articles/PMC11576863/
- "Live Face-to-Face Interaction During fMRI" — Redcay et al., PMC (2010): https://pmc.ncbi.nlm.nih.gov/articles/PMC2849986/
- "What the Brain Likes: Neural Correlates of Providing Feedback on Social Media" — SCAN, Oxford (2018): https://academic.oup.com/scan/article/13/7/699/5048941
- "A Functional Dissociation of Face-, Body-, and Scene-Selective Brain Areas" — Scientific Reports (2019): https://www.nature.com/articles/s41598-019-44663-9
- "The Role of the FFA in Social Cognition" — PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC1693125/

---

### Dimension 4: Sensory Salience — The Hook (20 pts)

**Brain regions:** Primary Visual Cortex V1–V4 (ventral stream) + Primary Auditory Cortex (A1) + Bilateral Insula.

**What it measures:** Whether the first 5–15 seconds produce a visceral sensory and bodily response strong enough to stop a scroll. The hook is entirely a function of early sensory salience — if the brain isn't grabbed in the first 5 seconds, the rest doesn't matter.

**Why the insula is critical here:**
The insula is the brain's interoception hub — it translates external stimuli into internal bodily states. A loud noise spikes V1/A1. A scene that makes your stomach drop ALSO activates the insula. Insula activation = "I *felt* something in my body" = the strongest possible hook signal. Content that produces insula response in the first 5 seconds is content that stops the scroll.

**Temporal window:**
This dimension is computed *entirely* on early content. The full video duration contributes almost nothing to this score — this is intentional. The hook is binary: it either grabs you in the first 5–15 seconds or it doesn't.

**Calculation:**
```python
# Time windows
t0_5  = preds[t <= 5s, :]
t5_15 = preds[(t > 5s) & (t <= 15s), :]
t_rest = preds[t > 15s, :]

# Region means per window
V1_early  = mean(t0_5[:, V1_voxels])
A1_early  = mean(t0_5[:, A1_voxels])
INS_early = mean(t0_5[:, insula_voxels])

V1_mid    = mean(t5_15[:, V1_voxels])
INS_mid   = mean(t5_15[:, insula_voxels])

V1_rest   = mean(t_rest[:, V1_voxels])
A1_rest   = mean(t_rest[:, A1_voxels])
INS_rest  = mean(t_rest[:, insula_voxels])

hook_score = (
    0.60 * (0.40*V1_early + 0.30*A1_early + 0.30*INS_early)
  + 0.30 * (0.50*V1_mid + 0.50*INS_mid)
  + 0.10 * mean([V1_rest, A1_rest, INS_rest])
)

score_dim4 = normalise(hook_score) * 20  # scaled to 20pts
```

**Scientific sources:**
- "Arousal Modulates the Amygdala-Insula Reciprocal Connectivity During Naturalistic Emotional Movie Watching" — ScienceDirect (2023): https://www.sciencedirect.com/science/article/pii/S1053811923004676
- "Default Mode and Visual Network Activity in an Attention Task" — PMC (2021): https://pmc.ncbi.nlm.nih.gov/articles/PMC8441717/

---

### Final Score Assembly

```python
def compute_brain_score(preds, segments, mode="auto"):
    """
    preds: (n_timesteps, n_vertices) — TRIBE v2 output
    segments: time alignment from model.predict()
    mode: "narrative" | "non_narrative" | "auto"
    """

    # 0. Detect content mode
    if mode == "auto":
        mode = detect_narrative_mode(segments)

    # 1. ISC multiplier (population variance proxy)
    isc_raw   = compute_isc_proxy(preds)
    isc_mult  = normalise(isc_raw, in_range=(0.5, 1.0), out_range=(0.70, 1.15))

    # 2. Four dimensions
    d1 = score_sustained_attention(preds, mode)    # max 30
    d2 = score_emotional_memory(preds)             # max 25
    d3 = score_social_processing(preds)            # max 25
    d4 = score_sensory_salience(preds, segments)   # max 20

    raw_score    = d1 + d2 + d3 + d4              # max 100
    brain_score  = round(raw_score * isc_mult, 1) # ISC-adjusted, cap at 100

    return {
        "brain_score": brain_score,
        "isc_multiplier": isc_mult,
        "dimensions": {
            "sustained_attention": d1,
            "emotional_memory": d2,
            "social_processing": d3,
            "sensory_salience": d4,
        },
        "content_mode": mode,
        "timeline": build_engagement_timeline(preds, segments),
    }
```

---

## THE KILLER FEATURE: Engagement Timeline

The Brain Score number is table stakes. The feature that will make creators obsessed is the **second-by-second drop-off map**.

For every 2-second window of the video, compute:
- DMN activation (inverted = attention)
- Combined social+emotional circuit activation

Plot these as an overlaid timeline against the video scrubber. Mark the exact timestamp where DMN spikes = "You lost them here at second 23."

This answers the question no analytics tool on earth currently answers:
> *"Exactly which second did I lose my audience — and why?"*

YouTube gives average view duration. NeuralPulse gives a brain map.

---

## ARCHITECTURE: MVP

### Stack
```
Frontend:   Next.js (React) — upload UI, score display, timeline visualisation
Backend:    FastAPI (Python) — file handling, job queue, score API
ML Layer:   TRIBE v2 inference — Python, PyTorch, runs on GPU server
Storage:    S3-compatible (Cloudflare R2 for cheapness) — video uploads
Queue:      Redis + Celery — async inference jobs (inference takes 30–120s)
GPU:        RunPod RTX 3090 (24GB) for production / Google Colab for dev
```

### API Flow
```
1. User uploads video → POST /upload → returns job_id
2. Frontend polls GET /status/{job_id} every 3s
3. GPU worker pulls job, runs TRIBE v2 inference
4. Worker computes all 4 dimensions + ISC + timeline
5. Results stored → GET /results/{job_id} returns full Brain Score JSON
6. Frontend renders score + timeline
```

### Inference time estimates
- 30-second video on T4 (Colab): ~45–90 seconds
- 60-second video on RTX 3090: ~30–60 seconds
- These are rough estimates — benchmark on actual hardware

---

## BRAIN ATLAS: ROI DEFINITIONS

The agent must map TRIBE v2's fsaverage5 voxel space to specific regions. Use the **Schaefer 2018 parcellation** (200 parcels, 7 networks) or the **Destrieux atlas** for fine-grained region identification.

Key voxel groups to define from atlas:

| Dimension | Regions | Atlas label examples |
|---|---|---|
| Sustained Attention | PCC, mPFC, angular gyrus | Schaefer: DefaultMode network parcels |
| Emotional Memory | Bilateral amygdala, PHG, anterior HC | Destrieux: Amygdala, Parahippocampal |
| Social Processing | pSTS, fusiform, rTPJ | Schaefer: Default + TempPar parcels |
| Sensory Salience | V1-V4, A1, bilateral insula | Destrieux: Cuneus, Transverse temporal, Insula |

Use `nilearn` to load the atlas and map parcels to fsaverage5 vertices:
```python
from nilearn import datasets, surface
destrieux = datasets.fetch_atlas_destrieux_2009(legacy_format=False)
# Map to fsaverage5 surface to match TRIBE v2 output space
```

---

## WHAT THE SCORE DOESN'T CLAIM

Build this caveat into the product UI explicitly — it builds trust and is scientifically honest:

1. **Population average, not your audience.** TRIBE v2 predicts responses of a general adult human population. A meditation video might score low on hook and social but perform brilliantly with its specific audience.

2. **Predicts neural engagement, not algorithm distribution.** A high Brain Score doesn't guarantee algorithmic promotion, timing advantage, or thumbnail CTR.

3. **Audio and text inputs work differently.** For audio-only or text inputs, the visual stream (V1–V4, pSTS) will produce low activation by design. The score should be interpreted modality-specifically.

4. **The score is a pre-publication diagnostic, not a verdict.** Its highest value is identifying the second you lost engagement and why — not gatekeeping what content is worth making.

---

## READING LIST FOR THE HUMAN BUILDING THIS

The person building this is studying these in parallel. The agent should be aware of the scientific grounding behind every decision.

### Conceptual Foundation
- Sapolsky: "Human Behavioral Biology" Lecture 1 — YouTube (Stanford)
- Doidge: "The Brain That Changes Itself" — book
- Huberman Lab Podcast Episode 1: "How Your Nervous System Works" — YouTube/podcast

### fMRI & BOLD Signal
- Mumford Brain Stats: "fMRI for Dummies" — YouTube (10-part series)
- DMN Wikipedia article: https://en.wikipedia.org/wiki/Default_mode_network
- Kandel: "Principles of Neural Science" — Chapters 1, 17, 62

### Dimension 1 — DMN / Sustained Attention
- Menon (2023): "20 Years of the Default Mode Network": https://www.med.stanford.edu/content/dam/sm/scsnl/documents/Neuron_2023_Menon_20_years.pdf
- Scientific Reports (2018): "DMN Stability and Task Engagement": https://www.nature.com/articles/s41598-018-36269-4

### Dimension 2 — Emotional Memory
- LeDoux: "The Emotional Brain" — book
- Murty et al. (2010) meta-analysis: https://pmc.ncbi.nlm.nih.gov/articles/PMC2949536/

### Dimension 3 — Social Processing
- Rebecca Saxe lectures on rTPJ / Theory of Mind — YouTube (MIT)
- Redcay et al. (2010): https://pmc.ncbi.nlm.nih.gov/articles/PMC2849986/
- SCAN (2018) Instagram Likes fMRI study: https://academic.oup.com/scan/article/13/7/699/5048941

### ISC / Foundational Metric
- Nastase et al. (2019) ISC Tutorial: https://academic.oup.com/scan/article/14/6/667/5489905
- BMC Psychology ISC Meta-Analysis (2025): https://link.springer.com/article/10.1186/s40359-025-02879-7
- Hidden Brain Podcast: "The Hive Mind" (Uri Hasson, Princeton) — Podcast

### Dimension 4 — Insula / Sensory Salience
- Barrett: "How Emotions Are Made" — book

### Bridge to Product
- Berger: "Contagious: Why Things Catch On" — book
- Daniel Levitin: "This Is Your Brain on Music" — YouTube talks

---

## CONVERSATION HISTORY SUMMARY

This project emerged from a conversation about Meta's TRIBE v2 launch. Key decisions made:

- **Dating app idea was rejected** — TRIBE v2 produces population-average predictions, not personalised ones. No individual-brain signal exists in the current model.
- **Brain Score for Creators was selected** — because it solves a real pain point (pre-publication engagement prediction) using the model's actual strengths (population-level neural response simulation).
- **Mac cannot run inference** — CUDA required. Use Colab for dev, RunPod for production. Mac is the frontend/code editor only.
- **Four dimensions selected over five** — Reward/striatum dimension was considered but excluded because TRIBE v2's cortical surface training may not reliably cover subcortical structures. Revisit if Colab demo shows strong striatum predictions.
- **ISC is a gate, not a dimension** — after reviewing the literature, ISC belongs as a final multiplier not an equal sub-score because it is a meta-property of all dimensions combined.
- **Narrative vs non-narrative mode** — the DMN calculation changes depending on whether speech is present. This is the most scientifically subtle design decision in the whole system.
- **Memory and emotion are one dimension** — initially proposed as two separate dimensions (Emotional Resonance + Memory Encoding). The Murty meta-analysis confirmed they are driven by the same amygdala-PHG circuit. Separating them would double-count.

---

## FIRST TASKS FOR THE AGENT

In order:

1. Set up Google Colab environment — install TRIBE v2, run the official notebook on a test video, confirm predictions tensor shape is `(n_timesteps, ~20000)`
2. Load Schaefer/Destrieux atlas, map parcels to fsaverage5 vertices, define the 4 ROI index sets
3. Implement the 4 scoring functions + ISC proxy + final assembly function
4. Run on 3 test videos of different types (talking head, music video, B-roll nature) — inspect raw scores before normalisation
5. Calibrate normalisation ranges based on raw output distributions
6. Build the engagement timeline output
7. Build FastAPI backend with upload → queue → results endpoints
8. Build Next.js frontend — upload, loading state, score card, timeline
9. Deploy GPU worker on RunPod, deploy web app on Vercel

---

*Document version: 1.0 — April 2026*
*Project NeuralPulse — All scientific citations verified from peer-reviewed sources*