# Generative AI Models vs. Feature Extractors
### Why You Can't Swap Gemma 4 Directly into TRIBE v2 — and How to Fix It

---

## Table of Contents

1. [What Is a Generative AI Model?](#1-what-is-a-generative-ai-model)
2. [What Is a Feature Extractor?](#2-what-is-a-feature-extractor)
3. [Key Architectural Differences](#3-key-architectural-differences)
4. [Why Gemma 4 Fails the TRIBE v2 Interface Contract](#4-why-gemma-4-fails-the-tribe-v2-interface-contract)
5. [How to Use a Generative Model as a Feature Extractor](#5-how-to-use-a-generative-model-as-a-feature-extractor)
6. [Python: Hooking Intermediate Layers with `register_forward_hook`](#6-python-hooking-intermediate-layers)
7. [When to Use Which](#7-when-to-use-which)

---

## 1. What Is a Generative AI Model?

A **generative AI model** is trained to produce new content — tokens, images, audio — that resembles its training distribution. The defining characteristic is that the model's *output* is a **prediction of the next element in a sequence**, not a fixed-size representation of an input.

### How It Works

Most generative language models (GPT family, Gemma, LLaMA, Mistral) are **autoregressive decoders**:

1. The input prompt is tokenized into a sequence of token IDs.
2. At each step, the model attends **only to previous tokens** (causal / masked self-attention).
3. The final layer projects the last hidden state through a **language-model head** (a linear layer of size `d_model → vocab_size`) to produce a probability distribution over the vocabulary.
4. The highest-probability token is sampled (or beam-searched) and appended to the sequence. Repeat.

```
Input:  [The   quick   brown   fox]
         ↓       ↓       ↓       ↓
        Embed  Embed   Embed   Embed
                     ↓
          Causal Transformer Layers (N)
                     ↓
           LM Head  (d_model → vocab_size)
                     ↓
Output:  probability distribution → next token "jumps"
```

### What It Outputs

| Output type | Shape | Meaning |
|---|---|---|
| Logits (per step) | `(B, vocab_size)` | Probability over next token |
| Generated token IDs | `(B, T_generated)` | The actual text/tokens produced |
| (Internal hidden states) | `(B, T, H)` per layer | *Not normally exposed* |

### Examples

| Model | Architecture | Primary use |
|---|---|---|
| GPT-4 / GPT-4o | Causal decoder | Text generation, chat |
| Gemma 4 | Causal decoder (MoE variant) | Text + multimodal generation |
| LLaMA 3 | Causal decoder | Text generation |
| Stable Diffusion | Diffusion + UNet decoder | Image generation |
| AudioLM | Autoregressive audio decoder | Speech/audio generation |

---

## 2. What Is a Feature Extractor?

A **feature extractor** is a model trained to produce a **dense, semantically rich representation** of an input. It does not generate new content; it compresses the input into a latent space that downstream tasks (classification, regression, brain encoding) can use.

### How It Works

Feature extractors are typically **bidirectional encoders** (or cross-modal encoders):

1. The full input is seen at once — no causal masking.
2. All tokens/patches/frames attend to each other in both directions.
3. The output is the **hidden state tensor at every layer**, not a next-token prediction.
4. These hidden states are passed downstream as-is.

```
Input:  [frame_1  frame_2  frame_3  ... frame_T]
              ↓        ↓        ↓             ↓
           Patch/Frame Embeddings
                         ↓
          Bidirectional Transformer Layers (N)
                         ↓
Output:  hidden_states[layer] → shape (B, T, H)  ← this is the feature
```

### What It Outputs

| Output type | Shape | Meaning |
|---|---|---|
| Hidden states (per layer) | `(B, T, H)` | Contextual representation of each token/frame |
| Pooled embedding | `(B, H)` | Global summary of the input |
| Layer stack | `(B, L, T, H)` | Full per-layer, per-token representations |

**For brain encoding (TRIBE v2 / neuralset format):**

| Dimension | Symbol | Meaning |
|---|---|---|
| Batch | `B` | Number of stimuli in the batch |
| Layers | `L` | Transformer layer index (depth) |
| Time | `T` | Temporal tokens aligned to stimulus time |
| Hidden | `H` | Feature dimension at that layer |

The final tensor passed to the brain encoder is typically shaped `(B, L, T, H)` — every layer, every timepoint, every feature.

### Examples

| Model | Modality | Architecture |
|---|---|---|
| CLIP (ViT) | Vision + Language | Bidirectional ViT encoder |
| Wav2Vec 2.0 | Audio | Bidirectional transformer encoder |
| VideoMAE | Video | Masked autoencoder (bidirectional ViT) |
| BERT | Text | Bidirectional transformer encoder |
| BiomedCLIP | Biomedical image + text | Dual bidirectional encoder |

---

## 3. Key Architectural Differences

| Property | Generative Model (e.g. Gemma 4) | Feature Extractor (e.g. CLIP, Wav2Vec) |
|---|---|---|
| **Primary training objective** | Next-token prediction (language modeling) | Contrastive / reconstruction / masked prediction |
| **Attention direction** | Causal (left-to-right only) | Bidirectional (full context) |
| **Standard output** | Token logits / generated text | Hidden state tensors |
| **Output shape** | `(B, vocab_size)` per step | `(B, T, H)` per layer |
| **Layer-wise access** | Not exposed by default | Often explicitly returned |
| **Temporal alignment** | None — output length grows with generation | Input-aligned — T matches input sequence length |
| **What is trained "on top"** | Sampling / decoding head | Task head or linear probe |
| **Inference mode** | Autoregressive loop | Single forward pass |

### The Core Conceptual Gap

```
Generative model:     input → [predict next] → [predict next] → ... → text
Feature extractor:    input → [encode all]   → fixed-size dense representation
```

A generative model is trying to answer: *"What comes next?"*
A feature extractor is trying to answer: *"What is this?"*

Brain encoding needs the latter — a rich description of what the stimulus *is* at every moment in time, not a prediction of what comes after it.

---

## 4. Why Gemma 4 Fails the TRIBE v2 Interface Contract

TRIBE v2 (and the underlying neuralset API) expects feature extractors to satisfy a specific **interface contract**:

### The Contract

```python
# What TRIBE v2 expects from any registered extractor
features = extractor(stimulus)
# features.shape == (B, L, T, H)
# Where:
#   B = batch size
#   L = number of transformer layers tapped
#   T = number of temporal tokens, aligned to stimulus time axis
#   H = hidden dimension at each layer
```

### Why Gemma 4 Breaks It

**Problem 1 — Wrong output by default.**
Calling `model(input_ids)` on Gemma 4 returns `CausalLMOutput` containing `logits` of shape `(B, T, vocab_size)`. There are no layer-wise hidden states unless you explicitly pass `output_hidden_states=True` — and even then, the temporal axis `T` corresponds to *input tokens*, not wall-clock stimulus time.

**Problem 2 — Causal masking corrupts temporal alignment.**
Gemma uses causal (masked) self-attention. This means token `t` can only see tokens `0..t-1`. For brain encoding, the representation of a frame at time `t` should incorporate context from the *entire* stimulus (past and future), because neural responses integrate information over a TR (typically 1–2 seconds). Causal representations are systematically half-informed.

**Problem 3 — No standardized layer-time output.**
TRIBE v2's neuralset extractors are written to return `(B, L, T, H)`. Gemma's `hidden_states` attribute (when enabled) returns a tuple of length `L+1` where each element is `(B, T_tokens, H)`. This must be stacked and temporally resampled to match the stimulus time axis — this isn't handled by the extractor interface automatically.

**Problem 4 — Architectural intent mismatch.**
Gemma is optimized via RLHF/SFT to *generate helpful text*. Its internal representations are shaped by that objective — they encode "what is likely to come next" more than "what is the latent perceptual content of this input." For brain encoding, you want representations that correlate with neural activity driven by stimulus *content*, not with generation quality.

### Summary

```
TRIBE v2 expects:     (B, L, T, H)  — layer-wise, time-aligned hidden states
Gemma 4 gives:        (B, T, vocab_size)  — next-token logits
                      or tuple of (B, T_tokens, H) per layer — not resampled, not time-aligned
```

The mismatch is not a bug you can patch with one line — it requires a deliberate **feature extraction wrapper**.

---

## 5. How to Use a Generative Model as a Feature Extractor

Despite the mismatch, you *can* extract meaningful features from a generative model by **tapping its intermediate layers**. This technique is well-established (it's how probing classifiers work, and how features are extracted from LLMs for neuroscience in papers like Toneva & Wehbe 2019).

### The Strategy

1. **Enable hidden state output** via `output_hidden_states=True`.
2. **Register forward hooks** on specific transformer layers to capture activations mid-pass.
3. **Run a single forward pass** (no autoregressive generation).
4. **Stack and resample** the captured hidden states to `(B, L, T, H)`.
5. **Temporally align** `T` to your stimulus time axis (e.g., upsample/downsample to match TR).

### What Layer to Tap?

| Layer range | What it tends to encode |
|---|---|
| Early layers (1–4) | Low-level token/patch features, syntax |
| Middle layers (8–16) | Semantic content, entities, scene structure |
| Late layers (20+) | Task-specific / generation-biased representations |

For brain encoding, **middle layers** typically correlate best with BOLD signals in higher visual/language areas. Early layers match primary sensory cortex. This is an empirical question — it varies by model and brain region.

---

## 6. Python: Hooking Intermediate Layers

The following snippet shows how to extract hidden states from any HuggingFace causal LM (e.g., Gemma) using `register_forward_hook`, then reshape them for TRIBE v2.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict

# ── 1. Load model ──────────────────────────────────────────────────────────────
model_name = "google/gemma-2-2b"  # or gemma-4 when available
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model.eval()

# ── 2. Hook registry ───────────────────────────────────────────────────────────
captured: Dict[str, torch.Tensor] = {}

def make_hook(layer_name: str):
    """Returns a forward hook that saves the output hidden state of a layer."""
    def hook(module, input, output):
        # For transformer layers, output is typically a tuple; first element is hidden state
        hidden = output[0] if isinstance(output, tuple) else output
        captured[layer_name] = hidden.detach().cpu()  # (B, T, H)
    return hook

# ── 3. Register hooks on layers of interest ────────────────────────────────────
LAYERS_TO_TAP = [8, 12, 16, 20]  # middle layers tend to be most brain-predictive
hooks = []

for layer_idx in LAYERS_TO_TAP:
    layer = model.model.layers[layer_idx]  # GemmaDecoderLayer
    h = layer.register_forward_hook(make_hook(f"layer_{layer_idx}"))
    hooks.append(h)

# ── 4. Single forward pass — no generation ────────────────────────────────────
text_stimulus = ["A red ball rolls across a wooden floor."]
inputs = tokenizer(text_stimulus, return_tensors="pt", padding=True)

with torch.no_grad():
    _ = model(**inputs)  # we only care about side-effects (captured dict)

# ── 5. Remove hooks (important — prevents memory leaks) ───────────────────────
for h in hooks:
    h.remove()

# ── 6. Stack into (B, L, T, H) ────────────────────────────────────────────────
layer_tensors: List[torch.Tensor] = [captured[f"layer_{i}"] for i in LAYERS_TO_TAP]
# Each tensor: (B, T_tokens, H)
features = torch.stack(layer_tensors, dim=1)  # (B, L, T_tokens, H)
print(f"Feature shape: {features.shape}")  # e.g. (1, 4, 12, 2048)

# ── 7. Temporal resampling to match stimulus TR ────────────────────────────────
# TRIBE v2 expects T to align with the number of TRs in the stimulus.
# If your stimulus is 10s at TR=1s, you need T=10.
# Use interpolation or mean-pooling to resample from T_tokens → T_tr.

import torch.nn.functional as F

T_tr = 10  # number of TRs you want to align to
B, L, T_tokens, H = features.shape
# Reshape for interpolate: treat (B*L, H, T_tokens) → (B*L, H, T_tr)
feat_flat = features.permute(0, 1, 3, 2).reshape(B * L, H, T_tokens).float()
feat_resampled = F.interpolate(feat_flat, size=T_tr, mode="linear", align_corners=False)
features_aligned = feat_resampled.reshape(B, L, H, T_tr).permute(0, 1, 3, 2)
# Final shape: (B, L, T_tr, H)
print(f"TRIBE-ready shape: {features_aligned.shape}")  # (1, 4, 10, 2048)
```

### Key Points from the Snippet

- **`register_forward_hook`** fires after each layer's forward pass, giving you the raw hidden states before they are passed to the next layer.
- **Always call `h.remove()`** after the forward pass. Leaving hooks registered causes them to accumulate on subsequent calls and creates memory leaks.
- **`output_hidden_states=True`** is an alternative — it returns all hidden states at once in the model output — but hooks give you more surgical control (e.g., you can tap attention weights, FFN outputs, etc.).
- **Temporal resampling is not optional** for TRIBE v2. The token axis (`T_tokens`) is determined by the tokenizer, not by stimulus time. You must explicitly map it to your TR axis.

---

## 7. When to Use Which

### Decision Guide

```
Do you need to generate new content (text, audio, video)?
    YES → Use a generative model (GPT, Gemma, etc.)
    NO  → Keep reading.

Do you need a fixed-size representation of an input for a downstream task?
    YES → Use a feature extractor natively designed for that modality.

Is your downstream task brain encoding in TRIBE v2?
    YES → You MUST produce (B, L, T, H) aligned to stimulus time.
          Use a purpose-built extractor (CLIP, Wav2Vec, VideoMAE) if possible.
          Use a generative model with hooks ONLY if:
            (a) no suitable encoder exists for your modality, OR
            (b) you have a research hypothesis about LLM representations in the brain.

Are you exploring whether LLM representations are brain-predictive (research use)?
    YES → Use a generative model with the hook-based wrapper above.
          Tap middle layers. Compare against a baseline encoder.
```

### Practical Recommendations for BoldSignal / TRIBE v2

| Modality | Recommended extractor | Notes |
|---|---|---|
| Static images | CLIP ViT-L/14 | Strong visual-semantic alignment |
| Video / dynamic stimuli | VideoMAE-Large | Temporal structure built in |
| Audio / speech | Wav2Vec 2.0 Large | Speech-optimized temporal features |
| Language / text | BERT-Large or RoBERTa | Bidirectional; best for encoding semantics |
| Language (research) | Gemma 4 via hooks | Middle layers; must resample to TR |
| Multimodal (video + audio + text) | TRIBE v2's built-in multi-encoder stack | Already handles alignment across modalities |

### The Bottom Line

| Use case | Model type |
|---|---|
| Generating descriptions, captions, summaries | Generative (Gemma, GPT) |
| Encoding stimuli for brain prediction | Feature extractor (CLIP, Wav2Vec, VideoMAE) |
| Probing what LLMs "know" about brain activity | Generative model with hook-based extraction |

Generative models are powerful, but they are **designed to produce, not to represent**. Feature extractors are designed to do the opposite. For brain encoding, you nearly always want the latter — or a carefully wrapped version of the former.

---

*Document context: BoldSignal / TRIBE v2 brain encoding pipeline. Prepared for reference when evaluating new backbone models.*
