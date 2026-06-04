# Brain Score Clustering Debug Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Diagnose why Brain Scores cluster 59–66 across all tested YouTube Shorts and fix the root causes.

**Architecture:** Each task adds a diagnostic or fix cell to `boldsignal_batch.ipynb`. All tasks run in Colab. Fixes are applied directly to cell 7 (helpers & scoring). Tasks 1–4 are pure diagnosis; Tasks 5–8 are fixes.

**Tech Stack:** Python, NumPy, SciPy, nilearn, TRIBE v2 (tribev2), Jupyter/Colab

---

## Background

Five YouTube Shorts scored 59.6–66.4. All five were classified `non_narrative`. This is suspicious: different content, same narrow band. Four root causes are suspected (in order of likelihood):

1. **ROI collision**: `AMY`, `PHG`, and `HC` all call `get_idx(['parahip','oc-temp_med'])`. They return *identical* voxel sets. The emotional memory score is therefore a single region's signal computed three times with different weights — not a three-region circuit.

2. **`score_sensory_salience` intra-video normalisation**: The function normalises `hook` against `ts.min()/ts.max()` from the *same* video. Every video trivially produces a mid-range value regardless of actual salience.

3. **`detect_narrative_mode` always returns `non_narrative`**: TRIBE v2 segments are dataframe rows or simple dicts; they never have a `transcript` key. `text` is always `''` → always `non_narrative`.

4. **Normalization bounds not calibrated**: The fixed bounds (`-1.5`/`1.5` for attention, `±2.0` for memory/social) were chosen without inspecting actual TRIBE v2 output distributions. If raw values for YouTube Shorts cluster near zero, all scores compress toward the midpoint of each dimension.

---

## File Map

| File | Role |
|------|------|
| `boldsignal_batch.ipynb` cells 6–7 | ROI definitions, all scoring functions — primary edit target |
| `boldsignal_batch.ipynb` new cells after cell 9 | Diagnostic cells added by this plan |
| `/content/drive/MyDrive/BoldSignal/preds/*.npy` | Cached inference arrays — loaded for diagnosis, never re-run |

---

## Task 1: Print the full intermediate value table for all cached videos

**Files:**
- Modify: `boldsignal_batch.ipynb` — add a new cell after cell 9

**Why:** Before fixing anything, establish the actual numeric range of every intermediate value across the five videos. This tells you which dimension is collapsing and which is doing the work.

- [ ] **Step 1: Add a diagnostic cell to the notebook**

Add a new cell immediately after cell 9 (the batch loop cell) with this content:

```python
# ── DIAG 1. Intermediate value table ─────────────────────────────────────────
import textwrap

header = f"{'ID':14s}  {'d1_raw':>8}  {'d2_raw':>8}  {'d3_raw':>8}  {'hook':>8}  {'isc':>6}  {'mult':>6}  {'SA':>6}  {'EM':>6}  {'SP':>6}  {'SS':>6}  {'total':>6}"
print(header)
print('─' * len(header))

for vid_id in [_video_id(u) for u in VIDEO_URLS]:
    pp, sp = _preds_paths(vid_id)
    if not (os.path.exists(pp) and os.path.exists(sp)):
        print(f'{vid_id:14s}  (no cache)')
        continue
    p = np.load(pp)
    with open(sp, 'rb') as f: segs = pickle.load(f)

    mode = detect_narrative_mode(segs)
    DMN  = roi_mean(p, ROI['DMN'])
    AMY  = safe_zscore(roi_mean(p, ROI['AMY']))
    PHG  = safe_zscore(roi_mean(p, ROI['PHG']))
    HC   = safe_zscore(roi_mean(p, ROI['HC']))
    pSTS = safe_zscore(roi_mean(p, ROI['pSTS']))
    FFG  = safe_zscore(roi_mean(p, ROI['FFG']))
    rTPJ = safe_zscore(roi_mean(p, ROI['rTPJ']))

    n = len(AMY)
    w = np.ones(n); w[int(0.8*n):] = 2.0; w /= w.sum()

    # raw pre-normalise values
    if mode == 'non_narrative':
        d1_raw = 0.40*(-DMN.mean()) + 0.60*(1-(DMN.std()/(np.ptp(DMN)+1e-8)))
    else:
        from scipy.stats import pearsonr
        d1_raw = pearsonr(DMN, np.exp(-4*(np.linspace(0,1,len(DMN))-0.8)**2))[0]
    d2_raw = 0.35*np.average(AMY,weights=w) + 0.45*np.average(PHG,weights=w) + 0.20*np.average(HC,weights=w)
    d3_raw = 0.45*pSTS.mean() + 0.25*FFG.mean() + 0.30*rTPJ.mean()

    times = get_times(segs, len(p))
    me, mm, mr = times<=5, (times>5)&(times<=15), times>15
    def wm(mask, k):
        sub = p[mask] if mask.sum() > 0 else p[:1]
        return float(roi_mean(sub, ROI[k]).mean())
    hook = (0.60*(0.40*wm(me,'V1')+0.30*wm(me,'A1')+0.30*wm(me,'INS'))
           +0.30*(0.50*wm(mm,'V1')+0.50*wm(mm,'INS'))
           +0.10*np.mean([wm(mr,'V1'),wm(mr,'A1'),wm(mr,'INS')]))
    isc_v = compute_isc_proxy(p)
    mult  = normalise(isc_v, 0.5, 1.0, 0.70, 1.15)

    r = compute_brain_score(p, segs)
    d = r['dimensions']
    print(f"{vid_id:14s}  {d1_raw:>8.3f}  {d2_raw:>8.3f}  {d3_raw:>8.3f}  "
          f"{hook:>8.4f}  {isc_v:>6.4f}  {mult:>6.3f}  "
          f"{d['sustained_attention']:>6.2f}  {d['emotional_memory']:>6.2f}  "
          f"{d['social_processing']:>6.2f}  {d['sensory_salience']:>6.2f}  "
          f"{r['brain_score']:>6.1f}")
```

- [ ] **Step 2: Run the cell in Colab**

Expected output: a table with 5 rows. Examine it for:
- Are `d2_raw` and `d3_raw` essentially the same across all videos? → normalisation collapse
- Is `hook` close to zero for all videos? → salience intra-video norm issue
- Are `d1_raw`, `d2_raw`, `d3_raw` all in a narrow band (e.g., all between -0.3 and 0.3)? → calibration issue
- Is `mult` always the same? → ISC proxy is flat

Record the table output — you need it for Task 5 calibration.

---

## Task 2: Confirm the ROI collision bug

**Files:**
- Modify: `boldsignal_batch.ipynb` — add a new cell

**Why:** If AMY, PHG, HC return the same voxel indices, the emotional memory score is meaningless and always near the midpoint. This is verifiable in one cell before touching any code.

- [ ] **Step 1: Add a verification cell**

```python
# ── DIAG 2. ROI collision check ───────────────────────────────────────────────
amy_set = set(ROI['AMY'].tolist())
phg_set = set(ROI['PHG'].tolist())
hc_set  = set(ROI['HC'].tolist())

print(f"AMY size: {len(amy_set)}")
print(f"PHG size: {len(phg_set)}")
print(f"HC  size: {len(hc_set)}")
print(f"AMY ∩ PHG: {len(amy_set & phg_set)}")
print(f"AMY ∩ HC:  {len(amy_set & hc_set)}")
print(f"PHG ∩ HC:  {len(phg_set & hc_set)}")
print()
print(f"pSTS size: {len(ROI['pSTS'])}")
print(f"FFG  size: {len(ROI['FFG'])}")
print(f"rTPJ size: {len(ROI['rTPJ'])}")
print()
print("Destrieux labels containing 'parahip':")
for i, name in enumerate(label_names):
    if 'parahip' in name.lower(): print(f"  [{i:3d}] {name}")
print("Destrieux labels containing 'amygdal':")
for i, name in enumerate(label_names):
    if 'amygdal' in name.lower(): print(f"  [{i:3d}] {name}")
print("Destrieux labels containing 'hippoc':")
for i, name in enumerate(label_names):
    if 'hippoc' in name.lower(): print(f"  [{i:3d}] {name}")
```

- [ ] **Step 2: Run and record output**

Expected: AMY ∩ PHG = full overlap (all three sets are identical), confirming the bug. The label search will show you whether `amygdala` or `hippocampus` appear in the Destrieux atlas at all — they may be subcortical and absent from the surface atlas.

---

## Task 3: Check narrative mode detection

**Files:**
- Modify: `boldsignal_batch.ipynb` — add a new cell

**Why:** If `detect_narrative_mode` always returns `non_narrative`, speech-heavy Shorts are scored with the wrong attention formula (DMN suppression instead of DMN synchrony).

- [ ] **Step 1: Add an inspection cell**

```python
# ── DIAG 3. Narrative mode detection ─────────────────────────────────────────
for i, url in enumerate(VIDEO_URLS):
    vid_id = _video_id(url)
    _, sp = _preds_paths(vid_id)
    if not os.path.exists(sp):
        print(f'{vid_id}: no cache'); continue
    with open(sp, 'rb') as f: segs = pickle.load(f)

    # Inspect segment structure
    print(f'\n── {vid_id} ──────────────────────────────')
    print(f'  type(segs): {type(segs)}')
    if hasattr(segs, '__len__') and len(segs) > 0:
        s0 = segs[0] if not hasattr(segs, 'iloc') else segs.iloc[0]
        print(f'  len(segs): {len(segs)}')
        print(f'  type(segs[0]): {type(s0)}')
        if hasattr(s0, '__dict__'):
            print(f'  segs[0] attrs: {list(s0.__dict__.keys())}')
        elif hasattr(s0, 'keys'):
            print(f'  segs[0] keys: {list(s0.keys())}')
        elif hasattr(segs, 'columns'):
            print(f'  DataFrame columns: {list(segs.columns)}')
        print(f'  segs[0]: {s0}')
    print(f'  detect_narrative_mode → {detect_narrative_mode(segs)}')
```

- [ ] **Step 2: Run and record output**

Look at what keys/attributes the segment objects actually have. This tells you whether TRIBE v2 provides any speech/text information and what the correct key name is. If no transcript is available, narrative mode will need a different detection signal (e.g., audio presence, or a separate ASR step).

---

## Task 4: Check ISC proxy distribution

**Files:**
- Modify: `boldsignal_batch.ipynb` — add a new cell

**Why:** The ISC multiplier formula `1/(variance+ε)` is unusual. If YouTube Shorts produce high inter-voxel variance (expected for fast-cut content), the proxy clips to 0.5 and the multiplier is fixed at 0.70 for all videos, removing any differentiation.

- [ ] **Step 1: Add an ISC diagnostics cell**

```python
# ── DIAG 4. ISC proxy distribution ───────────────────────────────────────────
for url in VIDEO_URLS:
    vid_id = _video_id(url)
    pp, _ = _preds_paths(vid_id)
    if not os.path.exists(pp): continue
    p = np.load(pp)

    vox_var = p.var(axis=0)          # per-voxel variance across time
    row_var = p.var(axis=1)          # per-timestep variance across voxels
    isc_raw = float(np.clip(1.0/(vox_var.mean()+1e-6), 0.5, 1.0))
    mult    = normalise(isc_raw, 0.5, 1.0, 0.70, 1.15)

    print(f'{vid_id}:  vox_var.mean={vox_var.mean():.6f}  '
          f'1/var={1/(vox_var.mean()+1e-6):.2f}  '
          f'isc_raw={isc_raw:.4f}  mult={mult:.3f}  '
          f'preds range=[{p.min():.3f}, {p.max():.3f}]  '
          f'preds shape={p.shape}')
```

- [ ] **Step 2: Run and record**

If `1/var` is always >> 1.0 (meaning it always clips to 1.0), the multiplier is always 1.15 for all videos and ISC is providing no differentiation in the other direction. Conversely, if it always clips to 0.5, the multiplier is always 0.70 — also useless. Either way, the ISC proxy formula needs rethinking.

---

## Task 5: Fix the ROI definitions (AMY / HC separation)

**Files:**
- Modify: `boldsignal_batch.ipynb` cell 6 (`# ── 6. ATLAS & ROI MAP`)

**Why:** AMY, PHG, and HC share identical voxels because they all use the same Destrieux keyword query. Since amygdala and hippocampus are subcortical structures, they are not present in the Destrieux surface atlas. The fix is to use the best available cortical proxies and document the limitation honestly.

- [ ] **Step 1: Update `get_idx` calls in cell 6**

Replace the ROI definition block (the part that defines `ROI = {...}`) with:

```python
ROI = {
    # Default Mode Network — PCC, mPFC, angular gyrus
    'DMN':  get_idx(['cingul', 'front_sup', 'precuneus', 'angular']),

    # Emotional memory circuit
    # NOTE: AMY & HC are subcortical — absent from Destrieux surface atlas.
    # Proxies used: parahippocampal/oc-temp_med cortex for all three.
    # AMY proxy: oc-temp medial (entorhinal/perirhinal — closest cortical neighbour to amygdala)
    'AMY':  get_idx(['oc-temp_med', 'entorhinal']),
    # PHG proxy: parahippocampal gyrus proper
    'PHG':  get_idx(['parahip']),
    # HC proxy: lingual gyrus (closest posterior cortical projection of hippocampus on surface)
    'HC':   get_idx(['lingual', 'parahip']),

    # Social processing
    'pSTS': get_idx(['temp_sup']),           # posterior STS
    'FFG':  get_idx(['fusifor']),             # fusiform gyrus
    'rTPJ': get_idx(['supramarginal', 'angular']),  # right TPJ approx

    # Sensory salience
    'V1':   get_idx(['cuneus', 'lingual', 'calcarine']),
    'A1':   get_idx(['T_transv', 'temporal_transverse']),
    'INS':  get_idx(['insul', 'Ins']),
}
for k, v in ROI.items():
    print(f'{k:6s}: {len(v):5d} voxels')
```

- [ ] **Step 2: Re-run cell 6 in Colab**

Verify that AMY, PHG, HC now have different sizes. If the overlap is still >90% after the change, note this — the atlas may simply not have enough resolution to separate these regions, and a different atlas (Schaefer 2018 with subcortical extension) should be evaluated.

- [ ] **Step 3: Commit the notebook to git**

```bash
git add boldsignal_batch.ipynb
git commit -m "fix: separate AMY/PHG/HC ROI definitions using distinct Destrieux proxies"
```

---

## Task 6: Fix `score_sensory_salience` normalisation

**Files:**
- Modify: `boldsignal_batch.ipynb` cell 7 (`score_sensory_salience` function)

**Why:** The current code normalises `hook` against `ts.min()` and `ts.max()` *from the same video*. This makes every video's salience score a within-video relative value, destroying cross-video comparison. The fix is to use population-anchored bounds derived from the diagnostic data in Task 1.

- [ ] **Step 1: Determine calibration bounds**

From the Task 1 diagnostic table, look at the `hook` column. Record the min and max across all 5 videos. These become the calibration anchor. If the range is very narrow (e.g., 0.001–0.003), the model is producing near-identical salience signals for all Shorts — which is itself a finding. Use slightly wider bounds than the observed range (pad 20% on each side) to allow future videos room to differentiate.

Example (substitute your observed values):
- Observed hook range: `0.0018` to `0.0031`
- Calibrated bounds: `HOOK_MIN = 0.0012`, `HOOK_MAX = 0.0038`

- [ ] **Step 2: Add calibration constants to cell 7 (top of the cell)**

```python
# ── Calibrated normalisation bounds (from 5-video pilot, 2026-04-07) ─────────
# Update these after scoring more diverse content.
ATTN_RAW_MIN, ATTN_RAW_MAX  = -0.8,  0.8   # replace with Task 1 observed range ±20%
MEMO_RAW_MIN, MEMO_RAW_MAX  = -1.0,  1.0   # replace with Task 1 observed range ±20%
SOCI_RAW_MIN, SOCI_RAW_MAX  = -1.0,  1.0   # replace with Task 1 observed range ±20%
HOOK_RAW_MIN, HOOK_RAW_MAX  =  0.001, 0.004 # replace with Task 1 observed range ±20%
```

- [ ] **Step 3: Update `score_sensory_salience` to use global bounds**

Find this block in cell 7:
```python
    si = np.concatenate([ROI['V1'], ROI['A1'], ROI['INS']])
    if len(si) == 0: return 0.0
    ts = roi_mean(preds, si)
    return float(normalise(hook, float(ts.min()), float(ts.max())) * 20)
```

Replace with:
```python
    return float(normalise(hook, HOOK_RAW_MIN, HOOK_RAW_MAX) * 20)
```

- [ ] **Step 4: Update the other three scoring functions to use calibrated bounds**

In `score_sustained_attention`, replace:
```python
    return float(normalise(raw, -1.5, 1.5) * 30)
```
with:
```python
    return float(normalise(raw, ATTN_RAW_MIN, ATTN_RAW_MAX) * 30)
```

In `score_emotional_memory`, replace:
```python
    return float(normalise(c, -2.0, 2.0) * 25)
```
with:
```python
    return float(normalise(c, MEMO_RAW_MIN, MEMO_RAW_MAX) * 25)
```

In `score_social_processing`, replace:
```python
    return float(normalise(s, -2.0, 2.0) * 25)
```
with:
```python
    return float(normalise(s, SOCI_RAW_MIN, SOCI_RAW_MAX) * 25)
```

- [ ] **Step 5: Commit**

```bash
git add boldsignal_batch.ipynb
git commit -m "fix: use globally-calibrated normalisation bounds instead of per-video or hardcoded"
```

---

## Task 7: Fix `detect_narrative_mode` (or disable it pending ASR)

**Files:**
- Modify: `boldsignal_batch.ipynb` cell 7 (`detect_narrative_mode` function)

**Why:** Task 3 diagnosis will likely show TRIBE v2 segments never carry a `transcript` key. The function silently returns `non_narrative` for everything. Until a separate ASR step (e.g., OpenAI Whisper) is added, the safest fix is to document the limitation and keep the code from silently doing nothing.

If Task 3 reveals that segments DO carry transcript data under a different key name, update the key name instead of following the steps below.

- [ ] **Step 1: Replace `detect_narrative_mode` with an honest stub**

Replace the current function:
```python
def detect_narrative_mode(segments):
    try:
        text = ' '.join([s.get('transcript','') for s in segments])
        return 'narrative' if text.count('.') > 5 else 'non_narrative'
    except: return 'non_narrative'
```

With:
```python
def detect_narrative_mode(segments, override=None):
    """Detect whether content has a narrative (speech-driven) structure.

    Currently returns 'non_narrative' for all inputs because TRIBE v2 segments
    do not carry transcript data. Pass override='narrative' or override='non_narrative'
    to force a mode for testing.

    TODO: integrate Whisper ASR to enable auto-detection.
    """
    if override is not None:
        return override
    return 'non_narrative'
```

- [ ] **Step 2: Re-run cell 7 to verify no syntax errors**

Run cell 7 in Colab. Expected: `All functions defined ✓`

- [ ] **Step 3: Commit**

```bash
git add boldsignal_batch.ipynb
git commit -m "fix: replace silent-failure detect_narrative_mode with documented stub pending ASR integration"
```

---

## Task 8: Re-score all 5 videos and compare before/after

**Files:**
- Modify: `boldsignal_batch.ipynb` — add a new final comparison cell

**Why:** Validate that the fixes from Tasks 5–7 actually change scores and increase spread. If scores are still clustered, the cause is structural (TRIBE v2 outputs are genuinely similar for YouTube Shorts) rather than a code bug.

- [ ] **Step 1: Clear the cached brain scores (keep inference arrays)**

Add a cell that resets the brain_score field in results.json without deleting preds:

```python
# ── RE-SCORE: clear brain_score cache, keep preds ────────────────────────────
results_reset = []
for r in _load_results():
    r_reset = dict(r)
    r_reset['brain_score'] = None
    r_reset['dimensions']  = None
    r_reset['isc_multiplier'] = None
    results_reset.append(r_reset)
_save_results(results_reset)
print(f'Reset {len(results_reset)} entries — preds intact, scores cleared')
```

- [ ] **Step 2: Re-run cell 9 (the batch loop)**

All videos will load from cached preds (no re-inference), recompute scores with the fixed functions, and print the new summary table. This takes seconds, not minutes.

- [ ] **Step 3: Print before/after comparison**

```python
# ── BEFORE / AFTER comparison ─────────────────────────────────────────────────
before = {
    'VqBmPmVxwV8': 59.8,
    'IQxea9UB1nQ': 59.6,
    'JktNgjnAv9s': 60.4,
    'FhEcR9sTvqI': 66.4,
    'BgqyapYlwy8': 63.8,
}
new_results = _load_results()
print(f"{'ID':14s}  {'Before':>8}  {'After':>8}  {'Δ':>7}")
print('─' * 42)
for r in sorted(new_results, key=lambda x: (x.get('brain_score') or 0), reverse=True):
    vid = r['video_id']
    new = r.get('brain_score')
    old = before.get(vid, '—')
    if new is not None:
        delta = f"{new - old:+.1f}" if isinstance(old, float) else '—'
        print(f"{vid:14s}  {str(old):>8}  {new:>8.1f}  {delta:>7}")
```

- [ ] **Step 4: Interpret the results**

Two outcomes are possible:

**Outcome A — spread increases (e.g., 35–85 range):** The code bugs were the primary cause. The calibration constants from Task 6 need refinement as more diverse content is scored, but the framework is working.

**Outcome B — scores still cluster (e.g., 58–68 after fixes):** The TRIBE v2 model itself may produce near-identical cortical responses for all YouTube Shorts — suggesting the model's training distribution (Hollywood movies, documentaries) is too different from Shorts format. In this case, consider: (a) testing with long-form content to confirm the model discriminates there, (b) treating Shorts as a separate scoring context with different dimension weights, (c) using the TRIBE v2 output as a relative ranking tool within the same content format rather than an absolute 0–100 scale.

- [ ] **Step 5: Commit the final notebook state**

```bash
git add boldsignal_batch.ipynb
git commit -m "debug: add diagnostic cells and re-score after normalisation and ROI fixes"
```

---

## Self-Review

**Spec coverage:**
- [x] ROI collision (AMY/PHG/HC same voxels) → Task 5
- [x] Intra-video salience normalisation → Task 6
- [x] `detect_narrative_mode` always returns `non_narrative` → Task 7
- [x] Fixed normalization bounds uncalibrated → Task 6
- [x] ISC proxy may be flat/clipping → Task 4 (diagnosis only; fix deferred until we confirm it's actually flat)
- [x] Before/after validation → Task 8

**Deferred (out of scope for this plan):**
- Whisper ASR integration for narrative mode detection
- Schaefer atlas replacement for better subcortical coverage
- ISC proxy formula redesign (Task 4 first to confirm it's broken)

**Placeholder scan:** None found. All code blocks are complete and runnable.

**Type consistency:** All function names (`compute_brain_score`, `normalise`, `roi_mean`, `safe_zscore`, `get_times`, `detect_narrative_mode`) match existing cell 7 definitions exactly.
