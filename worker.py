#!/usr/bin/env python3
"""Persistent worker: loads TRIBE v2 once, processes queue jobs sequentially."""

import os, sys, json, time, signal, subprocess, pickle, logging
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import zscore, pearsonr
import torch

sys.path.insert(0, str(Path(__file__).parent))
from db import init_db, get_next_pending, update_job, is_cancelled, set_worker_pid

BASE_DIR   = Path.home() / 'BoldSignal'
CACHE_DIR  = BASE_DIR / 'cache'
PREDS_DIR  = BASE_DIR / 'preds'
TARGET_FPS = 4

for _d in [CACHE_DIR, PREDS_DIR, BASE_DIR / 'logs']:
    _d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    handlers=[
        logging.FileHandler(str(BASE_DIR / 'logs' / 'worker.log')),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger('worker')

# ── Device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

# ── Atlas / ROI (built once at startup) ──────────────────────────────────────
log.info('Loading atlas...')
from nilearn import datasets, surface

_destrieux = datasets.fetch_atlas_destrieux_2009()
_fsavg5    = datasets.fetch_surf_fsaverage(mesh='fsaverage5')

if 'map_left' in _destrieux and 'map_right' in _destrieux:
    _lh = surface.load_surf_data(_destrieux['map_left'])
    _rh = surface.load_surf_data(_destrieux['map_right'])
else:
    _lh = surface.vol_to_surf(
        _destrieux['maps'], _fsavg5['pial_left'],
        interpolation='nearest_most_frequent').astype(int)
    _rh = surface.vol_to_surf(
        _destrieux['maps'], _fsavg5['pial_right'],
        interpolation='nearest_most_frequent').astype(int)

_all_labels  = np.concatenate([_lh, _rh])
_raw_labels  = _destrieux['labels']
_items       = _raw_labels.tolist() if hasattr(_raw_labels, 'tolist') else list(_raw_labels)
_label_names = [v.decode() if isinstance(v, bytes) else str(v) for v in _items]


def _get_idx(keywords):
    idx = []
    for i, name in enumerate(_label_names):
        if any(k.lower() in name.lower() for k in keywords):
            idx.extend(np.where(_all_labels == i)[0].tolist())
    return np.array(idx)


ROI = {
    'DMN':  _get_idx(['cingul', 'front_sup', 'precuneus', 'angular']),
    'AMY':  _get_idx(['temporal_pole']),
    'PHG':  _get_idx(['parahip']),
    'HC':   _get_idx(['oc-temp_med-Lingual']),
    'pSTS': _get_idx(['temp_sup']),
    'FFG':  _get_idx(['fusifor']),
    'rTPJ': _get_idx(['supramarginal', 'angular']),
    'V1':   _get_idx(['cuneus', 'calcarine']),
    'A1':   _get_idx(['T_transv', 'temporal_transverse']),
    'INS':  _get_idx(['insul', 'Ins']),
}
log.info('Atlas ready')

# ── Calibration bounds (pilot: 5 YouTube Shorts, 2026-04-07) ─────────────────
ATTN_RAW_MIN, ATTN_RAW_MAX =  0.377,  0.486
MEMO_RAW_MIN, MEMO_RAW_MAX = -0.144,  0.153
SOCI_RAW_MIN, SOCI_RAW_MAX = -0.083,  0.607
HOOK_RAW_MIN, HOOK_RAW_MAX =  0.024,  0.082


# ── Scoring ───────────────────────────────────────────────────────────────────
def _normalise(x, in_min=None, in_max=None, out_min=0.0, out_max=1.0):
    if in_min is None: in_min = float(np.min(x))
    if in_max is None: in_max = float(np.max(x))
    x = np.clip(x, in_min, in_max)
    return (x - in_min) / (in_max - in_min + 1e-8) * (out_max - out_min) + out_min


def _roi_mean(preds, idx):
    if len(idx) == 0:
        return np.zeros(preds.shape[0])
    valid = idx[idx < preds.shape[1]]
    return preds[:, valid].mean(axis=1)


def _safe_zscore(x):
    if len(x) == 0 or x.std() == 0:
        return np.zeros(max(len(x), 1))
    return zscore(x)


def _get_times(segments, n):
    try:
        return np.array([s.start for s in segments])
    except AttributeError:
        try:
            return np.array([s['onset'] for s in segments])
        except Exception:
            return np.arange(n, dtype=float)


def _score_sustained_attention(preds, mode):
    DMN = _roi_mean(preds, ROI['DMN'])
    if len(DMN) == 0:
        return 0.0
    if mode == 'non_narrative':
        raw = 0.40 * (-DMN.mean()) + 0.60 * (1 - (DMN.std() / (np.ptp(DMN) + 1e-8)))
        return float(_normalise(raw, ATTN_RAW_MIN, ATTN_RAW_MAX) * 30)
    tmpl = np.exp(-4 * (np.linspace(0, 1, len(DMN)) - 0.8) ** 2)
    r = pearsonr(DMN, tmpl)[0] if len(DMN) > 2 else 0.0
    return float(_normalise(r, -1.0, 1.0) * 30)


def _score_emotional_memory(preds):
    AMY = _safe_zscore(_roi_mean(preds, ROI['AMY']))
    PHG = _safe_zscore(_roi_mean(preds, ROI['PHG']))
    HC  = _safe_zscore(_roi_mean(preds, ROI['HC']))
    n   = len(AMY)
    w   = np.ones(n); w[int(0.8 * n):] = 2.0; w /= w.sum()
    c   = (0.35 * np.average(AMY, weights=w)
           + 0.45 * np.average(PHG, weights=w)
           + 0.20 * np.average(HC,  weights=w))
    return float(_normalise(c, MEMO_RAW_MIN, MEMO_RAW_MAX) * 25)


def _score_social_processing(preds):
    whole_mean = preds.mean(axis=1)
    whole_std  = preds.std(axis=1) + 1e-8

    def szm(k):
        return float(((_roi_mean(preds, ROI[k]) - whole_mean) / whole_std).mean())

    s = 0.45 * szm('pSTS') + 0.25 * szm('FFG') + 0.30 * szm('rTPJ')
    return float(_normalise(s, SOCI_RAW_MIN, SOCI_RAW_MAX) * 25)


def _score_sensory_salience(preds, segments):
    times = _get_times(segments, len(preds))

    def wm(mask, k):
        sub = preds[mask] if mask.sum() > 0 else preds[:1]
        return float(_roi_mean(sub, ROI[k]).mean())

    me = times <= 5
    mm = (times > 5) & (times <= 15)
    mr = times > 15
    hook = (0.60 * (0.40 * wm(me, 'V1') + 0.30 * wm(me, 'A1') + 0.30 * wm(me, 'INS'))
            + 0.30 * (0.50 * wm(mm, 'V1') + 0.50 * wm(mm, 'INS'))
            + 0.10 * np.mean([wm(mr, 'V1'), wm(mr, 'A1'), wm(mr, 'INS')]))
    return float(_normalise(hook, HOOK_RAW_MIN, HOOK_RAW_MAX) * 20)


def _isc_proxy(preds):
    vov = float(np.clip(preds.var(axis=0).mean(), 0.004, 0.024))
    t   = (vov - 0.004) / (0.024 - 0.004)
    return 1.15 - t * 0.45


def compute_brain_score(preds, segments):
    isc_mult = _isc_proxy(preds)
    d1 = _score_sustained_attention(preds, 'non_narrative')
    d2 = _score_emotional_memory(preds)
    d3 = _score_social_processing(preds)
    d4 = _score_sensory_salience(preds, segments)
    return {
        'brain_score':    min(round((d1 + d2 + d3 + d4) * isc_mult, 1), 100.0),
        'isc_multiplier': round(isc_mult, 3),
        'content_mode':   'non_narrative',
        'dimensions': {
            'sustained_attention': round(d1, 2),
            'emotional_memory':    round(d2, 2),
            'social_processing':   round(d3, 2),
            'sensory_salience':    round(d4, 2),
        },
    }


def build_engagement_timeline(preds, segments):
    times     = _get_times(segments, len(preds))
    attention = _normalise(-_roi_mean(preds, ROI['DMN']))
    emotional = _normalise(
        (_roi_mean(preds, ROI['AMY']) + _roi_mean(preds, ROI['PHG'])) / 2
    )
    return {
        'times':     times.tolist(),
        'attention': attention.tolist(),
        'emotional': emotional.tolist(),
    }


# ── Patch whisperx compute type for non-CUDA devices ─────────────────────────
if DEVICE != 'cuda':
    try:
        import whisperx.asr as _wasr
        _orig_load_model = _wasr.load_model

        def _patched_load_model(*args, **kwargs):
            if kwargs.get('compute_type') == 'float16':
                kwargs['compute_type'] = 'int8'
            return _orig_load_model(*args, **kwargs)

        _wasr.load_model = _patched_load_model
        log.info('Patched whisperx: float16 → int8 (non-CUDA device)')
    except Exception as _e:
        log.warning(f'Could not patch whisperx compute type: {_e}')

# ── Model (lazy, cached per config) ──────────────────────────────────────────
_model      = None
_model_conf = None


def get_model(config):
    global _model, _model_conf
    if _model is not None and _model_conf == config:
        return _model

    import importlib.util as _ilu
    import inspect

    # The local ./tribev2/ dir (cloned repo) has no top-level __init__.py so
    # Python treats it as a namespace package that shadows the real installed
    # package. Load the real package directly from its known path on disk,
    # bypassing import-path resolution entirely.
    _pkg_dir = Path(__file__).parent / 'tribev2' / 'tribev2'
    if 'tribev2' not in sys.modules or getattr(sys.modules.get('tribev2'), '__file__', None) is None:
        _spec = _ilu.spec_from_file_location(
            'tribev2', str(_pkg_dir / '__init__.py'),
            submodule_search_locations=[str(_pkg_dir)],
        )
        _pkg = _ilu.module_from_spec(_spec)
        sys.modules['tribev2'] = _pkg  # register before exec so relative imports resolve
        _spec.loader.exec_module(_pkg)
    TribeModel = sys.modules['tribev2'].TribeModel
    from huggingface_hub import login

    hf_token = os.environ.get('HF_TOKEN')
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)

    sig = inspect.signature(TribeModel.from_pretrained).parameters
    kw  = {'cache_folder': str(CACHE_DIR)}
    if 'cluster' in sig: kw['cluster'] = None
    if 'device'  in sig: kw['device']  = DEVICE
    if config == 'mini' and 'config_update' in sig:
        kw['config_update'] = {
            'data.text_feature.model_name':        'Qwen/Qwen3-0.6B',
            'data.text_feature.layers':            2 / 3,
            'data.video_feature.image.model_name': 'facebook/vjepa2-vitl-fpc64-256',
            'data.video_feature.image.layers':     2 / 3,
        }

    log.info(f'Loading TRIBE v2 ({config})...')
    _model = TribeModel.from_pretrained('facebook/tribev2', **kw)
    if hasattr(_model, '_model') and _model._model is not None:
        _model._model = _model._model.to(DEVICE)
    _model_conf = config
    log.info('Model ready')
    return _model


# ── Job processing ────────────────────────────────────────────────────────────
def _video_id(url):
    import re, hashlib
    m = re.search(r'(?:v=|shorts/|youtu\.be/)([\w-]+)', url)
    return m.group(1) if m else hashlib.md5(url.encode()).hexdigest()[:12]


def _run_subproc(cmd, job_id):
    """Run a subprocess; poll for cancellation every 0.5 s.
    Returns (success, stderr_tail). Writes stderr to a per-job log file to
    avoid pipe-buffer deadlocks with verbose tools like yt-dlp."""
    stderr_path = BASE_DIR / 'logs' / f'job_{job_id[:8]}.log'
    with open(stderr_path, 'w') as ef:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=ef)
        update_job(job_id, subproc_pid=proc.pid)
        while proc.poll() is None:
            if is_cancelled(job_id):
                proc.terminate()
                proc.wait()
                return False, 'cancelled'
            time.sleep(0.5)
    update_job(job_id, subproc_pid=None)
    with open(stderr_path) as f:
        tail = f.read()[-600:]
    return proc.returncode == 0, tail


def process_job(job):
    job_id = job['id']
    url    = job['url']
    config = job.get('config', 'mini')
    vid_id = _video_id(url)

    preds_path = PREDS_DIR / f'{vid_id}_preds.npy'
    segs_path  = PREDS_DIR / f'{vid_id}_segments.pkl'
    proc_path  = Path(f'/tmp/boldsignal_{vid_id}.mp4')
    raw_path   = Path(f'/tmp/boldsignal_{vid_id}_raw.mp4')

    update_job(job_id,
               status='running', video_id=vid_id,
               started_at=datetime.now().isoformat(), stage='starting')
    try:
        if not (preds_path.exists() and segs_path.exists()):
            # Download — use yt-dlp as a library to avoid subprocess FD issues
            if is_cancelled(job_id): return
            update_job(job_id, stage='downloading')
            try:
                import yt_dlp
            except ImportError:
                raise RuntimeError('yt-dlp not installed in this Python environment')
            cancelled = [False]

            def _cancel_hook(d):
                if is_cancelled(job_id):
                    cancelled[0] = True
                    raise yt_dlp.utils.DownloadCancelled()

            ydl_opts = {
                'format': 'bestvideo[ext=mp4][height<=480]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                'merge_output_format': 'mp4',
                'outtmpl': str(raw_path),
                'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
                'quiet': True,
                'no_warnings': True,
                'progress_hooks': [_cancel_hook],
            }
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
            except yt_dlp.utils.DownloadCancelled:
                return
            except Exception as e:
                raise RuntimeError(f'yt-dlp failed: {e}')
            if not raw_path.exists():
                if is_cancelled(job_id): return
                raise RuntimeError('yt-dlp finished but output file missing')

            # Preprocess
            if is_cancelled(job_id): return
            update_job(job_id, stage='preprocessing')
            ok, stderr = _run_subproc([
                'ffmpeg', '-y', '-i', str(raw_path),
                '-vf', f'fps={TARGET_FPS}',
                '-c:v', 'libx264', '-crf', '23',
                '-c:a', 'aac', str(proc_path),
            ], job_id)
            raw_path.unlink(missing_ok=True)
            if not ok or not proc_path.exists():
                if is_cancelled(job_id): return
                raise RuntimeError(f'ffmpeg failed:\n{stderr}')

            # Load model (cancellable before, not during)
            if is_cancelled(job_id): return
            update_job(job_id, stage='loading_model')
            model = get_model(config)

            # Feature extraction (video + audio + text inside TRIBE v2)
            if is_cancelled(job_id): return
            update_job(job_id, stage='extracting_features')
            log.info(f'  extracting features: {vid_id}')
            df_i = model.get_events_dataframe(video_path=str(proc_path))

            # Brain prediction
            if is_cancelled(job_id): return
            update_job(job_id, stage='predicting')
            log.info(f'  predicting: {vid_id}')
            preds_i, segs_i = model.predict(events=df_i)
            log.info(f'  preds shape: {preds_i.shape}')

            np.save(str(preds_path), preds_i)
            with open(segs_path, 'wb') as f:
                pickle.dump(segs_i, f)
            proc_path.unlink(missing_ok=True)

        else:
            if is_cancelled(job_id): return
            update_job(job_id, stage='loading_cache')
            preds_i = np.load(str(preds_path))
            with open(segs_path, 'rb') as f:
                segs_i = pickle.load(f)
            log.info(f'  loaded from cache: {vid_id}')

        if is_cancelled(job_id): return
        update_job(job_id, stage='scoring')
        r        = compute_brain_score(preds_i, segs_i)
        timeline = build_engagement_timeline(preds_i, segs_i)

        update_job(job_id,
                   status='done', stage='done',
                   brain_score=r['brain_score'],
                   isc_mult=r['isc_multiplier'],
                   content_mode=r['content_mode'],
                   dimensions=json.dumps(r['dimensions']),
                   timeline=json.dumps(timeline),
                   finished_at=datetime.now().isoformat())
        log.info(f'Done {vid_id}: {r["brain_score"]}/100')

    except Exception as e:
        if not is_cancelled(job_id):
            update_job(job_id, status='error', stage='error', error=str(e),
                       finished_at=datetime.now().isoformat())
        log.error(f'Error {vid_id}: {e}', exc_info=True)

    finally:
        proc_path.unlink(missing_ok=True)
        raw_path.unlink(missing_ok=True)


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    init_db()
    set_worker_pid(os.getpid())
    log.info(f'Worker started — PID {os.getpid()}, device={DEVICE}')

    while True:
        job = get_next_pending()
        if job:
            log.info(f'Processing: {job["url"][:70]}')
            process_job(job)
        else:
            time.sleep(2)


if __name__ == '__main__':
    main()
