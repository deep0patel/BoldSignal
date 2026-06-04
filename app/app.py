import atexit, os, sys, json, signal, subprocess
from pathlib import Path

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from db import init_db, add_jobs, get_all_jobs, cancel_job, remove_job, get_worker_pid, set_worker_pid

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title='BoldSignal', page_icon='🧠', layout='wide')

init_db()

# ── Worker management ─────────────────────────────────────────────────────────
def worker_alive():
    pid = get_worker_pid()
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def kill_worker():
    pid = get_worker_pid()
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass


def start_worker():
    kill_worker()
    proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).parent / 'worker.py')],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    set_worker_pid(proc.pid)


atexit.register(kill_worker)

if not worker_alive():
    start_worker()

# ── Chart colours (dark theme, matching notebook) ────────────────────────────
BG      = '#0D1117'
PANEL   = '#161B22'
BLUE    = '#4FC3F7'
PURPLE  = '#CE93D8'
RED     = '#FF6B6B'
GRID    = '#21262D'
TEXT    = '#C9D1D9'
SUBTEXT = '#8B949E'
DOT_COLORS = [BLUE, PURPLE, '#66BB6A', '#FF8A65']

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title('⚙️ Config')

    hf_token = os.environ.get('HF_TOKEN', '')
    if hf_token:
        st.success('HF Token: ✓ set')
    else:
        st.error('HF Token: ✗ not set')
        st.caption('Run before launching:')
        st.code('export HF_TOKEN=hf_your_token_here', language='bash')

    st.divider()

    config = st.radio('Model config', ['mini', 'max'], index=0)
    if config == 'mini':
        st.caption('Qwen3-0.6B + vjepa2-vitl\n~1.2 GB · Mac-safe')
    else:
        st.caption('Llama-3.2-3B + vjepa2-vitg\n~8 GB · full power')

    st.divider()

    alive = worker_alive()
    st.caption(f'Worker: {"🟢 running" if alive else "🔴 stopped"}')
    if alive:
        if st.button('⏹ Stop worker', use_container_width=True):
            kill_worker()
            st.rerun()
    else:
        if st.button('▶ Start worker', use_container_width=True):
            start_worker()
            st.rerun()

    st.caption(f'Data: `~/BoldSignal/`')

    st.divider()
    if st.button('📋 View worker log', use_container_width=True):
        log_path = Path.home() / 'BoldSignal' / 'logs' / 'worker.log'
        if log_path.exists():
            st.session_state['show_log'] = True
        else:
            st.info('No log file yet.')

# ── Header + URL submit ───────────────────────────────────────────────────────
st.title('🧠 BoldSignal')
st.caption('Brain engagement scoring via TRIBE v2 fMRI encoding')

col_input, col_btn = st.columns([5, 1])
with col_input:
    urls_raw = st.text_area(
        'YouTube URLs',
        placeholder='https://youtube.com/shorts/…\nhttps://youtu.be/…',
        height=110,
        label_visibility='collapsed',
    )
with col_btn:
    st.write('')
    st.write('')
    st.write('')
    if st.button('Add to Queue ▶', type='primary', use_container_width=True):
        urls = [u.strip() for u in urls_raw.strip().splitlines() if u.strip()]
        if urls:
            add_jobs(urls, config=config)
            st.toast(f'Added {len(urls)} job{"s" if len(urls) > 1 else ""} ({config})')
            st.rerun()
        else:
            st.warning('Paste at least one URL.')

st.divider()

# ── Log viewer ────────────────────────────────────────────────────────────────
if st.session_state.get('show_log'):
    log_path = Path.home() / 'BoldSignal' / 'logs' / 'worker.log'
    with st.expander('📋 Worker log (last 100 lines)', expanded=True):
        col_close, col_clear = st.columns([1, 1])
        with col_close:
            if st.button('✕ Close'):
                st.session_state['show_log'] = False
                st.rerun()
        with col_clear:
            if st.button('🗑 Clear log'):
                log_path.write_text('')
                st.toast('Log cleared.')
                st.rerun()
        lines = log_path.read_text().splitlines()
        st.code('\n'.join(lines[-100:]) if lines else '(empty)', language='text')

# ── Stage display labels ──────────────────────────────────────────────────────
STAGE_LABEL = {
    'queued':             '—',
    'starting':           '⚙ starting',
    'downloading':        '⬇ downloading',
    'preprocessing':      '✂ preprocessing',
    'loading_model':      '🔄 loading model',
    'extracting_features':'🎬 extracting (video/audio/text)',
    'predicting':         '🧠 predicting',
    'loading_cache':      '💾 loading cache',
    'scoring':            '📊 scoring',
    'done':               '✓ done',
    'error':              '✗ error',
    'cancelled':          '⊘ cancelled',
}

STATUS_ICON = {
    'pending':   '⏳',
    'running':   '▶',
    'done':      '✓',
    'error':     '✗',
    'cancelled': '⊘',
}

# ── Queue view (auto-refreshes every 2 s) ────────────────────────────────────
@st.fragment(run_every=2)
def queue_view():
    jobs = get_all_jobs()
    if not jobs:
        st.info('Queue is empty — paste URLs above to get started.')
        return

    pending_running = [j for j in jobs if j['status'] in ('pending', 'running')]
    done_jobs       = [j for j in jobs if j['status'] == 'done']
    other_jobs      = [j for j in jobs if j['status'] in ('error', 'cancelled')]

    # ── Active queue ──────────────────────────────────────────────────────────
    if pending_running:
        st.subheader(f'Queue — {len(pending_running)} active')
        for job in pending_running:
            c1, c2, c3, c4 = st.columns([0.4, 3.5, 2, 1])
            with c1:
                st.write(STATUS_ICON.get(job['status'], '?'))
            with c2:
                url = job['url']
                st.write(url if len(url) <= 65 else url[:62] + '…')
                st.caption(f"config: {job['config']}")
            with c3:
                st.caption(STAGE_LABEL.get(job['stage'], job['stage']))
            with c4:
                label  = 'Cancel' if job['status'] == 'running' else 'Remove'
                btn_id = f"act_{job['id']}"
                if st.button(label, key=btn_id, use_container_width=True):
                    if job['status'] == 'running':
                        pid = cancel_job(job['id'])
                        if pid:
                            try:
                                os.kill(pid, signal.SIGTERM)
                            except (ProcessLookupError, PermissionError):
                                pass
                    else:
                        remove_job(job['id'])
                    st.rerun()

    # ── Completed ─────────────────────────────────────────────────────────────
    if done_jobs:
        st.subheader(f'Completed — {len(done_jobs)} video{"s" if len(done_jobs) != 1 else ""}')
        for job in done_jobs:
            url   = job['url']
            label = (url if len(url) <= 65 else url[:62] + '…')
            score = job.get('brain_score') or 0
            with st.expander(f"✓ {label}  —  Brain Score: **{score}/100**"):
                _render_result(job)

    # ── Errors / cancelled ────────────────────────────────────────────────────
    if other_jobs:
        with st.expander(f'⚠ {len(other_jobs)} failed / cancelled'):
            for job in other_jobs:
                url = job['url']
                c1, c2 = st.columns([5, 1])
                with c1:
                    st.write(f"**{STATUS_ICON.get(job['status'], '')} {job['status']}** — "
                             f"{url if len(url) <= 65 else url[:62] + '…'}")
                    if job.get('error'):
                        st.caption(f"Error: {job['error']}")
                with c2:
                    if st.button('Remove', key=f"rm_{job['id']}", use_container_width=True):
                        remove_job(job['id'])
                        st.rerun()


def _render_result(job):
    dims     = json.loads(job['dimensions']) if job.get('dimensions') else {}
    timeline = json.loads(job['timeline'])   if job.get('timeline')   else {}

    col_r, col_t = st.columns(2)
    with col_r:
        if dims:
            fig = _radar_chart(job['brain_score'], job.get('isc_mult', 1.0), dims)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
    with col_t:
        if timeline:
            fig = _timeline_chart(job['brain_score'], timeline)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    with st.expander('Raw scores'):
        st.json({
            'brain_score':    job['brain_score'],
            'isc_multiplier': job.get('isc_mult'),
            'content_mode':   job.get('content_mode'),
            'dimensions':     dims,
        })


# ── Charts ────────────────────────────────────────────────────────────────────
def _radar_chart(brain_score, isc_mult, dims):
    dim_keys   = ['sustained_attention', 'emotional_memory',
                  'social_processing', 'sensory_salience']
    dim_labels = ['Attention\n(30)', 'Memory\n(25)', 'Social\n(25)', 'Salience\n(20)']
    dim_max    = [30, 25, 25, 20]
    scores     = [dims.get(k, 0) for k in dim_keys]
    pct        = [s / m * 100 for s, m in zip(scores, dim_max)]

    N      = len(dim_labels)
    angles = [n / N * 2 * np.pi for n in range(N)] + [0]
    pct_c  = pct + [pct[0]]

    fig = plt.figure(figsize=(5.5, 4.8), facecolor=BG)
    ax  = fig.add_subplot(111, polar=True, facecolor=PANEL)

    for ring in [25, 50, 75, 100]:
        ax.plot(angles, [ring] * (N + 1), color=GRID, lw=0.8, alpha=0.6)
    for angle in angles[:-1]:
        ax.plot([angle, angle], [0, 100], color=GRID, lw=0.8, alpha=0.5)

    ax.fill(angles, pct_c, color=BLUE, alpha=0.15)
    ax.plot(angles, pct_c, color=BLUE, lw=2.5)

    for angle, p, s, m, c in zip(angles[:-1], pct, scores, dim_max, DOT_COLORS):
        ax.scatter(angle, p, color=c, s=70, zorder=5)
        ax.text(angle, p + 17, f'{s:.1f}/{m}',
                ha='center', va='center', fontsize=9, color=c, fontweight='bold')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_labels, color=TEXT, fontsize=10)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], color=SUBTEXT, fontsize=7)
    ax.set_ylim(0, 125)
    ax.spines['polar'].set_color(GRID)
    ax.tick_params(colors=SUBTEXT)
    fig.suptitle(f'Score: {brain_score}/100  ·  ISC ×{isc_mult}',
                 color=TEXT, fontsize=11, fontweight='bold', y=0.98)
    fig.tight_layout()
    return fig


def _timeline_chart(brain_score, timeline):
    times     = np.array(timeline.get('times', []))
    attention = np.array(timeline.get('attention', []))
    emotional = np.array(timeline.get('emotional', []))

    if len(times) == 0:
        fig, ax = plt.subplots(facecolor=BG)
        ax.set_facecolor(PANEL)
        ax.text(0.5, 0.5, 'No timeline data', color=TEXT,
                ha='center', va='center', transform=ax.transAxes)
        return fig

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 4), sharex=True,
        facecolor=BG, gridspec_kw={'hspace': 0.08}
    )
    fig.suptitle(f'Engagement Timeline  ·  {brain_score}/100',
                 color=TEXT, fontsize=11, fontweight='bold', x=0.02, ha='left', y=0.99)

    for ax, signal, color, label in [
        (ax1, attention, BLUE,   'Attention'),
        (ax2, emotional, PURPLE, 'Emotional'),
    ]:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=SUBTEXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.grid(color=GRID, lw=0.6, alpha=0.8)
        ax.fill_between(times, signal, alpha=0.18, color=color)
        ax.plot(times, signal, color=color, lw=2)
        ax.set_ylabel(label, color=TEXT, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_xlim(times[0], times[-1])

    # Drop-off threshold line + shading
    ax1.axhline(0.4, color=RED, ls='--', lw=1.2, alpha=0.7)
    ax1.text(times[-1] * 0.98, 0.42, 'drop-off',
             color=RED, fontsize=8, ha='right', alpha=0.85)
    in_drop, seg_start = False, None
    for t, below in zip(times, attention < 0.4):
        if below and not in_drop:
            seg_start = t; in_drop = True
        elif not below and in_drop:
            ax1.axvspan(seg_start, t, color=RED, alpha=0.07)
            in_drop = False
    if in_drop:
        ax1.axvspan(seg_start, float(times[-1]), color=RED, alpha=0.07)

    ax2.set_xlabel('Time (s)', color=TEXT, fontsize=10)
    fig.subplots_adjust(top=0.88, hspace=0.08)
    return fig


queue_view()
