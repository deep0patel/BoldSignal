import sqlite3, os
from pathlib import Path
from datetime import datetime

DB_PATH = Path.home() / 'BoldSignal' / 'queue.db'


def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _conn() as c:
        c.execute('''CREATE TABLE IF NOT EXISTS jobs (
            id           TEXT PRIMARY KEY,
            url          TEXT NOT NULL,
            video_id     TEXT,
            status       TEXT DEFAULT 'pending',
            stage        TEXT DEFAULT 'queued',
            config       TEXT DEFAULT 'mini',
            brain_score  REAL,
            isc_mult     REAL,
            content_mode TEXT,
            dimensions   TEXT,
            timeline     TEXT,
            error        TEXT,
            subproc_pid  INTEGER,
            created_at   TEXT,
            started_at   TEXT,
            finished_at  TEXT
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY, value TEXT
        )''')


def add_jobs(urls, config='mini'):
    import uuid
    with _conn() as c:
        for url in urls:
            c.execute(
                'INSERT INTO jobs (id, url, config, created_at) VALUES (?, ?, ?, ?)',
                (str(uuid.uuid4()), url.strip(), config, datetime.now().isoformat())
            )


def get_all_jobs():
    with _conn() as c:
        return [dict(r) for r in c.execute(
            'SELECT * FROM jobs ORDER BY created_at'
        )]


def get_next_pending():
    with _conn() as c:
        r = c.execute(
            "SELECT * FROM jobs WHERE status='pending' ORDER BY created_at LIMIT 1"
        ).fetchone()
        return dict(r) if r else None


def update_job(job_id, **kw):
    if not kw:
        return
    sets = ', '.join(f'{k}=?' for k in kw)
    with _conn() as c:
        c.execute(f'UPDATE jobs SET {sets} WHERE id=?', [*kw.values(), job_id])


def is_cancelled(job_id):
    with _conn() as c:
        r = c.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
        return bool(r and r['status'] == 'cancelled')


def cancel_job(job_id):
    """Mark job cancelled; return subproc_pid if one is stored (for SIGTERM)."""
    with _conn() as c:
        r = c.execute(
            "SELECT subproc_pid, status FROM jobs WHERE id=?", (job_id,)
        ).fetchone()
        if r and r['status'] in ('pending', 'running'):
            c.execute("UPDATE jobs SET status='cancelled' WHERE id=?", (job_id,))
            return r['subproc_pid']
    return None


def remove_job(job_id):
    with _conn() as c:
        c.execute("DELETE FROM jobs WHERE id=?", (job_id,))


def get_worker_pid():
    with _conn() as c:
        r = c.execute("SELECT value FROM meta WHERE key='worker_pid'").fetchone()
        return int(r['value']) if r else None


def set_worker_pid(pid):
    with _conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO meta VALUES ('worker_pid', ?)", (str(pid),)
        )
