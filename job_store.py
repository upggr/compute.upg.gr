"""In-memory / SQLite job store for async run progress polling.

Coolify-friendly: no Redis required. Multi-worker deployments each keep their
own memory map; SQLite path is shared when a volume mounts static/data.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Optional

DEFAULT_DB_PATH = os.path.join('static', 'data', 'jobs.sqlite')
_LOCK = threading.Lock()
_MEMORY: Dict[str, Dict[str, Any]] = {}


def _utcnow() -> float:
    return time.time()


@contextmanager
def _db(path: str):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    try:
        conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                percent REAL NOT NULL,
                stage TEXT,
                result_json TEXT,
                error TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            '''
        )
        yield conn
        conn.commit()
    finally:
        conn.close()


def create_job(*, db_path: str = DEFAULT_DB_PATH, stage: str = 'queued') -> str:
    job_id = uuid.uuid4().hex[:16]
    now = _utcnow()
    record = {
        'job_id': job_id,
        'status': 'queued',
        'percent': 0.0,
        'stage': stage,
        'result': None,
        'error': None,
        'created_at': now,
        'updated_at': now,
    }
    with _LOCK:
        _MEMORY[job_id] = dict(record)
    try:
        with _db(db_path) as conn:
            conn.execute(
                'INSERT INTO jobs (job_id, status, percent, stage, result_json, error, created_at, updated_at) '
                'VALUES (?, ?, ?, ?, NULL, NULL, ?, ?)',
                (job_id, 'queued', 0.0, stage, now, now),
            )
    except sqlite3.Error:
        # Memory map remains the source of truth if SQLite is unavailable.
        pass
    return job_id


def update_job(
    job_id: str,
    *,
    status: Optional[str] = None,
    percent: Optional[float] = None,
    stage: Optional[str] = None,
    result: Any = None,
    error: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    with _LOCK:
        rec = _MEMORY.get(job_id) or {
            'job_id': job_id,
            'status': 'unknown',
            'percent': 0.0,
            'stage': None,
            'result': None,
            'error': None,
            'created_at': _utcnow(),
        }
        if status is not None:
            rec['status'] = status
        if percent is not None:
            rec['percent'] = float(max(0.0, min(100.0, percent)))
        if stage is not None:
            rec['stage'] = stage
        if result is not None:
            rec['result'] = result
        if error is not None:
            rec['error'] = error
        rec['updated_at'] = _utcnow()
        _MEMORY[job_id] = rec
        snap = dict(rec)
    try:
        with _db(db_path) as conn:
            conn.execute(
                '''
                INSERT INTO jobs (job_id, status, percent, stage, result_json, error, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    status=excluded.status,
                    percent=excluded.percent,
                    stage=excluded.stage,
                    result_json=excluded.result_json,
                    error=excluded.error,
                    updated_at=excluded.updated_at
                ''',
                (
                    job_id,
                    snap['status'],
                    snap['percent'],
                    snap.get('stage'),
                    json.dumps(snap.get('result')) if snap.get('result') is not None else None,
                    snap.get('error'),
                    snap.get('created_at') or _utcnow(),
                    snap['updated_at'],
                ),
            )
    except sqlite3.Error:
        pass


def get_job(job_id: str, *, db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    with _LOCK:
        if job_id in _MEMORY:
            return dict(_MEMORY[job_id])
    try:
        with _db(db_path) as conn:
            row = conn.execute(
                'SELECT job_id, status, percent, stage, result_json, error, created_at, updated_at '
                'FROM jobs WHERE job_id = ?',
                (job_id,),
            ).fetchone()
        if not row:
            return None
        result = json.loads(row[4]) if row[4] else None
        return {
            'job_id': row[0],
            'status': row[1],
            'percent': row[2],
            'stage': row[3],
            'result': result,
            'error': row[5],
            'created_at': row[6],
            'updated_at': row[7],
        }
    except sqlite3.Error:
        return None
