"""Persistent hall of fame for top verified Calabi-Yau candidates.

Stores the best-seen score for each content-addressed geometry across runs.
SQLite lives under static/data alongside run JSON artifacts — mount that
directory as a volume in production if you need the board to survive deploys.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

DEFAULT_DB_PATH = os.path.join('static', 'data', 'hall_of_fame.sqlite')
FEATURED_SEED_PATH = os.path.join('static', 'data', 'featured_candidates.json')


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _connect(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA journal_mode=WAL')
    return conn


@contextmanager
def _db(db_path: str):
    conn = _connect(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    with _db(db_path) as conn:
        conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS hall_of_fame (
                candidate_id   TEXT PRIMARY KEY,
                dataset_id     TEXT NOT NULL,
                dataset_name   TEXT,
                h11            INTEGER,
                h21            INTEGER,
                h31            INTEGER,
                euler_char     INTEGER,
                score          REAL NOT NULL,
                best_rank      INTEGER,
                verified       INTEGER NOT NULL DEFAULT 0,
                times_seen     INTEGER NOT NULL DEFAULT 1,
                first_seen_at  TEXT NOT NULL,
                last_seen_at   TEXT NOT NULL,
                last_run_id    TEXT,
                summary        TEXT,
                features_json  TEXT,
                raw_json       TEXT,
                tags_json      TEXT,
                viz_seed       INTEGER
            )
            '''
        )
        conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_hof_dataset_score '
            'ON hall_of_fame (dataset_id, verified DESC, score DESC)'
        )


def _row_to_candidate(row: sqlite3.Row) -> Dict[str, Any]:
    features = json.loads(row['features_json'] or '[]')
    tags = json.loads(row['tags_json'] or '[]')
    raw = json.loads(row['raw_json'] or '{}')
    return {
        'candidate_id': row['candidate_id'],
        'dataset_id': row['dataset_id'],
        'dataset_name': row['dataset_name'],
        'rank': row['best_rank'],
        'score': row['score'],
        'verified_target': bool(row['verified']),
        'h11': row['h11'],
        'h21': row['h21'],
        'h31': row['h31'],
        'euler_char': row['euler_char'],
        'times_seen': row['times_seen'],
        'first_seen_at': row['first_seen_at'],
        'last_seen_at': row['last_seen_at'],
        'last_run_id': row['last_run_id'],
        'summary': row['summary'],
        'features': features,
        'tags': tags,
        'raw': raw,
        'viz_seed': row['viz_seed'],
        'source': 'hall_of_fame',
    }


def get_candidate(candidate_id: str, db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    init_db(db_path)
    with _db(db_path) as conn:
        row = conn.execute(
            'SELECT * FROM hall_of_fame WHERE candidate_id = ?',
            (candidate_id,),
        ).fetchone()
    return _row_to_candidate(row) if row else None


def list_candidates(
    dataset_id: Optional[str] = None,
    verified_only: Optional[bool] = None,
    tag: Optional[str] = None,
    limit: int = 100,
    db_path: str = DEFAULT_DB_PATH,
) -> List[Dict[str, Any]]:
    init_db(db_path)
    clauses = []
    params: List[Any] = []
    if dataset_id:
        clauses.append('dataset_id = ?')
        params.append(dataset_id)
    if verified_only is True:
        clauses.append('verified = 1')
    elif verified_only is False:
        clauses.append('verified = 0')
    where = ('WHERE ' + ' AND '.join(clauses)) if clauses else ''
    params.append(max(1, min(int(limit), 500)))
    with _db(db_path) as conn:
        rows = conn.execute(
            f'''
            SELECT * FROM hall_of_fame
            {where}
            ORDER BY verified DESC, score DESC, last_seen_at DESC
            LIMIT ?
            ''',
            params,
        ).fetchall()
    candidates = [_row_to_candidate(r) for r in rows]
    if tag:
        candidates = [c for c in candidates if tag in (c.get('tags') or [])]
    return candidates


def upsert_candidate(record: Dict[str, Any], db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """Insert or improve an existing hall-of-fame entry.

    Keeps the higher score. Verified beats unverified when scores tie.
    Increments times_seen whenever the same id is seen again.
    """
    init_db(db_path)
    now = _utcnow()
    candidate_id = record['candidate_id']
    score = float(record.get('score') or 0.0)
    verified = 1 if record.get('verified_target') else 0
    features_json = json.dumps(record.get('features') or [], separators=(',', ':'))
    tags_json = json.dumps(record.get('tags') or [], separators=(',', ':'))
    raw_json = json.dumps(record.get('raw') or {}, separators=(',', ':'))

    with _db(db_path) as conn:
        existing = conn.execute(
            'SELECT * FROM hall_of_fame WHERE candidate_id = ?',
            (candidate_id,),
        ).fetchone()

        if existing is None:
            conn.execute(
                '''
                INSERT INTO hall_of_fame (
                    candidate_id, dataset_id, dataset_name,
                    h11, h21, h31, euler_char,
                    score, best_rank, verified, times_seen,
                    first_seen_at, last_seen_at, last_run_id,
                    summary, features_json, raw_json, tags_json, viz_seed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    candidate_id,
                    record.get('dataset_id'),
                    record.get('dataset_name'),
                    record.get('h11'),
                    record.get('h21'),
                    record.get('h31'),
                    record.get('euler_char'),
                    score,
                    record.get('rank'),
                    verified,
                    now,
                    now,
                    record.get('last_run_id'),
                    record.get('summary'),
                    features_json,
                    raw_json,
                    tags_json,
                    record.get('viz_seed'),
                ),
            )
        else:
            better_score = score > float(existing['score'])
            better_verified = verified and not existing['verified']
            keep_score = score if better_score else float(existing['score'])
            keep_rank = record.get('rank') if better_score else existing['best_rank']
            keep_verified = 1 if (verified or existing['verified']) else 0
            # Prefer richer payload when improving.
            use_new_payload = better_score or better_verified
            conn.execute(
                '''
                UPDATE hall_of_fame SET
                    score = ?,
                    best_rank = ?,
                    verified = ?,
                    times_seen = times_seen + 1,
                    last_seen_at = ?,
                    last_run_id = COALESCE(?, last_run_id),
                    summary = CASE WHEN ? THEN ? ELSE summary END,
                    features_json = CASE WHEN ? THEN ? ELSE features_json END,
                    raw_json = CASE WHEN ? THEN ? ELSE raw_json END,
                    tags_json = CASE WHEN ? THEN ? ELSE tags_json END,
                    h11 = COALESCE(?, h11),
                    h21 = COALESCE(?, h21),
                    h31 = COALESCE(?, h31),
                    euler_char = COALESCE(?, euler_char),
                    dataset_name = COALESCE(?, dataset_name),
                    viz_seed = COALESCE(?, viz_seed)
                WHERE candidate_id = ?
                ''',
                (
                    keep_score,
                    keep_rank,
                    keep_verified,
                    now,
                    record.get('last_run_id'),
                    use_new_payload, record.get('summary') or existing['summary'],
                    use_new_payload, features_json if use_new_payload else existing['features_json'],
                    use_new_payload, raw_json if use_new_payload else existing['raw_json'],
                    use_new_payload, tags_json if use_new_payload else existing['tags_json'],
                    record.get('h11'),
                    record.get('h21'),
                    record.get('h31'),
                    record.get('euler_char'),
                    record.get('dataset_name'),
                    record.get('viz_seed'),
                    candidate_id,
                ),
            )

    return get_candidate(candidate_id, db_path=db_path)


def promote_from_run(
    results: Dict[str, Any],
    run_id: str,
    canonical_id_fn,
    identity_payload_fn,
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    """Promote verified (and high-scoring) hits from a search run into the board.

    Returns how many records were upserted.
    """
    metadata = results.get('run_metadata') or {}
    dataset_id = metadata.get('dataset_id') or results.get('dataset_id') or 'kreuzer-skarke'
    dataset_name = metadata.get('dataset_name') or dataset_id
    top = results.get('top_results') or []
    promoted = 0

    for result in top:
        verified = bool(result.get('verified_target'))
        # Only promote verified targets into the hall of fame. Unverified
        # high-scorers stay in the per-run JSON, not the permanent board.
        if not verified:
            continue
        try:
            candidate_id = canonical_id_fn(dataset_id, result)
        except (ValueError, TypeError, KeyError):
            continue

        identity = identity_payload_fn(dataset_id, result)
        if dataset_id == 'cy5-folds':
            features = [
                ('h11', result.get('h11')),
                ('h21', result.get('h21')),
                ('h31', result.get('h31')),
            ]
        elif dataset_id == 'heterotic':
            features = [
                ('h11', result.get('h11')),
                ('h21', result.get('h21')),
                ('balance', round(result.get('hodge_balance', 0) or 0, 3)),
            ]
        else:
            features = [
                ('h11', result.get('h11')),
                ('h21', result.get('h21')),
                ('χ', result.get('euler_char')),
            ]

        viz_seed = int(hashlib.md5(candidate_id.encode('utf-8')).hexdigest()[:6], 16)
        upsert_candidate(
            {
                'candidate_id': candidate_id,
                'dataset_id': dataset_id,
                'dataset_name': dataset_name,
                'h11': identity.get('h11') or result.get('h11'),
                'h21': identity.get('h21') or result.get('h21'),
                'h31': identity.get('h31') or result.get('h31'),
                'euler_char': identity.get('euler_char') or result.get('euler_char'),
                'score': result.get('score'),
                'rank': result.get('rank'),
                'verified_target': True,
                'last_run_id': run_id,
                'summary': f"Verified target from run {run_id}",
                'features': features,
                'raw': result,
                'tags': ['verified', 'hall-of-fame', dataset_id],
                'viz_seed': viz_seed,
            },
            db_path=db_path,
        )
        promoted += 1
    return promoted


def seed_from_featured(
    featured_path: str = FEATURED_SEED_PATH,
    canonical_id_fn=None,
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    """One-time-ish seed: import curated featured JSON if hall is empty."""
    init_db(db_path)
    with _db(db_path) as conn:
        count = conn.execute('SELECT COUNT(*) AS n FROM hall_of_fame').fetchone()['n']
    if count > 0:
        return 0
    if not os.path.exists(featured_path):
        return 0

    with open(featured_path, 'r') as f:
        payload = json.load(f)

    seeded = 0
    for item in payload.get('candidates', []):
        dataset_id = item.get('dataset_id') or 'kreuzer-skarke'
        feature_map = {k: v for k, v in item.get('features') or []}
        record = {
            'h11': feature_map.get('h11'),
            'h21': feature_map.get('h21'),
            'h31': feature_map.get('h31'),
            'euler_char': feature_map.get('χ', feature_map.get('euler_char')),
            'hodge_balance': feature_map.get('balance'),
        }
        # Derive euler when missing for KS-like datasets.
        if record['euler_char'] is None and record['h11'] is not None and record['h21'] is not None:
            record['euler_char'] = int(2 * (record['h11'] - record['h21']))

        candidate_id = item.get('candidate_id')
        if canonical_id_fn is not None:
            try:
                candidate_id = canonical_id_fn(dataset_id, record)
            except Exception:
                candidate_id = item.get('candidate_id')

        upsert_candidate(
            {
                'candidate_id': candidate_id,
                'dataset_id': dataset_id,
                'dataset_name': item.get('dataset_name'),
                'h11': record.get('h11'),
                'h21': record.get('h21'),
                'h31': record.get('h31'),
                'euler_char': record.get('euler_char'),
                'score': item.get('score'),
                'rank': item.get('rank'),
                'verified_target': bool(item.get('verified_target')),
                'last_run_id': 'seed-featured',
                'summary': item.get('summary'),
                'features': item.get('features') or [],
                'raw': record,
                'tags': list(set((item.get('tags') or []) + ['featured-seed'])),
                'viz_seed': item.get('viz_seed'),
            },
            db_path=db_path,
        )
        seeded += 1
    return seeded
