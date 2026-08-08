"""SQLite store for offline-computed / curated Calabi–Yau geometry records.

The web container only *reads* this DB (plus upserts baked seed JSON on boot).
Heavy CYTools / PALP work runs offline; workers push records via
``scripts/geometry_worker_stub.py`` or ``scripts/ingest_geometry_db.py``.

DB path defaults to ``static/data/geometry.sqlite`` (Coolify volume).
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 2
DEFAULT_DB_PATH = os.path.join('static', 'data', 'geometry.sqlite')
KS_SAMPLE_PATH = os.path.join('data', 'ks_geometry_sample.json')
GEOMETRY_PACK_PATH = os.path.join('data', 'geometry_pack.json')

PIPELINE_STAGES = ('vertices', 'triangulated', 'intersections', 'periods')

# Prefer unique/curated/offline-complete over sample representatives.
_STATUS_RANK = {
    'unique': 50,
    'curated': 40,
    'representative': 30,
    'pending': 10,
    'failed': 0,
}

_STAGE_RANK = {
    'periods': 40,
    'intersections': 30,
    'triangulated': 20,
    'vertices': 10,
}

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


def _dumps(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        # triangulation / equation may already be plain text
        return value
    return json.dumps(value, separators=(',', ':'), ensure_ascii=False)


def _loads(value: Any, *, as_json: bool = True) -> Any:
    if value is None:
        return None
    if not as_json:
        return value
    if isinstance(value, (dict, list, int, float, bool)):
        return value
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    if text[0] in '[{':
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError, ValueError):
            return value
    return value


def _has_vertices(record: Optional[Dict[str, Any]]) -> bool:
    if not record:
        return False
    verts = record.get('polytope_vertices') or record.get('vertex_matrix')
    return isinstance(verts, (list, tuple)) and len(verts) > 0


def _has_triangulation(record: Optional[Dict[str, Any]]) -> bool:
    if not record:
        return False
    tri = record.get('triangulation')
    if tri is None:
        return False
    if isinstance(tri, str):
        return bool(tri.strip())
    if isinstance(tri, (list, tuple, dict)):
        return len(tri) > 0
    return True


def _has_intersections(record: Optional[Dict[str, Any]]) -> bool:
    if not record:
        return False
    ix = record.get('intersections')
    if ix is None and isinstance(record.get('extra'), dict):
        ix = record['extra'].get('intersections')
    if ix is None:
        return False
    if isinstance(ix, (list, tuple, dict)):
        return len(ix) > 0
    return True


def _has_periods(record: Optional[Dict[str, Any]]) -> bool:
    if not record:
        return False
    periods = record.get('periods')
    if periods is None:
        return False
    if isinstance(periods, (list, tuple, dict)):
        return len(periods) > 0
    if isinstance(periods, str):
        return bool(periods.strip())
    return True


def infer_stage(record: Optional[Dict[str, Any]]) -> Optional[str]:
    """Highest pipeline stage justified by stored fields (never invent periods)."""
    if not record:
        return None
    explicit = record.get('stage')
    if isinstance(explicit, str) and explicit.strip():
        stage = explicit.strip().lower()
        if stage in PIPELINE_STAGES:
            # Trust explicit stage only if data can support it; never claim
            # periods/intersections without payloads.
            if stage == 'periods' and not _has_periods(record):
                pass  # fall through to inferred
            elif stage == 'intersections' and not (
                _has_intersections(record) or _has_periods(record)
            ):
                pass
            else:
                return stage
    if _has_periods(record):
        return 'periods'
    if _has_intersections(record):
        return 'intersections'
    if _has_triangulation(record):
        return 'triangulated'
    if _has_vertices(record):
        return 'vertices'
    return None


def pipeline_note(stage: Optional[str]) -> str:
    """Human-readable note: what is filled vs still pending."""
    order = list(PIPELINE_STAGES)
    if not stage:
        return (
            'Pipeline pending: no vertices, triangulation, intersections, '
            'or periods stored yet (offline CYTools/PALP worker fills stages).'
        )
    try:
        idx = order.index(stage)
    except ValueError:
        return f'Pipeline stage={stage} (non-standard).'
    filled = order[: idx + 1]
    pending = order[idx + 1 :]
    parts = [f"filled: {', '.join(filled)}"]
    if pending:
        parts.append(f"pending: {', '.join(pending)}")
    else:
        parts.append('pending: none (full offline dump present)')
    return '; '.join(parts)


def stage_includes(current: Optional[str], required: str) -> bool:
    """True if ``current`` is at least as advanced as ``required``."""
    if not current:
        return False
    try:
        return PIPELINE_STAGES.index(current) >= PIPELINE_STAGES.index(required)
    except ValueError:
        return False


def _richness(record: Optional[Dict[str, Any]]) -> int:
    """Higher = prefer this hit when multiple share a Hodge key."""
    if not record:
        return -1
    score = _STATUS_RANK.get(str(record.get('status') or ''), 5)
    if _has_vertices(record):
        score += 100
    score += _STAGE_RANK.get(str(record.get('stage') or ''), 0)
    for key in (
        'hypersurface_equation',
        'triangulation',
        'configuration_matrix',
        'periods',
        'intersections',
        'weight_system',
        'ambient',
    ):
        if record.get(key) is not None:
            score += 5
    return score


def _ensure_pipeline_columns(conn: sqlite3.Connection) -> None:
    cols = {
        row['name']
        for row in conn.execute('PRAGMA table_info(geometries)').fetchall()
    }
    if 'stage' not in cols:
        conn.execute('ALTER TABLE geometries ADD COLUMN stage TEXT')
    if 'intersections' not in cols:
        conn.execute('ALTER TABLE geometries ADD COLUMN intersections TEXT')
    conn.execute(
        "INSERT INTO schema_meta (key, value) VALUES ('version', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (str(SCHEMA_VERSION),),
    )


def make_geometry_id(
    dataset_id: str,
    h11: int,
    h21: int,
    h31: Optional[int] = None,
    *,
    source: str = '',
    source_hash: Optional[str] = None,
    vertices: Any = None,
) -> str:
    """Content-addressed id: ``dataset:h11:h21[:h31]:source_hash``."""
    if source_hash is None:
        payload = {
            'dataset_id': dataset_id,
            'h11': h11,
            'h21': h21,
            'h31': h31,
            'source': source,
            'vertices': vertices,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(',', ':'), default=str)
        source_hash = hashlib.sha256(blob.encode('utf-8')).hexdigest()[:12]
    parts = [dataset_id, str(int(h11)), str(int(h21))]
    if h31 is not None:
        parts.append(str(int(h31)))
    parts.append(source_hash)
    return ':'.join(parts)


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    with _db(db_path) as conn:
        conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS schema_meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            '''
        )
        conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS geometries (
                id                      TEXT PRIMARY KEY,
                candidate_id            TEXT,
                dataset_id              TEXT NOT NULL,
                h11                     INTEGER NOT NULL,
                h21                     INTEGER NOT NULL,
                h31                     INTEGER,
                euler_char              INTEGER,
                source                  TEXT,
                status                  TEXT,
                note                    TEXT,
                vertex_matrix           TEXT,
                polytope_vertices       TEXT,
                triangulation           TEXT,
                hypersurface_equation   TEXT,
                weight_system           TEXT,
                configuration_matrix    TEXT,
                ambient                 TEXT,
                periods                 TEXT,
                orientifold             TEXT,
                stage                   TEXT,
                intersections           TEXT,
                extra                   TEXT,
                computed_at             TEXT,
                updated_at              TEXT NOT NULL
            )
            '''
        )
        conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_geom_hodge '
            'ON geometries (dataset_id, h11, h21, h31)'
        )
        conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_geom_candidate '
            'ON geometries (candidate_id)'
        )
        _ensure_pipeline_columns(conn)
        row = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'version'"
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO schema_meta (key, value) VALUES ('version', ?)",
                (str(SCHEMA_VERSION),),
            )


def _row_to_record(row: sqlite3.Row) -> Dict[str, Any]:
    triangulation_raw = row['triangulation']
    triangulation = _loads(triangulation_raw)
    keys = row.keys()
    stage_raw = row['stage'] if 'stage' in keys else None
    intersections_raw = row['intersections'] if 'intersections' in keys else None
    rec: Dict[str, Any] = {
        'id': row['id'],
        'candidate_id': row['candidate_id'],
        'dataset_id': row['dataset_id'],
        'h11': row['h11'],
        'h21': row['h21'],
        'h31': row['h31'],
        'euler_char': row['euler_char'],
        'source': row['source'],
        'status': row['status'],
        'note': row['note'],
        'vertex_matrix': _loads(row['vertex_matrix']),
        'polytope_vertices': _loads(row['polytope_vertices']),
        'triangulation': triangulation,
        'hypersurface_equation': row['hypersurface_equation'],
        'weight_system': _loads(row['weight_system']),
        'configuration_matrix': _loads(row['configuration_matrix']),
        'ambient': row['ambient'],
        'periods': _loads(row['periods']),
        'orientifold': _loads(row['orientifold']),
        'stage': stage_raw,
        'intersections': _loads(intersections_raw),
        'extra': _loads(row['extra']),
        'computed_at': row['computed_at'],
        'updated_at': row['updated_at'],
    }
    # Promote useful keys from extra into the top-level view for consumers.
    extra = rec.get('extra') or {}
    if isinstance(extra, dict):
        for key in (
            'name',
            'geometry_name',
            'polytope_id',
            'triangulation_id',
            'vertex_count',
            'facet_count',
            'point_count',
            'dual_point_count',
            'source_slice',
            'uniqueness',
            'geometry_uniqueness',
            'favourable',
            'reference',
            'reference_url',
            'pipeline_note',
        ):
            if rec.get(key) is None and extra.get(key) is not None:
                rec[key] = extra[key]
        if rec.get('intersections') is None and extra.get('intersections') is not None:
            rec['intersections'] = _loads(extra.get('intersections'))
        if rec.get('stage') is None and extra.get('stage') is not None:
            rec['stage'] = extra.get('stage')
    # Always re-infer stage from richness so stale/overclaimed stages are corrected.
    inferred = infer_stage(rec)
    if inferred:
        rec['stage'] = inferred
    rec['pipeline_note'] = pipeline_note(rec.get('stage'))
    return rec


def upsert_geometry(
    record: Dict[str, Any],
    *,
    db_path: str = DEFAULT_DB_PATH,
) -> Dict[str, Any]:
    """Insert or replace a geometry row. Returns the stored record."""
    init_db(db_path)
    dataset_id = str(record.get('dataset_id') or 'kreuzer-skarke')
    h11 = int(record['h11'])
    h21 = int(record['h21'])
    h31 = record.get('h31')
    if h31 is not None:
        h31 = int(h31)
    geom_id = record.get('id') or make_geometry_id(
        dataset_id,
        h11,
        h21,
        h31,
        source=str(record.get('source') or ''),
        source_hash=record.get('source_hash'),
        vertices=record.get('polytope_vertices') or record.get('vertex_matrix'),
    )
    now = _utcnow()
    computed_at = record.get('computed_at') or now
    status = record.get('status') or record.get('geometry_status') or 'pending'
    note = record.get('note')
    if status == 'representative' and not note:
        note = (
            'Representative polytope for this Hodge class — many distinct '
            'polytopes/triangulations can share the same Hodge numbers.'
        )

    # Fold leftover known keys into extra so nothing is dropped.
    known = {
        'id', 'candidate_id', 'dataset_id', 'h11', 'h21', 'h31', 'euler_char',
        'euler_characteristic', 'source', 'source_hash', 'status',
        'geometry_status', 'note', 'vertex_matrix', 'polytope_vertices',
        'triangulation', 'hypersurface_equation', 'weight_system',
        'configuration_matrix', 'ambient', 'periods', 'orientifold', 'extra',
        'computed_at', 'updated_at', 'name', 'geometry_name', 'polytope_id',
        'triangulation_id', 'vertex_count', 'facet_count', 'point_count',
        'dual_point_count', 'source_slice', 'uniqueness', 'geometry_uniqueness',
        'favourable', 'reference', 'reference_url', 'stage', 'intersections',
        'pipeline_note',
    }
    extra = dict(record.get('extra') or {}) if isinstance(record.get('extra'), dict) else {}
    for key, val in record.items():
        if key not in known and val is not None:
            extra.setdefault(key, val)
    for key in (
        'name', 'geometry_name', 'polytope_id', 'triangulation_id',
        'vertex_count', 'facet_count', 'point_count', 'dual_point_count',
        'source_slice', 'uniqueness', 'geometry_uniqueness', 'favourable',
        'reference', 'reference_url',
    ):
        if record.get(key) is not None:
            extra.setdefault(key, record[key])

    euler = record.get('euler_char')
    if euler is None:
        euler = record.get('euler_characteristic')

    verts = record.get('polytope_vertices')
    vmat = record.get('vertex_matrix')
    if verts is None and vmat is not None:
        verts = vmat
    if vmat is None and verts is not None:
        vmat = verts

    triangulation = record.get('triangulation')
    if triangulation is not None and not isinstance(triangulation, str):
        triangulation_store = _dumps(triangulation)
    else:
        triangulation_store = triangulation

    intersections = record.get('intersections')
    if intersections is None and isinstance(extra, dict):
        intersections = extra.pop('intersections', None)

    stage_probe = {
        'stage': record.get('stage'),
        'polytope_vertices': verts,
        'vertex_matrix': vmat,
        'triangulation': triangulation,
        'intersections': intersections,
        'periods': record.get('periods'),
        'extra': extra,
    }
    stage = infer_stage(stage_probe)
    if stage:
        extra['pipeline_note'] = pipeline_note(stage)

    with _db(db_path) as conn:
        conn.execute(
            '''
            INSERT INTO geometries (
                id, candidate_id, dataset_id, h11, h21, h31, euler_char,
                source, status, note, vertex_matrix, polytope_vertices,
                triangulation, hypersurface_equation, weight_system,
                configuration_matrix, ambient, periods, orientifold,
                stage, intersections, extra,
                computed_at, updated_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?
            )
            ON CONFLICT(id) DO UPDATE SET
                candidate_id = COALESCE(excluded.candidate_id, geometries.candidate_id),
                dataset_id = excluded.dataset_id,
                h11 = excluded.h11,
                h21 = excluded.h21,
                h31 = excluded.h31,
                euler_char = COALESCE(excluded.euler_char, geometries.euler_char),
                source = COALESCE(excluded.source, geometries.source),
                status = COALESCE(excluded.status, geometries.status),
                note = COALESCE(excluded.note, geometries.note),
                vertex_matrix = COALESCE(excluded.vertex_matrix, geometries.vertex_matrix),
                polytope_vertices = COALESCE(excluded.polytope_vertices, geometries.polytope_vertices),
                triangulation = COALESCE(excluded.triangulation, geometries.triangulation),
                hypersurface_equation = COALESCE(
                    excluded.hypersurface_equation, geometries.hypersurface_equation
                ),
                weight_system = COALESCE(excluded.weight_system, geometries.weight_system),
                configuration_matrix = COALESCE(
                    excluded.configuration_matrix, geometries.configuration_matrix
                ),
                ambient = COALESCE(excluded.ambient, geometries.ambient),
                periods = COALESCE(excluded.periods, geometries.periods),
                orientifold = COALESCE(excluded.orientifold, geometries.orientifold),
                stage = COALESCE(excluded.stage, geometries.stage),
                intersections = COALESCE(excluded.intersections, geometries.intersections),
                extra = COALESCE(excluded.extra, geometries.extra),
                computed_at = COALESCE(geometries.computed_at, excluded.computed_at),
                updated_at = excluded.updated_at
            ''',
            (
                geom_id,
                record.get('candidate_id'),
                dataset_id,
                h11,
                h21,
                h31,
                int(euler) if euler is not None else None,
                record.get('source'),
                status,
                note,
                _dumps(vmat) if not isinstance(vmat, str) else vmat,
                _dumps(verts) if not isinstance(verts, str) else verts,
                triangulation_store,
                record.get('hypersurface_equation'),
                _dumps(record.get('weight_system')),
                _dumps(record.get('configuration_matrix')),
                record.get('ambient'),
                _dumps(record.get('periods')),
                _dumps(record.get('orientifold')),
                stage,
                _dumps(intersections),
                _dumps(extra) if extra else None,
                computed_at,
                now,
            ),
        )
    got = get_by_id(geom_id, db_path=db_path)
    assert got is not None
    return got


def get_by_id(geom_id: str, *, db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    init_db(db_path)
    with _db(db_path) as conn:
        row = conn.execute(
            'SELECT * FROM geometries WHERE id = ?', (geom_id,)
        ).fetchone()
    return _row_to_record(row) if row else None


def get_by_candidate_id(
    candidate_id: str,
    *,
    db_path: str = DEFAULT_DB_PATH,
) -> Optional[Dict[str, Any]]:
    init_db(db_path)
    with _db(db_path) as conn:
        rows = conn.execute(
            'SELECT * FROM geometries WHERE candidate_id = ?',
            (candidate_id,),
        ).fetchall()
    if not rows:
        return None
    records = [_row_to_record(r) for r in rows]
    records.sort(key=_richness, reverse=True)
    return records[0]


def lookup_by_hodge(
    dataset_id: str,
    h11: int,
    h21: int,
    h31: Optional[int] = None,
    *,
    db_path: str = DEFAULT_DB_PATH,
) -> Optional[Dict[str, Any]]:
    """Return the best geometry hit for a Hodge key."""
    init_db(db_path)
    with _db(db_path) as conn:
        if h31 is not None:
            rows = conn.execute(
                '''
                SELECT * FROM geometries
                WHERE dataset_id = ? AND h11 = ? AND h21 = ?
                  AND (h31 IS NULL OR h31 = ?)
                ''',
                (dataset_id, int(h11), int(h21), int(h31)),
            ).fetchall()
        else:
            rows = conn.execute(
                '''
                SELECT * FROM geometries
                WHERE dataset_id = ? AND h11 = ? AND h21 = ?
                ''',
                (dataset_id, int(h11), int(h21)),
            ).fetchall()
    if not rows:
        return None
    records = [_row_to_record(r) for r in rows]
    records.sort(key=_richness, reverse=True)
    return records[0]


def list_geometries(
    *,
    dataset_id: Optional[str] = None,
    status: Optional[str] = None,
    source: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    db_path: str = DEFAULT_DB_PATH,
) -> List[Dict[str, Any]]:
    init_db(db_path)
    clauses: List[str] = []
    params: List[Any] = []
    if dataset_id:
        clauses.append('dataset_id = ?')
        params.append(dataset_id)
    if status:
        clauses.append('status = ?')
        params.append(status)
    if source:
        clauses.append('source = ?')
        params.append(source)
    where = ('WHERE ' + ' AND '.join(clauses)) if clauses else ''
    lim = max(1, min(int(limit), 500))
    off = max(0, int(offset))
    params.extend([lim, off])
    with _db(db_path) as conn:
        rows = conn.execute(
            f'''
            SELECT * FROM geometries
            {where}
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
            ''',
            params,
        ).fetchall()
    return [_row_to_record(r) for r in rows]


def count_geometries(*, db_path: str = DEFAULT_DB_PATH) -> int:
    init_db(db_path)
    with _db(db_path) as conn:
        row = conn.execute('SELECT COUNT(*) AS n FROM geometries').fetchone()
    return int(row['n']) if row else 0


def export_cytools_fields(record: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Subset useful for CYTools / export adapters."""
    if not record:
        return {}
    verts = record.get('polytope_vertices') or record.get('vertex_matrix')
    status = record.get('status') or record.get('geometry_status')
    if verts and not status:
        status = 'representative'
    payload: Dict[str, Any] = {
        'h11': record.get('h11'),
        'h21': record.get('h21'),
        'h31': record.get('h31'),
        'euler_char': record.get('euler_char'),
        'geometry_status': status,
        'geometry_source': record.get('source'),
        'geometry_db_id': record.get('id'),
    }
    if verts:
        payload['polytope_vertices'] = verts
        payload['vertex_matrix'] = record.get('vertex_matrix') or verts
        uniq = record.get('uniqueness') or record.get('geometry_uniqueness')
        payload['uniqueness'] = uniq or (
            'one polytope with these Hodge numbers; not unique'
            if status == 'representative'
            else uniq
        )
    if record.get('triangulation') is not None:
        payload['triangulation'] = record['triangulation']
    if record.get('hypersurface_equation') is not None:
        payload['hypersurface_equation'] = record['hypersurface_equation']
    if record.get('stage') is not None:
        payload['stage'] = record['stage']
    if record.get('pipeline_note') is not None:
        payload['pipeline_note'] = record['pipeline_note']
    if record.get('intersections') is not None:
        payload['intersections'] = record['intersections']
    if record.get('periods') is not None:
        payload['periods'] = record['periods']
    if record.get('note'):
        payload['note'] = record['note']
    return payload


def record_to_pack(record: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Shape a DB record like a geometry-pack / KS-sample blob for merge helpers."""
    if not record:
        return None
    pack: Dict[str, Any] = {
        'dataset_id': record.get('dataset_id'),
        'h11': record.get('h11'),
        'h21': record.get('h21'),
        'h31': record.get('h31'),
        'euler_char': record.get('euler_char'),
        'source': record.get('source'),
        'geometry_status': record.get('status'),
        'status': record.get('status'),
        'note': record.get('note'),
        'ambient': record.get('ambient'),
        'weight_system': record.get('weight_system'),
        'hypersurface_equation': record.get('hypersurface_equation'),
        'polytope_vertices': record.get('polytope_vertices'),
        'vertex_matrix': record.get('vertex_matrix'),
        'triangulation': record.get('triangulation'),
        'configuration_matrix': record.get('configuration_matrix'),
        'periods': record.get('periods'),
        'orientifold': record.get('orientifold'),
        'stage': record.get('stage'),
        'intersections': record.get('intersections'),
        'pipeline_note': record.get('pipeline_note'),
        'geometry_db_id': record.get('id'),
        'geometry_source': record.get('source'),
    }
    extra = record.get('extra') or {}
    if isinstance(extra, dict):
        for key in (
            'name', 'polytope_id', 'triangulation_id', 'vertex_count',
            'facet_count', 'point_count', 'dual_point_count', 'source_slice',
            'uniqueness', 'favourable', 'reference', 'reference_url',
        ):
            if extra.get(key) is not None:
                pack.setdefault(key, extra[key])
        if extra.get('uniqueness') is not None:
            pack.setdefault('geometry_uniqueness', extra['uniqueness'])
    if record.get('reference') is not None:
        pack.setdefault('reference', record['reference'])
    if record.get('reference_url') is not None:
        pack.setdefault('reference_url', record['reference_url'])
    if record.get('uniqueness') is not None:
        pack.setdefault('uniqueness', record['uniqueness'])
        pack.setdefault('geometry_uniqueness', record['uniqueness'])
    return pack


def merge_db_into_raw(
    raw: Optional[Dict[str, Any]],
    db_record: Optional[Dict[str, Any]],
    *,
    prefer_db_vertices: bool = True,
) -> Dict[str, Any]:
    """Merge DB geometry into raw, preferring DB when it has vertices."""
    out = dict(raw or {})
    pack = record_to_pack(db_record)
    if not pack:
        return out

    force = bool(prefer_db_vertices and _has_vertices(pack))
    # When DB is richer (has vertices), overwrite these keys from the DB hit.
    force_keys = {
        'polytope_vertices', 'vertex_matrix', 'geometry_status',
        'geometry_uniqueness', 'vertex_count', 'facet_count',
        'point_count', 'dual_point_count', 'ks_source_slice',
        'triangulation', 'triangulation_id', 'polytope_id',
    }

    mapping = {
        'ambient': 'ambient',
        'weight_system': 'weight_system',
        'hypersurface_equation': 'hypersurface_equation',
        'favourable': 'favourable',
        'polytope_vertices': 'polytope_vertices',
        'vertex_matrix': 'vertex_matrix',
        'triangulation': 'triangulation',
        'triangulation_id': 'triangulation_id',
        'polytope_id': 'polytope_id',
        'configuration_matrix': 'configuration_matrix',
        'geometry_status': 'geometry_status',
        'uniqueness': 'geometry_uniqueness',
        'vertex_count': 'vertex_count',
        'facet_count': 'facet_count',
        'point_count': 'point_count',
        'dual_point_count': 'dual_point_count',
        'source_slice': 'ks_source_slice',
        'periods': 'periods',
        'orientifold': 'orientifold',
    }
    for src, dst in mapping.items():
        val = pack.get(src)
        if val is None:
            continue
        if force and dst in force_keys:
            out[dst] = val
        elif out.get(dst) is None:
            out[dst] = val

    if pack.get('name') and (force or out.get('geometry_name') is None):
        out['geometry_name'] = pack['name']
    if pack.get('note') and (force or out.get('geometry_note') is None):
        out['geometry_note'] = pack['note']

    out['geometry_db_id'] = pack.get('geometry_db_id')
    out['geometry_source'] = pack.get('geometry_source') or pack.get('source')
    if out.get('polytope_vertices') or out.get('vertex_matrix'):
        out.setdefault(
            'geometry_status',
            pack.get('geometry_status') or 'representative',
        )
    return out


def ingest_ks_sample(
    json_path: str = KS_SAMPLE_PATH,
    *,
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    """Upsert polytopes from ``ks_geometry_sample.json``. Returns count upserted."""
    try:
        with open(json_path, 'r', encoding='utf-8') as fh:
            payload = json.load(fh)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return 0
    count = 0
    for item in payload.get('polytopes') or []:
        try:
            h11 = int(item['h11'])
            h21 = int(item['h21'])
        except (KeyError, TypeError, ValueError):
            continue
        dataset_id = str(item.get('dataset_id') or 'kreuzer-skarke')
        verts = item.get('polytope_vertices') or item.get('vertex_matrix')
        status = item.get('geometry_status') or 'representative'
        note = item.get('note') or (
            'Representative from KS HF sample — Hodge numbers do not uniquely '
            'fix a polytope.'
        )
        record = {
            'id': make_geometry_id(
                dataset_id, h11, h21, item.get('h31'),
                source='ks-hf-sample',
                vertices=verts,
            ),
            'dataset_id': dataset_id,
            'h11': h11,
            'h21': h21,
            'h31': item.get('h31'),
            'euler_char': item.get('euler_characteristic') or item.get('euler_char'),
            'source': 'ks-hf-sample',
            'status': status,
            'note': note,
            'polytope_vertices': item.get('polytope_vertices'),
            'vertex_matrix': item.get('vertex_matrix'),
            'extra': {
                k: item[k]
                for k in (
                    'name', 'vertex_count', 'facet_count', 'point_count',
                    'dual_point_count', 'source_slice', 'uniqueness',
                    'reference', 'reference_url',
                )
                if item.get(k) is not None
            },
        }
        upsert_geometry(record, db_path=db_path)
        count += 1
    return count


def ingest_geometry_pack(
    json_path: str = GEOMETRY_PACK_PATH,
    *,
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    """Upsert curated rows from ``geometry_pack.json``. Returns count upserted."""
    try:
        with open(json_path, 'r', encoding='utf-8') as fh:
            payload = json.load(fh)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return 0
    count = 0
    for item in payload.get('geometries') or []:
        try:
            h11 = int(item['h11'])
            h21 = int(item['h21'])
        except (KeyError, TypeError, ValueError):
            continue
        dataset_id = str(item.get('dataset_id') or 'kreuzer-skarke')
        verts = item.get('polytope_vertices') or item.get('vertex_matrix')
        has_verts = isinstance(verts, (list, tuple)) and len(verts) > 0
        status = item.get('geometry_status') or ('curated' if has_verts or item.get('hypersurface_equation') else 'pending')
        record = {
            'id': make_geometry_id(
                dataset_id, h11, h21, item.get('h31'),
                source='geometry-pack',
                vertices=verts or item.get('hypersurface_equation') or item.get('name'),
            ),
            'dataset_id': dataset_id,
            'h11': h11,
            'h21': h21,
            'h31': item.get('h31'),
            'euler_char': item.get('euler_char') or item.get('euler_characteristic'),
            'source': 'geometry-pack',
            'status': status,
            'note': item.get('note') or (
                f"Curated pack: {item['name']}" if item.get('name') else None
            ),
            'polytope_vertices': item.get('polytope_vertices'),
            'vertex_matrix': item.get('vertex_matrix'),
            'triangulation': item.get('triangulation'),
            'hypersurface_equation': item.get('hypersurface_equation'),
            'weight_system': item.get('weight_system'),
            'configuration_matrix': item.get('configuration_matrix'),
            'ambient': item.get('ambient'),
            'stage': item.get('stage'),
            'intersections': item.get('intersections'),
            'extra': {
                k: item[k]
                for k in (
                    'name', 'polytope_id', 'triangulation_id', 'favourable',
                    'reference', 'reference_url', 'uniqueness',
                    'showcase', 'showcase_note', 'periods_literature_pointer',
                )
                if item.get(k) is not None
            },
        }
        upsert_geometry(record, db_path=db_path)
        count += 1
    return count


def seed_baked_geometry(
    *,
    db_path: str = DEFAULT_DB_PATH,
    ks_path: str = KS_SAMPLE_PATH,
    pack_path: str = GEOMETRY_PACK_PATH,
) -> Dict[str, int]:
    """Idempotent boot seed: always upsert shipped JSON (refresh curated rows)."""
    init_db(db_path)
    n_ks = ingest_ks_sample(ks_path, db_path=db_path)
    n_pack = ingest_geometry_pack(pack_path, db_path=db_path)
    return {'ks_sample': n_ks, 'geometry_pack': n_pack, 'total': count_geometries(db_path=db_path)}


def resolve_geometry(
    *,
    candidate_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
    h11: Optional[int] = None,
    h21: Optional[int] = None,
    h31: Optional[int] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> Optional[Dict[str, Any]]:
    """Best DB hit: candidate_id first, then Hodge lookup."""
    if candidate_id:
        hit = get_by_candidate_id(candidate_id, db_path=db_path)
        if hit:
            return hit
    if dataset_id and h11 is not None and h21 is not None:
        return lookup_by_hodge(dataset_id, h11, h21, h31, db_path=db_path)
    return None
