from flask import Flask, render_template, jsonify, request, send_file, Response, abort
import io
import zipfile
import json
import os
import time
import uuid
import numpy as np
import hashlib
from collections import OrderedDict, defaultdict, deque
from datetime import datetime, timezone
from werkzeug.utils import safe_join
from cy_search import run_search, get_sample_results  # Demo implementation
from cy_search_real import (  # Real implementation
    run_real_search,
    list_available_datasets,
    CYSearchEngine,
    SYNTHETIC_RETRIEVAL_HONESTY,
)
from datasets_registry import DatasetRegistry, get_info_density_dataset
import hall_of_fame
import geometry_store
import physics_dossier
import physics_extensions
import job_store
import threading

app = Flask(__name__)

# Configure upload folder for results
RESULTS_DIR = 'static/data'
os.makedirs(RESULTS_DIR, exist_ok=True)
ANALYSIS_DIR = os.path.join('static', 'data', 'analysis')
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Request limits. These bound the work a single unauthenticated request can
# cause: without them one POST can occupy a gunicorn worker indefinitely.
# Public max is intentionally lower than an offline lab run would use.
MAX_N_CANDIDATES = 25000
# Sync requests above this force async job polling so gunicorn workers stay free.
FORCE_ASYNC_N_CANDIDATES = 5000
MAX_TOP_K = 1000
MAX_CUSTOM_ROWS = 10000
CANDIDATE_CACHE_MAXSIZE = 64

# Rate limits (per client IP, in-memory; generous for demos, bounded for abuse).
RATE_LIMIT_WINDOW_SEC = 60
RATE_LIMIT_HEAVY_MAX = 20
# TTL for ephemeral run artifacts under static/data (not HoF / geometry DBs).
RESULTS_TTL_SEC = 48 * 3600
RESULTS_KEEP_LAST_N = 40
CLEANUP_INTERVAL_SEC = 300
SAMPLE_RESULTS_TTL_SEC = 6 * 3600
WEIGHT_SET_TTL_SEC = 3600

# Bounded LRU cache: the key includes user-controlled params, so an unbounded
# dict here grows without limit as callers vary the seed.
CANDIDATE_CACHE = OrderedDict()
FEATURED_PATH = os.path.join('data', 'featured_candidates.json')
HALL_OF_FAME_PATH = os.path.join('static', 'data', 'hall_of_fame.sqlite')
GEOMETRY_DB_PATH = os.path.join('static', 'data', 'geometry.sqlite')

# Per-process caches (multi-worker safe: never mutate shared dataset singletons).
_RATE_LOCK = threading.Lock()
_RATE_HITS = defaultdict(deque)  # ip -> timestamps
_WEIGHT_SETS_LOCK = threading.Lock()
_WEIGHT_SETS = {}  # weight_set_id -> {weights, created_at}
_SAMPLE_CACHE_LOCK = threading.Lock()
_SAMPLE_CACHE = {}  # dataset_id -> {expires_at, payload}
_CLEANUP_LOCK = threading.Lock()
_LAST_CLEANUP_AT = 0.0

# Files / patterns that TTL cleanup must never delete.
_PROTECTED_DATA_NAMES = frozenset({
    'hall_of_fame.sqlite',
    'hall_of_fame.sqlite-wal',
    'hall_of_fame.sqlite-shm',
    'geometry.sqlite',
    'geometry.sqlite-wal',
    'geometry.sqlite-shm',
    'jobs.sqlite',
    'jobs.sqlite-wal',
    'jobs.sqlite-shm',
    'featured_candidates.json',
    'known_constructions.json',
    'metrics.json',
    'repro.md',
    'results_topk.csv',
})


def _utcnow_iso():
    """Timezone-aware UTC timestamp (datetime.utcnow() is deprecated in 3.12+)."""
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _bounded_int(value, default, minimum, maximum):
    """Coerce a request parameter to an int clamped to [minimum, maximum].

    Falls back to `default` when the value is missing or non-numeric, so that
    malformed input yields a usable request rather than a 500.
    """
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(parsed, maximum))


def _is_known_dataset(dataset_id):
    """Check a dataset id without raising (which would leak the registry keys)."""
    return any(d['id'] == dataset_id for d in list_available_datasets())


def _client_ip():
    forwarded = request.headers.get('X-Forwarded-For', '')
    if forwarded:
        return forwarded.split(',')[0].strip() or '0.0.0.0'
    return request.remote_addr or '0.0.0.0'


def _rate_limit_allow(max_hits=RATE_LIMIT_HEAVY_MAX, window_sec=RATE_LIMIT_WINDOW_SEC):
    """Simple IP token bucket. Returns True when the request is allowed."""
    ip = _client_ip()
    now = time.time()
    with _RATE_LOCK:
        bucket = _RATE_HITS[ip]
        while bucket and now - bucket[0] >= window_sec:
            bucket.popleft()
        if len(bucket) >= max_hits:
            return False
        bucket.append(now)
        return True


def _rate_limited_response():
    return jsonify({
        'status': 'error',
        'message': (
            f'Rate limit exceeded ({RATE_LIMIT_HEAVY_MAX} heavy requests per '
            f'{RATE_LIMIT_WINDOW_SEC}s). Retry shortly or use async jobs.'
        ),
    }), 429


def _maybe_cleanup_results(ttl_sec=RESULTS_TTL_SEC, keep_last_n=RESULTS_KEEP_LAST_N):
    """Delete old results_*.json / analysis artifacts; never touch HoF/geometry DBs."""
    global _LAST_CLEANUP_AT
    now = time.time()
    with _CLEANUP_LOCK:
        if now - _LAST_CLEANUP_AT < CLEANUP_INTERVAL_SEC:
            return
        _LAST_CLEANUP_AT = now

    cutoff = now - ttl_sec
    try:
        entries = []
        for name in os.listdir(RESULTS_DIR):
            if name in _PROTECTED_DATA_NAMES:
                continue
            if name.startswith('sample_results_'):
                continue
            path = os.path.join(RESULTS_DIR, name)
            if not os.path.isfile(path):
                continue
            if not (
                name.startswith('results_') and name.endswith('.json')
            ) and not name.endswith('.csv'):
                # Leave unknown files alone except explicit result JSON/CSV.
                if not (name.startswith('results_') or name.startswith('physics_')):
                    continue
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            entries.append((mtime, path, name))

        # Age-based delete, then trim surplus newest results_*.json.
        for mtime, path, name in entries:
            if mtime < cutoff:
                try:
                    os.remove(path)
                except OSError:
                    pass

        result_json = sorted(
            ((m, p) for m, p, n in entries
             if n.startswith('results_') and n.endswith('.json') and os.path.exists(p)),
            reverse=True,
        )
        for _, path in result_json[keep_last_n:]:
            try:
                os.remove(path)
            except OSError:
                pass

        if os.path.isdir(ANALYSIS_DIR):
            for name in os.listdir(ANALYSIS_DIR):
                path = os.path.join(ANALYSIS_DIR, name)
                if not os.path.isfile(path):
                    continue
                try:
                    if os.path.getmtime(path) < cutoff:
                        os.remove(path)
                except OSError:
                    pass
    except OSError:
        pass


def _validate_weight_dict(raw):
    """Validate a partial info-density weight map. Returns (weights, error_response)."""
    dataset = get_info_density_dataset()
    valid_keys = set(dataset.DEFAULT_WEIGHTS.keys())
    if not isinstance(raw, dict):
        return None, (jsonify({'status': 'error', 'message': 'Weights must be a JSON object'}), 400)
    invalid_keys = set(raw.keys()) - valid_keys
    if invalid_keys:
        return None, (jsonify({
            'status': 'error',
            'message': f'Invalid weight keys: {sorted(invalid_keys)}. Valid keys: {sorted(valid_keys)}',
        }), 400)
    cleaned = {}
    for key, value in raw.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            return None, (jsonify({
                'status': 'error',
                'message': f'Weight "{key}" must be a non-negative number',
            }), 400)
        cleaned[key] = float(value)
    return cleaned, None


def _store_weight_set(weights):
    weight_set_id = uuid.uuid4().hex[:12]
    now = time.time()
    with _WEIGHT_SETS_LOCK:
        expired = [k for k, v in _WEIGHT_SETS.items() if now - v['created_at'] > WEIGHT_SET_TTL_SEC]
        for k in expired:
            _WEIGHT_SETS.pop(k, None)
        _WEIGHT_SETS[weight_set_id] = {'weights': dict(weights), 'created_at': now}
    return weight_set_id


def _load_weight_set(weight_set_id):
    if not weight_set_id:
        return None
    now = time.time()
    with _WEIGHT_SETS_LOCK:
        rec = _WEIGHT_SETS.get(weight_set_id)
        if not rec:
            return None
        if now - rec['created_at'] > WEIGHT_SET_TTL_SEC:
            _WEIGHT_SETS.pop(weight_set_id, None)
            return None
        return dict(rec['weights'])


def _resolve_request_weights(params=None):
    """Resolve info-density weights for this request only (no global mutation).

    Precedence: body/query `weights` dict > `weight_set_id` > defaults.
    """
    dataset = get_info_density_dataset()
    params = params or {}
    resolved = dataset.DEFAULT_WEIGHTS.copy()
    weight_set_id = params.get('weight_set_id') or request.args.get('weight_set_id')
    stored = _load_weight_set(weight_set_id)
    if stored:
        resolved.update(stored)
    raw = params.get('weights')
    if raw is None and request.args.get('weights'):
        try:
            raw = json.loads(request.args.get('weights'))
        except (TypeError, ValueError, json.JSONDecodeError):
            raw = None
    if isinstance(raw, dict) and raw:
        cleaned, err = _validate_weight_dict(raw)
        if err:
            return None, err
        resolved.update(cleaned)
    return resolved, None


def _should_force_async(n_candidates, async_mode):
    if async_mode:
        return True
    return n_candidates > FORCE_ASYNC_N_CANDIDATES


def _safe_results_path(filename):
    """Resolve filename under RESULTS_DIR or return None if unsafe / missing dir escape."""
    if filename is None:
        return None
    # Reject obvious traversal tokens before join.
    normalized = filename.replace('\\', '/')
    if normalized.startswith('/') or any(part == '..' for part in normalized.split('/')):
        return None
    joined = safe_join(RESULTS_DIR, filename)
    if not joined:
        return None
    real_root = os.path.realpath(RESULTS_DIR)
    real_path = os.path.realpath(joined)
    if real_path != real_root and not real_path.startswith(real_root + os.sep):
        return None
    return real_path


# Topological invariants that identify a geometry, per dataset. Deliberately
# excludes run artifacts (rank, score, verified_target) so the same manifold
# gets the same id regardless of which search surfaced it.
IDENTITY_FIELDS = {
    'kreuzer-skarke': ('h11', 'h21', 'euler_char'),
    'cy5-folds': ('h11', 'h21', 'h31', 'euler_char'),
    'heterotic': ('h11', 'h21', 'euler_char'),
    'info-density': ('h11', 'h21', 'euler_char'),
    'f-theory-elliptic': ('h11', 'h21', 'euler_char'),
}


def canonical_id(dataset_id, result):
    """Build a stable, content-addressed id for a candidate.

    The old scheme was f"{dataset_id}-{rank:03d}", i.e. a position in one
    ranking: the same string named different manifolds for different seeds or
    n_candidates, so it could not be used to refer to a geometry at all.

    Here the id is derived from the invariants themselves, so it is stable
    across runs and reproducible by anyone holding the same numbers.
    """
    fields = IDENTITY_FIELDS.get(dataset_id, ('h11', 'h21', 'euler_char'))
    missing = [f for f in fields if result.get(f) is None]
    if missing:
        raise ValueError(f"cannot build id, missing invariants: {missing}")
    # Canonical text form -> hash. Sorted, explicit, and stable across versions.
    payload = f"{dataset_id}|" + "|".join(f"{f}={int(result[f])}" for f in fields)
    digest = hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]
    return f"{dataset_id}-{digest}"


def identity_payload(dataset_id, result):
    """The exact invariants an id commits to, for display and verification."""
    fields = IDENTITY_FIELDS.get(dataset_id, ('h11', 'h21', 'euler_char'))
    return {f: int(result[f]) for f in fields if result.get(f) is not None}


# Seed curated examples into the persistent board on first boot (empty DB only).
hall_of_fame.init_db(HALL_OF_FAME_PATH)
hall_of_fame.seed_from_featured(
    featured_path=FEATURED_PATH,
    canonical_id_fn=canonical_id,
    db_path=HALL_OF_FAME_PATH,
)
# Always ensure textbook/curated seeds exist even when the board is already populated.
hall_of_fame.ensure_featured_by_tags(
    featured_path=FEATURED_PATH,
    canonical_id_fn=canonical_id,
    db_path=HALL_OF_FAME_PATH,
    required_tags=['textbook', 'curated'],
)

# Geometry SQLite: create on boot and upsert baked KS sample + geometry pack.
# Offline-computed rows (other source hashes) are preserved; curated seeds refresh.
geometry_store.init_db(GEOMETRY_DB_PATH)
geometry_store.seed_baked_geometry(db_path=GEOMETRY_DB_PATH)


@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')


@app.route('/docs.html')
def docs():
    """Documentation page"""
    return render_template('docs.html')


@app.route('/results.html')
def results():
    """Demo results page"""
    return render_template('results.html')


@app.route('/about.html')
def about():
    """About page"""
    return render_template('about.html')


@app.route('/demo.html')
def demo():
    """Interactive demo page"""
    return render_template('demo.html')


@app.route('/candidates.html')
def candidates():
    """Candidate gallery / hall of fame"""
    return render_template('candidates.html')


@app.route('/candidate/<path:candidate_id>')
def candidate_page(candidate_id):
    """Shareable permanent detail page for one hall-of-fame geometry."""
    candidate = hall_of_fame.get_candidate(candidate_id, db_path=HALL_OF_FAME_PATH)
    if not candidate:
        # Fall back to API-shaped lookup across live cache / featured seed ids.
        return render_template('candidate.html', candidate=None, candidate_id=candidate_id), 404

    features = dict(candidate.get('features') or [])
    h11 = candidate.get('h11') if candidate.get('h11') is not None else features.get('h11')
    h21 = candidate.get('h21') if candidate.get('h21') is not None else features.get('h21')
    h31 = candidate.get('h31') if candidate.get('h31') is not None else features.get('h31')
    chi = candidate.get('euler_char')
    if chi is None:
        chi = features.get('χ', features.get('euler_char'))

    dataset_id = candidate.get('dataset_id') or 'kreuzer-skarke'
    dossier = physics_dossier.build_dossier(
        dataset_id=dataset_id,
        h11=h11,
        h21=h21,
        h31=h31,
        euler_char=chi,
        verified_target=candidate.get('verified_target'),
    )
    tabs = None
    mirror_partner = None
    if dossier.get('ok'):
        h11 = dossier['h11']
        h21 = dossier['h21']
        chi = dossier['euler_char']
        # Resolve Hodge-mirror partner on the board (CY3-style swap).
        if dataset_id != 'cy5-folds' and h11 is not None and h21 is not None:
            mirror_record = {
                'h11': int(h21),
                'h21': int(h11),
                'euler_char': int(-chi) if chi is not None else int(2 * (h21 - h11)),
            }
            try:
                mirror_cid = canonical_id(dataset_id, mirror_record)
            except ValueError:
                mirror_cid = None
            mirror_hit = hall_of_fame.get_candidate(mirror_cid, db_path=HALL_OF_FAME_PATH) if mirror_cid else None
            if mirror_hit is None:
                for item in hall_of_fame.list_candidates(
                    dataset_id=dataset_id, limit=500, db_path=HALL_OF_FAME_PATH
                ):
                    if item.get('h11') == mirror_record['h11'] and item.get('h21') == mirror_record['h21']:
                        mirror_hit = item
                        break
            mirror_partner = {
                'h11': mirror_record['h11'],
                'h21': mirror_record['h21'],
                'euler_char': mirror_record['euler_char'],
                'candidate_id': (mirror_hit or {}).get('candidate_id') or mirror_cid,
                'on_board': bool(mirror_hit),
                'display': (
                    f"h¹¹={mirror_record['h11']} · h²¹={mirror_record['h21']} · "
                    f"χ={mirror_record['euler_char']}"
                ),
            }

        construction = physics_dossier.construction_payload(
            dataset_id=dataset_id,
            candidate_id=candidate_id,
            raw=candidate.get('raw'),
            features=candidate.get('features'),
            tags=candidate.get('tags'),
            summary=candidate.get('summary'),
            h11=h11,
            h21=h21,
            h31=h31,
            euler_char=chi,
        )
        tabs = physics_dossier.build_tabs(
            dossier,
            construction=construction,
            tags=candidate.get('tags'),
            mirror_partner=mirror_partner,
        )

    neighbors = []
    if h11 is not None and h21 is not None:
        board = hall_of_fame.list_candidates(
            dataset_id=candidate.get('dataset_id'),
            limit=200,
            db_path=HALL_OF_FAME_PATH,
        )
        scored = []
        for item in board:
            ih11 = item.get('h11')
            ih21 = item.get('h21')
            if ih11 is None or ih21 is None:
                continue
            if item.get('candidate_id') == candidate_id:
                continue
            dist = physics_dossier.neighbor_distance(
                (int(h11), int(h21)), (int(ih11), int(ih21)))
            scored.append({
                'candidate_id': item.get('candidate_id'),
                'h11': int(ih11),
                'h21': int(ih21),
                'euler_char': item.get('euler_char'),
                'score': item.get('score'),
                'verified_target': item.get('verified_target'),
                'distance': round(dist, 3),
            })
        scored.sort(key=lambda row: row['distance'])
        neighbors = scored[:12]

    og_parts = []
    if h11 is not None:
        og_parts.append(f"h¹¹={h11}")
    if h21 is not None:
        og_parts.append(f"h²¹={h21}")
    if h31 is not None:
        og_parts.append(f"h³¹={h31}")
    if chi is not None:
        og_parts.append(f"χ={chi}")
    display_title = " · ".join(og_parts) if og_parts else candidate['candidate_id']
    og_title = f"{display_title} — upg-strings"
    og_description = (
        f"{candidate.get('dataset_name') or candidate.get('dataset_id')}: "
        f"{display_title}, "
        f"{'verified' if candidate.get('verified_target') else 'unverified'}, "
        f"best score {float(candidate.get('score') or 0):.4f}"
    )
    og_url = request.url
    return render_template(
        'candidate.html',
        candidate=candidate,
        candidate_id=candidate_id,
        display_title=display_title,
        h11=h11,
        h21=h21,
        h31=h31,
        chi=chi,
        dossier=dossier if dossier.get('ok') else None,
        tabs=tabs if tabs and tabs.get('ok') else None,
        neighbors=neighbors,
        og_title=og_title,
        og_description=og_description,
        og_url=og_url,
    )


@app.route('/eli5.html')
def eli5():
    """ELI5 page"""
    return render_template('eli5.html')


@app.route('/lookup.html')
def lookup():
    """Look up a geometry by its invariants"""
    return render_template('lookup.html', active_page='lookup')


@app.route('/render.html')
def render_page():
    """Full renderer page"""
    candidate_id = request.args.get('candidate_id', 'candidate')
    dataset_id = request.args.get('dataset_id', '')
    seed = request.args.get('seed', '4242')
    og_title = f"3D Render — {candidate_id}"
    if dataset_id:
        og_title += f" ({dataset_id})"
    og_description = "Interactive 3D render of a Calabi-Yau candidate."
    og_url = request.url
    og_image = request.url_root.rstrip('/') + '/static/assets/og-render.svg'

    return render_template(
        'render.html',
        candidate_id=candidate_id,
        dataset_id=dataset_id,
        seed=seed,
        og_title=og_title,
        og_description=og_description,
        og_url=og_url,
        og_image=og_image
    )


@app.route('/api/datasets')
def list_datasets():
    """List all available datasets"""
    datasets = list_available_datasets()
    return jsonify({'datasets': datasets})


@app.route('/api/run-demo', methods=['POST'])
def run_demo():
    """
    API endpoint to run the upg-strings search

    Accepts JSON payload:
    {
        "top_k": 100,
        "seed": 42,
        "verify": true,
        "n_candidates": 5000,
        "dataset_id": "kreuzer-skarke",  # or "cy5-folds", "heterotic", ...
        "use_real": true,
        "async": false,   # if true (or n_candidates > FORCE_ASYNC), return job_id
        "weights": {...},           # info-density only; request-scoped
        "weight_set_id": "..."      # from POST /api/info-density/weights
    }

    Public max n_candidates is MAX_N_CANDIDATES (25000). Requests above
    FORCE_ASYNC_N_CANDIDATES (5000) are forced async.
    """
    _maybe_cleanup_results()
    if not _rate_limit_allow():
        return _rate_limited_response()
    try:
        params = request.get_json() or {}
        top_k = _bounded_int(params.get('top_k'), 100, 1, MAX_TOP_K)
        seed = _bounded_int(params.get('seed'), 42, 0, 2**32 - 1)
        verify = params.get('verify', True)
        n_candidates = _bounded_int(params.get('n_candidates'), 5000, 10, MAX_N_CANDIDATES)
        dataset_id = params.get('dataset_id', 'kreuzer-skarke')
        use_real = params.get('use_real', True)  # Default to real implementation
        async_mode = _should_force_async(
            n_candidates, bool(params.get('async') or params.get('async_mode'))
        )
        weights = None
        if dataset_id == 'info-density':
            weights, err = _resolve_request_weights(params)
            if err:
                return err

        if not _is_known_dataset(dataset_id):
            return jsonify({
                'status': 'error',
                'message': f'Unknown dataset: {dataset_id}'
            }), 400

        if async_mode:
            job_id = job_store.create_job(stage='queued')

            def _worker():
                try:
                    job_store.update_job(job_id, status='running', percent=5, stage='init')
                    job_store.update_job(job_id, percent=20, stage='generate_candidates')
                    if use_real:
                        job_store.update_job(job_id, percent=40, stage='train_and_rank')
                        results = run_real_search(
                            top_k=top_k,
                            seed=seed,
                            n_candidates=n_candidates,
                            verify=verify,
                            dataset_id=dataset_id,
                            weights=weights,
                        )
                    else:
                        job_store.update_job(job_id, percent=40, stage='demo_search')
                        results = run_search(top_k=top_k, seed=seed, verify=verify)
                    job_store.update_job(job_id, percent=85, stage='persist')
                    run_id = _save_results(results)
                    payload = {
                        'status': 'success',
                        'run_id': run_id,
                        'results': results,
                        'results_url': f'/api/results/{run_id}',
                    }
                    job_store.update_job(
                        job_id,
                        status='completed',
                        percent=100,
                        stage='done',
                        result=payload,
                    )
                except Exception as exc:
                    job_store.update_job(
                        job_id,
                        status='failed',
                        percent=100,
                        stage='error',
                        error=str(exc),
                    )

            threading.Thread(target=_worker, daemon=True).start()
            return jsonify({
                'status': 'accepted',
                'job_id': job_id,
                'progress_url': f'/api/jobs/{job_id}',
                'forced_async': n_candidates > FORCE_ASYNC_N_CANDIDATES and not bool(
                    params.get('async') or params.get('async_mode')
                ),
                'message': 'Job queued; poll progress_url until status=completed',
            }), 202

        # Run the search - use real implementation by default
        if use_real:
            print(f"Running upg-strings: dataset={dataset_id}, {n_candidates} candidates, top_k={top_k}, seed={seed}")
            results = run_real_search(
                top_k=top_k,
                seed=seed,
                n_candidates=n_candidates,
                verify=verify,
                dataset_id=dataset_id,
                weights=weights,
            )
        else:
            print(f"Running DEMO: top_k={top_k}, seed={seed}")
            results = run_search(top_k=top_k, seed=seed, verify=verify)

        # Save results
        run_id = _save_results(results)

        return jsonify({
            'status': 'success',
            'run_id': run_id,
            'message': 'Demo completed successfully',
            'results': results,
            'results_url': f'/api/results/{run_id}'
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/jobs/<job_id>')
def get_job_progress(job_id):
    """Poll progress for an async /api/run-demo job."""
    job = job_store.get_job(job_id)
    if not job:
        return jsonify({'status': 'error', 'message': 'Job not found'}), 404
    return jsonify({
        'status': 'success',
        'job': {
            'job_id': job['job_id'],
            'status': job['status'],
            'percent': job['percent'],
            'stage': job.get('stage'),
            'error': job.get('error'),
            'result': job.get('result') if job['status'] == 'completed' else None,
        },
    })


@app.route('/api/batch', methods=['POST'])
def batch_api():
    """Bounded batch of identify / search jobs (max 50)."""
    _maybe_cleanup_results()
    if not _rate_limit_allow():
        return _rate_limited_response()
    params = request.get_json() or {}
    jobs = params.get('jobs') or params.get('requests') or []
    if not isinstance(jobs, list) or not jobs:
        return jsonify({'status': 'error', 'message': 'Provide a non-empty jobs array'}), 400
    if len(jobs) > 50:
        return jsonify({'status': 'error', 'message': 'Too many jobs (max 50)'}), 400

    results = []
    for idx, job in enumerate(jobs):
        if not isinstance(job, dict):
            results.append({'index': idx, 'status': 'error', 'message': 'job must be an object'})
            continue
        kind = (job.get('type') or job.get('op') or 'identify').lower()
        try:
            if kind == 'identify':
                # Reuse identify logic via internal call shape
                with app.test_request_context(
                    '/api/identify',
                    method='POST',
                    json=job.get('params') or job,
                ):
                    resp = identify()
                    payload = resp.get_json() if hasattr(resp, 'get_json') else resp[0].get_json()
                    status_code = resp.status_code if hasattr(resp, 'status_code') else resp[1]
                results.append({
                    'index': idx,
                    'type': 'identify',
                    'status': 'success' if status_code < 400 else 'error',
                    'http_status': status_code,
                    'result': payload,
                })
            elif kind == 'search':
                with app.test_request_context(
                    '/api/search',
                    method='POST',
                    json=job.get('params') or job,
                ):
                    resp = search_candidates()
                    payload = resp.get_json() if hasattr(resp, 'get_json') else resp[0].get_json()
                    status_code = resp.status_code if hasattr(resp, 'status_code') else resp[1]
                results.append({
                    'index': idx,
                    'type': 'search',
                    'status': 'success' if status_code < 400 else 'error',
                    'http_status': status_code,
                    'result': payload,
                })
            else:
                results.append({
                    'index': idx,
                    'status': 'error',
                    'message': f'Unsupported job type: {kind} (use identify|search)',
                })
        except Exception as exc:
            results.append({'index': idx, 'type': kind, 'status': 'error', 'message': str(exc)})

    return jsonify({
        'status': 'success',
        'count': len(results),
        'results': results,
    })


@app.route('/api/toy-soft', methods=['POST'])
def toy_soft_api():
    """Illustrative soft-parameter card — not derived from any CY."""
    params = request.get_json() or {}
    try:
        card = physics_extensions.toy_soft_parameter_card(
            A0=float(params.get('A0', 0.0)),
            m12=float(params.get('m12', params.get('m1_2', 500.0))),
            tan_beta=float(params.get('tan_beta', params.get('tanb', 10.0))),
            m0=float(params.get('m0', 500.0)),
        )
    except (TypeError, ValueError) as exc:
        return jsonify({'status': 'error', 'message': str(exc)}), 400
    return jsonify({'status': 'success', 'soft_toy_card': card})


@app.route('/api/results/<run_id>')
def get_results(run_id):
    """Get results for a specific run"""
    results = _load_results(run_id)
    if results is None:
        return jsonify({'error': 'Results not found'}), 404
    return jsonify(results)


@app.route('/api/sample-results')
def sample_results():
    """Cached sample results for display (avoids retraining RF on every hit)."""
    _maybe_cleanup_results()
    dataset_id = request.args.get('dataset_id', 'kreuzer-skarke')
    if not _is_known_dataset(dataset_id):
        return jsonify({'status': 'error', 'message': f'Unknown dataset: {dataset_id}'}), 400

    now = time.time()
    with _SAMPLE_CACHE_LOCK:
        cached = _SAMPLE_CACHE.get(dataset_id)
        if cached and cached['expires_at'] > now:
            return jsonify(cached['payload'])

    # Prefer durable fixture under static/data/ (immutable until TTL file mtime ages out).
    fixture_name = f'sample_results_{dataset_id}.json'
    fixture_path = _safe_results_path(fixture_name)
    if fixture_path and os.path.isfile(fixture_path):
        try:
            age = now - os.path.getmtime(fixture_path)
            if age < SAMPLE_RESULTS_TTL_SEC:
                with open(fixture_path, 'r') as f:
                    payload = json.load(f)
                with _SAMPLE_CACHE_LOCK:
                    _SAMPLE_CACHE[dataset_id] = {
                        'expires_at': now + SAMPLE_RESULTS_TTL_SEC,
                        'payload': payload,
                    }
                return jsonify(payload)
        except (OSError, json.JSONDecodeError):
            pass

    results = run_real_search(
        top_k=20,
        seed=42,
        n_candidates=1000,
        verify=True,
        dataset_id=dataset_id,
    )
    try:
        write_path = os.path.join(RESULTS_DIR, fixture_name)
        with open(write_path, 'w') as f:
            json.dump(results, f)
    except OSError:
        pass

    with _SAMPLE_CACHE_LOCK:
        _SAMPLE_CACHE[dataset_id] = {
            'expires_at': now + SAMPLE_RESULTS_TTL_SEC,
            'payload': results,
        }
    return jsonify(results)


def _save_results(results):
    run_id = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(RESULTS_DIR, f'results_{run_id}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    # Promote verified hits onto the permanent board.
    try:
        hall_of_fame.promote_from_run(
            results,
            run_id=run_id,
            canonical_id_fn=canonical_id,
            identity_payload_fn=identity_payload,
            db_path=HALL_OF_FAME_PATH,
        )
    except Exception:
        # Persistence of the run file already succeeded; hall-of-fame is best-effort.
        pass
    return run_id


def _load_results(run_id):
    results_file = os.path.join(RESULTS_DIR, f'results_{run_id}.json')
    if not os.path.exists(results_file):
        return None
    with open(results_file, 'r') as f:
        return json.load(f)


def _build_candidate_cards(dataset_id, seed=42, top_n=12, n_candidates=5000, weights=None):
    weight_key = tuple(sorted((weights or {}).items())) if weights else ()
    cache_key = (dataset_id, seed, top_n, n_candidates, weight_key)
    if cache_key in CANDIDATE_CACHE:
        CANDIDATE_CACHE.move_to_end(cache_key)
        return CANDIDATE_CACHE[cache_key]

    results = run_real_search(
        top_k=top_n,
        seed=seed,
        n_candidates=n_candidates,
        verify=True,
        dataset_id=dataset_id,
        weights=weights,
    )

    dataset = DatasetRegistry.get_dataset(dataset_id)
    metadata = dataset.get_metadata()

    candidates = []
    for result in results["top_results"]:
        # Content-addressed: stable across seeds/runs. The rank-based string is
        # retained separately because it is only meaningful within one run.
        candidate_id = canonical_id(dataset_id, result)
        rank_label = f"{dataset_id}-{result['rank']:03d}"
        if dataset_id == 'cy5-folds':
            feature_pairs = [
                ("h11", result.get("h11")),
                ("h21", result.get("h21")),
                ("h31", result.get("h31"))
            ]
        elif dataset_id == 'heterotic':
            feature_pairs = [
                ("h11", result.get("h11")),
                ("h21", result.get("h21")),
                ("balance", round(result.get("hodge_balance", 0), 3))
            ]
        elif dataset_id == 'info-density':
            feature_pairs = [
                ("h11", result.get("h11")),
                ("h21", result.get("h21")),
                ("info_ρ", round(result.get("info_density", 0), 3))
            ]
        else:
            feature_pairs = [
                ("h11", result.get("h11")),
                ("h21", result.get("h21")),
                ("χ", result.get("euler_char"))
            ]

        candidates.append({
            "candidate_id": candidate_id,
            "rank_label": rank_label,
            "identity": identity_payload(dataset_id, result),
            "identity_scheme": "upg-strings/sha256-invariants-v1",
            "rank": result.get("rank"),
            "score": result.get("score"),
            "verified_target": result.get("verified_target"),
            "dataset_id": dataset_id,
            "dataset_name": metadata.name,
            "target_description": metadata.target_description,
            "features": feature_pairs,
            "raw": result,
            "viz_seed": int(hashlib.md5(candidate_id.encode('utf-8')).hexdigest()[:6], 16)
        })

    payload = {
        "dataset": {
            "id": dataset_id,
            "name": metadata.name,
            "description": metadata.description,
            "target_description": metadata.target_description
        },
        "candidates": candidates
    }
    CANDIDATE_CACHE[cache_key] = payload
    while len(CANDIDATE_CACHE) > CANDIDATE_CACHE_MAXSIZE:
        CANDIDATE_CACHE.popitem(last=False)

    # Verified hits from live gallery browses also grow the permanent board.
    try:
        hall_of_fame.promote_from_run(
            {
                'dataset_id': dataset_id,
                'run_metadata': {
                    'dataset_id': dataset_id,
                    'dataset_name': metadata.name,
                },
                'top_results': results['top_results'],
            },
            run_id=f'live-{dataset_id}-{seed}-{top_n}',
            canonical_id_fn=canonical_id,
            identity_payload_fn=identity_payload,
            db_path=HALL_OF_FAME_PATH,
        )
    except Exception:
        pass

    return payload


@app.route('/api/candidates')
def top_candidates():
    dataset_id = request.args.get('dataset_id', 'kreuzer-skarke')
    seed = _bounded_int(request.args.get('seed'), 42, 0, 2**32 - 1)
    top_n = _bounded_int(request.args.get('top_n'), 12, 1, MAX_TOP_K)
    n_candidates = _bounded_int(request.args.get('n_candidates'), 5000, 10, MAX_N_CANDIDATES)

    if not _is_known_dataset(dataset_id):
        return jsonify({'status': 'error', 'message': f'Unknown dataset: {dataset_id}'}), 400

    payload = _build_candidate_cards(
        dataset_id=dataset_id,
        seed=seed,
        top_n=top_n,
        n_candidates=n_candidates
    )
    return jsonify({'status': 'success', **payload})


@app.route('/api/candidate/<path:candidate_id>')
def candidate_detail(candidate_id):
    # Permanent board first — shareable ids resolve here without re-running search.
    hof = hall_of_fame.get_candidate(candidate_id, db_path=HALL_OF_FAME_PATH)
    if hof:
        raw = hof.get('raw') or {}
        invariants = []
        if hof.get('euler_char') is not None:
            invariants.append({'label': 'Euler χ', 'value': hof.get('euler_char')})
        elif 'euler_char' in raw:
            invariants.append({'label': 'Euler χ', 'value': raw.get('euler_char')})
        for key, label in (
            ('hodge_balance', 'Hodge balance'),
            ('n_generations', 'Generations'),
            ('info_density', 'Info density'),
            ('times_seen', 'Times seen'),
        ):
            if key in raw and raw.get(key) is not None:
                val = raw.get(key)
                if isinstance(val, float):
                    val = round(val, 4)
                invariants.append({'label': label, 'value': val})
        if hof.get('times_seen') is not None:
            invariants.append({'label': 'Times seen', 'value': hof.get('times_seen')})
        if hof.get('first_seen_at'):
            invariants.append({'label': 'First seen', 'value': hof.get('first_seen_at')})
        if hof.get('last_seen_at'):
            invariants.append({'label': 'Last seen', 'value': hof.get('last_seen_at')})
        detail = hof.copy()
        detail.update({
            'invariants': invariants,
            'summary': hof.get('summary') or 'Hall of fame candidate',
            'detail_url': f'/candidate/{candidate_id}',
        })
        return jsonify({'status': 'success', 'candidate': detail})

    dataset_id = request.args.get('dataset_id', 'kreuzer-skarke')
    seed = _bounded_int(request.args.get('seed'), 42, 0, 2**32 - 1)
    top_n = _bounded_int(request.args.get('top_n'), 12, 1, MAX_TOP_K)
    n_candidates = _bounded_int(request.args.get('n_candidates'), 5000, 10, MAX_N_CANDIDATES)

    if not _is_known_dataset(dataset_id):
        return jsonify({'status': 'error', 'message': f'Unknown dataset: {dataset_id}'}), 400

    payload = _build_candidate_cards(
        dataset_id=dataset_id,
        seed=seed,
        top_n=top_n,
        n_candidates=n_candidates
    )

    for candidate in payload["candidates"]:
        if candidate["candidate_id"] == candidate_id:
            detail = candidate.copy()
            raw = detail.get("raw", {})
            invariants = []
            if "euler_char" in raw:
                invariants.append({"label": "Euler χ", "value": raw.get("euler_char")})
            if "hodge_balance" in raw:
                invariants.append({"label": "Hodge balance", "value": round(raw.get("hodge_balance", 0), 3)})
            if "n_generations" in raw:
                invariants.append({"label": "Generations", "value": raw.get("n_generations")})
            if "info_density" in raw:
                invariants.append({"label": "Info density", "value": round(raw.get("info_density", 0), 4)})
            if "hodge_entropy" in raw:
                invariants.append({"label": "Hodge entropy", "value": round(raw.get("hodge_entropy", 0), 4)})
            if "topo_efficiency" in raw:
                invariants.append({"label": "Topo efficiency", "value": round(raw.get("topo_efficiency", 0), 4)})
            if "flux_density" in raw:
                invariants.append({"label": "Flux density", "value": round(raw.get("flux_density", 0), 4)})
            if "vacuum_stability" in raw:
                invariants.append({"label": "Vacuum stability", "value": round(raw.get("vacuum_stability", 0), 4)})
            if "tadpole_charge" in raw:
                invariants.append({"label": "Tadpole (χ/24)", "value": round(raw.get("tadpole_charge", 0), 2)})

            detail.update({
                "invariants": invariants,
                "summary": f"Target: {detail.get('target_description')}",
                "detail_url": f'/candidate/{candidate_id}',
            })

            return jsonify({'status': 'success', 'candidate': detail})

    return jsonify({'status': 'error', 'message': 'Candidate not found'}), 404


@app.route('/api/featured-candidates')
def featured_candidates():
    """Hall of fame listing (seeded from featured JSON, grown by verified runs)."""
    dataset_id = request.args.get('dataset_id')
    tag = request.args.get('tag')
    verified = request.args.get('verified')
    verified_only = None
    if verified is not None:
        verified_only = verified.lower() == 'true'

    candidates = hall_of_fame.list_candidates(
        dataset_id=dataset_id or None,
        verified_only=verified_only,
        tag=tag,
        limit=_bounded_int(request.args.get('top_n'), 100, 1, 500),
        db_path=HALL_OF_FAME_PATH,
    )
    return jsonify({'status': 'success', 'candidates': candidates, 'source': 'hall_of_fame'})


@app.route('/api/geometry')
def api_geometry_list():
    """List geometry DB rows (bounded). Offline / seeded SQLite, not live CYTools."""
    rows = geometry_store.list_geometries(
        dataset_id=request.args.get('dataset_id') or None,
        status=request.args.get('status') or None,
        source=request.args.get('source') or None,
        limit=_bounded_int(request.args.get('limit'), 50, 1, 500),
        offset=_bounded_int(request.args.get('offset'), 0, 0, 100000),
        db_path=GEOMETRY_DB_PATH,
    )
    return jsonify({
        'status': 'success',
        'count': len(rows),
        'geometries': rows,
        'source': 'geometry_db',
    })


@app.route('/api/geometry/lookup')
def api_geometry_lookup():
    """Best geometry hit for a Hodge key (preferring richer / vertex-bearing rows)."""
    dataset_id = request.args.get('dataset_id') or 'kreuzer-skarke'
    try:
        h11 = int(request.args.get('h11'))
        h21 = int(request.args.get('h21'))
    except (TypeError, ValueError):
        return jsonify({
            'status': 'error',
            'message': 'h11 and h21 query parameters are required integers',
        }), 400
    h31 = None
    if request.args.get('h31') not in (None, ''):
        try:
            h31 = int(request.args.get('h31'))
        except (TypeError, ValueError):
            return jsonify({'status': 'error', 'message': 'h31 must be an integer'}), 400
    hit = geometry_store.lookup_by_hodge(
        dataset_id, h11, h21, h31, db_path=GEOMETRY_DB_PATH,
    )
    if not hit:
        return jsonify({
            'status': 'not_found',
            'message': 'No geometry DB row for this Hodge key',
            'query': {
                'dataset_id': dataset_id, 'h11': h11, 'h21': h21, 'h31': h31,
            },
        }), 404
    return jsonify({'status': 'success', 'geometry': hit, 'source': 'geometry_db'})


@app.route('/api/geometry/<path:candidate_id>')
def api_geometry_by_candidate(candidate_id):
    """Geometry linked to a Hall-of-Fame candidate id, else Hodge resolve from HoF."""
    hit = geometry_store.get_by_candidate_id(candidate_id, db_path=GEOMETRY_DB_PATH)
    if not hit:
        hof = hall_of_fame.get_candidate(candidate_id, db_path=HALL_OF_FAME_PATH)
        if hof and hof.get('h11') is not None and hof.get('h21') is not None:
            hit = geometry_store.lookup_by_hodge(
                hof.get('dataset_id') or 'kreuzer-skarke',
                int(hof['h11']),
                int(hof['h21']),
                hof.get('h31'),
                db_path=GEOMETRY_DB_PATH,
            )
    if not hit:
        return jsonify({
            'status': 'not_found',
            'message': 'No geometry DB row for this candidate',
            'candidate_id': candidate_id,
        }), 404
    return jsonify({
        'status': 'success',
        'candidate_id': candidate_id,
        'geometry': hit,
        'source': 'geometry_db',
    })


@app.route('/api/identify', methods=['POST'])
def identify():
    """Look up a geometry by its invariants rather than by rank position.

    This is the "I have an interesting manifold, is it in your system?" entry
    point. The caller supplies the invariants they already know (h11, h21, and
    for CY5 h31); we return the canonical id, the derived quantities, and how
    the manifold scores.

    IMPORTANT: the id returned is local to this system and is NOT a
    community-standard identifier. See `identifier_note` in the response.
    """
    params = request.get_json() or {}
    dataset_id = params.get('dataset_id', 'kreuzer-skarke')

    if not _is_known_dataset(dataset_id):
        return jsonify({'status': 'error', 'message': f'Unknown dataset: {dataset_id}'}), 400

    fields = IDENTITY_FIELDS.get(dataset_id, ('h11', 'h21', 'euler_char'))
    required = [f for f in fields if f != 'euler_char']

    provided = {}
    for field in required:
        if params.get(field) is None:
            return jsonify({
                'status': 'error',
                'message': f'Missing required invariant: {field}',
                'required': required
            }), 400
        try:
            provided[field] = int(params[field])
        except (TypeError, ValueError):
            return jsonify({'status': 'error', 'message': f'{field} must be an integer'}), 400

    if any(v < 0 for v in provided.values()):
        return jsonify({'status': 'error', 'message': 'Hodge numbers must be non-negative'}), 400

    # Euler characteristic is derived, not supplied, so it always agrees with
    # the Hodge numbers rather than being trusted from the caller.
    if dataset_id == 'cy5-folds':
        euler = 6 + 6 * (provided['h11'] - provided['h21'] + provided['h31'])
    else:
        euler = 2 * (provided['h11'] - provided['h21'])

    record = dict(provided)
    record['euler_char'] = euler

    supplied_euler = params.get('euler_char')
    euler_mismatch = None
    if supplied_euler is not None:
        try:
            if int(supplied_euler) != euler:
                euler_mismatch = (
                    f'Supplied euler_char={int(supplied_euler)} disagrees with '
                    f'the value derived from the Hodge numbers ({euler}). '
                    'Using the derived value.'
                )
        except (TypeError, ValueError):
            euler_mismatch = 'Supplied euler_char was not an integer; ignored.'

    cid = canonical_id(dataset_id, record)
    hof = hall_of_fame.get_candidate(cid, db_path=HALL_OF_FAME_PATH)
    dossier = physics_dossier.build_dossier(
        dataset_id=dataset_id,
        h11=record.get('h11'),
        h21=record.get('h21'),
        h31=record.get('h31'),
        euler_char=euler,
        verified_target=(hof or {}).get('verified_target'),
    )
    tabs = None
    if dossier.get('ok'):
        construction = physics_dossier.construction_payload(
            dataset_id=dataset_id,
            candidate_id=cid,
            raw=(hof or {}).get('raw') or record,
            features=(hof or {}).get('features'),
            tags=(hof or {}).get('tags'),
            summary=(hof or {}).get('summary'),
            h11=dossier['h11'],
            h21=dossier['h21'],
            h31=dossier.get('h31'),
            euler_char=dossier['euler_char'],
        )
        tabs = physics_dossier.build_tabs(
            dossier,
            construction=construction,
            tags=(hof or {}).get('tags'),
        )

    response = {
        'status': 'success',
        'candidate_id': cid,
        'identity': identity_payload(dataset_id, record),
        'identity_scheme': 'upg-strings/sha256-invariants-v1',
        'dataset_id': dataset_id,
        'detail_url': f'/candidate/{cid}',
        'in_hall_of_fame': bool(hof),
        'hall_of_fame': {
            'score': hof.get('score'),
            'verified_target': hof.get('verified_target'),
            'times_seen': hof.get('times_seen'),
            'last_seen_at': hof.get('last_seen_at'),
        } if hof else None,
        'derived': {
            'euler_char': euler,
            'abs_euler': abs(euler),
            'total_moduli': sum(provided.values()),
            'tadpole_charge_chi_over_24': round(abs(euler) / 24, 4),
        },
        'dossier': dossier if dossier.get('ok') else None,
        'tabs': tabs if tabs and tabs.get('ok') else None,
        'identifier_note': (
            'This id is local to upg-strings and is NOT a community-standard '
            'identifier. Hodge numbers do not uniquely determine a Calabi-Yau '
            'manifold: distinct geometries can share (h11, h21, chi). To refer '
            'to a specific geometry unambiguously, cite the Kreuzer-Skarke '
            'polytope (its vertex matrix) plus the triangulation used.'
        ),
        'uniqueness': 'non-unique: many manifolds may share these invariants',
    }
    if dossier.get('ok'):
        response['derived'].update({
            'n_generations': abs(euler) // 2,
            'picard_fuchs_order': dossier['h21'] + 1,
            'flux_density': dossier['scalars'].get('flux_density'),
            'vacuum_stability': dossier['scalars'].get('vacuum_stability'),
            'c2_J_proxy': physics_dossier.second_chern_proxy(
                dossier['h11'], dossier['h21']
            )['c2_J_proxy'],
        })
        if tabs and tabs.get('ok'):
            response['derived']['scan_readiness_pct'] = tabs['fluxes']['readiness']['pct']
            response['derived']['N_flux_est_sci'] = tabs['fluxes']['budget']['N_flux_est_sci']
    if euler_mismatch:
        response['warning'] = euler_mismatch

    return jsonify(response)


@app.route('/api/search', methods=['POST'])
def search_candidates():
    """Find ranked candidates near a set of invariants.

    Exact matches on (h11, h21) are rare in any one sampled run, so an exact
    lookup would almost always return nothing useful. Instead we rank the
    generated candidates by distance from the query and return the nearest,
    flagging which (if any) match exactly.

    Large n_candidates (> FORCE_ASYNC_N_CANDIDATES) must set async:true and
    poll /api/jobs/<id>; sync requests above the threshold are forced async.
    """
    _maybe_cleanup_results()
    if not _rate_limit_allow():
        return _rate_limited_response()
    params = request.get_json() or {}
    dataset_id = params.get('dataset_id', 'kreuzer-skarke')

    if not _is_known_dataset(dataset_id):
        return jsonify({'status': 'error', 'message': f'Unknown dataset: {dataset_id}'}), 400

    seed = _bounded_int(params.get('seed'), 42, 0, 2**32 - 1)
    n_candidates = _bounded_int(params.get('n_candidates'), 5000, 10, MAX_N_CANDIDATES)
    limit = _bounded_int(params.get('limit'), 10, 1, 100)
    async_mode = _should_force_async(
        n_candidates, bool(params.get('async') or params.get('async_mode'))
    )
    weights = None
    if dataset_id == 'info-density':
        weights, err = _resolve_request_weights(params)
        if err:
            return err

    query = {}
    for field in ('h11', 'h21', 'h31'):
        if params.get(field) is not None:
            try:
                query[field] = int(params[field])
            except (TypeError, ValueError):
                return jsonify({'status': 'error', 'message': f'{field} must be an integer'}), 400

    if not query:
        return jsonify({
            'status': 'error',
            'message': 'Provide at least one of h11, h21, h31 to search.'
        }), 400
    if any(v < 0 for v in query.values()):
        return jsonify({'status': 'error', 'message': 'Hodge numbers must be non-negative'}), 400

    def _run_search_body():
        # Search the full ranked set, not just the top slice, so a query can find
        # geometries the ranking scored poorly.
        payload = _build_candidate_cards(
            dataset_id=dataset_id, seed=seed,
            top_n=min(MAX_TOP_K, n_candidates), n_candidates=n_candidates,
            weights=weights,
        )

        scored = []
        for card in payload['candidates']:
            raw = card.get('raw', {})
            distance = 0.0
            for field, wanted in query.items():
                actual = raw.get(field)
                if actual is None:
                    distance = float('inf')
                    break
                distance += (float(actual) - wanted) ** 2
            if distance == float('inf'):
                continue
            scored.append((distance ** 0.5, card))

        scored.sort(key=lambda pair: (pair[0], pair[1].get('rank', 0)))

        matches = []
        for distance, card in scored[:limit]:
            matches.append({
                'candidate_id': card['candidate_id'],
                'identity': card.get('identity'),
                'rank': card.get('rank'),
                'score': card.get('score'),
                'verified_target': card.get('verified_target'),
                'features': card.get('features'),
                'distance': round(distance, 4),
                'exact_match': distance == 0.0,
            })

        return {
            'status': 'success',
            'query': query,
            'dataset_id': dataset_id,
            'searched': len(payload['candidates']),
            'exact_matches': sum(1 for m in matches if m['exact_match']),
            'matches': matches,
            'note': (
                'Results are the nearest candidates in this sampled run by '
                'Euclidean distance on the supplied invariants. Distance 0 means '
                'the invariants agree exactly, not that the manifold is the same.'
            ),
        }

    if async_mode:
        job_id = job_store.create_job(stage='queued')

        def _worker():
            try:
                job_store.update_job(job_id, status='running', percent=10, stage='search')
                body = _run_search_body()
                job_store.update_job(
                    job_id, status='completed', percent=100, stage='done', result=body
                )
            except Exception as exc:
                job_store.update_job(
                    job_id, status='failed', percent=100, stage='error', error=str(exc)
                )

        threading.Thread(target=_worker, daemon=True).start()
        return jsonify({
            'status': 'accepted',
            'job_id': job_id,
            'progress_url': f'/api/jobs/{job_id}',
            'message': 'Search queued; poll progress_url until status=completed',
        }), 202

    return jsonify(_run_search_body())


def _export_candidates(results):
    candidates = results.get("top_results", [])
    return candidates


def _make_export_payload(results, schema_name):
    metadata = results.get("run_metadata", {})
    candidates = []
    for cand in _export_candidates(results):
        enriched = dict(cand)
        dataset_id = (
            metadata.get('dataset_id') or cand.get('dataset_id') or 'kreuzer-skarke'
        )
        # Attach curated / sample geometry when Hodge matches.
        try:
            pack = physics_extensions.lookup_geometry_pack(
                dataset_id,
                int(cand.get('h11')),
                int(cand.get('h21')),
                cand.get('h31'),
            )
            if pack:
                enriched = physics_extensions.merge_geometry_into_raw(enriched, pack)
        except (TypeError, ValueError):
            pass
        # Prefer offline geometry DB when richer (vertices / CYTools dump).
        try:
            db_hit = geometry_store.resolve_geometry(
                candidate_id=cand.get('candidate_id'),
                dataset_id=dataset_id,
                h11=int(cand['h11']) if cand.get('h11') is not None else None,
                h21=int(cand['h21']) if cand.get('h21') is not None else None,
                h31=cand.get('h31'),
                db_path=GEOMETRY_DB_PATH,
            )
            if db_hit:
                enriched = geometry_store.merge_db_into_raw(enriched, db_hit)
                if schema_name.startswith('cytools'):
                    enriched.update(geometry_store.export_cytools_fields(db_hit))
        except (TypeError, ValueError):
            pass
        if schema_name.startswith('cytools'):
            enriched.update(physics_extensions.cytools_candidate_fields(enriched))
        candidates.append(enriched)
    return {
        "schema": schema_name,
        "run_metadata": metadata,
        "candidates": candidates,
    }


def _export_sage(results):
    payload = _export_candidates(results)
    return "candidates = " + repr(payload)


def _export_mathematica(results):
    candidates = _export_candidates(results)
    def format_value(value):
        if isinstance(value, bool):
            return "True" if value else "False"
        if value is None:
            return "Null"
        if isinstance(value, (int, float)):
            return str(value)
        return f"\"{str(value)}\""

    rows = []
    for candidate in candidates:
        items = []
        for key, value in candidate.items():
            items.append(f'"{key}" -> {format_value(value)}')
        rows.append("<|" + ", ".join(items) + "|>")
    return "candidates = {" + ", ".join(rows) + "};"


def _csv_cell(value):
    """Quote a CSV cell, neutralising spreadsheet formula injection.

    A leading =, +, - or @ makes Excel/Sheets evaluate the cell as a formula,
    so prefix those with a single quote before quoting.
    """
    if value is None:
        return '""'
    text = str(value)
    if text[:1] in ('=', '+', '-', '@', '\t', '\r'):
        text = "'" + text
    return '"' + text.replace('"', '""') + '"'


def _candidates_to_csv(candidates):
    if not candidates:
        return ""
    headers = sorted({key for candidate in candidates for key in candidate.keys()})
    lines = [",".join(_csv_cell(h) for h in headers)]
    for candidate in candidates:
        lines.append(",".join(_csv_cell(candidate.get(h, "")) for h in headers))
    return "\n".join(lines)


def _bundle_candidates(candidates, metadata):
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('candidates.json', json.dumps(candidates, indent=2))
        archive.writestr('candidates.csv', _candidates_to_csv(candidates))
        archive.writestr('metadata.json', json.dumps(metadata, indent=2))
    bundle.seek(0)
    return bundle


def _analysis_path(candidate_id):
    safe_id = candidate_id.replace('/', '_')
    return os.path.join(ANALYSIS_DIR, f'analysis_{safe_id}.json')


def _analyze_candidate(candidate):
    features = candidate.get("features", [])
    feature_map = {label: value for label, value in features if isinstance(label, str)}
    score = candidate.get("score")
    verified = candidate.get("verified_target")
    dataset_id = candidate.get("dataset_id") or 'kreuzer-skarke'

    h11 = candidate.get('h11', feature_map.get("h11"))
    h21 = candidate.get('h21', feature_map.get("h21"))
    h31 = candidate.get('h31', feature_map.get("h31"))
    euler = candidate.get("euler_char")
    if euler is None:
        euler = candidate.get("raw", {}).get("euler_char", feature_map.get("χ"))

    dossier = physics_dossier.build_dossier(
        dataset_id=dataset_id,
        h11=h11,
        h21=h21,
        h31=h31,
        euler_char=euler,
        verified_target=verified,
    )

    construction = None
    tabs = None
    if dossier.get('ok'):
        construction = physics_dossier.construction_payload(
            dataset_id=dataset_id,
            candidate_id=candidate.get('candidate_id') or 'unknown',
            raw=candidate.get('raw'),
            features=candidate.get('features'),
            tags=candidate.get('tags'),
            summary=candidate.get('summary'),
            h11=dossier['h11'],
            h21=dossier['h21'],
            h31=dossier.get('h31'),
            euler_char=dossier['euler_char'],
        )
        tabs = physics_dossier.build_tabs(
            dossier,
            construction=construction,
            tags=candidate.get('tags'),
        )

    def ratio(a, b):
        if a is None or b in (None, 0):
            return None
        return round(float(a) / float(b), 4)

    derived = {
        "h11_h21_ratio": ratio(h11, h21),
        "h21_h11_ratio": ratio(h21, h11),
        "h31_h11_ratio": ratio(h31, h11),
        "euler_abs": abs(euler) if euler is not None else None
    }
    if dossier.get('ok'):
        derived.update(dossier.get('scalars') or {})
        if tabs and tabs.get('ok'):
            derived['n_generations'] = tabs['phenomenology']['indices']['n_generations']
            derived['log_N_flux'] = tabs['fluxes']['budget'].get('log_N_flux')
            derived['N_flux_est_sci'] = tabs['fluxes']['budget'].get('N_flux_est_sci')

    complexity_index = None
    if h11 is not None and h21 is not None:
        complexity_index = round((h11 + h21) / 2, 3)
    stability_score = None
    if dossier.get('ok'):
        stability_score = dossier['scalars'].get('vacuum_stability')
    elif verified is not None:
        stability_score = 0.85 if verified else 0.45

    passed = sum(1 for c in dossier.get('checks', []) if c.get('ok'))
    total = len(dossier.get('checks') or [])
    if tabs and tabs.get('ok'):
        passed = sum(1 for c in tabs['certificates']['checks'] if c.get('ok'))
        total = len(tabs['certificates']['checks'])
    summary = (
        f"Topological certificate: {passed}/{total} necessary checks passed. "
        f"{dossier.get('caveat', '')}"
        if dossier.get('ok') else
        "Derived ratios and heuristic indicators computed for candidate."
    )

    analysis = {
        "candidate_id": candidate.get("candidate_id"),
        "dataset_id": dataset_id,
        "score": score,
        "verified_target": verified,
        "features": feature_map,
        "derived_metrics": derived,
        "complexity_index": complexity_index,
        "stability_score": stability_score,
        "dossier": dossier if dossier.get('ok') else None,
        "tabs": tabs if tabs and tabs.get('ok') else None,
        "construction": construction,
        "summary": summary,
        "generated_at": _utcnow_iso()
    }
    return analysis


@app.route('/api/export/<run_id>')
def export_results(run_id):
    export_format = request.args.get('format', 'json').lower()
    results = _load_results(run_id)
    if results is None:
        return jsonify({'error': 'Results not found'}), 404

    if export_format == 'json':
        content = json.dumps(results, indent=2)
        filename = f"results_{run_id}.json"
        mime = "application/json"
    elif export_format == 'csv':
        content = _candidates_to_csv(_export_candidates(results))
        filename = f"results_{run_id}.csv"
        mime = "text/csv"
    elif export_format == 'cytools':
        content = json.dumps(_make_export_payload(results, "cytools-candidates-v1"), indent=2)
        filename = f"cytools_{run_id}.json"
        mime = "application/json"
    elif export_format == 'cymetric':
        content = json.dumps(_make_export_payload(results, "cymetric-candidates-v1"), indent=2)
        filename = f"cymetric_{run_id}.json"
        mime = "application/json"
    elif export_format == 'sage':
        content = _export_sage(results)
        filename = f"candidates_{run_id}.sage"
        mime = "text/plain"
    elif export_format == 'mathematica':
        content = _export_mathematica(results)
        filename = f"candidates_{run_id}.wl"
        mime = "text/plain"
    else:
        return jsonify({'error': 'Unsupported export format'}), 400

    response = Response(content, mimetype=mime)
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"
    return response


@app.route('/api/export-gallery', methods=['POST'])
def export_gallery():
    params = request.get_json() or {}
    candidate_ids = params.get('candidate_ids', [])
    source = params.get('source', 'featured')
    dataset_id = params.get('dataset_id')
    seed = _bounded_int(params.get('seed'), 42, 0, 2**32 - 1)
    top_n = _bounded_int(params.get('top_n'), 12, 1, MAX_TOP_K)

    if not candidate_ids or not isinstance(candidate_ids, list):
        return jsonify({'error': 'No candidate ids provided'}), 400
    if len(candidate_ids) > 50:
        return jsonify({'error': 'Too many candidates selected (max 50).'}), 400

    candidates = []
    if source in ('featured', 'hall_of_fame'):
        for cid in candidate_ids:
            item = hall_of_fame.get_candidate(cid, db_path=HALL_OF_FAME_PATH)
            if item:
                candidates.append(item.get('raw') or item)
    elif source == 'live':
        if not dataset_id:
            return jsonify({'error': 'dataset_id required for live export'}), 400
        payload = _build_candidate_cards(dataset_id=dataset_id, seed=seed, top_n=top_n, n_candidates=5000)
        candidates = [c.get('raw', c) for c in payload.get('candidates', []) if c.get('candidate_id') in candidate_ids]
    else:
        return jsonify({'error': 'Unsupported source'}), 400

    if not candidates:
        return jsonify({'error': 'No matching candidates found'}), 404

    metadata = {
        "source": source,
        "dataset_id": dataset_id,
        "selection_count": len(candidates),
        "generated_at": _utcnow_iso()
    }

    bundle = _bundle_candidates(candidates, metadata)
    return send_file(bundle, mimetype='application/zip', as_attachment=True, download_name='gallery_selection.zip')


@app.route('/api/analyze-candidate', methods=['POST'])
def analyze_candidate():
    params = request.get_json() or {}
    candidate_id = params.get('candidate_id')
    source = params.get('source', 'featured')
    dataset_id = params.get('dataset_id')
    seed = _bounded_int(params.get('seed'), 42, 0, 2**32 - 1)
    top_n = _bounded_int(params.get('top_n'), 12, 1, MAX_TOP_K)

    if not candidate_id:
        return jsonify({'error': 'candidate_id required'}), 400

    candidate = None
    if source in ('featured', 'hall_of_fame'):
        candidate = hall_of_fame.get_candidate(candidate_id, db_path=HALL_OF_FAME_PATH)
    elif source == 'live':
        if not dataset_id:
            return jsonify({'error': 'dataset_id required for live analysis'}), 400
        payload = _build_candidate_cards(dataset_id=dataset_id, seed=seed, top_n=top_n, n_candidates=5000)
        candidate = next((c for c in payload.get('candidates', []) if c.get('candidate_id') == candidate_id), None)
    else:
        return jsonify({'error': 'Unsupported source'}), 400

    if not candidate:
        return jsonify({'error': 'Candidate not found'}), 404

    analysis = _analyze_candidate(candidate)
    analysis_file = _analysis_path(candidate_id)
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)

    return jsonify({'status': 'success', 'analysis': analysis})


@app.route('/api/analysis/<candidate_id>')
def get_analysis(candidate_id):
    analysis_file = _analysis_path(candidate_id)
    if not os.path.exists(analysis_file):
        return jsonify({'error': 'Analysis not found'}), 404
    with open(analysis_file, 'r') as f:
        analysis = json.load(f)
    return jsonify({'status': 'success', 'analysis': analysis})


@app.route('/api/analysis/<candidate_id>/bundle')
def download_analysis_bundle(candidate_id):
    analysis_file = _analysis_path(candidate_id)
    if not os.path.exists(analysis_file):
        return jsonify({'error': 'Analysis not found'}), 404

    with open(analysis_file, 'r') as f:
        analysis = json.load(f)

    candidates = [analysis]
    metadata = {
        "type": "candidate-analysis",
        "candidate_id": candidate_id,
        "generated_at": analysis.get("generated_at")
    }
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('analysis.json', json.dumps(analysis, indent=2))
        archive.writestr('analysis.csv', _candidates_to_csv(candidates))
        archive.writestr('summary.md', f"# Analysis {candidate_id}\n\n{analysis.get('summary')}\n")
        archive.writestr('metadata.json', json.dumps(metadata, indent=2))
    bundle.seek(0)
    return send_file(bundle, mimetype='application/zip', as_attachment=True, download_name=f'analysis_{candidate_id}.zip')


@app.route('/api/info-density/weights', methods=['GET', 'POST'])
def info_density_weights():
    """
    Request-scoped info-density weights (multi-worker safe).

    GET: defaults plus optional session/query weights
         (?weight_set_id=... or ?weights={...} JSON)
    POST: validate a weight map and return a weight_set_id. Does NOT mutate
          the process-global dataset singleton. Pass weight_set_id or weights
          on subsequent /api/run-demo, /api/search, or /api/export-physics calls.

    POST body (partial update supported):
    {
        "entropy": 0.20,
        "efficiency": 0.20,
        "compactness": 0.15,
        "balance": 0.10,
        "flux_density": 0.20,
        "vacuum_stability": 0.15
    }

    All weights should sum to 1.0 for normalized scoring.
    """
    dataset = get_info_density_dataset()
    defaults = dataset.DEFAULT_WEIGHTS.copy()

    if request.method == 'GET':
        resolved, err = _resolve_request_weights(dict(request.args))
        if err:
            return err
        return jsonify({
            'status': 'success',
            'weights': resolved,
            'default_weights': defaults,
            'weight_set_id': request.args.get('weight_set_id'),
            'note': (
                'Weights are request-scoped. POST returns a weight_set_id; pass it '
                '(or a weights object) on search/run-demo. Defaults are never mutated.'
            ),
        })

    try:
        new_weights = request.get_json() or {}
        cleaned, err = _validate_weight_dict(new_weights)
        if err:
            return err

        resolved = defaults.copy()
        resolved.update(cleaned)
        weight_set_id = _store_weight_set(cleaned)

        total = sum(resolved.values())
        warning = None
        if abs(total - 1.0) > 0.01:
            warning = f'Weights sum to {total:.3f}, not 1.0. Results may not be normalized.'

        response = {
            'status': 'success',
            'message': 'Weight set created (request-scoped; not applied globally)',
            'weight_set_id': weight_set_id,
            'weights': resolved,
            'default_weights': defaults,
            'ttl_seconds': WEIGHT_SET_TTL_SEC,
            'usage': {
                'run_demo': {'weight_set_id': weight_set_id},
                'or_inline': {'weights': cleaned},
            },
        }
        if warning:
            response['warning'] = warning
        return jsonify(response)

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/info-density/weights/reset', methods=['POST'])
def reset_info_density_weights():
    """Return default weights. Does not mutate global process state."""
    dataset = get_info_density_dataset()
    # Keep singleton in sync with defaults in case older code mutated it.
    dataset.weights = dataset.DEFAULT_WEIGHTS.copy()
    return jsonify({
        'status': 'success',
        'message': 'Defaults restored (no global weight poisoning)',
        'weights': dataset.DEFAULT_WEIGHTS.copy(),
        'default_weights': dataset.DEFAULT_WEIGHTS.copy(),
    })


@app.route('/api/export-physics', methods=['POST'])
def export_physics_data():
    """
    Export top-k candidates with all physics data for external analysis.

    Designed for researchers who want to run their own computations on
    the ranked candidates (vacuum energy calculations, flux analysis, etc.)

    Accepts JSON payload:
    {
        "dataset_id": "info-density",
        "top_k": 100,
        "seed": 42,
        "n_candidates": 5000,
        "format": "json",  // or "csv", "numpy"
        "weights": {...},
        "weight_set_id": "...",
        "async": false
    }

    Public max n_candidates is 25000; above 5000 requests are forced async.
    """
    _maybe_cleanup_results()
    if not _rate_limit_allow():
        return _rate_limited_response()
    try:
        params = request.get_json() or {}
        dataset_id = params.get('dataset_id', 'info-density')
        top_k = _bounded_int(params.get('top_k'), 100, 1, MAX_TOP_K)
        seed = _bounded_int(params.get('seed'), 42, 0, 2**32 - 1)
        n_candidates = _bounded_int(params.get('n_candidates'), 5000, 10, MAX_N_CANDIDATES)
        export_format = params.get('format', 'json').lower()
        async_mode = _should_force_async(
            n_candidates, bool(params.get('async') or params.get('async_mode'))
        )

        if not _is_known_dataset(dataset_id):
            return jsonify({'status': 'error', 'message': f'Unknown dataset: {dataset_id}'}), 400

        weights = None
        if dataset_id == 'info-density':
            weights, err = _resolve_request_weights(params)
            if err:
                return err

        def _build_export():
            results = run_real_search(
                top_k=top_k,
                seed=seed,
                n_candidates=n_candidates,
                verify=True,
                dataset_id=dataset_id,
                weights=weights,
            )

            candidates_data = []
            for r in results['top_results']:
                candidate = {
                    'rank': r.get('rank'),
                    'h11': r.get('h11'),
                    'h21': r.get('h21'),
                    'euler_characteristic': r.get('euler_char'),
                    'ml_score': r.get('score'),
                    'verified_target': r.get('verified_target')
                }

                if dataset_id == 'info-density':
                    candidate.update({
                        'tadpole_charge': r.get('tadpole_charge'),
                        'hodge_entropy': r.get('hodge_entropy'),
                        'topo_efficiency': r.get('topo_efficiency'),
                        'moduli_compactness': r.get('moduli_compactness'),
                        'hodge_balance': r.get('hodge_balance'),
                        'flux_density': r.get('flux_density'),
                        'vacuum_stability': r.get('vacuum_stability'),
                        'info_density': r.get('info_density')
                    })
                elif dataset_id == 'heterotic':
                    candidate.update({
                        'hodge_balance': r.get('hodge_balance'),
                        'n_generations': r.get('n_generations')
                    })
                elif dataset_id == 'cy5-folds':
                    candidate['h31'] = r.get('h31')

                candidates_data.append(candidate)

            export_payload = {
                'metadata': {
                    'dataset_id': dataset_id,
                    'dataset_name': results['run_metadata']['dataset'],
                    'total_candidates_searched': results['run_metadata']['total_candidates'],
                    'top_k': top_k,
                    'seed': seed,
                    'timestamp': results['run_metadata']['timestamp'],
                    'precision_at_k': results['performance_metrics']['precision_at_k'],
                    'checksum': results['run_metadata']['dataset_checksum']
                },
                'candidates': candidates_data
            }

            if dataset_id == 'info-density':
                export_payload['metadata']['weights'] = weights or get_info_density_dataset().DEFAULT_WEIGHTS.copy()

            return export_payload, candidates_data

        if async_mode and export_format == 'json':
            job_id = job_store.create_job(stage='queued')

            def _worker():
                try:
                    job_store.update_job(job_id, status='running', percent=15, stage='export_physics')
                    export_payload, _ = _build_export()
                    job_store.update_job(
                        job_id,
                        status='completed',
                        percent=100,
                        stage='done',
                        result={'status': 'success', 'export': export_payload},
                    )
                except Exception as exc:
                    job_store.update_job(
                        job_id, status='failed', percent=100, stage='error', error=str(exc)
                    )

            threading.Thread(target=_worker, daemon=True).start()
            return jsonify({
                'status': 'accepted',
                'job_id': job_id,
                'progress_url': f'/api/jobs/{job_id}',
                'message': 'Export queued; poll progress_url until status=completed',
            }), 202

        export_payload, candidates_data = _build_export()

        if export_format == 'json':
            return jsonify({'status': 'success', 'export': export_payload})

        elif export_format == 'csv':
            if not candidates_data:
                return jsonify({'status': 'error', 'message': 'No candidates to export'}), 400

            response = Response(_candidates_to_csv(candidates_data), mimetype='text/csv')
            response.headers['Content-Disposition'] = f'attachment; filename=physics_export_{dataset_id}.csv'
            return response

        elif export_format == 'numpy':
            if not candidates_data:
                return jsonify({'status': 'error', 'message': 'No candidates to export'}), 400

            numeric_keys = ['h11', 'h21', 'euler_characteristic', 'ml_score']
            if dataset_id == 'info-density':
                numeric_keys.extend(['tadpole_charge', 'hodge_entropy', 'topo_efficiency',
                                   'moduli_compactness', 'hodge_balance', 'flux_density',
                                   'vacuum_stability', 'info_density'])

            array_data = []
            for c in candidates_data:
                row = [float(c.get(k, 0) or 0) for k in numeric_keys]
                array_data.append(row)

            return jsonify({
                'status': 'success',
                'columns': numeric_keys,
                'data': array_data,
                'usage': 'import numpy as np; data = np.array(response["data"])'
            })

        else:
            return jsonify({'status': 'error', 'message': f'Unknown format: {export_format}'}), 400

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/score-custom', methods=['POST'])
def score_custom():
    """
    Score custom candidate data with a trained model.

    Accepts JSON payload:
    {
        "dataset_id": "kreuzer-skarke",
        "rows": [[...], [...]],
        "top_k": 20,
        "seed": 42,
        "verify": true
    }
    """
    _maybe_cleanup_results()
    if not _rate_limit_allow():
        return _rate_limited_response()
    try:
        params = request.get_json() or {}
        dataset_id = params.get('dataset_id', 'kreuzer-skarke')
        rows = params.get('rows', [])
        top_k = _bounded_int(params.get('top_k'), 20, 1, MAX_TOP_K)
        seed = _bounded_int(params.get('seed'), 42, 0, 2**32 - 1)
        verify = bool(params.get('verify', True))
        save_results = bool(params.get('save', False))

        if not rows or not isinstance(rows, list):
            return jsonify({'status': 'error', 'message': 'No input rows provided.'}), 400
        if len(rows) > MAX_CUSTOM_ROWS:
            return jsonify({
                'status': 'error',
                'message': f'Too many rows (max {MAX_CUSTOM_ROWS}).'
            }), 400
        if not _is_known_dataset(dataset_id):
            return jsonify({'status': 'error', 'message': f'Unknown dataset: {dataset_id}'}), 400

        dataset = DatasetRegistry.get_dataset(dataset_id)
        metadata = dataset.get_metadata()
        feature_dim = metadata.feature_dim

        parsed_rows = []
        for row in rows:
            if not isinstance(row, (list, tuple)):
                return jsonify({'status': 'error', 'message': 'Each row must be a list of numbers.'}), 400
            if len(row) != feature_dim:
                return jsonify({
                    'status': 'error',
                    'message': f'Expected {feature_dim} values per row for {dataset_id}.'
                }), 400
            try:
                parsed = [float(val) for val in row]
            except (TypeError, ValueError):
                return jsonify({
                    'status': 'error',
                    'message': 'Each row must contain only numbers.'
                }), 400
            if not all(np.isfinite(parsed)):
                return jsonify({
                    'status': 'error',
                    'message': 'Rows must not contain NaN or infinite values.'
                }), 400
            parsed_rows.append(parsed)

        custom_data = np.array(parsed_rows, dtype=np.float32)

        # Train model on synthetic dataset samples to score custom inputs
        # (target-defining columns are held out inside CYSearchEngine.train).
        weights = None
        if dataset_id == 'info-density':
            weights, err = _resolve_request_weights(params)
            if err:
                return err
        if weights is not None:
            train_candidates = dataset.generate_candidates(5000, seed, weights=weights)
        else:
            train_candidates = dataset.generate_candidates(5000, seed)
        train_labels = dataset.generate_labels(train_candidates, seed)
        engine = CYSearchEngine(dataset_id=dataset_id, random_seed=seed)
        engine.train(train_candidates, train_labels)

        scores = engine.rank_candidates(custom_data)
        top_k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[top_indices]

        if verify:
            labels = dataset.generate_labels(custom_data, seed)
            top_labels = labels[top_indices]
            true_positives = int(top_labels.sum())
            precision = true_positives / top_k if top_k else 0
            total_targets = int(labels.sum())
            recall = true_positives / total_targets if total_targets > 0 else 0
            first_hit_idx = None
            for idx, label in enumerate(top_labels):
                if label:
                    first_hit_idx = idx
                    break
            time_to_first_hit = first_hit_idx if first_hit_idx is not None else None
            baseline = float(labels.mean()) if len(labels) else 0.0
        else:
            top_labels = [None] * top_k
            true_positives = 0
            precision = None
            recall = None
            time_to_first_hit = None
            baseline = None

        results = {
            "run_metadata": {
                "timestamp": _utcnow_iso(),
                "dataset": metadata.name,
                "dataset_id": dataset_id,
                "dataset_description": metadata.description,
                "custom_input_count": len(custom_data),
                "model_type": "RandomForest",
                "random_seed": seed,
                "held_out_features": dataset.get_held_out_feature_names(),
                "model_features": dataset.get_model_feature_names(),
                "candidate_source": "synthetic_hodge_draws",
            },
            "performance_metrics": {
                "metric_kind": "synthetic_retrieval_vs_baseline",
                "precision_at_k": round(precision, 4) if precision is not None else None,
                "recall_at_k": round(recall, 4) if recall is not None else None,
                "time_to_first_hit": time_to_first_hit,
                "verified_count": int(true_positives),
                "total_top_k": top_k,
                "baseline_random_precision": (
                    round(baseline, 4) if baseline is not None else None
                ),
                "honesty": SYNTHETIC_RETRIEVAL_HONESTY,
                "leakage_note": dataset.leakage_note(),
                "verified_means": (
                    "Passes dataset target rule on synthetic labels "
                    "(not experimental physics verification)."
                ),
            },
            "timing": {
                "total_runtime_seconds": 0.0
            },
            "top_results": []
        }

        for idx in range(top_k):
            result = dataset.format_result(
                candidate=custom_data[top_indices[idx]],
                score=float(top_scores[idx]),
                verified=bool(top_labels[idx]) if verify else None,
                rank=idx + 1
            )
            results["top_results"].append(result)

        if save_results:
            run_id = _save_results(results)
            return jsonify({
                'status': 'success',
                'run_id': run_id,
                'results': results,
                'results_url': f'/api/results/{run_id}'
            })

        return jsonify({'status': 'success', 'results': results})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/data/<path:filename>')
def download_file(filename):
    """Download result files under RESULTS_DIR only (path-traversal safe)."""
    _maybe_cleanup_results()
    file_path = _safe_results_path(filename)
    if file_path is None:
        abort(403)
    if os.path.isfile(file_path):
        return send_file(file_path, as_attachment=True)
    abort(404)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5102, debug=False)
