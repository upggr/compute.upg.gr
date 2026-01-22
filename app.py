from flask import Flask, render_template, jsonify, request, send_file, Response
import io
import zipfile
import json
import os
import numpy as np
import hashlib
from datetime import datetime
from cy_search import run_search, get_sample_results  # Demo implementation
from cy_search_real import run_real_search, list_available_datasets, CYSearchEngine  # Real implementation
from datasets_registry import DatasetRegistry, get_info_density_dataset

app = Flask(__name__)

# Configure upload folder for results
RESULTS_DIR = 'static/data'
os.makedirs(RESULTS_DIR, exist_ok=True)
ANALYSIS_DIR = os.path.join('static', 'data', 'analysis')
os.makedirs(ANALYSIS_DIR, exist_ok=True)

CANDIDATE_CACHE = {}
FEATURED_PATH = os.path.join('static', 'data', 'featured_candidates.json')


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
    """Candidates gallery page"""
    return render_template('candidates.html')


@app.route('/eli5.html')
def eli5():
    """ELI5 page"""
    return render_template('eli5.html')


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
        "dataset_id": "kreuzer-skarke",  # or "cy5-folds", "heterotic"
        "use_real": true
    }

    Returns:
    {
        "status": "success",
        "run_id": "...",
        "message": "Demo completed",
        "results_url": "/api/results/..."
    }
    """
    try:
        params = request.get_json() or {}
        top_k = params.get('top_k', 100)
        seed = params.get('seed', 42)
        verify = params.get('verify', True)
        n_candidates = params.get('n_candidates', 5000)
        dataset_id = params.get('dataset_id', 'kreuzer-skarke')
        use_real = params.get('use_real', True)  # Default to real implementation

        # Run the search - use real implementation by default
        if use_real:
            print(f"Running upg-strings: dataset={dataset_id}, {n_candidates} candidates, top_k={top_k}, seed={seed}")
            results = run_real_search(
                top_k=top_k,
                seed=seed,
                n_candidates=n_candidates,
                verify=verify,
                dataset_id=dataset_id
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


@app.route('/api/results/<run_id>')
def get_results(run_id):
    """Get results for a specific run"""
    results = _load_results(run_id)
    if results is None:
        return jsonify({'error': 'Results not found'}), 404
    return jsonify(results)


@app.route('/api/sample-results')
def sample_results():
    """Get sample results for display (uses real implementation)"""
    # Use real implementation with small dataset for quick response
    dataset_id = request.args.get('dataset_id', 'kreuzer-skarke')
    results = run_real_search(
        top_k=20,
        seed=42,
        n_candidates=1000,
        verify=True,
        dataset_id=dataset_id
    )
    return jsonify(results)


def _save_results(results):
    run_id = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(RESULTS_DIR, f'results_{run_id}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    return run_id


def _load_results(run_id):
    results_file = os.path.join(RESULTS_DIR, f'results_{run_id}.json')
    if not os.path.exists(results_file):
        return None
    with open(results_file, 'r') as f:
        return json.load(f)


def _build_candidate_cards(dataset_id, seed=42, top_n=12, n_candidates=5000):
    cache_key = (dataset_id, seed, top_n, n_candidates)
    if cache_key in CANDIDATE_CACHE:
        return CANDIDATE_CACHE[cache_key]

    results = run_real_search(
        top_k=top_n,
        seed=seed,
        n_candidates=n_candidates,
        verify=True,
        dataset_id=dataset_id
    )

    dataset = DatasetRegistry.get_dataset(dataset_id)
    metadata = dataset.get_metadata()

    candidates = []
    for result in results["top_results"]:
        candidate_id = f"{dataset_id}-{result['rank']:03d}"
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
    return payload


@app.route('/api/candidates')
def top_candidates():
    dataset_id = request.args.get('dataset_id', 'kreuzer-skarke')
    seed = int(request.args.get('seed', 42))
    top_n = int(request.args.get('top_n', 12))
    n_candidates = int(request.args.get('n_candidates', 5000))

    payload = _build_candidate_cards(
        dataset_id=dataset_id,
        seed=seed,
        top_n=top_n,
        n_candidates=n_candidates
    )
    return jsonify({'status': 'success', **payload})


@app.route('/api/candidate/<candidate_id>')
def candidate_detail(candidate_id):
    dataset_id = request.args.get('dataset_id', 'kreuzer-skarke')
    seed = int(request.args.get('seed', 42))
    top_n = int(request.args.get('top_n', 12))
    n_candidates = int(request.args.get('n_candidates', 5000))

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
                "summary": f"Target: {detail.get('target_description')}"
            })

            return jsonify({'status': 'success', 'candidate': detail})

    # Fallback to featured candidates for curated gallery views
    if os.path.exists(FEATURED_PATH):
        with open(FEATURED_PATH, 'r') as f:
            featured = json.load(f)
        for candidate in featured.get('candidates', []):
            if candidate.get('candidate_id') == candidate_id:
                detail = candidate.copy()
                detail.update({
                    "invariants": [],
                    "summary": f"Target: {detail.get('summary', 'Featured candidate')}"
                })
                return jsonify({'status': 'success', 'candidate': detail})

    return jsonify({'status': 'error', 'message': 'Candidate not found'}), 404


@app.route('/api/featured-candidates')
def featured_candidates():
    if not os.path.exists(FEATURED_PATH):
        return jsonify({'status': 'error', 'message': 'Featured candidates not available'}), 404

    with open(FEATURED_PATH, 'r') as f:
        payload = json.load(f)

    candidates = payload.get('candidates', [])
    dataset_id = request.args.get('dataset_id')
    tag = request.args.get('tag')
    verified = request.args.get('verified')

    if dataset_id:
        candidates = [c for c in candidates if c.get('dataset_id') == dataset_id]
    if tag:
        candidates = [c for c in candidates if tag in c.get('tags', [])]
    if verified is not None:
        verified_bool = verified.lower() == 'true'
        candidates = [c for c in candidates if c.get('verified_target') == verified_bool]

    return jsonify({'status': 'success', 'candidates': candidates})


def _export_candidates(results):
    candidates = results.get("top_results", [])
    return candidates


def _make_export_payload(results, schema_name):
    metadata = results.get("run_metadata", {})
    return {
        "schema": schema_name,
        "run_metadata": metadata,
        "candidates": _export_candidates(results)
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


def _candidates_to_csv(candidates):
    if not candidates:
        return ""
    headers = sorted({key for candidate in candidates for key in candidate.keys()})
    lines = [",".join(headers)]
    for candidate in candidates:
        row = []
        for header in headers:
            value = candidate.get(header, "")
            value_str = "" if value is None else str(value).replace('"', '""')
            row.append(f'"{value_str}"')
        lines.append(",".join(row))
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
    dataset_id = candidate.get("dataset_id")

    h11 = feature_map.get("h11")
    h21 = feature_map.get("h21")
    h31 = feature_map.get("h31")
    euler = candidate.get("raw", {}).get("euler_char", feature_map.get("χ"))

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

    # Simple heuristic indicators (placeholders for deeper geometry)
    complexity_index = None
    if h11 is not None and h21 is not None:
        complexity_index = round((h11 + h21) / 2, 3)
    stability_score = None
    if verified is not None:
        stability_score = 0.85 if verified else 0.45

    summary = "Derived ratios and heuristic indicators computed for candidate."

    analysis = {
        "candidate_id": candidate.get("candidate_id"),
        "dataset_id": dataset_id,
        "score": score,
        "verified_target": verified,
        "features": feature_map,
        "derived_metrics": derived,
        "complexity_index": complexity_index,
        "stability_score": stability_score,
        "summary": summary,
        "generated_at": datetime.utcnow().isoformat() + "Z"
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
        candidates = _export_candidates(results)
        if not candidates:
            content = ""
        else:
            headers = sorted({key for candidate in candidates for key in candidate.keys()})
            lines = [",".join(headers)]
            for candidate in candidates:
                row = []
                for header in headers:
                    value = candidate.get(header, "")
                    value_str = "" if value is None else str(value).replace('"', '""')
                    row.append(f'"{value_str}"')
                lines.append(",".join(row))
            content = "\n".join(lines)
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
    seed = int(params.get('seed', 42))
    top_n = int(params.get('top_n', 12))

    if not candidate_ids or not isinstance(candidate_ids, list):
        return jsonify({'error': 'No candidate ids provided'}), 400
    if len(candidate_ids) > 50:
        return jsonify({'error': 'Too many candidates selected (max 50).'}), 400

    candidates = []
    if source == 'featured':
        if not os.path.exists(FEATURED_PATH):
            return jsonify({'error': 'Featured candidates not available'}), 404
        with open(FEATURED_PATH, 'r') as f:
            payload = json.load(f)
        candidates = [c for c in payload.get('candidates', []) if c.get('candidate_id') in candidate_ids]
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
        "generated_at": datetime.utcnow().isoformat() + "Z"
    }

    bundle = _bundle_candidates(candidates, metadata)
    return send_file(bundle, mimetype='application/zip', as_attachment=True, download_name='gallery_selection.zip')


@app.route('/api/analyze-candidate', methods=['POST'])
def analyze_candidate():
    params = request.get_json() or {}
    candidate_id = params.get('candidate_id')
    source = params.get('source', 'featured')
    dataset_id = params.get('dataset_id')
    seed = int(params.get('seed', 42))
    top_n = int(params.get('top_n', 12))

    if not candidate_id:
        return jsonify({'error': 'candidate_id required'}), 400

    candidate = None
    if source == 'featured':
        if not os.path.exists(FEATURED_PATH):
            return jsonify({'error': 'Featured candidates not available'}), 404
        with open(FEATURED_PATH, 'r') as f:
            payload = json.load(f)
        candidate = next((c for c in payload.get('candidates', []) if c.get('candidate_id') == candidate_id), None)
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
    Get or set custom weights for the info-density composite score.

    GET: Returns current weights
    POST: Set new weights (partial update supported)

    Accepts JSON payload for POST:
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

    if request.method == 'GET':
        return jsonify({
            'status': 'success',
            'weights': dataset.weights,
            'default_weights': dataset.DEFAULT_WEIGHTS
        })

    # POST - update weights
    try:
        new_weights = request.get_json() or {}

        # Validate weight keys
        valid_keys = set(dataset.DEFAULT_WEIGHTS.keys())
        invalid_keys = set(new_weights.keys()) - valid_keys
        if invalid_keys:
            return jsonify({
                'status': 'error',
                'message': f'Invalid weight keys: {invalid_keys}. Valid keys: {valid_keys}'
            }), 400

        # Validate weight values
        for key, value in new_weights.items():
            if not isinstance(value, (int, float)) or value < 0:
                return jsonify({
                    'status': 'error',
                    'message': f'Weight "{key}" must be a non-negative number'
                }), 400

        # Update weights
        dataset.set_weights(new_weights)

        # Check if weights sum to ~1.0 (warn but don't error)
        total = sum(dataset.weights.values())
        warning = None
        if abs(total - 1.0) > 0.01:
            warning = f'Weights sum to {total:.3f}, not 1.0. Results may not be normalized.'

        response = {
            'status': 'success',
            'message': 'Weights updated',
            'weights': dataset.weights
        }
        if warning:
            response['warning'] = warning

        return jsonify(response)

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/info-density/weights/reset', methods=['POST'])
def reset_info_density_weights():
    """Reset info-density weights to defaults"""
    dataset = get_info_density_dataset()
    dataset.weights = dataset.DEFAULT_WEIGHTS.copy()
    return jsonify({
        'status': 'success',
        'message': 'Weights reset to defaults',
        'weights': dataset.weights
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
        "format": "json"  // or "csv", "numpy"
    }

    Returns comprehensive physics data including:
    - All Hodge numbers and derived invariants
    - Tadpole charge (χ/24) for D3 brane counting
    - Flux density and vacuum stability metrics
    - ML ranking score
    """
    try:
        params = request.get_json() or {}
        dataset_id = params.get('dataset_id', 'info-density')
        top_k = int(params.get('top_k', 100))
        seed = int(params.get('seed', 42))
        n_candidates = int(params.get('n_candidates', 5000))
        export_format = params.get('format', 'json').lower()

        # Run search
        results = run_real_search(
            top_k=top_k,
            seed=seed,
            n_candidates=n_candidates,
            verify=True,
            dataset_id=dataset_id
        )

        # Build physics export payload
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

            # Add dataset-specific physics fields
            if dataset_id == 'info-density':
                candidate.update({
                    'tadpole_charge': r.get('tadpole_charge'),  # χ/24
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

        # Add weights if info-density
        if dataset_id == 'info-density':
            export_payload['metadata']['weights'] = get_info_density_dataset().weights

        # Format response
        if export_format == 'json':
            return jsonify({'status': 'success', 'export': export_payload})

        elif export_format == 'csv':
            if not candidates_data:
                return jsonify({'status': 'error', 'message': 'No candidates to export'}), 400

            headers = list(candidates_data[0].keys())
            lines = [','.join(headers)]
            for c in candidates_data:
                row = [str(c.get(h, '')) for h in headers]
                lines.append(','.join(row))

            response = Response('\n'.join(lines), mimetype='text/csv')
            response.headers['Content-Disposition'] = f'attachment; filename=physics_export_{dataset_id}.csv'
            return response

        elif export_format == 'numpy':
            # Return as JSON with array structure suitable for np.array()
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
    try:
        params = request.get_json() or {}
        dataset_id = params.get('dataset_id', 'kreuzer-skarke')
        rows = params.get('rows', [])
        top_k = int(params.get('top_k', 20))
        seed = int(params.get('seed', 42))
        verify = bool(params.get('verify', True))
        save_results = bool(params.get('save', False))

        if not rows or not isinstance(rows, list):
            return jsonify({'status': 'error', 'message': 'No input rows provided.'}), 400

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
            parsed_rows.append([float(val) for val in row])

        custom_data = np.array(parsed_rows, dtype=np.float32)

        # Train model on synthetic dataset samples to score custom inputs
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
        else:
            top_labels = [None] * top_k
            true_positives = 0
            precision = None
            recall = None
            time_to_first_hit = None

        results = {
            "run_metadata": {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "dataset": metadata.name,
                "dataset_id": dataset_id,
                "dataset_description": metadata.description,
                "custom_input_count": len(custom_data),
                "model_type": "RandomForest",
                "random_seed": seed
            },
            "performance_metrics": {
                "precision_at_k": round(precision, 4) if precision is not None else None,
                "recall_at_k": round(recall, 4) if recall is not None else None,
                "time_to_first_hit": time_to_first_hit,
                "verified_count": int(true_positives),
                "total_top_k": top_k
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
    """Download result files (CSV, JSON, MD)"""
    file_path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "File not found", 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5102, debug=False)
