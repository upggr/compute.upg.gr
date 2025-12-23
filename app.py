from flask import Flask, render_template, jsonify, request, send_file, Response
import json
import os
import numpy as np
import hashlib
from datetime import datetime
from cy_search import run_search, get_sample_results  # Demo implementation
from cy_search_real import run_real_search, list_available_datasets, CYSearchEngine  # Real implementation
from datasets_registry import DatasetRegistry

app = Flask(__name__)

# Configure upload folder for results
RESULTS_DIR = 'static/data'
os.makedirs(RESULTS_DIR, exist_ok=True)

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

            detail.update({
                "invariants": invariants,
                "summary": f"Target: {detail.get('target_description')}"
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
