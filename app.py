from flask import Flask, render_template, jsonify, request, send_file
import json
import os
from datetime import datetime
from cy_search import run_search, get_sample_results

app = Flask(__name__)

# Configure upload folder for results
RESULTS_DIR = 'static/data'
os.makedirs(RESULTS_DIR, exist_ok=True)


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


@app.route('/api/run-demo', methods=['POST'])
def run_demo():
    """
    API endpoint to run the CY-Search demo

    Accepts JSON payload:
    {
        "top_k": 100,
        "seed": 42,
        "verify": true
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

        # Run the search
        results = run_search(top_k=top_k, seed=seed, verify=verify)

        # Save results
        run_id = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(RESULTS_DIR, f'results_{run_id}.json')

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

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
    results_file = os.path.join(RESULTS_DIR, f'results_{run_id}.json')

    if not os.path.exists(results_file):
        return jsonify({'error': 'Results not found'}), 404

    with open(results_file, 'r') as f:
        results = json.load(f)

    return jsonify(results)


@app.route('/api/sample-results')
def sample_results():
    """Get sample results for display"""
    return jsonify(get_sample_results())


@app.route('/data/<path:filename>')
def download_file(filename):
    """Download result files (CSV, JSON, MD)"""
    file_path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "File not found", 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5102, debug=False)
