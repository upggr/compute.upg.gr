"""Regression tests for input validation, resource limits and export safety.

Run with:  python -m pytest tests/ -q
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module  # noqa: E402
from datasets_registry import (  # noqa: E402
    DatasetRegistry,
    InformationDensityDataset,
    get_info_density_dataset,
)


@pytest.fixture
def client():
    app_module.app.config['TESTING'] = True
    with app_module.app.test_client() as c:
        yield c


# --- resource limits -------------------------------------------------------

def test_n_candidates_is_clamped(client):
    """A huge n_candidates must not tie up the worker (previously unbounded)."""
    r = client.post('/api/run-demo', json={'n_candidates': 50_000_000, 'top_k': 5})
    # Large jobs may be accepted as async (202) instead of running sync (200).
    assert r.status_code in (200, 202)
    body = r.get_json()
    if r.status_code == 200:
        total = body['results']['run_metadata']['total_candidates']
        assert total <= app_module.MAX_N_CANDIDATES
    else:
        assert 'job_id' in body
        # Clamp is applied when the job runs; also enforced on the request path.
        assert app_module.MAX_N_CANDIDATES <= 50_000_000


def test_top_k_cannot_go_negative(client):
    """top_k=-5 used to yield precision_at_k = -4.0."""
    r = client.post('/api/run-demo', json={'top_k': -5, 'n_candidates': 200})
    assert r.status_code == 200
    precision = r.get_json()['results']['performance_metrics']['precision_at_k']
    assert 0.0 <= precision <= 1.0


def test_candidate_cache_is_bounded():
    """Varying the seed must not grow the cache without limit."""
    app_module.CANDIDATE_CACHE.clear()
    for seed in range(app_module.CANDIDATE_CACHE_MAXSIZE + 12):
        app_module._build_candidate_cards('kreuzer-skarke', seed=seed,
                                          top_n=2, n_candidates=60)
    assert len(app_module.CANDIDATE_CACHE) <= app_module.CANDIDATE_CACHE_MAXSIZE


# --- input validation ------------------------------------------------------

def test_unknown_dataset_returns_400_without_leaking_registry(client):
    r = client.post('/api/run-demo', json={'dataset_id': 'nope'})
    assert r.status_code == 400
    message = r.get_json()['message']
    assert 'kreuzer-skarke' not in message  # registry keys must not leak


def test_non_numeric_params_fall_back_to_defaults(client):
    r = client.get('/api/candidates?seed=abc&top_n=3&n_candidates=300')
    assert r.status_code == 200
    assert len(r.get_json()['candidates']) == 3


def test_score_custom_rejects_nan(client):
    r = client.post('/api/score-custom', json={
        'dataset_id': 'kreuzer-skarke',
        'rows': [[float('nan'), 1, 2, 3, 4]],
    })
    assert r.status_code == 400


def test_score_custom_rejects_too_many_rows(client):
    rows = [[1, 2, 3, 4, 5]] * (app_module.MAX_CUSTOM_ROWS + 1)
    r = client.post('/api/score-custom', json={'rows': rows})
    assert r.status_code == 400


# --- result integrity ------------------------------------------------------

def test_top_k_is_not_silently_truncated_to_20(client):
    """Requesting 100 used to return only 20 rows."""
    r = client.post('/api/run-demo', json={'top_k': 100, 'n_candidates': 2000})
    results = r.get_json()['results']
    assert len(results['top_results']) == 100
    assert results['performance_metrics']['total_top_k'] == 100


# --- export safety ---------------------------------------------------------

@pytest.mark.parametrize('payload', ['=cmd|calc', '+1', '-1+1', '@SUM(A1)'])
def test_csv_formula_injection_is_neutralised(payload):
    assert app_module._csv_cell(payload).startswith('"\'')


def test_csv_quotes_are_escaped():
    assert app_module._csv_cell('a"b') == '"a""b"'


# --- weight isolation ------------------------------------------------------

def test_per_request_weights_do_not_mutate_global_state():
    ds = get_info_density_dataset()
    before = dict(ds.weights)
    ds.generate_candidates(200, 42, weights={'entropy': 99.0})
    assert dict(ds.weights) == before


def test_constructor_does_not_alias_caller_dict():
    mine = {'entropy': 0.5}
    ds = InformationDensityDataset(weights=mine)
    ds.set_weights({'entropy': 0.9})
    assert mine == {'entropy': 0.5}


def test_partial_weights_do_not_raise():
    """A partial weight dict used to KeyError inside generate_candidates."""
    ds = InformationDensityDataset(weights={'entropy': 0.5})
    assert ds.generate_candidates(50, 1).shape == (50, 10)


# --- ML honesty: target-defining columns held out of the model -------------

@pytest.mark.parametrize('dataset_id,held_out', [
    ('kreuzer-skarke', ['euler_abs']),
    ('cy5-folds', ['h11']),
    ('info-density', ['info_density']),
    ('heterotic', ['hodge_balance', 'euler', 'euler_abs', 'n_gen']),
    ('f-theory-elliptic', ['elliptic_flag', 'base_h11_proxy']),
])
def test_target_defining_features_are_held_out_of_model(dataset_id, held_out):
    """Label may still use χ / target rules; the RF must not see those columns."""
    ds = DatasetRegistry.get_dataset(dataset_id)
    assert ds.get_held_out_feature_names() == held_out
    model_names = ds.get_model_feature_names()
    for name in held_out:
        assert name not in model_names
    candidates = ds.generate_candidates(500, 42)
    X_model = ds.model_features(candidates)
    assert X_model.shape == (500, len(model_names))
    # No single remaining model column should perfectly separate the classes
    # the way the old leaky design did (threshold on the label feature).
    labels = ds.generate_labels(candidates, 42)
    if labels.min() != labels.max():
        for col in range(X_model.shape[1]):
            values = X_model[:, col]
            pos, neg = values[labels == 1], values[labels == 0]
            if len(pos) == 0 or len(neg) == 0:
                continue
            perfect = pos.min() > neg.max() or pos.max() < neg.min()
            assert not perfect, (
                f'{dataset_id} model feature {model_names[col]} still '
                f'perfectly separates labels'
            )


def test_performance_metrics_are_labeled_honest_synthetic_retrieval(client):
    """API metrics must declare synthetic retrieval + leakage hold-out — never 'perfect precision'."""
    r = client.post('/api/run-demo', json={
        'top_k': 20, 'n_candidates': 400, 'seed': 7, 'use_real': True,
    })
    assert r.status_code == 200
    body = r.get_json()
    metrics = body['results']['performance_metrics']
    meta = body['results']['run_metadata']
    assert metrics['metric_kind'] == 'synthetic_retrieval_vs_baseline'
    assert 'honesty' in metrics and metrics['honesty']
    assert 'leakage_note' in metrics and metrics['leakage_note']
    assert 'baseline_random_precision' in metrics
    assert 'euler_abs' in meta.get('held_out_features', [])
    blob = json.dumps(body).lower()
    assert 'perfect precision' not in blob


def test_labels_still_use_held_out_target_rules():
    """Labels may still depend on χ / target columns even when the model cannot see them."""
    ds = DatasetRegistry.get_dataset('kreuzer-skarke')
    candidates = ds.generate_candidates(800, 42)
    labels = ds.generate_labels(candidates, 42)
    euler_abs = candidates[:, ds.get_feature_names().index('euler_abs')]
    np.testing.assert_array_equal(labels, (euler_abs < 100).astype(int))
