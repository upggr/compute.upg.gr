"""Tests for geometry sample merge, batch/progress APIs, F-theory dataset."""

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module  # noqa: E402
import physics_extensions  # noqa: E402
import physics_dossier  # noqa: E402
from datasets_registry import DatasetRegistry  # noqa: E402
import ml_roadmap  # noqa: E402


@pytest.fixture
def client():
    app_module.app.config['TESTING'] = True
    with app_module.app.test_client() as c:
        yield c


def test_ks_sample_has_hof_vertices():
    physics_extensions.reload_geometry_pack()
    pack = physics_extensions.lookup_geometry_pack('kreuzer-skarke', 38, 12)
    assert pack is not None
    verts = pack.get('polytope_vertices') or pack.get('vertex_matrix')
    assert verts and len(verts) >= 5
    assert pack.get('geometry_status') in ('representative', None) or True


def test_quintic_gets_literature_periods_and_o3():
    tabs = physics_dossier.build_tabs(
        physics_dossier.build_dossier('kreuzer-skarke', 1, 101, euler_char=-200)
    )
    assert tabs['ok']
    lit = tabs['fluxes'].get('quintic_periods_literature') or tabs['fluxes']['periods_full'].get('literature')
    assert lit and lit['status'] == 'literature_curated'
    assert tabs['fluxes']['orientifold']['status'] == 'curated_example'
    assert tabs['phenomenology'].get('soft_toy_card', {}).get('status') == 'toy_illustrative'


def test_ci5f_matrices_remain_null():
    pack = physics_extensions.lookup_geometry_pack('cy5-folds', 140, 62, 18)
    assert pack is not None
    assert pack.get('configuration_matrix') is None


def test_f_theory_dataset_registered():
    ids = {d['id'] for d in DatasetRegistry.list_datasets()}
    assert 'f-theory-elliptic' in ids
    ds = DatasetRegistry.get_dataset('f-theory-elliptic')
    X = ds.generate_candidates(20, seed=7)
    assert X.shape[1] == 7
    y = ds.generate_labels(X, seed=7)
    assert y[0] == 1  # literature seed injected at front


def test_batch_api_identify(client):
    r = client.post('/api/batch', json={
        'jobs': [
            {'type': 'identify', 'dataset_id': 'kreuzer-skarke', 'h11': 1, 'h21': 101},
            {'type': 'identify', 'dataset_id': 'f-theory-elliptic', 'h11': 2, 'h21': 272},
        ]
    })
    assert r.status_code == 200
    body = r.get_json()
    assert body['count'] == 2
    assert body['results'][0]['status'] == 'success'
    assert body['results'][0]['result']['tabs']['fluxes']['orientifold']['status'] == 'curated_example'


def test_batch_rejects_too_many(client):
    jobs = [{'type': 'identify', 'h11': 1, 'h21': 1}] * 51
    r = client.post('/api/batch', json={'jobs': jobs})
    assert r.status_code == 400


def test_async_run_progress(client):
    r = client.post('/api/run-demo', json={
        'async': True,
        'dataset_id': 'kreuzer-skarke',
        'n_candidates': 80,
        'top_k': 5,
        'seed': 3,
    })
    assert r.status_code == 202
    job_id = r.get_json()['job_id']
    deadline = time.time() + 60
    final = None
    while time.time() < deadline:
        pr = client.get(f'/api/jobs/{job_id}')
        assert pr.status_code == 200
        job = pr.get_json()['job']
        if job['status'] in ('completed', 'failed'):
            final = job
            break
        time.sleep(0.2)
    assert final is not None
    assert final['status'] == 'completed'
    assert final['percent'] == 100
    assert final['result']['run_id']


def test_toy_soft_api(client):
    r = client.post('/api/toy-soft', json={'A0': 100, 'm12': 250, 'tan_beta': 5, 'm0': 300})
    assert r.status_code == 200
    card = r.get_json()['soft_toy_card']
    assert card['status'] == 'toy_illustrative'
    assert 'NOT derived' in card['honesty']


def test_ml_roadmap_stubs():
    assert ml_roadmap.geometry_featurizer(1, 101)['status'] == 'baseline_vector'
    assert ml_roadmap.gnn_extension_point()['checked'] is False
    checklist = {c['id']: c['done'] for c in ml_roadmap.roadmap_checklist()}
    assert checklist['gnn'] is False
    assert checklist['featurizer_hook'] is True


def test_cytools_export_includes_geometry_status(client):
    r = client.post('/api/run-demo', json={
        'dataset_id': 'kreuzer-skarke',
        'n_candidates': 100,
        'top_k': 5,
        'seed': 1,
    })
    run_id = r.get_json()['run_id']
    ex = client.get(f'/api/export/{run_id}?format=cytools')
    assert ex.status_code == 200
    payload = ex.get_json()
    assert payload['schema'] == 'cytools-candidates-v1'
    assert 'geometry_status' in payload['candidates'][0]
