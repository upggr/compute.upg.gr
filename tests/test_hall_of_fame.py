"""Tests for persistent hall of fame and shareable candidate pages."""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module  # noqa: E402
import hall_of_fame  # noqa: E402


@pytest.fixture
def hof_db(tmp_path, monkeypatch):
    path = str(tmp_path / 'hof.sqlite')
    monkeypatch.setattr(app_module, 'HALL_OF_FAME_PATH', path)
    hall_of_fame.init_db(path)
    return path


@pytest.fixture
def client(hof_db):
    app_module.app.config['TESTING'] = True
    with app_module.app.test_client() as c:
        yield c


def test_upsert_keeps_best_score(hof_db):
    hall_of_fame.upsert_candidate({
        'candidate_id': 'kreuzer-skarke-aaa',
        'dataset_id': 'kreuzer-skarke',
        'dataset_name': 'KS',
        'h11': 10, 'h21': 5, 'euler_char': 10,
        'score': 0.5, 'rank': 3, 'verified_target': True,
        'features': [['h11', 10], ['h21', 5], ['χ', 10]],
        'raw': {'h11': 10, 'h21': 5, 'euler_char': 10},
        'tags': ['verified'],
        'viz_seed': 1,
    }, db_path=hof_db)
    hall_of_fame.upsert_candidate({
        'candidate_id': 'kreuzer-skarke-aaa',
        'dataset_id': 'kreuzer-skarke',
        'dataset_name': 'KS',
        'h11': 10, 'h21': 5, 'euler_char': 10,
        'score': 0.9, 'rank': 1, 'verified_target': True,
        'features': [['h11', 10], ['h21', 5], ['χ', 10]],
        'raw': {'h11': 10, 'h21': 5, 'euler_char': 10},
        'tags': ['verified'],
        'viz_seed': 1,
    }, db_path=hof_db)
    item = hall_of_fame.get_candidate('kreuzer-skarke-aaa', db_path=hof_db)
    assert item['score'] == 0.9
    assert item['rank'] == 1
    assert item['times_seen'] == 2


def test_promote_from_run_only_verified(hof_db):
    results = {
        'dataset_id': 'kreuzer-skarke',
        'run_metadata': {'dataset_id': 'kreuzer-skarke', 'dataset_name': 'KS'},
        'top_results': [
            {'rank': 1, 'score': 0.99, 'verified_target': True, 'h11': 38, 'h21': 12, 'euler_char': 52},
            {'rank': 2, 'score': 0.98, 'verified_target': False, 'h11': 40, 'h21': 10, 'euler_char': 60},
        ],
    }
    n = hall_of_fame.promote_from_run(
        results,
        run_id='test-run',
        canonical_id_fn=app_module.canonical_id,
        identity_payload_fn=app_module.identity_payload,
        db_path=hof_db,
    )
    assert n == 1
    listed = hall_of_fame.list_candidates(db_path=hof_db)
    assert len(listed) == 1
    assert listed[0]['verified_target'] is True


def test_candidate_page_renders(client, hof_db):
    cid = app_module.canonical_id(
        'kreuzer-skarke', {'h11': 38, 'h21': 12, 'euler_char': 52})
    hall_of_fame.upsert_candidate({
        'candidate_id': cid,
        'dataset_id': 'kreuzer-skarke',
        'dataset_name': 'Kreuzer-Skarke',
        'h11': 38, 'h21': 12, 'euler_char': 52,
        'score': 0.99, 'rank': 1, 'verified_target': True,
        'features': [['h11', 38], ['h21', 12], ['χ', 52]],
        'raw': {'h11': 38, 'h21': 12, 'euler_char': 52},
        'tags': ['verified'],
        'summary': 'test',
        'viz_seed': 42,
    }, db_path=hof_db)
    r = client.get(f'/candidate/{cid}')
    assert r.status_code == 200
    assert cid.encode() in r.data
    assert b'Copy link' in r.data
    assert b'analysis-tab' in r.data
    assert b'data-tab="moduli"' in r.data
    assert b'data-tab="fluxes"' in r.data
    assert b'data-tab="construction"' in r.data
    assert b'data-tab="certificates"' in r.data
    assert b'Open interactive renderer' not in r.data
    assert b'Loading analysis' in r.data


def test_featured_api_reads_hall_of_fame(client, hof_db):
    hall_of_fame.upsert_candidate({
        'candidate_id': 'kreuzer-skarke-xyz',
        'dataset_id': 'kreuzer-skarke',
        'dataset_name': 'KS',
        'h11': 1, 'h21': 2, 'euler_char': -2,
        'score': 0.8, 'rank': 4, 'verified_target': True,
        'features': [['h11', 1], ['h21', 2], ['χ', -2]],
        'raw': {},
        'tags': ['verified'],
        'viz_seed': 3,
    }, db_path=hof_db)
    r = client.get('/api/featured-candidates?dataset_id=kreuzer-skarke')
    assert r.status_code == 200
    data = r.get_json()
    assert data['source'] == 'hall_of_fame'
    assert any(c['candidate_id'] == 'kreuzer-skarke-xyz' for c in data['candidates'])


def test_ensure_textbook_seeds(hof_db):
    featured = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            'data', 'featured_candidates.json')
    if not os.path.exists(featured):
        featured = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                'static', 'data', 'featured_candidates.json')
    # Pretend board already has something
    hall_of_fame.upsert_candidate({
        'candidate_id': 'existing',
        'dataset_id': 'kreuzer-skarke',
        'h11': 10, 'h21': 5, 'euler_char': 10,
        'score': 0.1, 'verified_target': True,
        'features': [], 'raw': {}, 'tags': [],
    }, db_path=hof_db)
    n = hall_of_fame.ensure_featured_by_tags(
        featured_path=featured,
        canonical_id_fn=app_module.canonical_id,
        db_path=hof_db,
        required_tags=['textbook', 'curated'],
    )
    assert n >= 3
    listed = hall_of_fame.list_candidates(db_path=hof_db)
    assert any((c.get('h11'), c.get('h21')) == (1, 101) for c in listed)
    assert any((c.get('h11'), c.get('h21')) == (101, 1) for c in listed)
