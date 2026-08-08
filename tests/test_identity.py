"""Tests for candidate identity: content-addressed ids and /api/identify."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module  # noqa: E402


@pytest.fixture
def client():
    app_module.app.config['TESTING'] = True
    with app_module.app.test_client() as c:
        yield c


def test_id_is_stable_across_seeds_and_sizes():
    """The old rank-based id named different manifolds for different queries."""
    record = {'h11': 171, 'h21': 156, 'euler_char': 30}
    first = app_module.canonical_id('kreuzer-skarke', record)
    assert first == app_module.canonical_id('kreuzer-skarke', dict(record))
    # Run artifacts must not influence identity.
    noisy = dict(record, rank=7, score=0.1234, verified_target=False)
    assert app_module.canonical_id('kreuzer-skarke', noisy) == first


def test_different_invariants_give_different_ids():
    a = app_module.canonical_id('kreuzer-skarke', {'h11': 171, 'h21': 156, 'euler_char': 30})
    b = app_module.canonical_id('kreuzer-skarke', {'h11': 75, 'h21': 68, 'euler_char': 14})
    assert a != b


def test_same_invariants_differ_across_datasets():
    record = {'h11': 10, 'h21': 5, 'euler_char': 10}
    assert (app_module.canonical_id('kreuzer-skarke', record)
            != app_module.canonical_id('heterotic', record))


def test_identify_matches_id_produced_by_search(client):
    r = client.post('/api/identify', json={
        'dataset_id': 'kreuzer-skarke', 'h11': 171, 'h21': 156,
    })
    assert r.status_code == 200
    assert r.get_json()['candidate_id'] == app_module.canonical_id(
        'kreuzer-skarke', {'h11': 171, 'h21': 156, 'euler_char': 30})


def test_identify_derives_euler_rather_than_trusting_caller(client):
    r = client.post('/api/identify', json={
        'dataset_id': 'kreuzer-skarke', 'h11': 171, 'h21': 156, 'euler_char': 999,
    })
    body = r.get_json()
    assert body['derived']['euler_char'] == 30
    assert 'warning' in body


def test_identify_requires_invariants(client):
    r = client.post('/api/identify', json={'dataset_id': 'kreuzer-skarke', 'h11': 171})
    assert r.status_code == 400
    assert 'h21' in r.get_json()['message']


def test_identify_rejects_negative_hodge_numbers(client):
    r = client.post('/api/identify', json={
        'dataset_id': 'kreuzer-skarke', 'h11': -5, 'h21': 10})
    assert r.status_code == 400


def test_identify_states_ids_are_not_universal(client):
    """The response must not imply these ids are a community standard."""
    r = client.post('/api/identify', json={
        'dataset_id': 'kreuzer-skarke', 'h11': 171, 'h21': 156})
    body = r.get_json()
    assert 'NOT a community-standard' in body['identifier_note']
    assert 'non-unique' in body['uniqueness']


def test_cy5_uses_h31_in_identity(client):
    """CY5 identity must include h31, or distinct manifolds would collide."""
    a = app_module.canonical_id('cy5-folds', {'h11': 10, 'h21': 5, 'h31': 2, 'euler_char': 48})
    b = app_module.canonical_id('cy5-folds', {'h11': 10, 'h21': 5, 'h31': 9, 'euler_char': 90})
    assert a != b


def test_cards_expose_both_identity_and_rank_label(client):
    r = client.get('/api/candidates?dataset_id=kreuzer-skarke&seed=42&top_n=2&n_candidates=400')
    for card in r.get_json()['candidates']:
        assert card['candidate_id'].startswith('kreuzer-skarke-')
        assert card['rank_label'].startswith('kreuzer-skarke-')
        assert card['candidate_id'] != card['rank_label']
        assert set(card['identity']) >= {'h11', 'h21', 'euler_char'}
