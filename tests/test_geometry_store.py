"""Unit tests for offline geometry SQLite store and Flask lookup API."""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module  # noqa: E402
import geometry_store  # noqa: E402
import physics_dossier  # noqa: E402
import physics_extensions  # noqa: E402


@pytest.fixture
def geom_db(tmp_path, monkeypatch):
    path = str(tmp_path / 'geometry.sqlite')
    monkeypatch.setattr(app_module, 'GEOMETRY_DB_PATH', path)
    monkeypatch.setattr(geometry_store, 'DEFAULT_DB_PATH', path)
    geometry_store.init_db(path)
    return path


@pytest.fixture
def client(geom_db):
    # Re-seed into the temp DB so API tests see baked rows.
    geometry_store.seed_baked_geometry(db_path=geom_db)
    app_module.app.config['TESTING'] = True
    with app_module.app.test_client() as c:
        yield c


def test_upsert_and_lookup_by_hodge(geom_db):
    geometry_store.upsert_geometry(
        {
            'dataset_id': 'kreuzer-skarke',
            'h11': 7,
            'h21': 11,
            'euler_char': -8,
            'source': 'manual',
            'status': 'representative',
            'polytope_vertices': [[1, 0, 0, 0], [0, 1, 0, 0]],
            'note': 'test representative',
        },
        db_path=geom_db,
    )
    hit = geometry_store.lookup_by_hodge('kreuzer-skarke', 7, 11, db_path=geom_db)
    assert hit is not None
    assert hit['h11'] == 7
    assert hit['status'] == 'representative'
    assert hit['polytope_vertices'][0] == [1, 0, 0, 0]


def test_lookup_prefers_richer_status(geom_db):
    geometry_store.upsert_geometry(
        {
            'id': 'ks:1:2:a',
            'dataset_id': 'kreuzer-skarke',
            'h11': 1,
            'h21': 2,
            'source': 'ks-hf-sample',
            'status': 'representative',
            'polytope_vertices': [[0, 0, 0, 0]],
        },
        db_path=geom_db,
    )
    geometry_store.upsert_geometry(
        {
            'id': 'ks:1:2:b',
            'dataset_id': 'kreuzer-skarke',
            'h11': 1,
            'h21': 2,
            'source': 'cytools-offline',
            'status': 'unique',
            'polytope_vertices': [[1, 1, 1, 1], [2, 2, 2, 2]],
        },
        db_path=geom_db,
    )
    hit = geometry_store.lookup_by_hodge('kreuzer-skarke', 1, 2, db_path=geom_db)
    assert hit['status'] == 'unique'
    assert hit['source'] == 'cytools-offline'


def test_ingest_ks_sample_and_pack(geom_db):
    stats = geometry_store.seed_baked_geometry(db_path=geom_db)
    assert stats['ks_sample'] >= 1
    assert stats['geometry_pack'] >= 1
    hit = geometry_store.lookup_by_hodge(
        'kreuzer-skarke', 1, 101, db_path=geom_db,
    )
    assert hit is not None
    assert hit.get('polytope_vertices') or hit.get('vertex_matrix')


def test_get_by_candidate_id(geom_db):
    geometry_store.upsert_geometry(
        {
            'dataset_id': 'kreuzer-skarke',
            'candidate_id': 'kreuzer-skarke-deadbeef',
            'h11': 3,
            'h21': 5,
            'source': 'manual',
            'status': 'curated',
            'vertex_matrix': [[1, 0, 0, 0]],
        },
        db_path=geom_db,
    )
    hit = geometry_store.get_by_candidate_id(
        'kreuzer-skarke-deadbeef', db_path=geom_db,
    )
    assert hit is not None
    assert hit['h11'] == 3


def test_merge_prefers_db_vertices(geom_db):
    """DB vertices win over static pack / JSON when DB row is richer."""
    pack_like = {
        'dataset_id': 'kreuzer-skarke',
        'h11': 1,
        'h21': 101,
        'polytope_vertices': [[9, 9, 9, 9]],
        'vertex_matrix': [[9, 9, 9, 9]],
        'geometry_status': 'representative',
        'name': 'pack-placeholder',
    }
    raw = physics_extensions.merge_geometry_into_raw({}, pack_like)
    assert raw['polytope_vertices'] == [[9, 9, 9, 9]]

    db_record = geometry_store.upsert_geometry(
        {
            'dataset_id': 'kreuzer-skarke',
            'h11': 1,
            'h21': 101,
            'source': 'cytools-offline',
            'status': 'representative',
            'polytope_vertices': [[1, 0, 0, 0], [0, 1, 0, 0]],
            'vertex_matrix': [[1, 0, 0, 0], [0, 1, 0, 0]],
            'note': 'DB wins',
        },
        db_path=geom_db,
    )
    merged = geometry_store.merge_db_into_raw(raw, db_record)
    assert merged['polytope_vertices'] == [[1, 0, 0, 0], [0, 1, 0, 0]]
    assert merged['geometry_source'] == 'cytools-offline'


def test_construction_payload_uses_db(geom_db, monkeypatch):
    monkeypatch.setattr(geometry_store, 'DEFAULT_DB_PATH', geom_db)
    geometry_store.seed_baked_geometry(db_path=geom_db)
    c = physics_dossier.construction_payload(
        dataset_id='kreuzer-skarke',
        candidate_id='test-quintic',
        raw={'h11': 1, 'h21': 101, 'euler_char': -200},
        h11=1,
        h21=101,
        euler_char=-200,
    )
    assert c.get('geometry_db') is not None
    assert c['geometry_db']['source'] in (
        'ks-hf-sample', 'geometry-pack', 'cytools-offline',
    )
    present = c['present_geometry']
    assert present.get('polytope_vertices') or present.get('vertex_matrix')


def test_api_geometry_lookup_quintic(client):
    r = client.get(
        '/api/geometry/lookup?dataset_id=kreuzer-skarke&h11=1&h21=101'
    )
    assert r.status_code == 200
    body = r.get_json()
    assert body['status'] == 'success'
    geom = body['geometry']
    assert geom.get('polytope_vertices') or geom.get('vertex_matrix')


def test_api_geometry_lookup_hof_38_12(client):
    r = client.get(
        '/api/geometry/lookup?dataset_id=kreuzer-skarke&h11=38&h21=12'
    )
    assert r.status_code == 200
    geom = r.get_json()['geometry']
    assert geom.get('status') in ('representative', 'curated', 'unique')
    assert geom.get('polytope_vertices') or geom.get('vertex_matrix')


def test_api_geometry_list_bounded(client):
    r = client.get('/api/geometry?limit=5')
    assert r.status_code == 200
    body = r.get_json()
    assert body['status'] == 'success'
    assert len(body['geometries']) <= 5


def test_api_geometry_lookup_missing(client):
    r = client.get(
        '/api/geometry/lookup?dataset_id=kreuzer-skarke&h11=99999&h21=99999'
    )
    assert r.status_code == 404


def test_worker_stub_upsert(geom_db, tmp_path):
    dump = {
        'dataset_id': 'kreuzer-skarke',
        'h11': 4,
        'h21': 68,
        'source': 'cytools-offline',
        'status': 'representative',
        'polytope_vertices': [[1, 0, 0, 0]],
        'note': 'offline stub',
    }
    path = tmp_path / 'offline.json'
    path.write_text(json.dumps(dump), encoding='utf-8')
    # Call store the same way the stub does.
    stored = geometry_store.upsert_geometry(dump, db_path=geom_db)
    assert stored['source'] == 'cytools-offline'
    hit = geometry_store.lookup_by_hodge('kreuzer-skarke', 4, 68, db_path=geom_db)
    assert hit is not None
