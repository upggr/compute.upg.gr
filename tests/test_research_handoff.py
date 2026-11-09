"""Sourced handoff selection, provenance, validation and web endpoints."""

import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module  # noqa: E402
import research_handoff as handoff  # noqa: E402
from scripts import prepare_lean_handoff as calibration  # noqa: E402


@pytest.fixture
def catalog():
    return handoff.load_catalog()


@pytest.fixture
def client(tmp_path, monkeypatch):
    # These routes need only source files; the optional context links query a temp HoF.
    monkeypatch.setattr(app_module, 'HALL_OF_FAME_PATH', str(tmp_path / 'hof.sqlite'))
    app_module.app.config['TESTING'] = True
    with app_module.app.test_client() as client:
        yield client


def source_bytes():
    return (handoff.ROOT / handoff.SOURCE_FILE).read_bytes()


def test_catalog_has_sourced_records_not_synthetic_scores(catalog):
    source = json.loads(source_bytes())
    assert catalog['count'] == len(source['polytopes']) == 19
    assert catalog['small_euler_count'] == 11
    assert catalog['source']['sha256'] == hashlib.sha256(source_bytes()).hexdigest()
    assert [r['geometry']['euler_char'] for r in catalog['records'][:4]] == [0, -2, 6, 12]
    for record in catalog['records']:
        p = record['provenance']
        row = source['polytopes'][int(p['source_json_pointer'].split('/')[-1])]
        assert record['geometry']['polytope_vertices'] == row['polytope_vertices']
        assert record['geometry']['euler_char'] == row['euler_characteristic']
        assert p['source_record_sha256'] == hashlib.sha256(calibration.json_bytes(row)).hexdigest()
        assert record['checks']['passed'] is True
        assert record['hodge_numbers_recomputed'] is False
        assert record['reflexivity_recomputed'] is False
        assert record['formal_verification_status'] == 'not_performed'
        assert not {'score', 'rank', 'verified_target', 'intersections', 'tadpole_L'} & set(record)
        assert 'triangulation' in record['not_supplied']
        assert 'polytope_vertices' in record['geometry']


def test_rank_is_affine_and_exact():
    assert handoff.affine_rank([[9, 0, 0, 0], [9, 1, 0, 0], [9, 0, 1, 0], [9, 0, 0, 1], [9, 1, 1, 1]]) == 3
    assert handoff.affine_rank([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) == 4


def test_changed_source_is_not_silently_repaired():
    source = json.loads(source_bytes())
    source['polytopes'][0]['euler_characteristic'] = 123
    catalog = handoff.catalog_from_bytes(calibration.json_bytes(source))
    changed = next(r for r in catalog['records'] if r['geometry']['h11'] == 1 and r['geometry']['h21'] == 101)
    assert changed['geometry']['euler_char'] == 123
    assert not changed['checks']['passed']
    assert [c['id'] for c in changed['checks']['items'] if not c['passed']] == ['euler_identity']


@pytest.mark.parametrize('coordinate', [True, 1.0, '1', 10**10])
def test_non_integer_or_unbounded_vertices_are_rejected(coordinate):
    source = json.loads(source_bytes())
    source['polytopes'][0]['polytope_vertices'][0][0] = coordinate
    with pytest.raises(ValueError, match='exact integer coordinates'):
        handoff.catalog_from_bytes(calibration.json_bytes(source))


def test_duplicate_source_records_rejected():
    source = json.loads(source_bytes())
    source['polytopes'].append(copy.deepcopy(source['polytopes'][0]))
    with pytest.raises(ValueError, match='Duplicate'):
        handoff.catalog_from_bytes(calibration.json_bytes(source))


def test_batch_contains_only_selected_record_files_and_exact_source(catalog):
    ids = [r['record_id'] for r in catalog['records'][1:4]][::-1]
    data = handoff.build_batch(ids, catalog['source']['sha256'])
    assert data == handoff.build_batch(ids, catalog['source']['sha256'])
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        prefix = 'upg-ks-handoff/'
        expected = {'README.md', 'manifest.json', 'source/ks_geometry_sample.json', 'SHA256SUMS'} | {
            f'records/{i}.json' for i in ids
        }
        assert set(archive.namelist()) == {prefix + name for name in expected}
        manifest = json.loads(archive.read(prefix + 'manifest.json'))
        assert manifest['selected_record_ids'] == ids
        assert manifest['selection_order'] == 'client_selected_order_not_an_ml_ranking'
        assert manifest['formal_verification_status'] == 'not_performed'
        assert archive.read(prefix + 'source/ks_geometry_sample.json') == source_bytes()
        for line in archive.read(prefix + 'SHA256SUMS').decode().splitlines():
            digest, name = line.split('  ')
            assert hashlib.sha256(archive.read(prefix + name)).hexdigest() == digest
        for record in catalog['records']:
            if record['record_id'] in ids:
                assert json.loads(archive.read(prefix + f"records/{record['record_id']}.json")) == record


@pytest.mark.parametrize('ids', [None, {}, 'id', [], [None], [{}], ['../../app.py'], ['a', 'a'], ['id'] * 51])
def test_invalid_selection_rejected(catalog, ids):
    with pytest.raises(ValueError):
        handoff.build_batch(ids, catalog['source']['sha256'])


def test_stale_source_refused(catalog):
    with pytest.raises(handoff.StaleSource):
        handoff.build_batch([catalog['records'][0]['record_id']], '0' * 64)


def test_failed_record_cannot_be_exported(tmp_path):
    source = json.loads(source_bytes())
    source['polytopes'][0]['vertex_count'] += 1
    data = calibration.json_bytes(source)
    catalog = handoff.catalog_from_bytes(data)
    failed = next(r for r in catalog['records'] if not r['checks']['passed'])
    path = tmp_path / handoff.SOURCE_FILE
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    with pytest.raises(ValueError, match='fails the exact data checks'):
        handoff.build_batch([failed['record_id']], catalog['source']['sha256'], root=tmp_path)


def test_page_and_catalog_are_available_without_a_search(client):
    page = client.get('/handoff')
    assert page.status_code == 200
    html = page.get_data(as_text=True)
    assert 'Research handoff' in html
    assert 'not new ML discoveries' in html
    assert 'handoff-data' in html
    assert '/handoff/quintic.zip' in html
    body = client.get('/api/handoff/catalog').get_json()
    assert body['count'] == 19
    assert all(r['hodge_context_url'] is None for r in body['records'])
    for asset in ('/static/js/handoff.js', '/static/css/handoff.css'):
        response = client.get(asset)
        assert response.status_code == 200
        response.close()


def test_batch_api_and_checksum_header(client, catalog):
    ids = [r['record_id'] for r in catalog['records'][:5]]
    response = client.post('/api/handoff/batch', json={'record_ids': ids, 'source_sha256': catalog['source']['sha256']})
    assert response.status_code == 200
    assert response.mimetype == 'application/zip'
    assert 'attachment;' in response.headers['Content-Disposition']
    assert response.headers['X-Content-SHA256'] == hashlib.sha256(response.data).hexdigest()
    assert response.data == handoff.build_batch(ids, catalog['source']['sha256'])
    stale = client.post('/api/handoff/batch', json={'record_ids': ids, 'source_sha256': '0' * 64})
    assert stale.status_code == 409


@pytest.mark.parametrize('payload', [None, [], {'record_ids': []}, {'record_ids': {}, 'source_sha256': '0' * 64}])
def test_batch_api_bad_input(client, payload):
    assert client.post('/api/handoff/batch', json=payload).status_code == 400


def test_batch_api_bounds_body_size(client):
    assert client.post('/api/handoff/batch', data=' ' * 16385, content_type='application/json').status_code == 413


def test_batch_api_bounds_streamed_body_without_content_length(client):
    from werkzeug.test import EnvironBuilder

    builder = EnvironBuilder(
        path='/api/handoff/batch', method='POST', data=' ' * 16385,
        content_type='application/json',
    )
    try:
        environ = builder.get_environ()
        environ.pop('CONTENT_LENGTH', None)
        environ['wsgi.input_terminated'] = True
        assert client.open(environ).status_code == 413
    finally:
        builder.close()


def test_hodge_context_link_does_not_merge_gallery_claims(client, catalog):
    record = catalog['records'][0]
    geometry = record['geometry']
    cid = app_module.canonical_id('kreuzer-skarke', geometry)
    app_module.hall_of_fame.upsert_candidate({
        'candidate_id': cid, 'dataset_id': 'kreuzer-skarke',
        'h11': geometry['h11'], 'h21': geometry['h21'], 'euler_char': geometry['euler_char'],
        'score': 1.0, 'verified_target': True,
        'raw': {'intersections': {'fabricated_test_value': 999}},
    }, db_path=app_module.HALL_OF_FAME_PATH)
    response = client.get('/api/handoff/catalog').get_json()
    linked = next(r for r in response['records'] if r['record_id'] == record['record_id'])
    assert linked.pop('hodge_context_url') == f'/candidate/{cid}'
    assert linked == record


def test_online_quintic_matches_offline_zip_and_reports_negative_control(client):
    response = client.get('/handoff/quintic.zip')
    assert response.status_code == 200
    assert response.data == calibration.build_archive()
    assert response.headers['X-Content-SHA256'] == hashlib.sha256(response.data).hexdigest()
    body = client.get('/api/handoff/calibration').get_json()
    assert body['report']['positive']['passed']
    assert not body['report']['negative_control']['passed']
    assert body['report']['negative_control_rejected_as_expected']
    assert body['report']['formal_verification_status'] == 'not_performed'


def test_docker_ships_handoff_runtime_assets():
    docker = (handoff.ROOT / 'Dockerfile').read_text()
    assert 'COPY research_handoff.py .' in docker
    assert 'COPY examples/lean_handoff/ examples/lean_handoff/' in docker
    assert 'COPY scripts/ scripts/' in docker
    ignore = (handoff.ROOT / '.dockerignore').read_text()
    assert '!examples/lean_handoff/README.md' in ignore
