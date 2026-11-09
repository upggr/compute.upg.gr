"""The shareable pilot is reproducible, standalone, and honest about its scope."""

import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.lean_handoff import check as pilot  # noqa: E402
from scripts import prepare_lean_handoff as handoff  # noqa: E402


@pytest.fixture
def files():
    return handoff.build_files()


@pytest.fixture
def record(files):
    return json.loads(files['quintic.json'])


def test_positive_is_a_curated_calibration_not_a_search_hit(record):
    report = pilot.check_record(record)
    assert report['passed']
    assert len(report['checks']) == 5
    assert report['formal_verification_status'] == 'not_performed'
    assert record['data_origin'] == 'curated_textbook_not_ml_ranked'
    g = record['geometry']
    assert (g['h11'], g['h21'], g['euler_char'], g['triple_intersection_H3']) == (1, 101, -200, 5)
    assert abs(g['euler_char']) >= 100  # Deliberately not a rare KS target.
    assert not {'verified_target', 'score', 'tadpole_L', 'c2_J_proxy'} & (set(record) | set(g))
    assert report['checks'][-1]['determinants'] == [1, -1, 1, -1, 1]


def test_negative_changes_only_euler_and_fails_exactly_that_claim(files, record):
    negative = json.loads(files['negative_euler.json'])
    assert negative['geometry']['euler_char'] == -198
    report = pilot.check_bytes(files['negative_euler.json'])
    assert not report['passed']
    assert [c['id'] for c in report['checks'] if not c['passed']] == ['euler_identity']
    negative['geometry']['euler_char'] = -200
    assert negative == record


def test_provenance_binds_actual_curated_source(files, record):
    provenance = record['provenance']
    source_bytes = (handoff.ROOT / provenance['source_file']).read_bytes()
    assert hashlib.sha256(source_bytes).hexdigest() == provenance['source_file_sha256']
    assert hashlib.sha256(files['source_geometry.json']).hexdigest() == provenance['source_record_sha256']
    index = int(provenance['source_json_pointer'].split('/')[-1])
    seed = json.loads(source_bytes)['geometries'][index]
    assert seed == json.loads(files['source_geometry.json'])
    assert seed['polytope_id'] == 'simplex-P4'
    assert record['geometry']['ambient_fan']['ray_matrix'] == seed['vertex_matrix']


@pytest.mark.parametrize('key,value', [
    ('h11', True), ('h11', 1.0), ('h21', '101'), ('euler_char', -200.0),
    ('complex_dimension', 3.0), ('degree', False), ('triple_intersection_H3', 5.0),
])
def test_non_integer_values_rejected(record, key, value):
    record['geometry'][key] = value
    with pytest.raises(ValueError):
        pilot.check_record(record)


def test_invalid_shape_and_invented_certification_rejected(record):
    bad = copy.deepcopy(record)
    bad['geometry']['ambient_fan']['ray_matrix'][0] = [1, 0, 0]
    with pytest.raises(ValueError):
        pilot.check_record(bad)
    record['formal_verification_status'] = 'certified'
    with pytest.raises(ValueError):
        pilot.check_record(record)


def test_corrupted_rays_and_intersection_are_not_ignored(record):
    record['geometry']['ambient_fan']['ray_matrix'][0][0] = 2
    record['geometry']['triple_intersection_H3'] = 6
    report = pilot.check_record(record)
    assert {c['id'] for c in report['checks'] if not c['passed']} == {
        'fan_ray_sum', 'ambient_cone_determinants', 'hyperplane_intersection',
    }


@pytest.mark.parametrize('data', [b'{"schema": 1, "schema": 2}', b'{"schema": NaN}'])
def test_ambiguous_json_rejected(data):
    with pytest.raises(ValueError):
        pilot.check_bytes(data)


def test_archive_is_deterministic_self_contained_and_checksummed(files, tmp_path):
    archive_bytes = handoff.build_archive()
    assert archive_bytes == handoff.build_archive()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        assert set(archive.namelist()) == {f'{handoff.BUNDLE_NAME}/{name}' for name in files}
        archive.extractall(tmp_path)
    directory = tmp_path / handoff.BUNDLE_NAME
    for line in (directory / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ')
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
    stored = json.loads((directory / 'check_report.json').read_bytes())
    assert stored['positive'] == pilot.check_bytes(files['quintic.json'])
    assert stored['negative_control'] == pilot.check_bytes(files['negative_euler.json'])
    for name, expected_exit in [('quintic.json', 0), ('negative_euler.json', 1)]:
        proc = subprocess.run(
            [sys.executable, str(directory / 'check.py'), str(directory / name)],
            cwd=tmp_path, capture_output=True, text=True, timeout=10,
        )
        assert proc.returncode == expected_exit, proc.stderr
        assert json.loads(proc.stdout)['passed'] is (expected_exit == 0)
    malformed = tmp_path / 'malformed.json'
    malformed.write_text('{}')
    proc = subprocess.run(
        [sys.executable, str(directory / 'check.py'), str(malformed)],
        cwd=tmp_path, capture_output=True, text=True, timeout=10,
    )
    assert proc.returncode == 2
    assert json.loads(proc.stdout)['passed'] is False


def test_exporter_does_not_silently_overwrite_files(tmp_path):
    output = tmp_path / 'pilot.zip'
    output.write_bytes(b'keep this file')
    proc = subprocess.run(
        [sys.executable, str(handoff.ROOT / 'scripts/prepare_lean_handoff.py'), '--output', str(output)],
        cwd=tmp_path, capture_output=True, text=True, timeout=10,
    )
    assert proc.returncode == 1
    assert output.read_bytes() == b'keep this file'


def test_ambiguous_source_is_rejected(tmp_path):
    source = json.loads((handoff.ROOT / handoff.SOURCE_FILE).read_bytes())
    seed = next(r for r in source['geometries'] if r.get('polytope_id') == 'simplex-P4')
    source['geometries'].append(copy.deepcopy(seed))
    path = tmp_path / handoff.SOURCE_FILE
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match='exactly one'):
        handoff.build_files(tmp_path)
