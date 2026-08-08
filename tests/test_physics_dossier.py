"""Tests for topological physics dossier."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import physics_dossier  # noqa: E402


def test_euler_identity_cy3():
    d = physics_dossier.build_dossier('kreuzer-skarke', 73, 95)
    assert d['ok']
    assert d['euler_char'] == 2 * (73 - 95)
    assert d['euler_consistent'] is True


def test_dataset_target_low_euler():
    d = physics_dossier.build_dossier('kreuzer-skarke', 73, 95, euler_char=-44)
    target = next(c for c in d['checks'] if c['id'] == 'dataset_target')
    assert target['ok'] is True  # |-44| < 100


def test_dataset_target_fails_large_euler():
    d = physics_dossier.build_dossier('kreuzer-skarke', 200, 50)
    assert d['euler_char'] == 300
    target = next(c for c in d['checks'] if c['id'] == 'dataset_target')
    assert target['ok'] is False


def test_inconsistent_supplied_euler_flagged():
    d = physics_dossier.build_dossier('kreuzer-skarke', 10, 5, euler_char=999)
    assert d['euler_consistent'] is False
    identity = next(c for c in d['checks'] if c['id'] == 'euler_identity')
    assert identity['ok'] is False


def test_tadpole_and_mirror():
    d = physics_dossier.build_dossier('kreuzer-skarke', 38, 12)
    assert d['scalars']['tadpole_L'] == round(abs(2 * (38 - 12)) / 24, 4)
    assert d['scalars']['mirror_h11'] == 12
    assert d['scalars']['mirror_h21'] == 38


def test_build_tabs_moduli_and_flux_budget():
    d = physics_dossier.build_dossier('kreuzer-skarke', 38, 12)
    tabs = physics_dossier.build_tabs(d, tags=['verified'])
    assert tabs['ok']
    assert tabs['moduli']['counts']['kahler_moduli_h11'] == 38
    assert tabs['moduli']['counts']['complex_structure_moduli_h21'] == 12
    assert tabs['fluxes']['budget']['tadpole_L'] == d['scalars']['tadpole_L']
    assert 'Periods' in tabs['fluxes']['requires_for_full_scan'][0]
    assert 'Standard Model' in tabs['phenomenology']['not_computed'][0]
    assert tabs['phenomenology']['tags'] == ['verified']
    assert tabs['certificates']['checks']


def test_construction_payload_no_fake_equations():
    c = physics_dossier.construction_payload(
        'kreuzer-skarke',
        'kreuzer-skarke-abc',
        raw={'h11': 38, 'h21': 12, 'euler_char': 52},
        features=[['h11', 38], ['h21', 12], ['χ', 52]],
        tags=['featured-seed'],
        summary='Low |χ| candidate',
    )
    assert c['present_geometry'] == {}
    assert any(u['id'] == 'hypersurface_equation' for u in c['unavailable'])
    assert 'Kreuzer' in c['reconstruct_howto']
    assert 'h11' in c['raw_keys']
    assert c['feature_map']['h11'] == 38


def test_construction_payload_keeps_real_geometry_keys():
    c = physics_dossier.construction_payload(
        'kreuzer-skarke',
        'id',
        raw={'polytope_id': 'KS-123', 'h11': 1},
    )
    assert c['present_geometry']['polytope_id'] == 'KS-123'
    # polytope_id is present; vertex matrix still unavailable
    assert any(u['id'] == 'polytope_vertices' for u in c['unavailable'])
