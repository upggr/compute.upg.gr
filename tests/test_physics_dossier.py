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
    assert tabs['fluxes']['budget']['log_N_flux'] is not None
    assert tabs['fluxes']['budget']['N_flux_est_sci']
    assert tabs['moduli']['stabilization']
    assert tabs['phenomenology']['indices']['n_generations'] == abs(d['euler_char']) // 2
    assert 'Periods' in tabs['fluxes']['requires_for_full_scan'][0]
    assert 'Standard Model' in tabs['phenomenology']['not_computed'][0]
    assert tabs['phenomenology']['tags'] == ['verified']
    assert tabs['certificates']['checks']
    assert any(c['id'] == 'generation_index' for c in tabs['certificates']['checks'])


def test_flux_vacua_estimate_stirling():
    est = physics_dossier.flux_vacua_estimate(12, abs(2 * (38 - 12)) / 24.0)
    assert est['K'] == 13
    assert est['log_N_flux'] is not None
    assert est['N_flux_est_sci']


def test_curated_quintic_construction():
    c = physics_dossier.construction_payload(
        'kreuzer-skarke',
        'quintic-demo',
        h11=1,
        h21=101,
        euler_char=-200,
    )
    assert c['curated']['name'].startswith('Quintic')
    assert 'hypersurface_equation' in c['present_geometry']
    assert c['workplan']
    assert not any(u['id'] == 'hypersurface_equation' for u in c['unavailable'])


def test_construction_payload_no_fake_equations():
    c = physics_dossier.construction_payload(
        'kreuzer-skarke',
        'kreuzer-skarke-abc',
        raw={'h11': 38, 'h21': 12, 'euler_char': 52},
        features=[['h11', 38], ['h21', 12], ['χ', 52]],
        tags=['featured-seed'],
        summary='Low |χ| candidate',
        h11=38,
        h21=12,
        euler_char=52,
    )
    assert 'hypersurface_equation' not in c['present_geometry']
    assert any(u['id'] == 'hypersurface_equation' for u in c['unavailable'])
    assert 'Kreuzer' in c['reconstruct_howto']
    assert 'h11' in c['raw_keys']
    assert c['feature_map']['h11'] == 38
    assert len(c['workplan']) >= 3


def test_construction_payload_keeps_real_geometry_keys():
    c = physics_dossier.construction_payload(
        'kreuzer-skarke',
        'id',
        raw={'polytope_id': 'KS-123', 'h11': 1},
    )
    assert c['present_geometry']['polytope_id'] == 'KS-123'
    # polytope_id is present; vertex matrix still unavailable
    assert any(u['id'] == 'polytope_vertices' for u in c['unavailable'])


def test_three_generation_index():
    d = physics_dossier.build_dossier('kreuzer-skarke', 4, 1)  # χ = 6
    tabs = physics_dossier.build_tabs(d)
    assert d['euler_char'] == 6
    assert tabs['phenomenology']['indices']['n_generations'] == 3
    assert tabs['phenomenology']['indices']['three_generation_target'] is True

def test_info_density_target_not_auto_pass():
    d = physics_dossier.build_dossier('info-density', 38, 12)
    target = next(c for c in d['checks'] if c['id'] == 'dataset_target')
    assert target['ok'] is False
    assert 'ranking run' in target['detail']


def test_cy5_diamond_kind():
    d = physics_dossier.build_dossier('cy5-folds', 140, 62, h31=18)
    assert d['ok']
    assert d['diamond']['kind'] == 'cy5'
    assert d['h31'] == 18
    assert 'CY5' in d['diamond']['note']


def test_known_constructions_json_loads():
    physics_dossier.load_known_constructions(force_reload=True)
    c = physics_dossier.lookup_known_construction('kreuzer-skarke', 4, 68)
    assert c is not None
    assert 'Tetraquadric' in c['name']


def test_mirror_partner_in_tabs():
    d = physics_dossier.build_dossier('kreuzer-skarke', 1, 101)
    partner = {
        'h11': 101, 'h21': 1, 'euler_char': 200,
        'candidate_id': 'demo', 'on_board': True, 'display': 'mirror',
    }
    tabs = physics_dossier.build_tabs(d, mirror_partner=partner)
    assert tabs['moduli']['mirror']['partner']['on_board'] is True
    assert tabs['moduli']['mirror']['h11'] == 101


def test_analyze_includes_tabs():
    # Mirror what _analyze_candidate attaches without importing Flask app.
    d = physics_dossier.build_dossier('kreuzer-skarke', 38, 12, euler_char=52, verified_target=True)
    construction = physics_dossier.construction_payload(
        'kreuzer-skarke', 'demo', h11=38, h21=12, euler_char=52,
    )
    tabs = physics_dossier.build_tabs(d, construction=construction, tags=['verified'])
    assert tabs['ok']
    assert 'moduli' in tabs
    assert tabs['phenomenology']['indices']['n_generations'] == 26
    assert tabs['fluxes']['budget']['log_N_flux'] is not None


def test_period_and_intersection_proxies():
    p = physics_dossier.period_structure(12)
    assert p['picard_fuchs_order'] == 13
    assert p['h3_betti'] == 26
    ix = physics_dossier.intersection_proxies(3)
    assert ix['symmetric_triple_independent'] == 10


def test_scan_readiness_and_links():
    d = physics_dossier.build_dossier('kreuzer-skarke', 1, 101, euler_char=-200)
    c = physics_dossier.construction_payload(
        'kreuzer-skarke', 'q', h11=1, h21=101, euler_char=-200,
    )
    tabs = physics_dossier.build_tabs(d, construction=c)
    assert tabs['fluxes']['readiness']['total'] >= 6
    assert tabs['fluxes']['readiness']['score'] >= 3
    assert tabs['overview']['external_links']
    assert tabs['construction']['external_links']
    assert tabs['moduli']['counts']['picard_fuchs_order'] == 102
    assert any(ch['id'] == 'scan_readiness' for ch in tabs['certificates']['checks'])


def test_heterotic_sketch_on_pheno_tab():
    d = physics_dossier.build_dossier('heterotic', 73, 70)
    tabs = physics_dossier.build_tabs(d)
    assert tabs['phenomenology']['heterotic']['three_generation_target'] is True
    assert tabs['phenomenology']['heterotic']['checklist']
