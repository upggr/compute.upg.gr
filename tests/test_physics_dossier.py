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
    assert 'period' in tabs['fluxes']['requires_for_full_scan'][0].lower()
    assert any('Yukawa' in x or 'soft' in x.lower() for x in tabs['phenomenology']['pending_geometry'])
    assert tabs['fluxes']['lattice_miniscan']['K'] == 13
    assert tabs['fluxes']['orientifold']['base_L'] is not None
    assert tabs['fluxes']['periods_full']['picard_fuchs_order'] == 13
    assert tabs['phenomenology']['soft_terms']['terms']
    assert tabs['phenomenology']['gauge']['steps']
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


def test_geometry_pack_fills_quintic_vertices():
    c = physics_dossier.construction_payload(
        'kreuzer-skarke', 'q', h11=1, h21=101, euler_char=-200,
    )
    assert 'vertex_matrix' in c['present_geometry'] or 'polytope_vertices' in c['present_geometry']
    assert 'hypersurface_equation' in c['present_geometry']
    assert not any(u['id'] == 'hypersurface_equation' for u in c['unavailable'])


def test_flux_miniscan_mirror_quintic():
    import physics_extensions
    scan = physics_extensions.flux_lattice_miniscan(1, 200 / 24.0)
    assert scan['status'] == 'computed'
    assert scan['counted'] > 0
    assert scan['K'] == 2


def test_cy5_pack_rejects_string_config_matrix():
    import physics_extensions
    physics_extensions.reload_geometry_pack()
    physics_dossier.load_known_constructions(force_reload=True)
    # Placeholder strings must never count as a present configuration matrix.
    assert not physics_extensions.is_real_configuration_matrix(
        'attach real CI5F matrix in raw when available'
    )
    assert physics_extensions.is_real_configuration_matrix([[3, 3], [1, 1]])
    merged = physics_extensions.merge_geometry_into_raw(
        {},
        {'configuration_matrix': 'fake string', 'ambient': 'P'},
    )
    assert 'configuration_matrix' not in merged
    assert merged['ambient'] == 'P'

    c = physics_dossier.construction_payload(
        'cy5-folds',
        'cy5-folds-cab01a33852a',
        h11=140,
        h21=62,
        h31=18,
        features=[['h11', 140], ['h21', 62], ['h31', 18]],
    )
    assert 'configuration_matrix' not in c['present_geometry']
    assert any(u['id'] == 'configuration_matrix' for u in c['unavailable'])
    assert 'pending' in c['reconstruct_howto'].lower() or 'still' in c['reconstruct_howto'].lower()
    note = c['present_geometry'].get('geometry_note') or (c.get('curated') or {}).get('note') or ''
    assert 'matrix' in note.lower() or 'CI5F' in note


def test_featured_cy5_triples_have_honest_pack_entries():
    import physics_extensions
    physics_extensions.reload_geometry_pack()
    for h11, h21, h31 in ((140, 62, 18), (151, 41, 22), (131, 55, 29), (112, 48, 33)):
        pack = physics_extensions.lookup_geometry_pack('cy5-folds', h11, h21, h31)
        assert pack is not None, (h11, h21, h31)
        assert pack.get('configuration_matrix') is None
        assert 'invent' in (pack.get('note') or '').lower() or 'needs' in (pack.get('note') or '').lower()
        curated = physics_dossier.lookup_known_construction('cy5-folds', h11, h21)
        assert curated is not None


def test_heterotic_chi6_constructions():
    physics_dossier.load_known_constructions(force_reload=True)
    for h11, h21, chi in ((73, 70, 6), (89, 92, -6)):
        c = physics_dossier.lookup_known_construction('heterotic', h11, h21)
        assert c is not None
        assert '3' in c['note'] or 'generation' in c['note'].lower()
        d = physics_dossier.build_dossier('heterotic', h11, h21)
        assert d['euler_char'] == chi
        assert abs(chi) // 2 == 3
