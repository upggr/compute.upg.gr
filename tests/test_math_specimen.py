"""Mathematics analysis tab: geometric specimen framing."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math_specimen  # noqa: E402
import physics_dossier  # noqa: E402
import physics_extensions  # noqa: E402


@pytest.fixture
def client():
    import app as app_module
    app_module.app.config['TESTING'] = True
    with app_module.app.test_client() as c:
        yield c


def test_mathematics_tab_quintic_honesty_and_mirror():
    physics_extensions.reload_geometry_pack()
    d = physics_dossier.build_dossier('kreuzer-skarke', 1, 101, euler_char=-200)
    construction = physics_dossier.construction_payload(
        'kreuzer-skarke',
        'kreuzer-skarke-66d611d18a9d',
        h11=1,
        h21=101,
        euler_char=-200,
    )
    partner = {
        'h11': 101,
        'h21': 1,
        'euler_char': 200,
        'candidate_id': 'kreuzer-skarke-mirror-demo',
        'on_board': True,
        'display': 'h¹¹=101 · h²¹=1 · χ=200',
    }
    tabs = physics_dossier.build_tabs(
        d, construction=construction, mirror_partner=partner,
    )
    assert tabs['ok']
    math_tab = tabs['mathematics']
    assert math_tab['id'] == 'mathematics'
    assert math_specimen.SPECIMEN_FRAMING in math_tab['specimen']['framing']
    assert 'citeable geometric specimen' in math_tab['specimen']['framing'].lower()
    assert 'proof of string theory' not in math_tab['specimen']['framing'].lower()
    assert math_tab['invariants']['h11'] == 1
    assert math_tab['invariants']['h21'] == 101
    assert math_tab['invariants']['euler_char'] == -200
    assert math_tab['invariants']['b3'] == 204
    assert math_tab['invariants']['picard_fuchs_order'] == 102
    assert math_tab['invariants']['mirror_h11'] == 101
    assert math_tab['invariants']['mirror_h21'] == 1
    assert math_tab['mirror_symmetry']['partner']['on_board'] is True
    assert math_tab['mirror_symmetry']['level'] == 'hof_partner'
    assert math_tab['enumerative']['periods_status'] == 'literature_curated'
    assert math_tab['enumerative']['periods_literature']
    assert '#mathematics' in math_tab['cite']['deep_links']['mathematics']
    assert '#certificates' in math_tab['cite']['deep_links']['certificates']
    assert '@misc{' in math_tab['cite']['bibtex']
    assert math_tab['found'] is True
    assert math_tab['found_badge'] == 'FOUND'
    assert math_tab['match']['headline'] == 'SPECIMEN FOUND'
    assert math_tab['match']['reasons']


def test_mathematics_pending_periods_non_quintic():
    d = physics_dossier.build_dossier('kreuzer-skarke', 99, 50)
    construction = physics_dossier.construction_payload(
        'kreuzer-skarke', 'bare-test', h11=99, h21=50, euler_char=98,
    )
    tabs = physics_dossier.build_tabs(d, construction=construction, tags=[])
    math_tab = tabs['mathematics']
    assert math_tab['enumerative']['periods_status'] == 'pending'
    assert 'CYTools' in math_tab['enumerative']['handoff']
    assert math_tab['mirror_symmetry']['level'] == 'hodge_swap_only'
    assert math_tab['found'] is False
    assert math_tab['found_badge'] is None
    assert math_tab['match']['headline'] == 'Specimen pending'


def test_candidate_page_has_mathematics_tab(client):
    r = client.get('/candidate/kreuzer-skarke-66d611d18a9d')
    assert r.status_code == 200
    html = r.data.decode('utf-8')
    assert 'data-tab="mathematics"' in html
    assert 'tab-mathematics' in html
    assert 'tab-found-badge' in html or 'FOUND' in html
    assert 'SPECIMEN FOUND' in html or 'found-banner-yes' in html
    assert 'citeable geometric specimen' in html.lower()
    assert 'proof of string theory' not in html.lower()
    assert 'analysis-tab-found' in html
