"""Citeable dossiers, quintic showcase pipeline, and draft note routes."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import physics_dossier  # noqa: E402
import physics_extensions  # noqa: E402


def test_honest_pipeline_checklist_quintic_literature():
    physics_extensions.reload_geometry_pack()
    c = physics_dossier.construction_payload(
        'kreuzer-skarke',
        'kreuzer-skarke-66d611d18a9d',
        h11=1,
        h21=101,
        euler_char=-200,
    )
    assert c.get('showcase') is True
    checklist = c.get('pipeline_checklist') or []
    by_stage = {s['stage']: s for s in checklist}
    assert by_stage['vertices']['filled'] is True
    assert by_stage['triangulated']['filled'] is True
    assert by_stage['intersections']['filled'] is True
    assert by_stage['intersections']['status'] == 'literature'
    assert by_stage['periods']['filled'] is False
    assert 'CdOGP' in by_stage['periods']['detail'] or 'literature' in by_stage['periods']['detail'].lower()


def test_build_tabs_exposes_rich_pipeline_showcase_and_math():
    d = physics_dossier.build_dossier('kreuzer-skarke', 1, 101)
    construction = physics_dossier.construction_payload(
        'kreuzer-skarke', 'kreuzer-skarke-66d611d18a9d',
        h11=1, h21=101, euler_char=-200,
    )
    tabs = physics_dossier.build_tabs(d, construction=construction)
    assert tabs['ok']
    assert tabs['model_building']['showcase'] is True
    assert tabs['model_building']['periods_literature']
    assert any(s.get('detail') for s in tabs['model_building']['geometry_pipeline']['checklist'])
    assert tabs['construction']['pipeline_checklist']
    assert tabs['mathematics']['id'] == 'mathematics'
    assert 'specimen' in tabs['mathematics']['specimen']['framing'].lower() or 'specimen' in tabs['mathematics']['specimen']['headline'].lower()
