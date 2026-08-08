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
