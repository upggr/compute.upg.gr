"""Phase 1 model-building: exclusions, cards, geometry stage, APIs."""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import model_cards  # noqa: E402
import model_exclusions  # noqa: E402


REQUIRED_KEYS = ('ok', 'rules_out', 'assumptions', 'detail')


def _by_id(certs, cert_id):
    return next(c for c in certs if c['id'] == cert_id)


def test_heterotic_chi6_does_not_rule_out_standard_embedding_3gen():
    # χ = 2(4-1) = 6
    certs = model_exclusions.evaluate('heterotic', 4, 1)
    het = _by_id(certs, 'heterotic_standard_embedding_3gen')
    assert het['ok'] is True
    assert abs(het.get('euler_abs', 6)) == 6 or '|χ|=6' in het['detail'] or 'χ|=6' in het['detail']


def test_heterotic_chi8_rules_out_standard_embedding_3gen():
    # χ = 2(5-1) = 8
    certs = model_exclusions.evaluate('heterotic', 5, 1)
    het = _by_id(certs, 'heterotic_standard_embedding_3gen')
    assert het['ok'] is False
    assert '3-generation' in het['rules_out'].lower() or '3 generation' in het['rules_out'].lower()


def test_tadpole_rules_out_when_L_below_stated_minimum():
    # |χ|=2 → L = 2/24 ≈ 0.0833 < 1.0
    certs = model_exclusions.evaluate(
        'kreuzer-skarke', 2, 1, tadpole_L_min=1.0,
    )
    tad = _by_id(certs, 'ks_tadpole_budget')
    assert tad['ok'] is False
    assert any('1' in a or 'L_min' in a or 'minimum' in a.lower() for a in tad['assumptions'])
    assert 'L=' in tad['detail'] or 'L =' in tad['detail'] or '0.08' in tad['detail']


def test_tadpole_passes_when_L_meets_minimum():
    # Quintic |χ|=200 → L ≈ 8.333 ≥ 1
    certs = model_exclusions.evaluate(
        'kreuzer-skarke', 1, 101, tadpole_L_min=1.0,
    )
    tad = _by_id(certs, 'ks_tadpole_budget')
    assert tad['ok'] is True


def test_every_exclusion_has_required_fields():
    certs = model_exclusions.evaluate('kreuzer-skarke', 1, 101)
    assert len(certs) >= 1
    for c in certs:
        for key in REQUIRED_KEYS:
            assert key in c, f'missing {key} in {c.get("id")}'
        assert isinstance(c['ok'], bool)
        assert isinstance(c['assumptions'], list)
        assert len(c['assumptions']) >= 1
        assert isinstance(c['rules_out'], str)
        assert isinstance(c['detail'], str)


# --- model cards -----------------------------------------------------------

def test_model_cards_load_at_least_three():
    cards = model_cards.load_cards(force_reload=True)
    assert len(cards) >= 3
    for card in cards:
        assert card['reference_url'] or card['arxiv']
        assert card['title']
        assert card['framework'] in (
            'iib-flux', 'heterotic', 'f-theory', 'other',
        )
        assert 'spectrum_summary' in card
        assert card['honesty']


def test_model_cards_match_quintic_and_heterotic():
    q = model_cards.list_for_hodge('kreuzer-skarke', 1, 101)
    assert any('CdOGP' in c['title'] or 'quintic' in c['title'].lower() for c in q)
    h = model_cards.lookup('heterotic', 73, 70)
    assert h is not None
    assert h['framework'] == 'heterotic'
    assert 'arxiv.org' in (h.get('reference_url') or '') or h.get('arxiv')


def test_model_card_lookup_miss():
    assert model_cards.lookup('kreuzer-skarke', 99999, 99999) is None
