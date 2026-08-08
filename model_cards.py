"""Literature model cards for string model-building (cited spectra only).

Loads ``data/model_cards.json`` into memory (optional SQLite ingest later).
Cards are class-level literature pointers — never invented soft spectra.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

DEFAULT_CARDS_PATH = os.path.join('data', 'model_cards.json')

_CARDS: List[Dict[str, Any]] = []
_LOADED_FROM: Optional[str] = None


def _normalize_card(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'id': str(raw['id']),
        'dataset_id': str(raw.get('dataset_id') or ''),
        'h11': int(raw['h11']) if raw.get('h11') is not None else None,
        'h21': int(raw['h21']) if raw.get('h21') is not None else None,
        'h31': int(raw['h31']) if raw.get('h31') is not None else None,
        'candidate_id': raw.get('candidate_id'),
        'framework': str(raw.get('framework') or 'other'),
        'title': str(raw.get('title') or ''),
        'reference': str(raw.get('reference') or ''),
        'reference_url': raw.get('reference_url'),
        'arxiv': raw.get('arxiv'),
        'assumptions': list(raw.get('assumptions') or []),
        'spectrum_summary': str(raw.get('spectrum_summary') or ''),
        'geometry_status': str(raw.get('geometry_status') or 'hodge-only'),
        'honesty': str(raw.get('honesty') or ''),
    }


def load_cards(
    path: str = DEFAULT_CARDS_PATH,
    *,
    force_reload: bool = False,
) -> List[Dict[str, Any]]:
    """Load and cache model cards from JSON."""
    global _CARDS, _LOADED_FROM
    if (
        not force_reload
        and _CARDS
        and _LOADED_FROM == os.path.abspath(path)
    ):
        return list(_CARDS)

    cards: List[Dict[str, Any]] = []
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            payload = json.load(fh)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        _CARDS = []
        _LOADED_FROM = os.path.abspath(path)
        return []

    raw_list = payload.get('cards') if isinstance(payload, dict) else payload
    if not isinstance(raw_list, list):
        raw_list = []
    for item in raw_list:
        if not isinstance(item, dict) or 'id' not in item:
            continue
        if item.get('h11') is None or item.get('h21') is None:
            continue
        if not item.get('reference_url') and not item.get('arxiv'):
            # Require a citable URL or arXiv id.
            continue
        cards.append(_normalize_card(item))

    _CARDS = cards
    _LOADED_FROM = os.path.abspath(path)
    return list(_CARDS)


def all_cards(*, path: str = DEFAULT_CARDS_PATH) -> List[Dict[str, Any]]:
    return load_cards(path)


def list_for_hodge(
    dataset_id: str,
    h11: int,
    h21: int,
    h31: Optional[int] = None,
    *,
    path: str = DEFAULT_CARDS_PATH,
    include_dataset_agnostic: bool = False,
) -> List[Dict[str, Any]]:
    """Cards matching dataset + Hodge key (optional h31 exact match when set)."""
    cards = load_cards(path)
    out: List[Dict[str, Any]] = []
    for card in cards:
        if int(card['h11']) != int(h11) or int(card['h21']) != int(h21):
            continue
        if h31 is not None and card.get('h31') is not None:
            if int(card['h31']) != int(h31):
                continue
        if card['dataset_id'] == dataset_id:
            out.append(card)
        elif include_dataset_agnostic and not card['dataset_id']:
            out.append(card)
    return out


def lookup(
    dataset_id: str,
    h11: int,
    h21: int,
    h31: Optional[int] = None,
    *,
    path: str = DEFAULT_CARDS_PATH,
    card_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Return one matching card (by id if given, else first Hodge match)."""
    if card_id:
        for card in load_cards(path):
            if card['id'] == card_id:
                return card
        return None
    matches = list_for_hodge(dataset_id, h11, h21, h31, path=path)
    return matches[0] if matches else None
