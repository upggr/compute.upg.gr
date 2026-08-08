#!/usr/bin/env python3
"""Refresh data/ks_geometry_sample.json from HF calabi-yau-data/polytopes-4d slices.

Downloads ONLY small/medium vertex-count parquet partitions, extracts one
representative polytope per target Hodge pair, writes static JSON, and deletes
parquet caches. Does not ship the full 473M database.

Requires: pandas, pyarrow (dev-only; not needed at Coolify runtime).
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'data' / 'ks_cache'
OUT = ROOT / 'data' / 'ks_geometry_sample.json'
HF_BASE = (
    'https://huggingface.co/datasets/calabi-yau-data/polytopes-4d/resolve/main'
)

SLICES = [
    'polytopes-4d-05-vertices.parquet',
    'polytopes-4d-06-vertices.parquet',
    'polytopes-4d-07-vertices.parquet',
    'polytopes-4d-08-vertices.parquet',
    'polytopes-4d-09-vertices.parquet',
    'polytopes-4d-10-vertices.parquet',
]

TARGETS: Dict[Tuple[int, int], str] = {
    (1, 101): 'Quintic class (textbook / KS)',
    (101, 1): 'Mirror quintic class (textbook / KS)',
    (2, 83): 'Bicubic Hodge class',
    (4, 68): 'Tetraquadric Hodge class',
    (1, 149): 'CICY P5[2,4] Hodge class',
    (1, 103): 'Weighted octic Hodge class',
    (19, 19): 'Self-mirror Hodge class',
    (2, 86): 'P3xP1 hypersurface Hodge class',
    (38, 12): 'HoF featured KS (38,12)',
    (25, 26): 'HoF featured KS (25,26)',
    (14, 62): 'HoF featured KS (14,62)',
    (44, 33): 'HoF featured KS (44,33)',
}


def _verts_to_list(vertices: Any) -> List[List[int]]:
    out: List[List[int]] = []
    for row in vertices:
        if hasattr(row, 'tolist'):
            out.append([int(x) for x in row.tolist()])
        else:
            out.append([int(x) for x in row])
    return out


def main() -> None:
    import pandas as pd

    CACHE.mkdir(parents=True, exist_ok=True)
    hits: Dict[Tuple[int, int], Dict[str, Any]] = {}
    counts: Dict[Tuple[int, int], int] = {}

    for fname in SLICES:
        path = CACHE / fname
        if not path.exists():
            url = f'{HF_BASE}/{fname}'
            print(f'downloading {fname} ...')
            urllib.request.urlretrieve(url, path)
        df = pd.read_parquet(
            path,
            columns=[
                'h11',
                'h12',
                'vertices',
                'vertex_count',
                'euler_characteristic',
                'point_count',
                'dual_point_count',
                'facet_count',
            ],
        )
        for key, label in TARGETS.items():
            match = df[(df['h11'] == key[0]) & (df['h12'] == key[1])]
            if len(match) == 0:
                continue
            counts[key] = counts.get(key, 0) + int(len(match))
            if key in hits:
                continue
            row = match.iloc[0]
            hits[key] = {
                'dataset_id': 'kreuzer-skarke',
                'h11': int(key[0]),
                'h21': int(key[1]),
                'name': label,
                'polytope_vertices': _verts_to_list(row['vertices']),
                'vertex_matrix': _verts_to_list(row['vertices']),
                'vertex_count': int(row['vertex_count']),
                'facet_count': int(row['facet_count']),
                'point_count': int(row['point_count']),
                'dual_point_count': int(row['dual_point_count']),
                'euler_characteristic': int(row['euler_characteristic']),
                'source_slice': fname,
                'geometry_status': 'representative',
                'uniqueness': 'one polytope with these Hodge numbers; not unique',
                'note': (
                    'Real reflexive 4-polytope vertices from Hugging Face '
                    'calabi-yau-data/polytopes-4d (Kreuzer–Skarke). '
                    'ONE representative — not unique at the Hodge level.'
                ),
                'reference': (
                    'Kreuzer–Skarke arXiv:hep-th/0002240; '
                    'HF calabi-yau-data/polytopes-4d'
                ),
            }

    payload = {
        'version': 1,
        'note': (
            'Sidecar sample of REAL Kreuzer–Skarke reflexive 4-polytope vertices '
            'for textbook + featured Hall-of-Fame Hodge pairs. NOT the full 473M DB.'
        ),
        'source': 'https://huggingface.co/datasets/calabi-yau-data/polytopes-4d',
        'citation': 'Kreuzer:2000xy (hep-th/0002240)',
        'polytopes': [hits[k] for k in sorted(hits)],
        'match_counts_in_downloaded_slices': {
            f'{a},{b}': counts.get((a, b), 0) for a, b in sorted(counts)
        },
        'missing_targets': [
            {'h11': a, 'h21': b, 'label': TARGETS[(a, b)]}
            for (a, b) in TARGETS
            if (a, b) not in hits
        ],
    }
    OUT.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(f'wrote {OUT} ({len(payload["polytopes"])} polytopes)')
    for path in CACHE.glob('*.parquet'):
        path.unlink()
        print(f'deleted cache {path.name}')


if __name__ == '__main__':
    main()
