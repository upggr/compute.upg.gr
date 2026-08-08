#!/usr/bin/env python3
"""Stub CLI for offline CYTools / PALP geometry workers.

This script does **not** run CYTools. Compute polytopes / triangulations /
periods on a machine that has CYTools (or PALP), write a JSON dump matching
``scripts/geometry_record.schema.json``, then upsert into the geometry SQLite DB:

  # On the offline / CYTools host:
  python scripts/geometry_worker_stub.py \\
      --input /tmp/my_polytope.json \\
      --db static/data/geometry.sqlite

  # Or push the sqlite / JSON onto the web host volume and re-ingest.

Example input JSON
------------------
{
  "dataset_id": "kreuzer-skarke",
  "h11": 2,
  "h21": 86,
  "euler_char": -168,
  "source": "cytools-offline",
  "status": "representative",
  "note": "One FRST representative; not unique for this Hodge pair.",
  "polytope_vertices": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]],
  "vertex_matrix": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]],
  "triangulation": {"kind": "frst", "id": "example"},
  "intersections": {"triple_summary": "offline-provided; not invented"},
  "periods": null,
  "stage": "triangulated",
  "candidate_id": null,
  "extra": {"cytools_version": "1.x", "polytope_id": "offline-demo"}
}

Pipeline stages (inferred from richness if ``stage`` omitted)
-------------------------------------------------------------
``vertices`` → ``triangulated`` → ``intersections`` → ``periods``

Honesty: never invent soft spectra, Yukawas, unique polytopes, or numerical
periods. Only upsert fields that an offline CYTools/PALP run actually produced.
When multiple polytopes share Hodge numbers, set ``status=representative`` and
explain non-uniqueness in ``note``.

Batch mode: pass a JSON array of records, or ``{"geometries":[...]}``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import geometry_store  # noqa: E402


def _load_records(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as fh:
        payload = json.load(fh)
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        if 'geometries' in payload and isinstance(payload['geometries'], list):
            return [r for r in payload['geometries'] if isinstance(r, dict)]
        if 'polytopes' in payload and isinstance(payload['polytopes'], list):
            return [r for r in payload['polytopes'] if isinstance(r, dict)]
        return [payload]
    raise ValueError('Input must be a JSON object or array of geometry records')


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Upsert offline-computed geometry JSON into geometry.sqlite',
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--input', '-i', required=True,
        help='JSON file: one record, array, or {geometries:[...]}',
    )
    parser.add_argument(
        '--db',
        default=geometry_store.DEFAULT_DB_PATH,
        help='SQLite path (default: static/data/geometry.sqlite)',
    )
    parser.add_argument(
        '--default-source',
        default='cytools-offline',
        help='Filled when a record omits source',
    )
    parser.add_argument(
        '--default-status',
        default='representative',
        help='Filled when a record omits status',
    )
    args = parser.parse_args()

    records = _load_records(args.input)
    if not records:
        print('No records found in input', file=sys.stderr)
        return 1

    geometry_store.init_db(args.db)
    upserted = 0
    for rec in records:
        if 'h11' not in rec or 'h21' not in rec:
            print(f'Skip record missing h11/h21: {rec!r}', file=sys.stderr)
            continue
        rec = dict(rec)
        rec.setdefault('dataset_id', 'kreuzer-skarke')
        rec.setdefault('source', args.default_source)
        rec.setdefault('status', args.default_status)
        stored = geometry_store.upsert_geometry(rec, db_path=args.db)
        upserted += 1
        print(
            f"upserted id={stored['id']} status={stored.get('status')} "
            f"stage={stored.get('stage')} source={stored.get('source')}"
        )

    print(f'Done: {upserted}/{len(records)} → {args.db}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
