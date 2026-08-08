#!/usr/bin/env python3
"""Ingest baked geometry JSON into static/data/geometry.sqlite (idempotent).

Usage:
  python scripts/ingest_geometry_db.py
  python scripts/ingest_geometry_db.py --db /path/to/geometry.sqlite
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import geometry_store  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--db',
        default=geometry_store.DEFAULT_DB_PATH,
        help='SQLite path (default: static/data/geometry.sqlite)',
    )
    parser.add_argument(
        '--ks-sample',
        default=geometry_store.KS_SAMPLE_PATH,
        help='Path to ks_geometry_sample.json',
    )
    parser.add_argument(
        '--geometry-pack',
        default=geometry_store.GEOMETRY_PACK_PATH,
        help='Path to geometry_pack.json',
    )
    args = parser.parse_args()
    stats = geometry_store.seed_baked_geometry(
        db_path=args.db,
        ks_path=args.ks_sample,
        pack_path=args.geometry_pack,
    )
    print(
        f"Ingested ks_sample={stats['ks_sample']} "
        f"geometry_pack={stats['geometry_pack']} "
        f"total_rows={stats['total']} → {args.db}"
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
