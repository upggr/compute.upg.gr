#!/usr/bin/env python3
"""Package a reproducible, narrowly scoped quintic handoff. No web/DB imports."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lean_handoff.check import (  # noqa: E402
    BASIS, CONES, EQUATION, METADATA, check_bytes,
)

BUNDLE_NAME = 'upg-quintic-lean-pilot'
SOURCE_FILE = 'data/geometry_pack.json'


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + '\n').encode('utf-8')


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def build_files(root=ROOT):
    root = Path(root)
    source_bytes = (root / SOURCE_FILE).read_bytes()
    source = json.loads(source_bytes)
    matches = [
        (index, row) for index, row in enumerate(source['geometries'])
        if row.get('polytope_id') == 'simplex-P4'
    ]
    if len(matches) != 1:
        raise ValueError('Expected exactly one curated simplex-P4 source record')
    index, seed = matches[0]
    if (seed.get('dataset_id') != 'kreuzer-skarke' or seed.get('ambient') != 'P^4'
            or (seed.get('h11'), seed.get('h21')) != (1, 101)
            or seed.get('vertex_matrix') != seed.get('polytope_vertices')):
        raise ValueError('Curated source is no longer the expected quintic fixture; review before sharing')

    snapshot = json_bytes(seed)
    record = {
        **METADATA,
        'geometry': {
            'complex_dimension': 3,
            'ambient': 'P^4 over C',
            'equation': EQUATION,
            'degree': 5,
            'weights': seed['weight_system'],
            'h11': seed['h11'],
            'h21': seed['h21'],
            'euler_char': 2 * (seed['h11'] - seed['h21']),
            'intersection_basis': BASIS,
            'triple_intersection_H3': seed['intersections']['triple_intersection_H3'],
            'ambient_fan': {
                'lattice': 'N = Z^4',
                'ray_matrix': seed['vertex_matrix'],
                'maximal_cones': CONES,
            },
        },
        'provenance': {
            'source_repository': 'https://github.com/upggr/compute.upg.gr',
            'source_file': SOURCE_FILE,
            'source_json_pointer': f'/geometries/{index}',
            'source_file_sha256': sha256(source_bytes),
            'source_record_file': 'source_geometry.json',
            'source_record_sha256': sha256(snapshot),
            'reference': seed['reference'],
            'reference_url': seed['reference_url'],
        },
        'not_supplied': [
            'CYTools/PALP computation or FRST dump',
            'numerical periods or monodromy',
            'K3 fibration or CM fiber',
            'Mukai lattice',
            'orientifold, branes, fluxes or tadpole cancellation',
            'Ricci-flat metric or cymetric training data',
        ],
    }
    negative = copy.deepcopy(record)
    negative['geometry']['euler_char'] = -198
    positive_bytes = json_bytes(record)
    negative_bytes = json_bytes(negative)
    positive_report = check_bytes(positive_bytes)
    negative_report = check_bytes(negative_bytes)
    failed = [c['id'] for c in negative_report['checks'] if not c['passed']]
    if not positive_report['passed'] or failed != ['euler_identity']:
        raise ValueError('Pilot checks did not match the expected positive/negative outcomes')

    assets = root / 'examples' / 'lean_handoff'
    # Explicit allowlist: never package local DBs, run artifacts or environment files.
    files = {name: (assets / name).read_bytes() for name in ('README.md', 'check.py', 'record.schema.json')}
    files.update({
        'source_geometry.json': snapshot,
        'quintic.json': positive_bytes,
        'negative_euler.json': negative_bytes,
        'check_report.json': json_bytes({
            'formal_verification_status': 'not_performed',
            'positive': positive_report,
            'negative_control': negative_report,
            'negative_control_expected_failures': ['euler_identity'],
            'negative_control_rejected_as_expected': True,
        }),
    })
    files['SHA256SUMS'] = ''.join(
        f'{sha256(content)}  {name}\n' for name, content in sorted(files.items())
    ).encode('ascii')
    return files


def build_archive(root=ROOT):
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_STORED) as archive:
        for name, content in sorted(build_files(root).items()):
            info = zipfile.ZipInfo(f'{BUNDLE_NAME}/{name}', date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content)
    return output.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'dist' / f'{BUNDLE_NAME}.zip')
    parser.add_argument('--force', action='store_true', help='Replace an existing generated ZIP')
    args = parser.parse_args()
    try:
        content = build_archive()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('wb' if args.force else 'xb') as fh:
            fh.write(content)
    except (OSError, ValueError) as exc:
        parser.exit(1, f'{exc}\n')
    print(f'Created {args.output}\nSHA-256: {sha256(content)}')
    print('Five exact-integer checks passed; corrupted Euler fixture rejected. Lean verification: NOT performed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
