"""Bounded research handoff from the sourced KS sidecar, without ML or DB merges.

Records retain their own vertices and provenance. Hodge matches must never be
used to attach another construction's triangulation/intersections to this data.
"""

from __future__ import annotations

import io
import json
import zipfile
from fractions import Fraction
from pathlib import Path

from scripts.prepare_lean_handoff import json_bytes, sha256

ROOT = Path(__file__).resolve().parent
SOURCE_FILE = 'data/ks_geometry_sample.json'
MAX_SELECTION = 50
SCHEMA = 'upg-research-handoff-v1'
SCOPE = (
    'Sourced KS sample records, not new ML discoveries. Ordering is within this '
    'small local sample, not a global rarity ranking. Hodge/Euler labels come '
    'from the source and have not been recomputed from the vertices. Integer '
    'checks are not geometry certification or Lean proofs.'
)
MISSING = [
    'triangulation', 'intersection tensor and basis', 'periods / monodromy',
    'K3 fibration / CM fiber', 'Mukai lattice',
    'orientifold / branes / fluxes / tadpole cancellation', 'Ricci-flat metric',
]
BATCH_README = '''# upg-strings: selected KS sample records

This is a batch from the local sourced Kreuzer-Skarke sidecar, NOT an ML top-k
geometry discovery or a claim of global rarity. The manifest records the chosen
order; inspect each record and its source snapshot before deeper computation.
No ML scores or textbook intersections have been attached by Hodge matching.
Any Hodge-class gallery links on the website are context only, not identification
of these particular polytopes or evidence of a model score for them.

## Contents and provenance

- manifest.json: selected IDs/order, source hash, scope and missing inputs.
- records/*.json: vertex data, source labels, provenance and integer checks.
- source/ks_geometry_sample.json: exact bytes of the original local sidecar.
- SHA256SUMS: hashes of every other file (including this README).

Verify with `shasum -a 256 -c SHA256SUMS` (macOS) or
`sha256sum -c SHA256SUMS` (Linux). Hashes detect altered bytes; they do not
certify geometry or the sender. Record IDs are shortened hashes of canonical
JSON source records, NOT universal polytope IDs or lattice-isomorphism classes.
The full source-record SHA-256 is also provided. To reproduce that hash, serialize
the pointed-to source row with Python json.dumps(row, indent=2, sort_keys=True,
ensure_ascii=True, allow_nan=False), append a newline, and encode UTF-8.
The source snapshot includes unselected rows for provenance; only IDs listed
in the manifest and records/ are part of this selection.

## What has been checked?

On the stored integer data only: Euler-label consistency chi = 2(h11-h21),
vertex count, distinct vertices, agreement of the two source vertex arrays,
and full affine rank 4 (exact rational elimination). The data loader requires
integer coordinates in four columns. Hodge numbers, reflexivity, triangulations,
and smooth Calabi-Yau geometry have NOT been independently computed here.

Vertices are copied in the source's basis/polar convention. Confirm which
polytope/lattice a downstream CYTools/PALP import expects before taking a dual
or triangulating. The same Hodge pair does not identify a unique geometry.
A matrix alone is not a complete CYTools triangulation or cymetric input.

## Missing, rather than invented

No triangulation, intersection tensor/basis, numerical periods/monodromy,
K3/CM fibration data, Mukai lattice, brane/orientifold/flux setup, tadpole
cancellation, or Ricci-flat metric is supplied by this sample handoff.
In particular, |chi(CY3)|/24 is not an F-theory CY4 D3-tadpole budget.
A SymPy/Arb monodromy experiment needs additional analytic data first.

## Suggested next step

Choose one record by its full source hash. Independently recompute its Hodge
numbers and an explicit triangulation/intersection basis offline, agree on a
precise Lean theorem, and return the input hash, pinned tools, commands, proof
source, assumptions, axiom report and logs. The separate quintic calibration
kit on /handoff tests the bridge and negative control; it does not fill these
gaps. No Lean build is performed by this download endpoint.
'''


class StaleSource(ValueError):
    """The client selected from a different source snapshot."""


def affine_rank(vertices):
    """Rank of vertex differences, using exact rational row elimination."""
    rows = [[Fraction(x - y) for x, y in zip(v, vertices[0])] for v in vertices[1:]]
    rank = 0
    for column in range(4):
        pivot = next((i for i in range(rank, len(rows)) if rows[i][column]), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        value = rows[rank][column]
        rows[rank] = [x / value for x in rows[rank]]
        for i in range(rank + 1, len(rows)):
            factor = rows[i][column]
            rows[i] = [x - factor * y for x, y in zip(rows[i], rows[rank])]
        rank += 1
    return rank


def _record(raw, index, source):
    if not isinstance(raw, dict) or raw.get('dataset_id') != 'kreuzer-skarke':
        raise ValueError('Expected a KS source record')
    for key in ('h11', 'h21', 'euler_characteristic', 'vertex_count'):
        if type(raw.get(key)) is not int:
            raise ValueError(f'Source {key} must be an exact integer')
    if min(raw['h11'], raw['h21']) < 0:
        raise ValueError('Source Hodge numbers must be non-negative')
    vertices = raw.get('polytope_vertices')
    if (not isinstance(vertices, list) or not 5 <= len(vertices) <= 100
            or any(not isinstance(v, list) or len(v) != 4
                   or any(type(x) is not int or abs(x) > 10**9 for x in v) for v in vertices)):
        raise ValueError('Expected 5–100 rows of four bounded exact integer coordinates')
    h11, h21, chi = raw['h11'], raw['h21'], raw['euler_characteristic']
    rank = affine_rank(vertices)
    checks = [
        {'id': 'euler_identity', 'label': 'Euler labels: chi = 2(h11 - h21)',
         'passed': chi == 2 * (h11 - h21)},
        {'id': 'vertex_count', 'label': 'Declared vertex count matches the matrix',
         'passed': raw['vertex_count'] == len(vertices)},
        {'id': 'distinct_vertices', 'label': 'Vertex rows are distinct',
         'passed': len({tuple(v) for v in vertices}) == len(vertices)},
        {'id': 'vertex_arrays_agree', 'label': 'Source vertex arrays agree',
         'passed': json_bytes(raw.get('vertex_matrix')) == json_bytes(vertices)},
        {'id': 'affine_rank', 'label': 'Affine rank is 4 (exact rational arithmetic)',
         'passed': rank == 4, 'observed': rank},
    ]
    digest = sha256(json_bytes(raw))
    return {
        'schema': SCHEMA,
        'record_id': f'ks-sample-{digest[:16]}',
        'dataset_id': 'kreuzer-skarke',
        'data_origin': 'sourced_ks_sidecar_not_ml_ranked',
        'geometry_status': 'representative',
        'formal_verification_status': 'not_performed',
        'hodge_numbers_recomputed': False,
        'reflexivity_recomputed': False,
        'geometry': {
            'h11': h11, 'h21': h21, 'euler_char': chi,
            'vertex_count': len(vertices), 'polytope_vertices': vertices,
            'vertex_convention': 'Unmodified source basis/polar convention; confirm before downstream import.',
        },
        'provenance': {
            'source_file': SOURCE_FILE,
            'source_file_sha256': source['sha256'],
            'source_json_pointer': f'/polytopes/{index}',
            'source_record_sha256': digest,
            'source_url': source['url'],
            'source_slice': raw.get('source_slice'),
            'reference': raw.get('reference'),
            'reference_url': raw.get('reference_url'),
        },
        'checks': {'kind': 'exact_script_checks_not_lean',
                   'passed': all(c['passed'] for c in checks), 'items': checks},
        'not_supplied': list(MISSING),
    }


def catalog_from_bytes(data):
    payload = json.loads(data)
    rows = payload.get('polytopes') if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not 1 <= len(rows) <= 500:
        raise ValueError('Expected a bounded, nonempty KS sample')
    source = {
        'file': SOURCE_FILE, 'sha256': sha256(data),
        'url': payload.get('source'), 'citation': payload.get('citation'),
    }
    records = [_record(row, index, source) for index, row in enumerate(rows)]
    if len({r['record_id'] for r in records}) != len(records):
        raise ValueError('Duplicate KS sample record IDs')
    records.sort(key=lambda r: (
        abs(r['geometry']['euler_char']), r['geometry']['h11'],
        r['geometry']['h21'], r['record_id'],
    ))
    return {
        'schema': SCHEMA, 'scope': SCOPE, 'source': source,
        'count': len(records),
        'small_euler_count': sum(abs(r['geometry']['euler_char']) < 100 for r in records),
        'default_order': 'abs_euler_ascending_then_h11_h21_id',
        'formal_verification_status': 'not_performed',
        'records': records,
    }


def load_catalog(root=ROOT):
    return catalog_from_bytes((Path(root) / SOURCE_FILE).read_bytes())


def build_batch(record_ids, source_sha256, root=ROOT):
    """Export exact selected records, refusing unknown/duplicate IDs or stale data."""
    if (not isinstance(record_ids, list) or not 1 <= len(record_ids) <= MAX_SELECTION
            or any(not isinstance(i, str) or len(i) > 80 for i in record_ids)
            or len(set(record_ids)) != len(record_ids)):
        raise ValueError('Select 1–50 distinct record IDs')
    if (not isinstance(source_sha256, str) or len(source_sha256) != 64
            or any(c not in '0123456789abcdef' for c in source_sha256)):
        raise ValueError('source_sha256 must be a lowercase SHA-256 digest')
    source_bytes = (Path(root) / SOURCE_FILE).read_bytes()
    catalog = catalog_from_bytes(source_bytes)
    if source_sha256 != catalog['source']['sha256']:
        raise StaleSource('The source sample changed. Reload the handoff page and select again.')
    indexed = {r['record_id']: r for r in catalog['records']}
    if any(i not in indexed for i in record_ids):
        raise ValueError('Selection contains an unknown KS sample record ID')
    selected = [indexed[i] for i in record_ids]
    if not all(r['checks']['passed'] for r in selected):
        raise ValueError('A selected record fails the exact data checks; inspect it before export')
    files = {
        'README.md': BATCH_README.encode('utf-8'),
        'source/ks_geometry_sample.json': source_bytes,
        'manifest.json': json_bytes({
            'schema': SCHEMA, 'scope': SCOPE, 'source': catalog['source'],
            'selection_order': 'client_selected_order_not_an_ml_ranking',
            'selected_record_ids': record_ids, 'selection_count': len(selected),
            'formal_verification_status': 'not_performed',
            'not_supplied': MISSING,
        }),
    }
    files.update({f"records/{r['record_id']}.json": json_bytes(r) for r in selected})
    files['SHA256SUMS'] = ''.join(
        f'{sha256(data)}  {name}\n' for name, data in sorted(files.items())
    ).encode('ascii')
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_STORED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(f'upg-ks-handoff/{name}', date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    return output.getvalue()
