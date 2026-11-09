#!/usr/bin/env python3
"""Exact arithmetic checks for a quintic handoff, NOT a Lean/geometry verifier.

Standalone, standard library only. Exit 0: checks pass; 1: a mathematical
check fails; 2: malformed/unsupported input. Never imports the Flask app.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations, permutations
from pathlib import Path

SCHEMA = 'upg-quintic-lean-pilot-v1'
EQUATION = 'x0^5 + x1^5 + x2^5 + x3^5 + x4^5 = 0'
BASIS = 'H = c1(O_P4(1)) restricted to X'
CONES = [list(c) for c in combinations(range(5), 4)]
METADATA = {
    'schema': SCHEMA,
    'example_id': 'fermat-quintic-p4',
    'data_origin': 'curated_textbook_not_ml_ranked',
    'geometry_status': 'representative',
    'formal_verification_status': 'not_performed',
}
PROVENANCE_KEYS = {
    'source_repository', 'source_file', 'source_json_pointer',
    'source_file_sha256', 'source_record_file', 'source_record_sha256',
    'reference', 'reference_url',
}


def _object(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f'{name}: missing or unexpected fields')


def _integers(values, length, name):
    # bool is an int subclass in Python; floats such as 1.0 are not accepted.
    if (not isinstance(values, list) or len(values) != length
            or any(type(v) is not int for v in values)):
        raise ValueError(f'{name}: expected {length} exact integers')


def validate_record(record):
    """Validate this fixed pilot format, not a general geometry schema."""
    _object(record, set(METADATA) | {'geometry', 'provenance', 'not_supplied'}, 'record')
    if any(record[k] != v for k, v in METADATA.items()):
        raise ValueError('Unsupported pilot metadata or certification claim')
    g = record['geometry']
    _object(g, {
        'complex_dimension', 'ambient', 'equation', 'degree', 'weights',
        'h11', 'h21', 'euler_char', 'intersection_basis',
        'triple_intersection_H3', 'ambient_fan',
    }, 'geometry')
    if (g['complex_dimension'] != 3 or g['ambient'] != 'P^4 over C'
            or g['equation'] != EQUATION or g['intersection_basis'] != BASIS):
        raise ValueError('Only the declared Fermat quintic / hyperplane basis is supported')
    for key in ('complex_dimension', 'degree', 'h11', 'h21', 'euler_char', 'triple_intersection_H3'):
        _integers([g[key]], 1, key)
    if g['degree'] != 5 or g['h11'] < 1 or g['h21'] < 0:
        raise ValueError('Expected degree 5, h11 >= 1, and h21 >= 0')
    _integers(g['weights'], 5, 'weights')
    if g['weights'] != [1] * 5:
        raise ValueError('Only ordinary projective P^4 weights are supported')

    fan = g['ambient_fan']
    _object(fan, {'lattice', 'ray_matrix', 'maximal_cones'}, 'ambient_fan')
    if fan['lattice'] != 'N = Z^4':
        raise ValueError('Expected ambient fan rays in N = Z^4')
    rays = fan['ray_matrix']
    if not isinstance(rays, list) or len(rays) != 5:
        raise ValueError('Expected five ambient fan rays')
    for row in rays:
        _integers(row, 4, 'ray_matrix row')
    cones = fan['maximal_cones']
    if not isinstance(cones, list) or len(cones) != 5:
        raise ValueError('Expected five maximal cones')
    for cone in cones:
        _integers(cone, 4, 'maximal_cone')
    if cones != CONES:
        raise ValueError('Expected all four-ray subsets in lexicographic order (zero-based indices)')

    p = record['provenance']
    _object(p, PROVENANCE_KEYS, 'provenance')
    if any(not isinstance(v, str) or not v for v in p.values()):
        raise ValueError('Provenance fields must be nonempty strings')
    for key in ('source_file_sha256', 'source_record_sha256'):
        value = p[key]
        if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError(f'{key}: expected a lowercase SHA-256 digest')
    omitted = record['not_supplied']
    if (not isinstance(omitted, list) or not omitted
            or any(not isinstance(v, str) or not v for v in omitted)
            or len(set(omitted)) != len(omitted)):
        raise ValueError('not_supplied must list distinct, nonempty strings')


def determinant(matrix):
    """Small square integer determinant by the Leibniz formula; no floats."""
    total = 0
    for perm in permutations(range(len(matrix))):
        inversions = sum(perm[i] > perm[j] for i in range(len(perm)) for j in range(i + 1, len(perm)))
        term = (-1) ** inversions
        for i, j in enumerate(perm):
            term *= matrix[i][j]
        total += term
    return total


def check_record(record):
    validate_record(record)
    g = record['geometry']
    fan = g['ambient_fan']
    rays = fan['ray_matrix']
    dets = [determinant([rays[i] for i in cone]) for cone in fan['maximal_cones']]
    equalities = [
        ('euler_identity', 'chi = 2 * (h11 - h21)', g['euler_char'], 2 * (g['h11'] - g['h21'])),
        ('anticanonical_degree', 'degree = sum(weights)', g['degree'], sum(g['weights'])),
        ('hyperplane_intersection', 'H^3 on a degree-d hypersurface in P^4 equals d',
         g['triple_intersection_H3'], g['degree']),
        ('fan_ray_sum', 'The five declared ambient fan rays sum to zero',
         [sum(row[j] for row in rays) for j in range(4)], [0] * 4),
    ]
    checks = [
        {'id': key, 'statement': statement, 'observed': lhs, 'expected': rhs, 'passed': lhs == rhs}
        for key, statement, lhs, rhs in equalities
    ]
    checks.append({
        'id': 'ambient_cone_determinants',
        'statement': 'Each declared maximal ambient cone has determinant +1 or -1',
        'determinants': dets,
        'passed': all(abs(d) == 1 for d in dets),
    })
    return {
        'check_kind': 'exact_integer_script_checks',
        'formal_verification_status': 'not_performed',
        'scope': 'Arithmetic on declared inputs; not a derivation of Hodge numbers or a proof of physical viability.',
        'passed': all(c['passed'] for c in checks),
        'checks': checks,
    }


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON field: {key}')
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f'Non-finite JSON value: {value}')


def check_bytes(data):
    record = json.loads(data, object_pairs_hook=_unique_keys, parse_constant=_reject_constant)
    return {'input_sha256': hashlib.sha256(data).hexdigest(), **check_record(record)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='quintic.json or negative_euler.json')
    args = parser.parse_args()
    try:
        report = check_bytes(args.input.read_bytes())
    except (OSError, ValueError) as exc:
        print(json.dumps({'passed': False, 'error': str(exc), 'formal_verification_status': 'not_performed'}))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
