"""Deeper analysis layers that finish the remaining dossier gaps.

These are still honest: combinatorial / symbolic / curated-geometry results,
never invented soft spectra or unique polytopes for arbitrary Hodge pairs.
"""

from __future__ import annotations

import itertools
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

GEOMETRY_PACK_PATH = os.path.join('data', 'geometry_pack.json')
_LEGACY_GEOMETRY_PACK = os.path.join('static', 'data', 'geometry_pack.json')
_PACK_CACHE: Optional[Dict[str, Any]] = None


def _load_pack(path: Optional[str] = None, *, force_reload: bool = False) -> Dict[str, Any]:
    global _PACK_CACHE
    if _PACK_CACHE is not None and path is None and not force_reload:
        return _PACK_CACHE
    for candidate in ([path] if path else []) + [GEOMETRY_PACK_PATH, _LEGACY_GEOMETRY_PACK]:
        if not candidate:
            continue
        try:
            with open(candidate, 'r', encoding='utf-8') as fh:
                payload = json.load(fh)
            if path is None:
                _PACK_CACHE = payload
            return payload
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    empty = {'version': 0, 'geometries': []}
    if path is None:
        _PACK_CACHE = empty
    return empty


def reload_geometry_pack() -> Dict[str, Any]:
    """Clear cache and reload geometry pack from disk (tests / hot reload)."""
    return _load_pack(force_reload=True)

def lookup_geometry_pack(
    dataset_id: str, h11: int, h21: int, h31: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """Return curated geometry blob for a Hodge key, if present."""
    pack = _load_pack()
    for item in pack.get('geometries') or []:
        if (item.get('dataset_id') or 'kreuzer-skarke') != dataset_id:
            continue
        if int(item.get('h11', -1)) != int(h11) or int(item.get('h21', -1)) != int(h21):
            continue
        if h31 is not None and item.get('h31') is not None and int(item['h31']) != int(h31):
            continue
        return item
    return None


def flux_lattice_miniscan(h21: int, tadpole_L: float, *, max_k: int = 5) -> Dict[str, Any]:
    """Enumerate a toy integer flux lattice for small K.

    Model: K = h²¹+1 flux integers (n_1..n_K) with ∑ n_i² ≤ floor(2L).
    This is a pedagogical lattice toy — not the physical H³ flux lattice.
    """
    K = int(h21) + 1
    L = max(float(tadpole_L), 0.0)
    budget = int(math.floor(2 * L))
    if K > max_k:
        return {
            'status': 'skipped',
            'reason': f'K={K} too large for in-process enumeration (max_k={max_k})',
            'K': K,
            'budget_sum_sq': budget,
            'counted': None,
            'sample': [],
            'honesty': 'toy_lattice',
            'tex': r'\sum_{i=1}^{K} n_i^2 \le \lfloor 2L\rfloor',
        }

    # Bound each |n_i| by sqrt(budget)
    lim = int(math.floor(math.sqrt(budget))) if budget >= 0 else 0
    counted = 0
    samples: List[List[int]] = []
    # Include the zero flux and scan orthant-symmetric representatives.
    ranges = [range(-lim, lim + 1) for _ in range(K)]
    for tup in itertools.product(*ranges):
        s2 = sum(v * v for v in tup)
        if s2 <= budget:
            counted += 1
            if len(samples) < 12:
                samples.append(list(tup))

    return {
        'status': 'computed',
        'K': K,
        'budget_sum_sq': budget,
        'L': round(L, 6),
        'counted': counted,
        'sample': samples,
        'honesty': 'toy_lattice',
        'note': (
            'Toy integer lattice with ∑n² ≤ ⌊2L⌋. Physical IIB fluxes live on '
            'H³(X,ℤ) with tadpole and Dirac quantization — this is a size check, '
            'not a vacuum scan over periods.'
        ),
        'tex': r'\#\{n\in\mathbb{Z}^K:\|n\|_2^2\le\lfloor 2L\rfloor\}',
    }


def orientifold_tadpole_sketch(euler: int) -> Dict[str, Any]:
    """Full IIB / O3–O7 tadpole identity with unknown charges marked."""
    L = abs(int(euler)) / 24.0
    return {
        'status': 'partial',
        'base_L': round(L, 6),
        'tex': (
            r'N_{D3} + \tfrac{1}{2}N_{\mathrm{flux}}'
            r'= \tfrac{\chi(X)}{24} + \tfrac{\chi(D_{O7})}{12} + \cdots'
        ),
        'components': [
            {
                'id': 'chi_term',
                'label': 'Geometric tadpole χ(X)/24',
                'value': round(L, 6),
                'status': 'exact',
            },
            {
                'id': 'o7',
                'label': 'O7 contribution χ(D_O7)/12',
                'value': None,
                'status': 'needs_involution',
            },
            {
                'id': 'o3',
                'label': 'O3 plane charges',
                'value': None,
                'status': 'needs_involution',
            },
            {
                'id': 'mobile_d3',
                'label': 'Mobile D3 count N_D3',
                'value': None,
                'status': 'free_after_fluxes',
            },
        ],
        'note': (
            'Orientifold involution and O-plane loci are not stored. '
            'We expose the exact χ/24 piece and leave O3/O7 as explicit unknowns.'
        ),
    }


def periods_structure_full(h21: int) -> Dict[str, Any]:
    """Symbolic period / Picard–Fuchs structure from Hodge data."""
    K = int(h21) + 1
    b3 = 2 * K
    period_components = [f'\\varpi_{i}' for i in range(K)]
    return {
        'picard_fuchs_order': K,
        'b3': b3,
        'period_vector_tex': r'\Pi = (' + ','.join(period_components) + r')',
        'pf_operator_tex': r'\mathcal{L}_{\mathrm{PF}}(\theta)\,\\varpi_0 = 0'
        r'\quad(\deg\mathcal{L}=h^{2,1}+1)',
        'yukawa_from_periods_tex': (
            r'C_{ijk} = \int_X \Omega\wedge\partial_i\partial_j\partial_k\Omega'
        ),
        'prepotential_tex': (
            r'F = \tfrac12 Y^I\\mathcal{F}_I(Y)'
            r'\quad\text{(special geometry; needs periods)}'
        ),
        'status': 'symbolic',
        'note': (
            'Period integrals are not evaluated numerically here. '
            'We give the exact dimensions and the standard special-geometry identities.'
        ),
    }


def soft_terms_symbolic() -> Dict[str, Any]:
    """Gravity-mediation soft-term skeleton (symbolic only)."""
    return {
        'status': 'symbolic',
        'mediation': 'gravity / modulus mediation sketch',
        'terms': [
            {
                'name': 'Universal scalar mass',
                'tex': r'm_0^2 \sim m_{3/2}^2',
                'status': 'symbolic',
            },
            {
                'name': 'Gaugino mass',
                'tex': r'M_{1/2} \sim m_{3/2}',
                'status': 'symbolic',
            },
            {
                'name': 'A-terms',
                'tex': r'A_0 \sim m_{3/2}',
                'status': 'symbolic',
            },
            {
                'name': 'μ / Bμ problem',
                'tex': r'\mu\sim m_{3/2},\quad B\mu\sim m_{3/2}^2',
                'status': 'symbolic',
            },
            {
                'name': 'Gravitino mass',
                'tex': r'm_{3/2} = e^{K/2}|W|',
                'status': 'needs_moduli_vevs',
            },
        ],
        'note': (
            'No soft spectrum is computed: that needs stabilized moduli vevs, '
            'gauge kinetic functions, and a mediation model. Formulas are shown '
            'so the missing inputs are explicit.'
        ),
    }


def yukawa_structure(h11: int, h21: int, dataset_id: str) -> Dict[str, Any]:
    """Combinatorial Yukawa / coupling counting proxies."""
    h11_i, h21_i = int(h11), int(h21)
    if dataset_id == 'heterotic':
        return {
            'status': 'combinatorial',
            'tex': r'Y_{abc}\sim\int H^1(V)^\otimes3',
            'counts': {
                'complex_structure_moduli': h21_i,
                'kahler_moduli': h11_i,
                'bundle_moduli_unknown': True,
            },
            'note': (
                'Heterotic Yukawas need bundle cohomology H¹(X,V) etc. '
                'Only Hodge dimensions of the base are known here.'
            ),
        }
    # IIB / closed-string style: triple intersections on Kähler side
    trip = (h11_i * (h11_i + 1) * (h11_i + 2)) // 6
    return {
        'status': 'combinatorial',
        'tex': r'C_{ijk}=\int_X J_i\wedge J_j\wedge J_k',
        'counts': {
            'kahler_moduli': h11_i,
            'symmetric_triple_unknowns': trip,
            'complex_structure_yukawas_need_periods': True,
        },
        'note': (
            'Closed-string triple intersections need the Kähler ring; '
            'open-string/MSSM Yukawas need D-brane/bundle data.'
        ),
    }


def gauge_embedding_sketch(dataset_id: str, euler: int) -> Dict[str, Any]:
    """Gauge / SM embedding checklist."""
    n_gen = abs(int(euler)) // 2
    if dataset_id == 'heterotic':
        steps = [
            {
                'label': 'E₈×E₈ or SO(32) heterotic',
                'status': 'assumed_framework',
                'tex': r'E_8\times E_8',
            },
            {
                'label': 'Commutant / structure group',
                'status': 'needs_bundle',
                'tex': r'H\subset E_8,\ G=[E_8,H]',
            },
            {
                'label': 'Net chirality',
                'status': 'exact_index',
                'tex': rf'n_{{\mathrm{{gen}}}}=|\chi|/2={n_gen}',
            },
            {
                'label': 'SM gauge factors SU(3)×SU(2)×U(1)',
                'status': 'needs_bundle',
                'tex': r'SU(3)_C\times SU(2)_L\times U(1)_Y\subset G',
            },
        ]
    else:
        steps = [
            {
                'label': 'IIB / F-theory gauge from 7-branes',
                'status': 'needs_geometry',
                'tex': r'G\subset E_8\ \text{(local F-theory)}',
            },
            {
                'label': 'Intersecting brane / quiver SM',
                'status': 'needs_geometry',
                'tex': r'[D_a]\cdot[D_b]\cdot[D_c]',
            },
            {
                'label': 'Generation index from topology',
                'status': 'exact_index',
                'tex': r'|\chi|/2=' + str(n_gen) + r'\ \text{(heterotic-like index; IIB uses intersections)}',
            },
        ]
    return {
        'status': 'checklist',
        'n_generations': n_gen,
        'steps': steps,
        'note': 'Gauge embeddings are not unique at the Hodge level; this is a roadmap.',
    }


def is_real_configuration_matrix(value: Any) -> bool:
    """True only for list/tuple matrices — never placeholder strings."""
    if not isinstance(value, (list, tuple)) or not value:
        return False
    for row in value:
        if isinstance(row, (list, tuple)):
            if not row:
                return False
            if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in row):
                return False
        elif not (isinstance(row, (int, float)) and not isinstance(row, bool)):
            return False
    return True


def merge_geometry_into_raw(
    raw: Optional[Dict[str, Any]],
    pack: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Merge curated geometry pack fields into raw without clobbering existing."""
    out = dict(raw or {})
    if not pack:
        return out
    mapping = {
        'ambient': 'ambient',
        'weight_system': 'weight_system',
        'hypersurface_equation': 'hypersurface_equation',
        'favourable': 'favourable',
        'polytope_vertices': 'polytope_vertices',
        'vertex_matrix': 'vertex_matrix',
        'triangulation': 'triangulation',
        'triangulation_id': 'triangulation_id',
        'polytope_id': 'polytope_id',
        'configuration_matrix': 'configuration_matrix',
    }
    for src, dst in mapping.items():
        val = pack.get(src)
        if val is None or out.get(dst) is not None:
            continue
        if src == 'configuration_matrix' and not is_real_configuration_matrix(val):
            continue
        out[dst] = val
    if pack.get('name') and 'geometry_name' not in out:
        out['geometry_name'] = pack['name']
    if pack.get('note') and 'geometry_note' not in out:
        out['geometry_note'] = pack['note']
    return out
