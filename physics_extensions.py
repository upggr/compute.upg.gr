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
KS_SAMPLE_PATH = os.path.join('data', 'ks_geometry_sample.json')
_PACK_CACHE: Optional[Dict[str, Any]] = None
_KS_SAMPLE_CACHE: Optional[Dict[str, Any]] = None


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


def _load_ks_sample(*, force_reload: bool = False) -> Dict[str, Any]:
    global _KS_SAMPLE_CACHE
    if _KS_SAMPLE_CACHE is not None and not force_reload:
        return _KS_SAMPLE_CACHE
    try:
        with open(KS_SAMPLE_PATH, 'r', encoding='utf-8') as fh:
            _KS_SAMPLE_CACHE = json.load(fh)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        _KS_SAMPLE_CACHE = {'version': 0, 'polytopes': []}
    return _KS_SAMPLE_CACHE


def reload_geometry_pack() -> Dict[str, Any]:
    """Clear cache and reload geometry pack from disk (tests / hot reload)."""
    global _KS_SAMPLE_CACHE
    _KS_SAMPLE_CACHE = None
    _load_ks_sample(force_reload=True)
    return _load_pack(force_reload=True)


def lookup_ks_sample(
    dataset_id: str, h11: int, h21: int
) -> Optional[Dict[str, Any]]:
    """Return a labeled KS polytope representative from the sidecar sample."""
    if dataset_id not in ('kreuzer-skarke', 'info-density', 'heterotic'):
        return None
    sample = _load_ks_sample()
    for item in sample.get('polytopes') or []:
        if int(item.get('h11', -1)) != int(h11) or int(item.get('h21', -1)) != int(h21):
            continue
        return item
    return None


def lookup_geometry_pack(
    dataset_id: str, h11: int, h21: int, h31: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """Return curated geometry blob for a Hodge key, merged with KS sample vertices."""
    pack = _load_pack()
    found: Optional[Dict[str, Any]] = None
    for item in pack.get('geometries') or []:
        if (item.get('dataset_id') or 'kreuzer-skarke') != dataset_id:
            continue
        if int(item.get('h11', -1)) != int(h11) or int(item.get('h21', -1)) != int(h21):
            continue
        if h31 is not None and item.get('h31') is not None and int(item['h31']) != int(h31):
            continue
        found = dict(item)
        break

    ks = lookup_ks_sample(dataset_id, h11, h21)
    if ks:
        if found is None:
            found = dict(ks)
        else:
            # Prefer textbook ambient / equations; fill missing vertices from sample.
            for key in (
                'polytope_vertices',
                'vertex_matrix',
                'vertex_count',
                'facet_count',
                'point_count',
                'dual_point_count',
                'geometry_status',
                'uniqueness',
                'source_slice',
            ):
                if found.get(key) is None and ks.get(key) is not None:
                    found[key] = ks[key]
            if not found.get('polytope_vertices') and ks.get('polytope_vertices'):
                found['polytope_vertices'] = ks['polytope_vertices']
                found['vertex_matrix'] = ks.get('vertex_matrix') or ks['polytope_vertices']
            note_bits = [found.get('note'), ks.get('note')]
            found['note'] = ' '.join(n for n in note_bits if n)
            found.setdefault('geometry_status', ks.get('geometry_status', 'representative'))
            found.setdefault('uniqueness', ks.get('uniqueness'))
    return found


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


def quintic_orientifold_curated(h11: int, h21: int) -> Optional[Dict[str, Any]]:
    """Curated IIB O3/O7 sketch for the quintic / mirror-quintic class only."""
    if not ((h11, h21) in ((1, 101), (101, 1))):
        return None
    chi = 2 * (int(h11) - int(h21))
    L = abs(chi) / 24.0
    return {
        'status': 'curated_example',
        'applies_to': f'quintic class (h11,h21)=({h11},{h21})',
        'base_L': round(L, 6),
        'tex': (
            r'N_{D3}+\tfrac12 N_{\mathrm{flux}}'
            r'=\tfrac{\chi(X)}{24}+\tfrac{\chi(D_{O7})}{12}+Q_{O3}'
        ),
        'components': [
            {
                'id': 'chi_term',
                'label': 'χ(X)/24 for this Hodge class',
                'value': round(L, 6),
                'status': 'exact_for_class',
            },
            {
                'id': 'o7_schematic',
                'label': 'O7 divisor contribution (schematic)',
                'value': None,
                'status': 'literature_schematic',
                'detail': (
                    'Standard IIB O3/O7 setups on the quintic class introduce an '
                    'anti-holomorphic involution and O7 loci; χ(D_O7)/12 is not '
                    'fixed by Hodge numbers alone.'
                ),
            },
            {
                'id': 'o3_schematic',
                'label': 'O3 plane charge sum (schematic)',
                'value': None,
                'status': 'literature_schematic',
                'detail': (
                    'O3 charges depend on fixed-point loci of the involution. '
                    'Shown as an explicit unknown — not a vacuum census.'
                ),
            },
        ],
        'note': (
            'CURATED EXAMPLE for the quintic / mirror-quintic Hodge class only, '
            'following the schematic IIB O3–O7 tadpole used in the string-pheno '
            'literature. Not a general algorithm for arbitrary KS polytopes.'
        ),
        'references': [
            'IIB orientifold tadpole reviews (e.g. Blumenhagen–Cvetic–Lüst–Weigand)',
            'Candelas–Horowitz–Strominger–Witten quintic compactification',
        ],
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


def quintic_periods_literature(h11: int, h21: int) -> Optional[Dict[str, Any]]:
    """Candelas–de la Ossa–Green–Parkes style PF content for quintic / mirror.

    Numerical period values are NOT computed as physical outputs — we expose
    the published operator / special-point structure and a clearly labeled
    toy evaluation of the classical PF monodromy parameter.
    """
    if (h11, h21) not in ((1, 101), (101, 1)):
        return None
    # Mirror-family coordinate z; PF for the mirror quintic (CdOGP):
    # θ^4 ϖ − 5z (5θ+1)(5θ+2)(5θ+3)(5θ+4) ϖ = 0
    # with θ = z d/dz. Large-complex-structure / Fermat / conifold special points.
    z_toy = 1.0 / 5.0**5  # famous conifold locus z = 5^{-5} in standard normalization
    return {
        'status': 'literature_curated',
        'applies_to': f'quintic / mirror-quintic class (h11,h21)=({h11},{h21})',
        'picard_fuchs_order': 4 if h21 == 1 else 102,
        'mirror_family_pf_tex': (
            r'\theta^4\varpi - 5z(5\theta+1)(5\theta+2)(5\theta+3)(5\theta+4)\varpi = 0'
            r',\quad \theta=z\frac{d}{dz}'
        ),
        'special_points': [
            {
                'id': 'large_complex_structure',
                'z': 0,
                'note': 'LCS point; classical period expansion in z',
            },
            {
                'id': 'conifold',
                'z': '5^{-5}',
                'z_float_toy': z_toy,
                'note': (
                    'Standard conifold locus in the one-parameter mirror family '
                    '(toy float shown for the published closed form 5^{-5} only).'
                ),
            },
            {
                'id': 'fermat_gorris',
                'note': 'Fermat / Gepner-like orbifold point in the mirror family',
            },
        ],
        'yukawa_tex': r'Y_{zzz} \propto 1/(1-5^5 z) \quad\text{(mirror family; published)}',
        'honesty': (
            'Literature-backed formulas for the one-parameter mirror-quintic family '
            '(Candelas–de la Ossa–Green–Parkes). Not a numerical period engine for '
            'arbitrary KS polytopes. The float 5^{-5} is the known closed-form locus, '
            'not a fitted vacuum.'
        ),
        'references': [
            'Candelas, de la Ossa, Green, Parkes, Nucl. Phys. B359 (1991) 21',
            'Greene–Plesser mirror construction',
        ],
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


def toy_soft_parameter_card(
    A0: float = 0.0,
    m12: float = 500.0,
    tan_beta: float = 10.0,
    m0: float = 500.0,
) -> Dict[str, Any]:
    """Illustrative mSUGRA-style soft pattern from user inputs.

    NOT derived from any Calabi–Yau geometry on this site.
    """
    a0 = float(A0)
    m_half = float(m12)
    tb = float(tan_beta)
    m_0 = float(m0)
    # Purely algebraic illustrative relations — not RGE-evolved physical masses.
    return {
        'status': 'toy_illustrative',
        'honesty': (
            'Illustrative MSSM soft pattern from user-chosen A0, m1/2, tanβ, m0. '
            'NOT derived from this CY, NOT RGE-evolved, NOT physical GeV predictions.'
        ),
        'inputs': {
            'A0': a0,
            'm12': m_half,
            'tan_beta': tb,
            'm0': m_0,
            'units': 'user-chosen illustrative units (often called GeV in textbooks)',
        },
        'symbolic_outputs': [
            {'name': 'scalar_mass_proxy', 'tex': r'm_0', 'value': m_0},
            {'name': 'gaugino_mass_proxy', 'tex': r'm_{1/2}', 'value': m_half},
            {'name': 'A_term_proxy', 'tex': r'A_0', 'value': a0},
            {
                'name': 'higgsino_mu_proxy',
                'tex': r'\mu_{\mathrm{toy}}\sim m_{1/2}',
                'value': m_half,
            },
            {
                'name': 'Bmu_proxy',
                'tex': r'B\mu_{\mathrm{toy}}\sim m_{1/2}^2',
                'value': m_half * m_half,
            },
            {
                'name': 'tan_beta',
                'tex': r'\tan\beta',
                'value': tb,
            },
        ],
        'yukawa_texture_placeholder': {
            'status': 'generation_indices_only',
            'matrix_shape': [3, 3],
            'note': (
                'Yukawa texture shown as generation-index placeholders Y_{ij}; '
                'no numerical SM Yukawas are claimed.'
            ),
            'Y_u_tex': r'Y^u_{ij},\ i,j=1,2,3',
            'Y_d_tex': r'Y^d_{ij},\ i,j=1,2,3',
            'Y_e_tex': r'Y^e_{ij},\ i,j=1,2,3',
        },
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
        'geometry_status': 'geometry_status',
        'uniqueness': 'geometry_uniqueness',
        'vertex_count': 'vertex_count',
        'facet_count': 'facet_count',
        'point_count': 'point_count',
        'dual_point_count': 'dual_point_count',
        'source_slice': 'ks_source_slice',
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
    if out.get('polytope_vertices') or out.get('vertex_matrix'):
        out.setdefault('geometry_status', pack.get('geometry_status') or 'representative')
    else:
        out.setdefault('geometry_status', 'pending')
    return out


def cytools_candidate_fields(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Fields for cytools-candidates-v1 export adapters."""
    raw = raw or {}
    vertices = raw.get('polytope_vertices') or raw.get('vertex_matrix')
    status = raw.get('geometry_status')
    if vertices and not status:
        status = 'representative'
    if not vertices:
        status = status or 'pending'
    payload = {
        'h11': raw.get('h11'),
        'h21': raw.get('h21'),
        'h31': raw.get('h31'),
        'euler_char': raw.get('euler_char'),
        'geometry_status': status,
    }
    if vertices:
        payload['polytope_vertices'] = vertices
        payload['vertex_matrix'] = raw.get('vertex_matrix') or vertices
        payload['uniqueness'] = raw.get('geometry_uniqueness') or (
            'one polytope with these Hodge numbers; not unique'
        )
    else:
        payload['note'] = (
            'Hodge-only export; attach a KS polytope vertex matrix + triangulation '
            'before CYTools toric constructions.'
        )
    return payload