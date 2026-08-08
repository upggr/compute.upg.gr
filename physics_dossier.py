"""Physics dossier for a Calabi–Yau candidate (shareable certificate).

Computes identities, tadpole/flux proxies, and necessary-condition checks from
Hodge numbers alone. These are *proxies and identities*, not a claim that the
geometry is phenomenologically viable — Hodge data do not uniquely fix a
manifold.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import geometry_store
import math_specimen
import model_cards
import model_exclusions
import physics_extensions


def _f(x) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _i(x) -> Optional[int]:
    v = _f(x)
    return int(v) if v is not None else None


def derive_euler(dataset_id: str, h11: int, h21: int, h31: Optional[int] = None) -> int:
    if dataset_id == 'cy5-folds' and h31 is not None:
        return int(6 + 6 * (h11 - h21 + h31))
    return int(2 * (h11 - h21))


def physics_scalars(h11: int, h21: int, euler: Optional[int] = None) -> Dict[str, float]:
    """Scalar proxies shared with the information-density ranking story."""
    if euler is None:
        euler = 2 * (h11 - h21)
    euler_abs = abs(euler)
    h_total = h11 + h21 + 1e-10
    p11 = h11 / h_total
    p21 = h21 / h_total
    hodge_entropy = -(p11 * math.log(p11 + 1e-10) + p21 * math.log(p21 + 1e-10))
    hodge_entropy_norm = hodge_entropy / math.log(2)
    topo_efficiency = euler_abs / h_total
    moduli_compactness = 1.0 / (1.0 + math.log1p(h_total))
    hodge_balance = 1.0 - abs(h11 - h21) / h_total

    K = h21 + 1
    L = max(euler_abs / 24.0, 1.0)
    log_flux = 2 * K * math.log(2 * math.pi * L + 1e-10) - (K * math.log(K + 1e-10) - K)
    flux_density = 1.0 / (1.0 + math.exp(-log_flux / 100.0))

    tadpole_headroom = max(1.0 - euler_abs / (24 * h11 + 1e-10), 0.0)
    stabilization_ratio = min(h11, h21) / (max(h11, h21) + 1e-10)
    moduli_penalty = math.exp(-h_total / 200.0)
    vacuum_stability = (
        0.4 * tadpole_headroom + 0.4 * stabilization_ratio + 0.2 * moduli_penalty
    )

    return {
        'euler_char': float(euler),
        'euler_abs': float(euler_abs),
        'tadpole_L': round(euler_abs / 24.0, 4),
        'total_moduli': float(h11 + h21),
        'hodge_entropy': round(hodge_entropy_norm, 4),
        'topo_efficiency': round(topo_efficiency, 4),
        'moduli_compactness': round(moduli_compactness, 4),
        'hodge_balance': round(hodge_balance, 4),
        'flux_density': round(flux_density, 4),
        'vacuum_stability': round(vacuum_stability, 4),
        'tadpole_headroom': round(tadpole_headroom, 4),
        'mirror_h11': float(h21),
        'mirror_h21': float(h11),
        'mirror_euler': float(-euler),
    }


def dataset_target_check(dataset_id: str, h11: int, h21: int, euler: int) -> Dict[str, Any]:
    """Necessary condition used as the search target for that dataset."""
    if dataset_id == 'cy5-folds':
        ok = h11 > 100
        rule = 'h^{1,1} > 100'
        rule_tex = r'h^{1,1} > 100'
        detail = f'h¹¹={h11}'
    elif dataset_id == 'heterotic':
        total = h11 + h21 + 1e-10
        balance = 1.0 - abs(h11 - h21) / total
        ok = balance >= 0.9
        rule = 'h^{1,1} ≈ h^{2,1} (balance ≥ 0.9)'
        rule_tex = r'1 - \dfrac{|h^{1,1}-h^{2,1}|}{h^{1,1}+h^{2,1}} \ge 0.9'
        detail = f'balance={balance:.3f}'
    elif dataset_id == 'info-density':
        ok = False
        rule = 'top decile by information-density composite (run-dependent)'
        rule_tex = r'\text{info-density percentile (run-dependent)}'
        detail = (
            'Cannot certify from Hodge numbers alone — percentile requires a '
            'full ranking run. Proxies (entropy, compactness, flux density) '
            'are shown for guidance only.'
        )
    elif dataset_id == 'f-theory-elliptic':
        ok = h11 <= 10 or (h11, h21) in ((2, 272), (272, 2), (19, 19))
        rule = 'elliptic-friendly proxy (h¹¹≤10 or literature elliptic seed)'
        rule_tex = r'h^{1,1}\le 10\ \text{or literature elliptic seed}'
        detail = f'h¹¹={h11}, h²¹={h21} (proxy — not a Weierstrass model DB)'
    else:
        ok = abs(euler) < 100
        rule = '|χ| < 100'
        rule_tex = r'|\chi| < 100'
        detail = f'|χ|={abs(euler)}'
    return {
        'id': 'dataset_target',
        'label': 'Dataset target rule',
        'rule': rule,
        'rule_tex': rule_tex,
        'ok': ok,
        'detail': detail,
    }


def build_dossier(
    dataset_id: str,
    h11: Optional[int],
    h21: Optional[int],
    h31: Optional[int] = None,
    euler_char: Optional[int] = None,
    verified_target: Optional[bool] = None,
) -> Dict[str, Any]:
    h11_i, h21_i = _i(h11), _i(h21)
    if h11_i is None or h21_i is None:
        return {
            'ok': False,
            'error': 'Need integer h11 and h21 to build a dossier.',
        }

    h31_i = _i(h31)
    derived_euler = derive_euler(dataset_id or 'kreuzer-skarke', h11_i, h21_i, h31_i)
    supplied = _i(euler_char)
    euler_consistent = supplied is None or supplied == derived_euler
    euler = derived_euler

    scalars = physics_scalars(h11_i, h21_i, euler)
    checks: List[Dict[str, Any]] = [
        {
            'id': 'euler_identity',
            'label': 'Euler identity',
            'rule': (
                'χ = 6 + 6(h¹¹ − h²¹ + h³¹)' if dataset_id == 'cy5-folds'
                else 'χ = 2(h¹¹ − h²¹)'
            ),
            'rule_tex': (
                r'\chi = 6 + 6(h^{1,1} - h^{2,1} + h^{3,1})'
                if dataset_id == 'cy5-folds'
                else r'\chi = 2(h^{1,1} - h^{2,1})'
            ),
            'ok': euler_consistent,
            'detail': (
                f'derived χ={derived_euler}'
                + ('' if supplied is None else f', supplied χ={supplied}')
            ),
        },
        dataset_target_check(dataset_id or 'kreuzer-skarke', h11_i, h21_i, euler),
        {
            'id': 'tadpole_positive',
            'label': 'Tadpole charge defined',
            'rule': 'L = |χ|/24 ≥ 0',
            'rule_tex': r'L = |\chi|/24 \ge 0',
            'ok': scalars['tadpole_L'] >= 0,
            'detail': f'L={scalars["tadpole_L"]}',
        },
        {
            'id': 'moduli_positive',
            'label': 'Positive Hodge numbers',
            'rule': 'h¹¹ ≥ 1 and h²¹ ≥ 1',
            'rule_tex': r'h^{1,1} \ge 1,\quad h^{2,1} \ge 1',
            'ok': h11_i >= 1 and h21_i >= 1,
            'detail': f'h¹¹={h11_i}, h²¹={h21_i}',
        },
    ]
    if verified_target is not None:
        checks.append({
            'id': 'verified_flag',
            'label': 'Search verification flag',
            'rule': 'verified_target from ranking run',
            'ok': bool(verified_target),
            'detail': 'yes' if verified_target else 'no',
        })

    diamond = {
        'h11': h11_i,
        'h21': h21_i,
        'h31': h31_i,
        'euler': euler,
        'kind': 'cy5' if dataset_id == 'cy5-folds' else 'cy3',
        # Simplified CY3 Hodge diamond corners commonly quoted in intros:
        # h^{0,0}=1, h^{3,0}=1, h^{1,1}, h^{2,1}, and symmetries.
        'h00': 1,
        'h30': 1,
        'h03': 1,
        'h33': 1,
        'note': (
            'CY5: showing h¹¹ / h²¹ / h³¹ only — full fivefold Hodge diamond '
            'needs more Hodge numbers than we store.'
            if dataset_id == 'cy5-folds' else
            'Simplified CY3 Hodge diamond (corners + h¹¹ / h²¹).'
        ),
    }

    if dataset_id == 'cy5-folds':
        identities = [
            {
                'name': 'Euler characteristic (CY5)',
                'tex': r'\chi = 6 + 6(h^{1,1} - h^{2,1} + h^{3,1})',
                'value': euler,
            },
            {
                'name': 'D3 tadpole charge (formal)',
                'tex': r'L = |\chi|/24',
                'value': scalars['tadpole_L'],
            },
        ]
    else:
        identities = [
            {
                'name': 'Euler characteristic (CY3)',
                'tex': r'\chi = 2(h^{1,1} - h^{2,1})',
                'value': euler,
            },
            {
                'name': 'D3 tadpole charge',
                'tex': r'L = |\chi|/24',
                'value': scalars['tadpole_L'],
            },
            {
                'name': 'Mirror map (Hodge)',
                'tex': r'(h^{1,1}, h^{2,1}) \mapsto (h^{2,1}, h^{1,1})',
                'value': f"({int(scalars['mirror_h11'])}, {int(scalars['mirror_h21'])})",
            },
            {
                'name': 'Flux vacua proxy (log-density)',
                'tex': r'\log N_{\mathrm{flux}} \sim 2K\log(2\pi L) - \log K!\quad(K=h^{2,1}+1)',
                'value': scalars['flux_density'],
            },
        ]

    return {
        'ok': True,
        'dataset_id': dataset_id,
        'h11': h11_i,
        'h21': h21_i,
        'h31': h31_i,
        'euler_char': euler,
        'euler_consistent': euler_consistent,
        'diamond': diamond,
        'scalars': scalars,
        'checks': checks,
        'identities': identities,
        'caveat': (
            'Hodge numbers do not uniquely determine a Calabi–Yau. Many distinct '
            'polytopes/triangulations share (h¹¹, h²¹, χ). This page is a '
            'topological certificate for the invariants, not a uniqueness proof.'
        ),
    }


def neighbor_distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _raw_keys(raw: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(raw, dict):
        return []
    return sorted(str(k) for k in raw.keys())


def _fmt_sci(x: float) -> str:
    """Scientific notation string safe for huge flux-vacua estimates."""
    if x <= 0 or not math.isfinite(x):
        return '0'
    exp = int(math.floor(math.log10(x)))
    mant = x / (10 ** exp)
    return f'{mant:.3f}e{exp:+d}'


def flux_vacua_estimate(h21: int, tadpole_L: float) -> Dict[str, Any]:
    """Bousso–Polchinski / Denef–Douglas-style log-density estimate.

    Exact inputs: K = h²¹+1 (flux superpotential monomials / CS periods count
    proxy) and L = |χ|/24. The vacua *count* is an asymptotic proxy, not a scan.
    """
    K = int(h21) + 1
    L = max(float(tadpole_L), 1e-12)
    if K <= 0:
        return {
            'K': K,
            'L': round(L, 6),
            'log_N_flux': None,
            'N_flux_est': None,
            'N_flux_est_sci': None,
            'stirling_note': 'K must be positive',
        }
    # Stirling: log K! ≈ K log K − K + ½ log(2πK)
    log_k_fact = K * math.log(K) - K + 0.5 * math.log(2 * math.pi * K)
    log_N = 2 * K * math.log(2 * math.pi * L) - log_k_fact
    # Avoid overflow in exp; keep sci string from log10.
    log10_N = log_N / math.log(10.0)
    if log10_N > 300:
        n_sci = f'10^{log10_N:.2f}'
        n_est = None
    else:
        n_est = math.exp(log_N)
        n_sci = _fmt_sci(n_est)
    return {
        'K': K,
        'L': round(L, 6),
        'log_N_flux': round(log_N, 4),
        'log10_N_flux': round(log10_N, 4),
        'N_flux_est': n_est,
        'N_flux_est_sci': n_sci,
        'stirling_note': (
            r'log N ∼ 2K log(2πL) − log K!  (Stirling); asymptotic landscape proxy'
        ),
    }


def second_chern_proxy(h11: int, h21: int) -> Dict[str, Any]:
    """Common KS-style linear proxy for c₂·J scale (not an intersection number)."""
    c2 = 12 * h11 + 6 * h21
    return {
        'c2_J_proxy': int(c2),
        'formula': '12 h¹¹ + 6 h²¹',
        'honesty': 'proxy',
        'note': (
            'Linear stand-in used in ranking features; a real c₂·J needs the '
            'resolved toric geometry and a chosen Kähler class.'
        ),
    }


def period_structure(h21: int) -> Dict[str, Any]:
    """Combinatorial sizes of the period / Picard–Fuchs problem from h²¹."""
    order = int(h21) + 1
    return {
        'picard_fuchs_order': order,
        'period_vector_dim': order,
        'formula': 'K = h²¹ + 1',
        'note': (
            'The GVW flux superpotential is built from periods; the number of '
            'independent periods is h³(X)=2(h²¹+1) for a CY3, often summarized '
            'via K=h²¹+1 for the flux monomial count proxy used above.'
        ),
        'h3_betti': 2 * (int(h21) + 1),
    }


def intersection_proxies(h11: int) -> Dict[str, Any]:
    """Combinatorial sizes related to the Kähler / intersection ring."""
    h = int(h11)
    return {
        'kahler_cone_dim': h,
        'triple_intersection_tensor_entries': h ** 3,
        'symmetric_triple_independent': (h * (h + 1) * (h + 2)) // 6,
        'note': (
            'Counts of unknown intersection numbers if the ring were fully generic. '
            'Real CY3 intersection rings are highly constrained; these are size '
            'estimates for the computational problem, not computed κ_ijk.'
        ),
    }


def external_geometry_links(
    dataset_id: str, h11: int, h21: int, h31: Optional[int] = None
) -> List[Dict[str, str]]:
    """Deep-links to public catalogs (lookup aids — not unique geometry IDs)."""
    links: List[Dict[str, str]] = []
    if dataset_id in ('kreuzer-skarke', 'info-density', 'heterotic'):
        links.append({
            'label': 'Kreuzer–Skarke CY data (TU Wien)',
            'url': 'http://hep.itp.tuwien.ac.at/~kreuzer/CY/',
            'detail': f'Search polytopes with h11={h11}, h12={h21} in the online forms / dumps.',
        })
        links.append({
            'label': 'KS list paper (arXiv)',
            'url': 'https://arxiv.org/abs/hep-th/0002240',
            'detail': 'Original reflexive polytope enumeration.',
        })
    if dataset_id == 'cy5-folds':
        links.append({
            'label': 'CI5F paper (arXiv:2310.15966)',
            'url': 'https://arxiv.org/abs/2310.15966',
            'detail': (
                f'Constructive CI5F census (≤4 projective factors). Featured '
                f'(h¹¹,h²¹,h³¹)=({h11},{h21},{h31}) is outside that h¹¹ range.'
            ),
        })
        links.append({
            'label': 'CI5F Dropbox dataset (cicy5f.json)',
            'url': (
                'https://www.dropbox.com/scl/fo/z7ii5idt6qxu36e0b8azq/h'
                '?rlkey=0qfhx3tykytduobpld510gsfy&dl=0'
            ),
            'detail': '27 068 configuration matrices + partial Hodge data from the paper.',
        })
        links.append({
            'label': 'cymetric / CI5F resources',
            'url': 'https://github.com/pythoncymetric/cymetric',
            'detail': (
                f'Look for configurations compatible with '
                f'(h¹¹,h²¹,h³¹)=({h11},{h21},{h31}).'
            ),
        })
    links.append({
        'label': 'CYTools (geometry toolkit)',
        'url': 'https://arxiv.org/abs/2211.03823',
        'detail': 'Offline: polytopes, triangulations, intersection numbers.',
    })
    return links


def scan_readiness(
    construction: Optional[Dict[str, Any]],
    h11: int,
    h21: int,
    euler: int,
) -> Dict[str, Any]:
    """How many flux-scan prerequisites we already have vs still missing."""
    present = set((construction or {}).get('present_geometry') or {})
    curated = bool((construction or {}).get('curated'))
    items = [
        {
            'id': 'hodge',
            'label': 'Hodge invariants locked',
            'ok': True,
            'detail': f'(h¹¹,h²¹,χ)=({h11},{h21},{euler})',
        },
        {
            'id': 'tadpole',
            'label': 'Tadpole budget L=|χ|/24',
            'ok': True,
            'detail': f'L={abs(euler)/24:.4f}',
        },
        {
            'id': 'ambient_or_curated',
            'label': 'Ambient / curated construction class',
            'ok': curated or ('ambient' in present),
            'detail': 'Textbook class or stored ambient' if (curated or 'ambient' in present) else 'Missing',
        },
        {
            'id': 'equation',
            'label': 'Hypersurface / CICY equation',
            'ok': 'hypersurface_equation' in present,
            'detail': 'Present' if 'hypersurface_equation' in present else 'Not stored',
        },
        {
            'id': 'vertices',
            'label': 'Polytope vertex matrix',
            'ok': 'polytope_vertices' in present or 'vertex_matrix' in present,
            'detail': 'Present' if ('polytope_vertices' in present or 'vertex_matrix' in present) else 'Needs PALP/CYTools',
        },
        {
            'id': 'triangulation',
            'label': 'Fine regular star triangulation',
            'ok': 'triangulation' in present or 'triangulation_id' in present,
            'detail': 'Present' if ('triangulation' in present or 'triangulation_id' in present) else 'Needs CYTools',
        },
        {
            'id': 'periods',
            'label': 'Periods / prepotential',
            'ok': True,
            'detail': f'Symbolic PF structure available (K=h²¹+1={h21 + 1}); numerical periods not evaluated',
        },
        {
            'id': 'orientifold',
            'label': 'Orientifold O3/O7 data',
            'ok': False,
            'detail': 'χ/24 term known; O3/O7 charges need involution',
        },
    ]
    done = sum(1 for i in items if i['ok'])
    return {
        'checklist': items,
        'score': done,
        'total': len(items),
        'pct': round(100.0 * done / len(items), 1),
        'note': (
            'Readiness for a *real* flux vacuum scan. Green items are available '
            'from Hodge/curated metadata; red items need an external geometry pipeline.'
        ),
    }


def heterotic_model_sketch(h11: int, h21: int, euler: int) -> Dict[str, Any]:
    """Topological checklist for heterotic model building (not a bundle)."""
    gens = generation_index(euler)
    return {
        'n_generations': gens['n_generations'],
        'three_generation_target': gens['three_generation_target'],
        'anomaly_cancellation_sketch': (
            'Heterotic anomaly cancellation schematically requires c₂(V) = c₂(TX) '
            '(plus five-branes). c₂(TX) needs intersection data — only a c₂·J '
            'ranking proxy is available here.'
        ),
        'bundle_status': 'unavailable',
        'checklist': [
            {
                'label': 'Base CY3 with these Hodge numbers',
                'ok': False,
                'detail': 'Pick KS/CICY geometry (non-unique at Hodge level)',
            },
            {
                'label': 'Stable holomorphic vector bundle / monad',
                'ok': False,
                'detail': 'Not stored — needed for spectrum',
            },
            {
                'label': f'Net chirality target n_gen={gens["n_generations"]}',
                'ok': True,
                'detail': 'Fixed by |χ|/2 once index(V)=χ(X)/2 setup is chosen',
            },
            {
                'label': 'Yukawa / soft terms',
                'ok': False,
                'detail': 'Need bundle cohomology + moduli vevs',
            },
        ],
    }


def generation_index(euler: int) -> Dict[str, Any]:
    """Net chiral generation index |χ|/2 (heterotic / topological index proxy)."""
    n_gen = abs(int(euler)) // 2
    return {
        'n_generations': n_gen,
        'formula': '|χ|/2',
        'three_generation_target': abs(int(euler)) == 6,
        'note': (
            'For heterotic compactifications the net number of chiral generations '
            'is |χ|/2 when the gauge bundle index equals the Euler characteristic '
            'index. This is a topological necessary condition, not a full model.'
        ),
    }


def stabilization_map(h11: int, h21: int) -> List[Dict[str, Any]]:
    """Which moduli sectors can be fixed by which mechanisms (in principle)."""
    return [
        {
            'sector': 'Complex structure',
            'count': h21,
            'mechanism': 'G₃ flux (tree-level GVW superpotential)',
            'status': 'in_principle',
            'detail': f'{h21} moduli; flux lattice dimension proxy K = h²¹+1',
        },
        {
            'sector': 'Kähler',
            'count': h11,
            'mechanism': 'Non-perturbative (gaugino condensation / instantons)',
            'status': 'needs_extra',
            'detail': f'{h11} moduli; fluxes alone do not stabilize volume moduli',
        },
        {
            'sector': 'Dilaton / axio-dilaton',
            'count': 1,
            'mechanism': 'Fluxes + non-perturbative corrections',
            'status': 'needs_extra',
            'detail': 'Universal modulus; needs a concrete flux+np setup',
        },
    ]


# Famous constructions: loaded from static JSON when present; tiny built-in
# fallback keeps quintic/mirror/bicubic working without the data file.
_KNOWN_FALLBACK: Dict[Tuple[str, int, int], Dict[str, Any]] = {
    ('kreuzer-skarke', 1, 101): {
        'name': 'Quintic threefold in ℂP⁴',
        'ambient': 'P^4',
        'weight_system': [1, 1, 1, 1, 1],
        'hypersurface_equation': 'Generic degree-5 hypersurface ∑_{|α|=5} c_α x^α = 0',
        'favourable': True,
        'reference': 'Candelas–Horowitz–Strominger–Witten; Greene–Plesser mirror',
        'note': (
            'Textbook example. Many distinct quintics share (h¹¹,h²¹)=(1,101); '
            'this is a construction class, not a unique vacuum.'
        ),
    },
    ('kreuzer-skarke', 101, 1): {
        'name': 'Mirror quintic',
        'ambient': 'Toric mirror of P^4[5] (Greene–Plesser orbifold / KS dual)',
        'weight_system': [1, 1, 1, 1, 1],
        'hypersurface_equation': 'Mirror family of the quintic (orbifold + resolution)',
        'favourable': True,
        'reference': 'Greene–Plesser; Candelas–de la Ossa–Green–Parkes',
        'note': 'Hodge mirror of the quintic: (h¹¹,h²¹)=(101,1), χ=+200.',
    },
    ('kreuzer-skarke', 2, 83): {
        'name': 'Bicubic in P²×P²',
        'ambient': 'P^2 × P^2',
        'weight_system': None,
        'hypersurface_equation': 'Bidegree (3,3) hypersurface',
        'favourable': True,
        'reference': 'Classic CICY / toric hypersurface example',
        'note': 'Common pedagogical example with h¹¹=2, h²¹=83, χ=−162.',
    },
}

_KNOWN_CACHE: Optional[Dict[Tuple[str, int, int], Dict[str, Any]]] = None
KNOWN_CONSTRUCTIONS_PATH = os.path.join('data', 'known_constructions.json')
_LEGACY_KNOWN_PATH = os.path.join('static', 'data', 'known_constructions.json')


def load_known_constructions(
    path: Optional[str] = None,
    *,
    force_reload: bool = False,
) -> Dict[Tuple[str, int, int], Dict[str, Any]]:
    """Load curated construction classes from JSON (with built-in fallback)."""
    global _KNOWN_CACHE
    if _KNOWN_CACHE is not None and not force_reload:
        return _KNOWN_CACHE

    candidates = []
    if path:
        candidates.append(path)
    candidates.extend([KNOWN_CONSTRUCTIONS_PATH, _LEGACY_KNOWN_PATH])

    loaded: Dict[Tuple[str, int, int], Dict[str, Any]] = dict(_KNOWN_FALLBACK)
    for candidate_path in candidates:
        try:
            with open(candidate_path, 'r', encoding='utf-8') as fh:
                payload = json.load(fh)
            for item in payload.get('constructions') or []:
                ds = item.get('dataset_id') or 'kreuzer-skarke'
                h11 = item.get('h11')
                h21 = item.get('h21')
                if h11 is None or h21 is None:
                    continue
                entry = {k: v for k, v in item.items() if k not in ('dataset_id', 'h11', 'h21')}
                loaded[(ds, int(h11), int(h21))] = entry
            break
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue

    _KNOWN_CACHE = loaded
    return loaded


# Back-compat alias used by tests / callers that expect a mapping.
KNOWN_CONSTRUCTIONS = _KNOWN_FALLBACK


def lookup_known_construction(
    dataset_id: str, h11: int, h21: int
) -> Optional[Dict[str, Any]]:
    return load_known_constructions().get((dataset_id, int(h11), int(h21)))

def honest_pipeline_checklist(
    dataset_id: str,
    h11: Optional[int],
    h21: Optional[int],
    present_geometry: Optional[Dict[str, Any]] = None,
    *,
    pack: Optional[Dict[str, Any]] = None,
    stage: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Honest vertices → triangulation → intersections → periods checklist.

    Never marks numerical periods filled unless a periods payload exists.
    Quintic / mirror classes may mark intersections as literature/combinatorial
    and periods as literature-only (CdOGP), without inventing CYTools dumps.
    """
    pg = dict(present_geometry or {})
    if pack:
        for key in (
            'polytope_vertices', 'vertex_matrix', 'triangulation',
            'intersections', 'periods', 'ambient',
        ):
            if pg.get(key) is None and pack.get(key) is not None:
                pg[key] = pack[key]

    inferred = stage or geometry_store.infer_stage(pg)
    quintic_class = (
        (dataset_id or '').startswith('kreuzer')
        and h11 is not None
        and h21 is not None
        and (int(h11), int(h21)) in ((1, 101), (101, 1))
    )
    lit_periods = bool(
        quintic_class or (pack or {}).get('periods_literature_pointer')
    )
    ix = pg.get('intersections')
    ix_lit = isinstance(ix, dict) and str(ix.get('status') or '').startswith('literature')

    labels = {
        'vertices': 'Vertices',
        'triangulated': 'Triangulation / ambient',
        'intersections': 'Intersections',
        'periods': 'Periods',
    }
    details = {
        'vertices': (
            'Curated / stored polytope vertices present.'
            if geometry_store._has_vertices(pg)
            else 'No vertex matrix stored yet (offline worker or pack).'
        ),
        'triangulated': (
            (
                'Ambient ℂP⁴ note: smooth projective ambient — no toric FRST required '
                'for the textbook quintic class.'
                if quintic_class and (h11, h21) == (1, 101)
                else (
                    'Greene–Plesser / mirror ambient note present.'
                    if quintic_class
                    else 'Triangulation or ambient resolution note present.'
                )
            )
            if geometry_store._has_triangulation(pg) or geometry_store.stage_includes(
                inferred, 'triangulated'
            )
            else 'Triangulation / ambient resolution still pending.'
        ),
        'intersections': (
            (
                'Literature / combinatorial only'
                + (
                    f" (κ(H³)={ix.get('triple_intersection_H3')})"
                    if isinstance(ix, dict) and ix.get('triple_intersection_H3') is not None
                    else ''
                )
                + ' — not a CYTools dump.'
            )
            if ix_lit
            else (
                'Offline intersection numbers stored.'
                if geometry_store._has_intersections(pg)
                else (
                    'Combinatorial / symbolic only unless literature or offline dump '
                    'is attached — Hodge numbers alone do not fix intersections.'
                )
            )
        ),
        'periods': (
            (
                'CdOGP literature formulas surfaced on the Fluxes tab '
                '(Picard–Fuchs / special points). Not a numerical period engine.'
            )
            if lit_periods and not geometry_store._has_periods(pg)
            else (
                'Offline / stored period payload present.'
                if geometry_store._has_periods(pg)
                else 'Numerical periods pending offline worker — not invented here.'
            )
        ),
    }

    out: List[Dict[str, Any]] = []
    for name in geometry_store.PIPELINE_STAGES:
        filled = geometry_store.stage_includes(inferred, name)
        # Literature periods are a showcase overlay, not a filled offline stage.
        if name == 'periods' and lit_periods and not geometry_store._has_periods(pg):
            filled = False
            status = 'literature'
        elif name == 'intersections' and ix_lit:
            filled = True
            status = 'literature'
        elif filled:
            status = 'have'
        else:
            status = 'pending'
        out.append({
            'stage': name,
            'label': labels.get(name, name),
            'filled': filled,
            'status': status,
            'detail': details[name],
        })
    return out


def construction_workplan(
    dataset_id: str, h11: int, h21: int, euler: int, h31: Optional[int] = None
) -> List[Dict[str, str]]:
    """Concrete next-step recipe with numbers filled in (still needs external tools)."""
    L = abs(euler) / 24.0
    steps = [
        {
            'id': 'invariants',
            'title': 'Lock topological invariants',
            'detail': f'Use (h¹¹,h²¹,χ)=({h11},{h21},{euler}) as search keys.',
        },
    ]
    if dataset_id == 'kreuzer-skarke':
        steps.extend([
            {
                'id': 'polytope',
                'title': 'Find reflexive 4-polytopes with these Hodge numbers',
                'detail': (
                    'PALP / KS database / CYTools: filter polytopes whose '
                    f'favourable hypersurfaces give h¹¹={h11}, h²¹={h21}.'
                ),
            },
            {
                'id': 'triangulate',
                'title': 'Choose a fine regular star triangulation',
                'detail': 'Fixes the toric ambient and the resolved CY hypersurface.',
            },
            {
                'id': 'periods',
                'title': 'Compute periods / prepotential',
                'detail': (
                    f'Needed for a real flux scan on the {h21}-dimensional '
                    'complex-structure moduli space.'
                ),
            },
            {
                'id': 'tadpole',
                'title': 'Impose D3 tadpole',
                'detail': f'Target budget L = |χ|/24 = {L:.4f}.',
            },
        ])
    elif dataset_id == 'cy5-folds':
        steps.extend([
            {
                'id': 'config',
                'title': 'Look up a real CI5F configuration matrix',
                'detail': (
                    f'Compatible with (h¹¹,h²¹,h³¹)=({h11},{h21},{h31}). '
                    'Start from arXiv:2310.15966 / Dropbox cicy5f.json (checked: '
                    'featured HoF triples are not in that ≤4-factor census). '
                    'Attach a literature-backed multi-degree matrix before claiming '
                    'a concrete fivefold — never invent one from Hodge numbers alone.'
                ),
            },
            {
                'id': 'kahler',
                'title': 'Large-volume Kähler cone data',
                'detail': f'h¹¹={h11} suggests many Kähler moduli (LVS-style searches).',
            },
        ])
    elif dataset_id == 'heterotic':
        n_gen = abs(euler) // 2
        steps.extend([
            {
                'id': 'geometry',
                'title': 'Pick a concrete CY3 with these Hodge numbers',
                'detail': 'KS / CICY / toric construction — Hodge pair is not unique.',
            },
            {
                'id': 'bundle',
                'title': 'Build a stable holomorphic vector bundle',
                'detail': (
                    f'Target net chirality ~ |χ|/2 = {n_gen} generations '
                    '(index theorem); needs monad/extension/spectral cover data.'
                ),
            },
        ])
    else:
        steps.append({
            'id': 'backend',
            'title': 'Attach geometry from a heavier backend',
            'detail': 'Store polytope / bundle / flux data in raw_json for this id.',
        })
    return steps


def construction_payload(
    dataset_id: str,
    candidate_id: str,
    raw: Optional[Dict[str, Any]] = None,
    features: Optional[List] = None,
    tags: Optional[List] = None,
    summary: Optional[str] = None,
    h11: Optional[int] = None,
    h21: Optional[int] = None,
    h31: Optional[int] = None,
    euler_char: Optional[int] = None,
) -> Dict[str, Any]:
    """Honest construction metadata: show what we have, never invent equations."""
    raw = raw or {}
    feature_map = {}
    for item in features or []:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            feature_map[str(item[0])] = item[1]

    h11_i = _i(h11 if h11 is not None else raw.get('h11', feature_map.get('h11')))
    h21_i = _i(h21 if h21 is not None else raw.get('h21', feature_map.get('h21')))
    h31_i = _i(h31 if h31 is not None else raw.get('h31', feature_map.get('h31')))
    euler_i = _i(
        euler_char if euler_char is not None
        else raw.get('euler_char', feature_map.get('χ', feature_map.get('euler_char')))
    )
    if euler_i is None and h11_i is not None and h21_i is not None:
        euler_i = derive_euler(dataset_id or 'kreuzer-skarke', h11_i, h21_i, h31_i)

    known_geometry_keys = (
        'polytope_id', 'polytope_hash', 'polytope_vertices', 'vertex_matrix',
        'triangulation', 'triangulation_id', 'hypersurface_equation',
        'weight_system', 'favourable', 'ambient', 'configuration_matrix',
        'geometry_name', 'geometry_note', 'geometry_status', 'geometry_uniqueness',
        'vertex_count', 'facet_count', 'point_count', 'dual_point_count',
        'ks_source_slice', 'geometry_source', 'geometry_db_id', 'periods',
        'orientifold', 'stage', 'intersections', 'pipeline_note',
        'showcase', 'showcase_note', 'periods_literature_pointer',
    )

    def _present_from(src: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k in known_geometry_keys:
            if k not in src or src[k] is None:
                continue
            if k == 'configuration_matrix' and not physics_extensions.is_real_configuration_matrix(
                src[k]
            ):
                continue
            out[k] = src[k]
        return out

    present_geometry = _present_from(raw)

    curated = None
    pack = None
    db_geometry = None
    if h11_i is not None and h21_i is not None:
        curated = lookup_known_construction(dataset_id or 'kreuzer-skarke', h11_i, h21_i)
        pack = physics_extensions.lookup_geometry_pack(
            dataset_id or 'kreuzer-skarke', h11_i, h21_i, h31_i
        )
        raw = physics_extensions.merge_geometry_into_raw(raw, pack)
        # Prefer SQLite geometry DB when present / richer (has vertices).
        db_geometry = geometry_store.resolve_geometry(
            candidate_id=candidate_id,
            dataset_id=dataset_id or 'kreuzer-skarke',
            h11=h11_i,
            h21=h21_i,
            h31=h31_i,
        )
        if db_geometry:
            raw = geometry_store.merge_db_into_raw(raw, db_geometry)
        present_geometry = _present_from(raw)
        if curated:
            for key in (
                'ambient', 'weight_system', 'hypersurface_equation', 'favourable',
            ):
                if key not in present_geometry and curated.get(key) is not None:
                    present_geometry[key] = curated[key]
            if curated.get('note') and 'geometry_note' not in present_geometry:
                present_geometry['geometry_note'] = curated['note']
        if pack and physics_extensions.is_real_configuration_matrix(
            pack.get('configuration_matrix')
        ):
            present_geometry.setdefault('configuration_matrix', pack['configuration_matrix'])
        if pack and pack.get('name'):
            present_geometry.setdefault('geometry_name', pack['name'])
        if pack and pack.get('note'):
            present_geometry.setdefault('geometry_note', pack['note'])

    unavailable = [
        {
            'id': 'polytope_vertices',
            'label': 'Reflexive polytope vertex matrix',
            'reason': 'Not stored in hall-of-fame records yet.',
        },
        {
            'id': 'triangulation',
            'label': 'Fine regular star triangulation',
            'reason': 'Required to fix a toric ambient / CY hypersurface.',
        },
        {
            'id': 'hypersurface_equation',
            'label': 'Hypersurface / CICY equation',
            'reason': 'Not recoverable from Hodge numbers alone.',
        },
        {
            'id': 'configuration_matrix',
            'label': 'Configuration matrix (CICY / CI5F)',
            'reason': (
                'No real multi-degree / CI5F matrix is stored for this Hodge class; '
                'look up a matching entry in the literature database.'
            ),
        },
    ]
    # Drop unavailable entries that we unexpectedly do have (raw or curated).
    have_vertices = (
        'polytope_vertices' in present_geometry or 'vertex_matrix' in present_geometry
    )
    unavailable = [
        u for u in unavailable
        if u['id'] not in present_geometry
        and not (u['id'] == 'polytope_vertices' and have_vertices)
    ]

    if dataset_id == 'kreuzer-skarke':
        reconstruct = (
            'Kreuzer–Skarke: look up a reflexive 4-polytope with these Hodge '
            'numbers (PALP / KS database), choose a triangulation, then form '
            'the anticanonical hypersurface. Many polytopes can share the same '
            '(h¹¹, h²¹, χ) — the content-addressed id here is not a KS polytope id.'
        )
        source_url = 'http://hep.itp.tuwien.ac.at/~kreuzer/CY/'
    elif dataset_id == 'cy5-folds':
        has_matrix = 'configuration_matrix' in present_geometry
        reconstruct = (
            'CY5 / CI5F: recover a complete-intersection fivefold configuration '
            'compatible with (h¹¹, h²¹, h³¹). '
            + (
                'A curated configuration matrix is attached below.'
                if has_matrix else
                'Featured Hodge triples here are pedagogical class keys — a real '
                'CI5F configuration matrix from the CI5F database is still pending.'
            )
        )
        source_url = None
    elif dataset_id == 'heterotic':
        reconstruct = (
            'Heterotic: Hodge balance is only a necessary filter. A full model '
            'needs a stable holomorphic vector bundle (or monad) on a concrete '
            'geometry — not present in stored fields. χ=±6 only flags a '
            'three-generation index target, not a bundle construction.'
        )
        source_url = None
    else:
        reconstruct = (
            'Reconstruction needs dataset-specific geometric data beyond Hodge '
            'numbers. Extension point: attach construction payloads from a '
            'heavier backend.'
        )
        source_url = None

    workplan: List[Dict[str, str]] = []
    if h11_i is not None and h21_i is not None and euler_i is not None:
        workplan = construction_workplan(
            dataset_id or 'kreuzer-skarke', h11_i, h21_i, euler_i, h31_i
        )

    honesty = (
        'Exact: stored metadata keys and Hodge invariants. '
        'Curated: textbook constructions when (h¹¹,h²¹) matches a known class. '
        'Geometry DB: offline / seeded SQLite hits preferred when richer (vertices). '
        'Unavailable: unique polytope vertices / triangulation unless stored.'
    )

    geometry_db_meta = None
    if db_geometry:
        geometry_db_meta = {
            'id': db_geometry.get('id'),
            'source': db_geometry.get('source'),
            'status': db_geometry.get('status'),
            'note': db_geometry.get('note'),
            'stage': db_geometry.get('stage'),
            'pipeline_note': db_geometry.get('pipeline_note'),
            'intersections': db_geometry.get('intersections'),
            'reference': db_geometry.get('reference'),
            'reference_url': db_geometry.get('reference_url'),
        }
        if present_geometry.get('geometry_source') is None:
            present_geometry['geometry_source'] = db_geometry.get('source')
        if present_geometry.get('geometry_status') is None:
            present_geometry['geometry_status'] = db_geometry.get('status')
        if present_geometry.get('stage') is None and db_geometry.get('stage'):
            present_geometry['stage'] = db_geometry.get('stage')
        if present_geometry.get('pipeline_note') is None and db_geometry.get('pipeline_note'):
            present_geometry['pipeline_note'] = db_geometry.get('pipeline_note')
        if present_geometry.get('intersections') is None and db_geometry.get('intersections'):
            present_geometry['intersections'] = db_geometry.get('intersections')

    citations: List[Dict[str, str]] = []
    seen_cite: set = set()

    def _add_citation(label: Optional[str], url: Optional[str], source: str) -> None:
        if not label and not url:
            return
        key = (label or '', url or '')
        if key in seen_cite:
            return
        # Prefer URL'd citations: skip label-only duplicates of an existing URL row.
        if not url:
            for existing in citations:
                if existing.get('url') and (
                    (label and label == existing.get('label'))
                    or (label and label in (existing.get('label') or ''))
                    or ((existing.get('label') or '') in (label or ''))
                ):
                    return
        seen_cite.add(key)
        citations.append({
            'label': label or url or 'Reference',
            'url': url or '',
            'source': source,
        })

    if curated:
        _add_citation(curated.get('reference'), curated.get('reference_url'), 'curated')
    if pack:
        _add_citation(pack.get('reference'), pack.get('reference_url'), 'geometry-pack')
    if db_geometry:
        _add_citation(
            db_geometry.get('reference'),
            db_geometry.get('reference_url'),
            'geometry-db',
        )
    if source_url:
        _add_citation('Dataset / catalog source', source_url, 'dataset')

    showcase = False
    showcase_note = None
    if pack and pack.get('showcase'):
        showcase = True
        showcase_note = pack.get('showcase_note')
    if curated and curated.get('showcase'):
        showcase = True
        showcase_note = showcase_note or curated.get('showcase_note')
    if candidate_id == 'kreuzer-skarke-66d611d18a9d':
        showcase = True
        showcase_note = showcase_note or (
            'Flagship linking example: textbook quintic class (h¹¹,h²¹)=(1,101). '
            'Cite or share this dossier — class-level, not a unique polytope proof.'
        )

    return {
        'candidate_id': candidate_id,
        'dataset_id': dataset_id,
        'summary': summary,
        'tags': list(tags or []),
        'feature_map': feature_map,
        'raw_keys': _raw_keys(raw),
        'present_geometry': present_geometry,
        'unavailable': unavailable,
        'reconstruct_howto': reconstruct,
        'source_url': source_url,
        'curated': curated,
        'citations': citations,
        'workplan': workplan,
        'honesty': honesty,
        'geometry_db': geometry_db_meta,
        'showcase': showcase,
        'showcase_note': showcase_note,
        'pipeline_checklist': honest_pipeline_checklist(
            dataset_id or 'kreuzer-skarke',
            h11_i,
            h21_i,
            present_geometry,
            pack=pack,
        ),
    }


def build_tabs(
    dossier: Dict[str, Any],
    *,
    construction: Optional[Dict[str, Any]] = None,
    tags: Optional[List] = None,
    mirror_partner: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Organize dossier scalars into analysis tabs with honesty labels."""
    if not dossier.get('ok'):
        return {'ok': False, 'error': dossier.get('error', 'dossier unavailable')}

    s = dossier['scalars']
    h11, h21 = dossier['h11'], dossier['h21']
    euler = dossier['euler_char']
    dataset_id = dossier.get('dataset_id') or 'kreuzer-skarke'
    vacua = flux_vacua_estimate(h21, s['tadpole_L'])
    gens = generation_index(euler)
    c2 = second_chern_proxy(h11, h21)
    stab = stabilization_map(h11, h21)
    periods = period_structure(h21)
    periods_full = physics_extensions.periods_structure_full(h21)
    intersections = intersection_proxies(h11)
    links = external_geometry_links(dataset_id, h11, h21, dossier.get('h31'))
    readiness = scan_readiness(construction, h11, h21, euler)
    # Recompute readiness after construction may have gained pack geometry.
    if construction:
        readiness = scan_readiness(construction, h11, h21, euler)
    miniscan = physics_extensions.flux_lattice_miniscan(h21, s['tadpole_L'])
    orientifold = physics_extensions.orientifold_tadpole_sketch(euler)
    quintic_o3 = physics_extensions.quintic_orientifold_curated(h11, h21)
    if quintic_o3:
        orientifold = quintic_o3
    soft = physics_extensions.soft_terms_symbolic()
    soft_toy = physics_extensions.toy_soft_parameter_card()
    yukawas = physics_extensions.yukawa_structure(h11, h21, dataset_id)
    gauge = physics_extensions.gauge_embedding_sketch(dataset_id, euler)
    periods_literature = physics_extensions.quintic_periods_literature(h11, h21)
    if periods_literature:
        periods_full = dict(periods_full)
        periods_full['literature'] = periods_literature
        periods_full['status'] = 'literature_curated'
        periods_full['note'] = periods_literature['honesty']
        periods_full['mirror_family_pf_tex'] = periods_literature['mirror_family_pf_tex']
        periods_full['special_points'] = periods_literature['special_points']
        periods_full['references'] = periods_literature['references']

    mirror_block = {
        'h11': int(s['mirror_h11']),
        'h21': int(s['mirror_h21']),
        'euler': int(s['mirror_euler']),
        'note': 'Mirror swaps h¹¹ ↔ h²¹ and sends χ → −χ at the Hodge level.',
        'partner': mirror_partner,
        'compare': {
            'same_abs_euler': True,
            'moduli_swap': f'({h11},{h21}) ↔ ({int(s["mirror_h11"])},{int(s["mirror_h21"])})',
            'kahler_vs_cs': (
                f'Here Kähler-heavy' if h11 > h21 else
                ('CS-heavy' if h21 > h11 else 'balanced')
            ),
        },
    }
    if dataset_id == 'cy5-folds':
        mirror_block['note'] = (
            'CY5 mirror maps are richer than a simple (h¹¹,h²¹) swap; '
            'Hodge-level swap below is only a partial guide.'
        )

    moduli = {
        'id': 'moduli',
        'title': 'Moduli',
        'honesty': (
            'Exact counts from Hodge numbers for a CY3: dim of Kähler moduli '
            'space is h¹¹; complex-structure moduli space dimension is h²¹ '
            '(before quotienting by discrete symmetries / flux stabilization). '
            'Stabilization map is mechanism taxonomy, not a solved potential.'
        ),
        'counts': {
            'kahler_moduli_h11': h11,
            'complex_structure_moduli_h21': h21,
            'h31': dossier.get('h31'),
            'total_moduli': int(s['total_moduli']),
            'moduli_compactness': s['moduli_compactness'],
            'hodge_balance': s['hodge_balance'],
            'picard_fuchs_order': periods['picard_fuchs_order'],
            'h3_betti': periods['h3_betti'],
            'triple_intersection_unknowns': intersections['symmetric_triple_independent'],
        },
        'periods': periods,
        'intersections': intersections,
        'mirror': mirror_block,
        'stabilization': stab,
        'meaning': [
            'h¹¹ counts independent Kähler (size) deformations of even cycles.',
            'h²¹ counts complex-structure (shape) deformations.',
            f'Period / Picard–Fuchs problem size proxy: K = h²¹+1 = {periods["picard_fuchs_order"]}.',
            'G₃ fluxes can fix complex structure in principle; Kähler moduli '
            'need non-perturbative effects — see stabilization map below.',
        ],
    }

    fluxes = {
        'id': 'fluxes',
        'title': 'Fluxes / tadpole',
        'honesty': (
            'Exact: tadpole budget L = |χ|/24 from the Euler characteristic. '
            'Computed proxy: Stirling / Bousso–Polchinski log-density and '
            'asymptotic N_flux estimate. Not a vacuum scan over a real flux lattice.'
        ),
        'budget': {
            'euler_char': euler,
            'tadpole_L': s['tadpole_L'],
            'tadpole_headroom': s['tadpole_headroom'],
            'flux_density_proxy': s['flux_density'],
            'K_proxy': vacua['K'],
            'log_N_flux': vacua['log_N_flux'],
            'log10_N_flux': vacua.get('log10_N_flux'),
            'N_flux_est_sci': vacua['N_flux_est_sci'],
            'picard_fuchs_order': periods['picard_fuchs_order'],
            'h3_betti': periods['h3_betti'],
        },
        'vacua_estimate': vacua,
        'periods': periods,
        'periods_full': periods_full,
        'lattice_miniscan': miniscan,
        'orientifold': orientifold,
        'quintic_periods_literature': periods_literature,
        'readiness': readiness,
        'constraints': [
            {
                'name': 'D3 tadpole (IIB sketch)',
                'tex': r'N_{D3} + N_{\mathrm{flux}} = L = |\chi|/24',
                'status': 'exact_budget',
                'detail': f"L = {s['tadpole_L']}; headroom proxy = {s['tadpole_headroom']}",
            },
            {
                'name': 'Orientifold-extended tadpole',
                'tex': orientifold['tex'],
                'status': 'partial',
                'detail': orientifold['note'],
            },
            {
                'name': 'Period / flux monomial count',
                'tex': r'K = h^{2,1}+1,\quad b_3 = 2(h^{2,1}+1)',
                'status': 'exact_count',
                'detail': (
                    f"K={periods['picard_fuchs_order']}, "
                    f"b₃={periods['h3_betti']} (CY3 Betti)"
                ),
            },
            {
                'name': 'Flux vacua asymptotic estimate',
                'tex': r'\log N_{\mathrm{flux}} \sim 2K\log(2\pi L) - \log K!',
                'status': 'proxy',
                'detail': (
                    f"K = {vacua['K']}; log N ≈ {vacua['log_N_flux']}; "
                    f"N ~ {vacua['N_flux_est_sci']}; "
                    f"ranking sigmoid = {s['flux_density']}"
                ),
            },
            {
                'name': 'Toy flux-lattice count',
                'tex': miniscan['tex'],
                'status': miniscan['status'],
                'detail': (
                    f"counted={miniscan.get('counted')} · "
                    f"{miniscan.get('note') or miniscan.get('reason')}"
                ),
            },
        ],
        'requires_for_full_scan': [
            'Numerical periods / prepotential on the complex-structure moduli space',
            'Orientifold involution with explicit O3/O7 charges',
            'Physical H³ flux lattice (beyond the toy ∑n² bound)',
            'A concrete triangulation / hypersurface when not in the geometry pack',
        ],
    }

    pheno = {
        'id': 'phenomenology',
        'title': 'Phenomenology',
        'honesty': (
            'Topological indices are exact formulas; soft terms and Yukawas are '
            'symbolic / combinatorial until moduli vevs and bundles/branes exist.'
        ),
        'proxies': {
            'hodge_balance': s['hodge_balance'],
            'vacuum_stability_proxy': s['vacuum_stability'],
            'topo_efficiency': s['topo_efficiency'],
            'hodge_entropy': s['hodge_entropy'],
            'c2_J_proxy': c2['c2_J_proxy'],
        },
        'indices': gens,
        'chern': c2,
        'heterotic': (
            heterotic_model_sketch(h11, h21, euler)
            if dataset_id == 'heterotic' else None
        ),
        'soft_terms': soft,
        'soft_toy_card': soft_toy,
        'yukawas': yukawas,
        'gauge': gauge,
        'dataset_target': next(
            (c for c in dossier['checks'] if c['id'] == 'dataset_target'), None
        ),
        'tags': list(tags or []),
        'computed': [
            {
                'name': 'Net generation index',
                'value': gens['n_generations'],
                'status': 'exact_formula',
                'detail': gens['note'],
            },
            {
                'name': 'Three-generation necessary condition',
                'value': 'yes' if gens['three_generation_target'] else 'no',
                'status': 'exact_check',
                'detail': '|χ| = 6 ⇔ |χ|/2 = 3 generations (heterotic index sketch)',
            },
            {
                'name': 'c₂·J scale proxy',
                'value': c2['c2_J_proxy'],
                'status': 'proxy',
                'detail': c2['note'],
            },
            {
                'name': 'Symmetric triple-intersection unknowns (generic)',
                'value': intersections['symmetric_triple_independent'],
                'status': 'combinatorial',
                'detail': intersections['note'],
            },
            {
                'name': 'Soft-term skeleton',
                'value': soft['mediation'],
                'status': 'symbolic',
                'detail': soft['note'],
            },
            {
                'name': 'Yukawa structure',
                'value': yukawas['status'],
                'status': yukawas['status'],
                'detail': yukawas['note'],
            },
            {
                'name': 'Gauge embedding roadmap',
                'value': gauge['status'],
                'status': 'checklist',
                'detail': gauge['note'],
            },
        ],
        'pending_geometry': [
            'Numerical soft masses (need moduli vevs + mediation)',
            'Explicit SM Yukawa matrices (need branes/bundles + wavefunctions)',
            'Unique gauge embedding (need concrete geometry)',
        ],
    }

    # Extra certificate checks derived here (keep dossier.checks intact + append).
    extra_checks = [
        {
            'id': 'generation_index',
            'label': 'Generation index defined',
            'rule': 'n_gen = |χ|/2 ≥ 0',
            'rule_tex': r'n_{\mathrm{gen}} = |\chi|/2 \ge 0',
            'ok': gens['n_generations'] >= 0,
            'detail': f"n_gen={gens['n_generations']}",
        },
        {
            'id': 'euler_even_cy3',
            'label': 'Euler parity (CY3)',
            'rule': 'χ even for CY3 Hodge identity χ=2(h¹¹−h²¹)',
            'rule_tex': r'\chi = 2(h^{1,1}-h^{2,1})\ \text{even}',
            'ok': dataset_id == 'cy5-folds' or (euler % 2 == 0),
            'detail': f'χ={euler}',
        },
    ]
    all_checks = list(dossier['checks']) + extra_checks

    certificates = {
        'id': 'certificates',
        'title': 'Certificates',
        'honesty': (
            'Machine-checkable identities and necessary conditions on the '
            'stored invariants. These are certificates, not uniqueness theorems '
            'or existence proofs for string vacua.'
        ),
        'checks': all_checks + [
            {
                'id': 'period_count',
                'label': 'Period count identity',
                'rule': 'K = h²¹+1 and b₃ = 2(h²¹+1) for CY3',
                'rule_tex': r'K = h^{2,1}+1,\quad b_3 = 2(h^{2,1}+1)',
                'ok': dataset_id == 'cy5-folds' or periods['h3_betti'] == 2 * (h21 + 1),
                'detail': f"K={periods['picard_fuchs_order']}, b₃={periods['h3_betti']}",
            },
            {
                'id': 'scan_readiness',
                'label': 'Flux-scan readiness',
                'rule': f'{readiness["score"]}/{readiness["total"]} prerequisites available',
                'rule_tex': rf'\text{{readiness }} {readiness["score"]}/{readiness["total"]}',
                'ok': readiness['score'] >= 3,
                'detail': f"{readiness['pct']}% — see Fluxes / Construction tabs",
            },
        ],
        'identities': dossier['identities'] + [
            {
                'name': 'Net generation index',
                'tex': r'n_{\mathrm{gen}} = |\chi|/2',
                'value': gens['n_generations'],
            },
            {
                'name': 'Period / Betti count (CY3)',
                'tex': r'b_3 = 2(h^{2,1}+1)',
                'value': periods['h3_betti'],
            },
            {
                'name': 'Flux vacua log-count (Stirling proxy)',
                'tex': r'\log N_{\mathrm{flux}} \sim 2K\log(2\pi L)-\log K!',
                'value': vacua['log_N_flux'],
            },
        ],
        'euler_consistent': dossier['euler_consistent'],
        'readiness': readiness,
    }

    overview = {
        'id': 'overview',
        'title': 'Overview',
        'honesty': dossier.get('caveat', ''),
        'diamond': dossier['diamond'],
        'identities': dossier['identities'],
        'scalars_highlight': {
            'tadpole_L': s['tadpole_L'],
            'total_moduli': int(s['total_moduli']),
            'hodge_balance': s['hodge_balance'],
            'flux_density': s['flux_density'],
            'n_generations': gens['n_generations'],
            'N_flux_est_sci': vacua['N_flux_est_sci'],
            'scan_readiness_pct': readiness['pct'],
            'picard_fuchs_order': periods['picard_fuchs_order'],
        },
        'readiness': readiness,
        'external_links': links,
    }

    out_construction = dict(construction or {
        'honesty': 'No construction context supplied.',
        'unavailable': [],
        'present_geometry': {},
        'raw_keys': [],
        'feature_map': {},
        'tags': [],
        'reconstruct_howto': '',
        'curated': None,
        'workplan': [],
    })
    out_construction['external_links'] = links
    out_construction['readiness'] = readiness

    # --- Model-building tab (exclusions + literature cards + pipeline) -----
    exclusions = model_exclusions.evaluate(
        dataset_id, h11, h21, dossier.get('h31'), euler,
    )
    cards = model_cards.list_for_hodge(
        dataset_id, h11, h21, dossier.get('h31'),
    )
    stage = None
    pipeline = None
    intersections_offline = None
    if construction:
        pg = construction.get('present_geometry') or {}
        gdb = construction.get('geometry_db') or {}
        stage = pg.get('stage') or gdb.get('stage')
        pipeline = pg.get('pipeline_note') or gdb.get('pipeline_note')
        intersections_offline = pg.get('intersections') or gdb.get('intersections')
        if stage is None:
            stage = geometry_store.infer_stage({
                'polytope_vertices': pg.get('polytope_vertices'),
                'vertex_matrix': pg.get('vertex_matrix'),
                'triangulation': pg.get('triangulation'),
                'intersections': intersections_offline,
                'periods': pg.get('periods'),
            })
            pipeline = geometry_store.pipeline_note(stage)
    pg = (construction or {}).get('present_geometry') or {}
    stage_checklist = honest_pipeline_checklist(
        dataset_id,
        h11,
        h21,
        pg,
        stage=stage,
    )
    # Prefer richer checklist also on Construction tab.
    out_construction['pipeline_checklist'] = (
        (construction or {}).get('pipeline_checklist') or stage_checklist
    )
    showcase = bool((construction or {}).get('showcase'))
    showcase_note = (construction or {}).get('showcase_note')
    if periods_literature:
        showcase = showcase or (h11, h21) in ((1, 101), (101, 1))
        showcase_note = showcase_note or (
            'Flagship quintic / mirror class: CdOGP literature periods on Fluxes; '
            'pipeline checklist is honest about what is curated vs offline-pending.'
        )

    model_building = {
        'id': 'model-building',
        'title': 'Model-building',
        'honesty_banner': (
            'Model-building aids: topological exclusions and literature cards. '
            'Spectra only when cited from published references.'
        ),
        'honesty': (
            'Exclusions are necessary conditions under stated assumptions only. '
            'Model cards cite published literature. Geometry stages reflect '
            'offline / seeded SQLite richness — never invented periods or spectra.'
        ),
        'exclusions': exclusions,
        'cards': cards,
        'model_cards': cards,  # alias for older callers
        'showcase': showcase,
        'showcase_note': showcase_note,
        'periods_literature': periods_literature,
        'geometry_pipeline': {
            'stage': stage,
            'pipeline_note': pipeline or geometry_store.pipeline_note(stage),
            'checklist': stage_checklist,
            'intersections': intersections_offline or pg.get('intersections'),
        },
    }

    mathematics = math_specimen.mathematics_payload(
        dossier,
        construction=out_construction,
        mirror_partner=mirror_partner,
        mirror_block=mirror_block,
        periods=periods,
        periods_full=periods_full,
        periods_literature=periods_literature,
        external_links=links,
        pipeline_stage=stage,
        pipeline_note=pipeline or geometry_store.pipeline_note(stage),
        pipeline_checklist=stage_checklist,
        candidate_id=(construction or {}).get('candidate_id'),
        tags=tags,
    )
    # Surface FOUND on Model-building when exclusions/cards/showcase hit too.
    mb_reasons = []
    if showcase:
        mb_reasons.append('Flagship showcase / literature periods')
    if any(
        (ex or {}).get('ok') and (ex or {}).get('id') == 'heterotic_standard_embedding_3gen'
        for ex in exclusions
    ):
        mb_reasons.append('Passes heterotic |χ|=6 three-generation necessary condition')
    if cards:
        mb_reasons.append(f'{len(cards)} literature model card(s) matched')
    model_building['found'] = bool(mb_reasons)
    model_building['found_badge'] = 'FOUND' if mb_reasons else None
    model_building['match'] = {
        'found': bool(mb_reasons),
        'headline': 'MODEL HIT FOUND' if mb_reasons else 'No model-card hit',
        'blurb': (
            'Literature cards and/or topological necessary conditions match this Hodge class.'
            if mb_reasons
            else 'No literature model card or showcase condition matched.'
        ),
        'reasons': mb_reasons,
        'tab_badge': 'FOUND' if mb_reasons else None,
    }

    return {
        'ok': True,
        'dataset_id': dataset_id,
        'overview': overview,
        'moduli': moduli,
        'fluxes': fluxes,
        'phenomenology': pheno,
        'mathematics': mathematics,
        'model_building': model_building,
        'construction': out_construction,
        'certificates': certificates,
    }
