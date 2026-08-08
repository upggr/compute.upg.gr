"""Physics dossier for a Calabi–Yau candidate (shareable certificate).

Computes identities, tadpole/flux proxies, and necessary-condition checks from
Hodge numbers alone. These are *proxies and identities*, not a claim that the
geometry is phenomenologically viable — Hodge data do not uniquely fix a
manifold.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


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
        detail = f'h¹¹={h11}'
    elif dataset_id == 'heterotic':
        total = h11 + h21 + 1e-10
        balance = 1.0 - abs(h11 - h21) / total
        ok = balance >= 0.9
        rule = 'h^{1,1} ≈ h^{2,1} (balance ≥ 0.9)'
        detail = f'balance={balance:.3f}'
    elif dataset_id == 'info-density':
        ok = True
        rule = 'top decile by information-density composite (run-dependent)'
        detail = 'proxy scalars shown below; percentile needs a full ranking run'
    else:
        ok = abs(euler) < 100
        rule = '|χ| < 100'
        detail = f'|χ|={abs(euler)}'
    return {'id': 'dataset_target', 'label': 'Dataset target rule', 'rule': rule, 'ok': ok, 'detail': detail}


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
            'ok': scalars['tadpole_L'] >= 0,
            'detail': f'L={scalars["tadpole_L"]}',
        },
        {
            'id': 'moduli_positive',
            'label': 'Positive Hodge numbers',
            'rule': 'h¹¹ ≥ 1 and h²¹ ≥ 1',
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
        # Simplified CY3 Hodge diamond corners commonly quoted in intros:
        # h^{0,0}=1, h^{3,0}=1, h^{1,1}, h^{2,1}, and symmetries.
        'h00': 1,
        'h30': 1,
        'h03': 1,
        'h33': 1,
    }

    identities = [
        {
            'name': 'Euler characteristic (CY3)',
            'tex': r'\chi = 2(h^{1,1} - h^{2,1})',
            'value': euler if dataset_id != 'cy5-folds' else None,
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


# Famous constructions keyed by (dataset_id, h11, h21). Only cite when unique-
# enough textbook examples exist — never invent polytopes for arbitrary Hodge.
KNOWN_CONSTRUCTIONS: Dict[Tuple[str, int, int], Dict[str, Any]] = {
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


def lookup_known_construction(
    dataset_id: str, h11: int, h21: int
) -> Optional[Dict[str, Any]]:
    return KNOWN_CONSTRUCTIONS.get((dataset_id, int(h11), int(h21)))


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
                'title': 'Recover a CI5F configuration matrix',
                'detail': (
                    f'Compatible with (h¹¹,h²¹,h³¹)=({h11},{h21},{h31}). '
                    'Configuration matrices are not stored on this page yet.'
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
        'weight_system', 'favourable', 'ambient',
    )
    present_geometry = {
        k: raw[k] for k in known_geometry_keys if k in raw and raw[k] is not None
    }

    curated = None
    if h11_i is not None and h21_i is not None:
        curated = lookup_known_construction(dataset_id or 'kreuzer-skarke', h11_i, h21_i)
        if curated:
            # Merge curated fields into present_geometry only when raw lacks them.
            for key in (
                'ambient', 'weight_system', 'hypersurface_equation', 'favourable',
            ):
                if key not in present_geometry and curated.get(key) is not None:
                    present_geometry[key] = curated[key]

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
    ]
    # Drop unavailable entries that we unexpectedly do have (raw or curated).
    unavailable = [u for u in unavailable if u['id'] not in present_geometry]

    if dataset_id == 'kreuzer-skarke':
        reconstruct = (
            'Kreuzer–Skarke: look up a reflexive 4-polytope with these Hodge '
            'numbers (PALP / KS database), choose a triangulation, then form '
            'the anticanonical hypersurface. Many polytopes can share the same '
            '(h¹¹, h²¹, χ) — the content-addressed id here is not a KS polytope id.'
        )
        source_url = 'http://hep.itp.tuwien.ac.at/~kreuzer/CY/'
    elif dataset_id == 'cy5-folds':
        reconstruct = (
            'CY5 / CI5F: recover a complete-intersection fivefold configuration '
            'compatible with (h¹¹, h²¹, h³¹). Configuration matrices are not '
            'stored on this page yet.'
        )
        source_url = None
    elif dataset_id == 'heterotic':
        reconstruct = (
            'Heterotic: Hodge balance is only a necessary filter. A full model '
            'needs a stable holomorphic vector bundle (or monad) on a concrete '
            'geometry — not present in stored fields.'
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
        'Unavailable: unique polytope vertices / triangulation unless stored in raw.'
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
        'workplan': workplan,
        'honesty': honesty,
    }


def build_tabs(
    dossier: Dict[str, Any],
    *,
    construction: Optional[Dict[str, Any]] = None,
    tags: Optional[List] = None,
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
            'total_moduli': int(s['total_moduli']),
            'moduli_compactness': s['moduli_compactness'],
            'hodge_balance': s['hodge_balance'],
        },
        'mirror': {
            'h11': int(s['mirror_h11']),
            'h21': int(s['mirror_h21']),
            'euler': int(s['mirror_euler']),
            'note': 'Mirror swaps h¹¹ ↔ h²¹ and sends χ → −χ at the Hodge level.',
        },
        'stabilization': stab,
        'meaning': [
            'h¹¹ counts independent Kähler (size) deformations of even cycles.',
            'h²¹ counts complex-structure (shape) deformations.',
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
        },
        'vacua_estimate': vacua,
        'constraints': [
            {
                'name': 'D3 tadpole (IIB sketch)',
                'tex': r'N_{D3} + N_{\mathrm{flux}} = L = |\chi|/24',
                'status': 'exact_budget',
                'detail': f"L = {s['tadpole_L']}; headroom proxy = {s['tadpole_headroom']}",
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
        ],
        'requires_for_full_scan': [
            'Periods / prepotential on the complex-structure moduli space',
            'Orientifold involution and O3/O7 charges',
            'Quantized flux lattice and tadpole cancellation with mobile D3s',
            'A concrete triangulation / hypersurface (see Construction)',
        ],
    }

    pheno = {
        'id': 'phenomenology',
        'title': 'Phenomenology',
        'honesty': (
            'Topological indices and necessary-condition proxies from Hodge data. '
            'Generation index |χ|/2 is exact as a topological formula; matching '
            'the Standard Model still needs bundles/fluxes. No soft spectra.'
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
        ],
        'not_computed': [
            'Standard Model gauge embeddings',
            'Soft SUSY-breaking spectra',
            'Yukawa couplings / fermion masses',
            'Full flux vacuum enumeration over a concrete lattice',
        ],
    }

    # Extra certificate checks derived here (keep dossier.checks intact + append).
    extra_checks = [
        {
            'id': 'generation_index',
            'label': 'Generation index defined',
            'rule': 'n_gen = |χ|/2 ≥ 0',
            'ok': gens['n_generations'] >= 0,
            'detail': f"n_gen={gens['n_generations']}",
        },
        {
            'id': 'euler_even_cy3',
            'label': 'Euler parity (CY3)',
            'rule': 'χ even for CY3 Hodge identity χ=2(h¹¹−h²¹)',
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
        'checks': all_checks,
        'identities': dossier['identities'] + [
            {
                'name': 'Net generation index',
                'tex': r'n_{\mathrm{gen}} = |\chi|/2',
                'value': gens['n_generations'],
            },
            {
                'name': 'Flux vacua log-count (Stirling proxy)',
                'tex': r'\log N_{\mathrm{flux}} \sim 2K\log(2\pi L)-\log K!',
                'value': vacua['log_N_flux'],
            },
        ],
        'euler_consistent': dossier['euler_consistent'],
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
        },
    }

    return {
        'ok': True,
        'dataset_id': dataset_id,
        'overview': overview,
        'moduli': moduli,
        'fluxes': fluxes,
        'phenomenology': pheno,
        'construction': construction or {
            'honesty': 'No construction context supplied.',
            'unavailable': [],
            'present_geometry': {},
            'raw_keys': [],
            'feature_map': {},
            'tags': [],
            'reconstruct_howto': '',
            'curated': None,
            'workplan': [],
        },
        'certificates': certificates,
    }
