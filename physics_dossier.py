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


def construction_payload(
    dataset_id: str,
    candidate_id: str,
    raw: Optional[Dict[str, Any]] = None,
    features: Optional[List] = None,
    tags: Optional[List] = None,
    summary: Optional[str] = None,
) -> Dict[str, Any]:
    """Honest construction metadata: show what we have, never invent equations."""
    raw = raw or {}
    feature_map = {}
    for item in features or []:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            feature_map[str(item[0])] = item[1]

    known_geometry_keys = (
        'polytope_id', 'polytope_hash', 'polytope_vertices', 'vertex_matrix',
        'triangulation', 'triangulation_id', 'hypersurface_equation',
        'weight_system', 'favourable', 'ambient',
    )
    present_geometry = {
        k: raw[k] for k in known_geometry_keys if k in raw and raw[k] is not None
    }

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
    # Drop unavailable entries that we unexpectedly do have.
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
        'honesty': (
            'Exact: stored metadata keys and Hodge invariants. '
            'Unavailable: hypersurface equations and polytope vertices '
            '(unless listed under present geometry).'
        ),
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

    moduli = {
        'id': 'moduli',
        'title': 'Moduli',
        'honesty': (
            'Exact counts from Hodge numbers for a CY3: dim of Kähler moduli '
            'space is h¹¹; complex-structure moduli space dimension is h²¹ '
            '(before quotienting by discrete symmetries / flux stabilization). '
            'These are dimension proxies, not a solved moduli potential.'
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
        'meaning': [
            'h¹¹ counts independent Kähler (size) deformations of even cycles.',
            'h²¹ counts complex-structure (shape) deformations.',
            'Stabilization in flux compactifications typically needs enough '
            'fluxes relative to these counts — see the Fluxes tab.',
        ],
    }

    fluxes = {
        'id': 'fluxes',
        'title': 'Fluxes / tadpole',
        'honesty': (
            'Exact: tadpole budget L = |χ|/24 from the Euler characteristic. '
            'Proxy: Bousso–Polchinski-inspired log flux-density scalar used in '
            'ranking. Not a vacuum scan.'
        ),
        'budget': {
            'euler_char': euler,
            'tadpole_L': s['tadpole_L'],
            'tadpole_headroom': s['tadpole_headroom'],
            'flux_density_proxy': s['flux_density'],
            'K_proxy': h21 + 1,
        },
        'constraints': [
            {
                'name': 'D3 tadpole (IIB sketch)',
                'tex': r'N_{D3} + N_{\mathrm{flux}} = L = |\chi|/24',
                'status': 'exact_budget',
            },
            {
                'name': 'Flux vacua log-density proxy',
                'tex': r'\log N_{\mathrm{flux}} \sim 2K\log(2\pi L) - \log K!',
                'status': 'proxy',
                'detail': f"K = h²¹+1 = {h21 + 1}; normalized proxy = {s['flux_density']}",
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
            'Only necessary-condition proxies from Hodge data and the dataset '
            'search target. No gauge group, no soft SUSY spectrum, no Yukawas.'
        ),
        'proxies': {
            'hodge_balance': s['hodge_balance'],
            'vacuum_stability_proxy': s['vacuum_stability'],
            'topo_efficiency': s['topo_efficiency'],
            'hodge_entropy': s['hodge_entropy'],
        },
        'dataset_target': next(
            (c for c in dossier['checks'] if c['id'] == 'dataset_target'), None
        ),
        'tags': list(tags or []),
        'not_computed': [
            'Standard Model gauge embeddings',
            'Soft SUSY-breaking spectra',
            'Yukawa couplings / fermion masses',
            'Full flux vacuum enumeration',
        ],
    }

    certificates = {
        'id': 'certificates',
        'title': 'Certificates',
        'honesty': (
            'Machine-checkable identities and necessary conditions on the '
            'stored invariants. These are certificates, not uniqueness theorems '
            'or existence proofs for string vacua.'
        ),
        'checks': dossier['checks'],
        'identities': dossier['identities'],
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
        },
        'certificates': certificates,
    }
