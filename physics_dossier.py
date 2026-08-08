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
