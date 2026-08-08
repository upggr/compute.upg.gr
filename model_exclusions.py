"""Topological necessary-condition exclusions for model-building certificates.

Honesty: these are *necessary* conditions under stated assumptions only.
They never claim sufficiency (“can host a model”) from Hodge numbers alone —
only that a class of models is *ruled out* when a check fails (`ok=False`).
`ok=True` means the check does **not** rule out that class under Y.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _euler(
    dataset_id: str,
    h11: int,
    h21: int,
    h31: Optional[int] = None,
    euler: Optional[int] = None,
) -> int:
    if euler is not None:
        return int(euler)
    if dataset_id == 'cy5-folds' and h31 is not None:
        return int(6 + 6 * (h11 - h21 + h31))
    return int(2 * (h11 - h21))


def evaluate(
    dataset_id: str,
    h11: int,
    h21: int,
    h31: Optional[int] = None,
    euler: Optional[int] = None,
    *,
    tadpole_L_min: Optional[float] = 1.0,
) -> List[Dict[str, Any]]:
    """Return exclusion / landmark certificates for a Hodge key.

    Parameters
    ----------
    tadpole_L_min:
        Stated minimum tadpole budget L = |χ|/24. If None, the tadpole
        exclusion certificate is omitted. Default 1.0 (one flux/D3 unit in
        the crude |χ|/24 units used elsewhere on the site).
    """
    chi = _euler(dataset_id, int(h11), int(h21), h31, euler)
    chi_abs = abs(chi)
    L = chi_abs / 24.0
    certs: List[Dict[str, Any]] = []

    # --- Heterotic standard-embedding 3-generation necessary condition -----
    n_gen = chi_abs // 2
    het_ok = chi_abs == 6
    certs.append({
        'id': 'heterotic_standard_embedding_3gen',
        'label': 'Heterotic standard-embedding 3-generation index',
        'ok': het_ok,
        'rules_out': (
            'Standard-embedding heterotic models with net chirality '
            'n_gen = |χ|/2 = 3 generations'
        ),
        'assumptions': [
            'Heterotic E₈×E₈ (or Spin(32)/ℤ₂) on a CY3',
            'Standard embedding: gauge connection = spin connection '
            '(or index equals the Euler characteristic index)',
            'Net chiral generation count n_gen = |χ|/2',
            'Target phenomenology: exactly three net generations',
        ],
        'detail': (
            f'χ={chi}, |χ|={chi_abs}, n_gen=|χ|/2={n_gen}. '
            + (
                '|χ|=6 so the 3-generation index condition is satisfied '
                '(necessary only — still needs a concrete geometry + bundle).'
                if het_ok else
                f'|χ|≠6 so standard-embedding 3-generation models are ruled out '
                f'(would give n_gen={n_gen}, not 3).'
            )
        ),
        'euler_char': chi,
        'euler_abs': chi_abs,
        'n_generations': n_gen,
        'kind': 'exclusion',
    })

    # --- IIB / KS tadpole budget -------------------------------------------
    if tadpole_L_min is not None:
        L_min = float(tadpole_L_min)
        tad_ok = L >= L_min
        certs.append({
            'id': 'ks_tadpole_budget',
            'label': 'IIB/KS D3 tadpole budget L = |χ|/24',
            'ok': tad_ok,
            'rules_out': (
                f'Flux + D3 constructions that require tadpole budget '
                f'L ≥ L_min={L_min:g}'
            ),
            'assumptions': [
                'Type IIB / O3–O7 orientifold sketch on a CY3',
                'Tadpole identity L = |χ|/24 (Euler characteristic budget)',
                f'Stated minimum budget L_min={L_min:g} for the model class under study',
                'No additional localized sources that enlarge the effective budget',
            ],
            'detail': (
                f'L=|χ|/24={L:.6g} with |χ|={chi_abs}. '
                + (
                    f'L ≥ L_min={L_min:g}: does not rule out models needing that budget.'
                    if tad_ok else
                    f'L < L_min={L_min:g}: rules out flux+D3 models that need at least '
                    f'that budget under the stated assumptions.'
                )
            ),
            'tadpole_L': round(L, 6),
            'tadpole_L_min': L_min,
            'euler_abs': chi_abs,
            'kind': 'exclusion',
        })

    # --- Self-mirror landmark (not an exclusion) ---------------------------
    if chi == 0:
        certs.append({
            'id': 'self_mirror_locus',
            'label': 'Self-mirror Hodge locus (χ=0)',
            'ok': True,
            'rules_out': (
                'Nothing ruled out — landmark note only '
                '(χ=0 self-mirror Hodge numbers)'
            ),
            'assumptions': [
                'CY3 Hodge identity χ = 2(h^{1,1} − h^{2,1})',
                'Self-mirror at the Hodge level means h^{1,1}=h^{2,1}',
            ],
            'detail': (
                f'h11={h11}, h21={h21} ⇒ χ=0. Landmark for the self-mirror locus; '
                'not an exclusion. Many distinct geometries can share these Hodge numbers.'
            ),
            'euler_char': 0,
            'kind': 'landmark',
        })

    # --- CY5 incomplete phenomenology claim --------------------------------
    if dataset_id == 'cy5-folds':
        # Full CY5 phenomenology needs the complete Hodge diamond / more than
        # (h11,h21[,h31]) sketches we expose here.
        incomplete = h31 is None
        cy5_ok = not incomplete
        certs.append({
            'id': 'cy5_full_phenomenology_claim',
            'label': 'CY5 full-phenomenology claim from partial Hodge data',
            'ok': cy5_ok,
            'rules_out': (
                'Claims of complete CY5 phenomenology derived from the '
                'partial Hodge data available on this site alone'
            ),
            'assumptions': [
                'Calabi–Yau fivefold compactification',
                'Full phenomenology requires the complete Hodge diamond, '
                'configuration data, and fluxes/branes — not Hodge triples alone',
            ],
            'detail': (
                (
                    f'h31={h31} present; still only a necessary Hodge sketch — '
                    'this check does not certify a vacuum, but does not rule out '
                    'further offline model-building from a complete diamond.'
                )
                if cy5_ok else
                'h31 missing: incomplete Hodge diamond on this page → cannot claim '
                'full CY5 phenomenology from the stored invariants alone.'
            ),
            'h31': h31,
            'kind': 'exclusion',
        })

    return certs
