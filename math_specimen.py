"""Mathematics-tab payload: citeable geometric specimen framing.

Honest presentation for geometers — Hodge class ± construction metadata,
mirror swap, period/GW handoff notes. Never invents intersection numbers,
curve counts, or numerical periods.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import geometry_store


SPECIMEN_FRAMING = (
    'This page is a citeable geometric specimen (Hodge class ± construction data), '
    'not a proof of string theory.'
)


def _truncate(value: Any, limit: int = 48) -> Any:
    """Keep toric payloads displayable without dumping huge matrices twice."""
    if isinstance(value, list) and len(value) > limit:
        return {
            'preview': value[:limit],
            'truncated': True,
            'total_rows': len(value),
            'note': f'Showing first {limit} of {len(value)} rows; use geometry JSON export for full data.',
        }
    return value


def _toric_status(construction: Optional[Dict[str, Any]], pg: Dict[str, Any]) -> str:
    gdb = (construction or {}).get('geometry_db') or {}
    status = gdb.get('status') or pg.get('geometry_status')
    if status in ('representative', 'curated', 'seeded', 'literature'):
        return str(status)
    if (construction or {}).get('curated'):
        return 'curated'
    if geometry_store._has_vertices(pg):
        return 'representative'
    return 'absent'


def evaluate_specimen_match(
    *,
    construction: Optional[Dict[str, Any]] = None,
    mirror_partner: Optional[Dict[str, Any]] = None,
    periods_literature: Optional[Dict[str, Any]] = None,
    tags: Optional[List] = None,
    has_vertices: bool = False,
    toric_status: str = 'absent',
    periods_status: str = 'pending',
) -> Dict[str, Any]:
    """Decide whether this candidate is a showcaseable math specimen.

    FOUND means we have *extra* geometric / literature substance beyond a bare
    Hodge triple.
    """
    construction = construction or {}
    tags_l = [str(t).lower() for t in (tags or [])]
    curated = construction.get('curated') or {}
    partner = mirror_partner or {}
    reasons: List[str] = []

    if curated or 'textbook' in tags_l or 'curated' in tags_l:
        name = curated.get('name') or curated.get('geometry_name') or 'textbook / curated class'
        reasons.append(f'Named construction: {name}')
    if has_vertices or toric_status in ('curated', 'representative', 'seeded', 'literature'):
        reasons.append(
            f'Toric / polytope data present (status={toric_status or "vertices"})'
        )
    if periods_literature or periods_status == 'literature_curated':
        reasons.append('Literature Picard–Fuchs / periods notes (e.g. CdOGP)')
    if periods_status == 'stored':
        reasons.append('Offline periods payload stored in geometry DB')
    if partner.get('on_board') and partner.get('candidate_id'):
        reasons.append(
            f'Mirror partner on Hall of Fame ({partner.get("candidate_id")})'
        )
    if construction.get('showcase'):
        reasons.append('Flagship showcase entry')

    found = bool(reasons)
    if found:
        headline = 'SPECIMEN FOUND'
        blurb = (
            'This Hodge class matches showcase conditions — citeable geometric '
            'substance beyond bare invariants.'
        )
    else:
        headline = 'Specimen pending'
        blurb = (
            'Bare Hodge specimen only. Add curated construction, polytope vertices, '
            'literature periods, or a mirror on the board to unlock SPECIMEN FOUND.'
        )
    return {
        'found': found,
        'headline': headline,
        'blurb': blurb,
        'reasons': reasons,
        'tab_badge': 'FOUND' if found else None,
    }


def mathematics_payload(
    dossier: Dict[str, Any],
    *,
    construction: Optional[Dict[str, Any]] = None,
    mirror_partner: Optional[Dict[str, Any]] = None,
    mirror_block: Optional[Dict[str, Any]] = None,
    periods: Optional[Dict[str, Any]] = None,
    periods_full: Optional[Dict[str, Any]] = None,
    periods_literature: Optional[Dict[str, Any]] = None,
    external_links: Optional[List[Dict[str, str]]] = None,
    pipeline_stage: Optional[str] = None,
    pipeline_note: Optional[str] = None,
    pipeline_checklist: Optional[List[Dict[str, Any]]] = None,
    candidate_id: Optional[str] = None,
    tags: Optional[List] = None,
) -> Dict[str, Any]:
    """Build the Mathematics analysis tab for a candidate dossier."""
    h11 = int(dossier['h11'])
    h21 = int(dossier['h21'])
    h31 = dossier.get('h31')
    euler = int(dossier['euler_char'])
    dataset_id = dossier.get('dataset_id') or 'kreuzer-skarke'
    periods = periods or {}
    periods_full = periods_full or {}
    construction = construction or {}
    pg = dict(construction.get('present_geometry') or {})
    gdb = construction.get('geometry_db') or {}
    curated = construction.get('curated') or {}

    b3 = periods.get('h3_betti')
    if b3 is None and dataset_id != 'cy5-folds':
        b3 = 2 * (h21 + 1)
    pf_order = periods.get('picard_fuchs_order')
    if pf_order is None and dataset_id != 'cy5-folds':
        pf_order = h21 + 1

    mirror = dict(mirror_block or {})
    if not mirror:
        mirror = {
            'h11': h21,
            'h21': h11,
            'euler': -euler,
            'partner': mirror_partner,
            'note': 'Mirror swaps h¹¹ ↔ h²¹ and sends χ → −χ at the Hodge level.',
        }
    elif mirror_partner is not None and 'partner' not in mirror:
        mirror['partner'] = mirror_partner

    partner = mirror.get('partner') or mirror_partner
    partner_on_board = bool(partner and partner.get('on_board') and partner.get('candidate_id'))
    if partner_on_board:
        mirror_level = 'hof_partner'
        mirror_honesty = (
            'Hodge-level mirror swap with a Hall-of-Fame partner permalink. '
            'This is not a constructed mirror map of periods or Gromov–Witten invariants.'
        )
    else:
        mirror_level = 'hodge_swap_only'
        mirror_honesty = (
            'Only the Hodge-level swap (h¹¹,h²¹,χ) ↔ (h²¹,h¹¹,−χ) is known here. '
            'No explicit Greene–Plesser / toric mirror construction is claimed unless '
            'curated construction notes say so.'
        )

    has_vertices = geometry_store._has_vertices(pg)
    weight_system = (
        pg.get('weight_system')
        or curated.get('weight_system')
        or gdb.get('weight_system')
    )
    config_matrix = (
        pg.get('configuration_matrix')
        or curated.get('configuration_matrix')
        or gdb.get('configuration_matrix')
    )
    vertices = pg.get('polytope_vertices') or pg.get('vertex_matrix')
    toric_status = _toric_status(construction, pg)
    toric_links = [
        link for link in (external_links or [])
        if 'kreuzer' in (link.get('label') or '').lower()
        or 'cytools' in (link.get('label') or '').lower()
        or 'tuwien' in (link.get('url') or '').lower()
    ]
    if not toric_links and external_links:
        toric_links = list(external_links[:3])

    lit = periods_literature
    if lit:
        periods_status = 'literature_curated'
        enumerative_summary = (
            'CdOGP / literature Picard–Fuchs structure is curated for this Hodge class. '
            'Numerical period integrals and Gromov–Witten numbers are not computed on this site.'
        )
        periods_handoff = (
            'Literature PF formulas available; numerical periods / GW still hand off to '
            'CYTools, Sage, or published tables.'
        )
    elif geometry_store._has_periods(pg):
        periods_status = 'stored'
        enumerative_summary = (
            'A periods payload is stored in the geometry DB. Treat it as offline worker '
            'output — this tab does not recompute period integrals.'
        )
        periods_handoff = 'Inspect stored periods via geometry JSON; verify offline.'
    else:
        periods_status = 'pending'
        enumerative_summary = (
            'Periods pending — hand off to CYTools/Sage. Pipeline stage reflects stored '
            'geometry only; no invented curve counts or Yukawa numbers.'
        )
        periods_handoff = 'Periods pending — hand off to CYTools/Sage.'

    stage = pipeline_stage
    if stage is None:
        stage = geometry_store.infer_stage({
            'polytope_vertices': pg.get('polytope_vertices'),
            'vertex_matrix': pg.get('vertex_matrix'),
            'triangulation': pg.get('triangulation'),
            'intersections': pg.get('intersections') or gdb.get('intersections'),
            'periods': pg.get('periods'),
        })
    pnote = pipeline_note or geometry_store.pipeline_note(stage)

    cid = candidate_id or construction.get('candidate_id') or 'candidate'
    canonical = f'https://compute.upg.gr/candidate/{cid}'
    hodge_label = f'(h^{{1,1}},h^{{2,1}})=({h11},{h21})'
    if h31 is not None:
        hodge_label = f'(h^{{1,1}},h^{{2,1}},h^{{3,1}})=({h11},{h21},{h31})'
    bib_key = 'upgstrings_math_' + ''.join(
        ch if ch.isalnum() else '_' for ch in cid
    )
    cite = {
        'framing': SPECIMEN_FRAMING,
        'canonical_url': canonical,
        'deep_links': {
            'mathematics': f'{canonical}#mathematics',
            'certificates': f'{canonical}#certificates',
        },
        'markdown': (
            f'[Geometric specimen {hodge_label} — upg-strings]({canonical}#mathematics)'
        ),
        'markdown_deep': (
            f'[Mathematics]({canonical}#mathematics) · '
            f'[Certificates]({canonical}#certificates)'
        ),
        'bibtex': (
            f'@misc{{{bib_key},\n'
            f'  title = {{upg-strings geometric specimen: {hodge_label}}},\n'
            f'  author = {{Kokkinis, Ioannis}},\n'
            f'  howpublished = {{\\url{{{canonical}}}}},\n'
            f'  note = {{Citeable Hodge class ± construction metadata; χ={euler}}},\n'
            f'  year = {{2026}}\n'
            f'}}'
        ),
        'plain': (
            f'Ioannis Kokkinis, upg-strings geometric specimen {hodge_label} '
            f'(χ={euler}). {canonical}#mathematics. '
            f'Topological / combinatorial specimen.'
        ),
    }

    has_ix = geometry_store._has_intersections(pg) or bool(
        (pg.get('intersections') or gdb.get('intersections'))
    )
    match = evaluate_specimen_match(
        construction=construction,
        mirror_partner=partner if isinstance(partner, dict) else mirror_partner,
        periods_literature=lit,
        tags=tags if tags is not None else construction.get('tags'),
        has_vertices=has_vertices,
        toric_status=toric_status,
        periods_status=periods_status,
    )
    handoff = [
        {
            'id': 'download_geometry',
            'label': 'Download / export geometry JSON',
            'detail': (
                f'GET /api/geometry/{cid} or /api/geometry/lookup for this Hodge class; '
                f'analysis bundle at /api/analysis/{cid}/bundle.'
            ),
            'ready': True,
        },
        {
            'id': 'open_cytools',
            'label': 'Open in CYTools / PALP',
            'detail': (
                'Import vertices / weight system when present; triangulate favourable '
                'hypersurface and compute intersection ring offline.'
                if has_vertices or weight_system
                else 'No vertices/weights stored — look up KS polytope externally first.'
            ),
            'ready': bool(has_vertices or weight_system),
        },
        {
            'id': 'macaulay2',
            'label': 'Macaulay2 / Sage algebraic checks',
            'detail': (
                'Use ambient / configuration matrix when present; otherwise reconstruct '
                'from KS reflexive polytope data.'
            ),
            'ready': bool(config_matrix or curated.get('ambient') or pg.get('ambient')),
        },
        {
            'id': 'periods_gw',
            'label': 'Periods / Gromov–Witten handoff',
            'detail': periods_handoff,
            'ready': periods_status in ('literature_curated', 'stored'),
        },
        {
            'id': 'missing',
            'label': 'Still missing on this site',
            'detail': '; '.join(
                part for part, ok in (
                    ('unique polytope (Hodge non-unique)', not has_vertices),
                    ('triangulation', not geometry_store._has_triangulation(pg)),
                    ('intersection numbers', not has_ix),
                    ('numerical periods', periods_status == 'pending'),
                )
                if ok
            ) or 'Core Hodge specimen is present; richer geometry may still be representative-only.',
            'ready': False,
        },
    ]

    return {
        'id': 'mathematics',
        'title': 'Mathematics',
        'found': match['found'],
        'found_badge': match['tab_badge'],
        'match': match,
        'honesty': (
            'Pure-mathematics framing of stored Hodge data and construction metadata. '
            'No invented theorems, intersection numbers, or curve counts.'
        ),
        'specimen': {
            'headline': match['headline'],
            'framing': SPECIMEN_FRAMING,
            'found': match['found'],
            'reasons': match['reasons'],
            'blurb': match['blurb'],
            'detail': (
                f'{hodge_label}, χ={euler}, dataset={dataset_id}. '
                'Share the permalink as a topological / combinatorial specimen; '
                'certificates tab lists machine-checkable identities.'
            ),
        },
        'invariants': {
            'h11': h11,
            'h21': h21,
            'h31': h31,
            'euler_char': euler,
            'b3': b3,
            'picard_fuchs_order': pf_order,
            'period_count_proxy': pf_order,
            'mirror_h11': mirror.get('h11'),
            'mirror_h21': mirror.get('h21'),
            'mirror_euler': mirror.get('euler'),
            'mirror_partner': partner,
        },
        'mirror_symmetry': {
            'level': mirror_level,
            'honesty': mirror_honesty,
            'explanation': (
                'Mirror symmetry exchanges Kähler and complex-structure deformations at '
                'the level of Hodge numbers for Calabi–Yau threefolds. A full mirror map '
                'identifies period data / quantum-corrected Kähler geometry with the '
                'complex-structure VHS of the mirror — that map is not computed here.'
            ),
            'formulas': [
                {
                    'name': 'Hodge mirror swap',
                    'tex': r'(h^{1,1},h^{2,1})\leftrightarrow(h^{2,1},h^{1,1})',
                },
                {
                    'name': 'Euler sign flip',
                    'tex': r'\chi(X^\circ)=-\chi(X)',
                },
                {
                    'name': 'This specimen',
                    'tex': (
                        rf'({h11},{h21},\chi={euler})'
                        rf'\leftrightarrow({mirror.get("h11")},{mirror.get("h21")},'
                        rf'\chi={mirror.get("euler")})'
                    ),
                },
            ],
            'partner': partner,
            'note': mirror.get('note'),
        },
        'enumerative': {
            'summary': enumerative_summary,
            'periods_status': periods_status,
            'handoff': periods_handoff,
            'pipeline_stage': stage,
            'pipeline_note': pnote,
            'checklist': pipeline_checklist or [],
            'periods_literature': lit,
            'periods_full': {
                'status': periods_full.get('status'),
                'note': periods_full.get('note'),
                'picard_fuchs_order': periods_full.get('picard_fuchs_order') or pf_order,
                'b3': periods_full.get('b3') or b3,
                'mirror_family_pf_tex': periods_full.get('mirror_family_pf_tex'),
                'special_points': periods_full.get('special_points'),
                'references': periods_full.get('references'),
                'period_vector_tex': periods_full.get('period_vector_tex'),
                'pf_operator_tex': periods_full.get('pf_operator_tex'),
            } if periods_full else None,
            'gw_note': (
                'Gromov–Witten / enumerative invariants would use the intersection ring '
                'and (quantum-corrected) periods. This site does not enumerate curves.'
            ),
        },
        'toric': {
            'status': toric_status,
            'has_vertices': has_vertices,
            'vertices': _truncate(vertices) if has_vertices else None,
            'weight_system': weight_system,
            'configuration_matrix': _truncate(config_matrix) if config_matrix else None,
            'ambient': pg.get('ambient') or curated.get('ambient'),
            'hypersurface_equation': (
                pg.get('hypersurface_equation') or curated.get('hypersurface_equation')
            ),
            'geometry_source': (
                gdb.get('source') or pg.get('geometry_source') or curated.get('name')
            ),
            'note': (
                'Toric / combinatorial data are classification input (KS reflexive '
                'polytopes, weights, configuration matrices). Status '
                f'“{toric_status}” means Hodge numbers alone do not uniquify a polytope '
                'unless a curated construction says otherwise.'
                if toric_status != 'absent'
                else (
                    'No vertices, weight system, or configuration matrix stored for this '
                    'entry. Use Kreuzer–Skarke / external catalogs with the Hodge key.'
                )
            ),
            'external_links': toric_links,
        },
        'hodge_theory': {
            'blurb': (
                'For a Calabi–Yau threefold the intermediate Jacobian / Hodge filtration '
                'on H³ gives a variation of Hodge structure (VHS) over the complex-structure '
                'moduli space. Exact moduli dimensions follow from Hodge numbers alone.'
            ),
            'kahler_moduli_dim': h11,
            'complex_structure_moduli_dim': h21,
            'h31': h31,
            'formulas': [
                {
                    'name': 'Kähler moduli dimension',
                    'tex': rf'\dim\mathcal{{M}}_{{K}} = h^{{1,1}} = {h11}',
                },
                {
                    'name': 'Complex-structure moduli dimension',
                    'tex': rf'\dim\mathcal{{M}}_{{CS}} = h^{{2,1}} = {h21}',
                },
                {
                    'name': 'Middle cohomology (CY3)',
                    'tex': rf'b_3 = 2(h^{{2,1}}+1) = {b3}'
                    if dataset_id != 'cy5-folds' and b3 is not None
                    else r'b_3\ \text{from full Hodge diamond (CY5)}',
                },
            ],
            'vhs_note': (
                f'Period domain / PF problem size proxy K = h²¹+1 = {pf_order} '
                '(symbolic; not a solved VHS).'
                if pf_order is not None
                else 'VHS data beyond Hodge numbers are not stored here.'
            ),
        },
        'cite': cite,
        'handoff': handoff,
    }
