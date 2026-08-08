# Candidate detail: analysis tabs

Date: 2026-08-08  
Status: implementing (async design lock — ship incremental honest tabs)

## Intent

Move “hand off to heavier tools” analyses onto the candidate page itself, as
tabs. Deepen what Hodge data actually supports; never fake proofs, equations,
or particle spectra.

## Assumptions

- Stored fields are typically: `h11`, `h21`, optional `h31`, `euler_char`,
  `dataset_id`, `verified_target`, `score`, `features`, `raw`, `tags`.
- No polytope vertex matrices or triangulation ids are stored today.
- Auto-analysis on load and inline 3D preview must remain (no renderer CTA
  regression). Neighbor link contrast stays light on dark theme.

## Approaches considered

1. **Pure template split** — rearrange existing HTML into tabs. Fast, but no
   structured honesty labels for Construction / Phenomenology.
2. **Recommended: dossier tab payload + tabbed UI** — `physics_dossier`
   returns per-tab exact/proxy/unavailable fields; template renders tabs.
3. **Separate API per tab** — overkill for static Hodge certificates.

## Tab map

| Tab | Computes / shows | Honesty |
|-----|------------------|---------|
| Overview | Diamond, identities summary, auto-analysis | Exact identities + ranking proxies |
| Moduli | Counts: Kähler≈h¹¹, CS≈h²¹; mirror; compactness | Counts are exact for CY3 Hodge; geometric moduli spaces need more data |
| Fluxes | L=\|χ\|/24, headroom, log-density proxy, scan prerequisites | Budget exact from χ; vacua count is a proxy |
| Phenomenology | Dataset target, balance, vacuum-stability proxy, tags | Proxies only — no soft spectra |
| Construction | Dataset meta, raw keys present, reconstruction recipe | Unavailable equations labeled explicitly |
| Certificates | PASS/FAIL checks + identity list | Machine-checkable certificates, not theorems |

Landscape, student lab, and 3D preview stay as sections below the tabs.

## Extension points

- Construction tab lists remaining `unavailable` items when raw/curated data
  lack vertices or triangulations.
- Fluxes still lists `requires_for_full_scan` for periods / O3–O7 / lattice.
- Phenomenology still lists soft spectra / Yukawas / gauge embeddings as
  not_computed (need bundles/fluxes).

## Filled in (2026-08-08 deepen pass)

- Stirling `log N_flux` + scientific `N_flux` estimate on Fluxes / Overview
- Stabilization map on Moduli (CS vs Kähler vs dilaton mechanisms)
- Generation index `|χ|/2` + 3-generation check on Phenomenology / Certificates
- `c₂·J` ranking proxy
- Curated textbook constructions (quintic, mirror, bicubic) when Hodge matches
- Numbered construction workplan with invariants filled in

## Files

- `physics_dossier.py` — `build_tabs` / enrich dossier
- `templates/candidate.html` — tab UI
- `static/css/style.css` — tab styles
- `app.py` — pass construction context
- `tests/test_physics_dossier.py` — tab payload tests
