# Candidate detail: topological certificate + landscape + student lab

Date: 2026-08-08  
Status: implementing (user: do all three)

## Scope

Every `/candidate/<id>` page includes:

1. **Scientist dossier** — Hodge diamond, identities (χ, tadpole, mirror, flux proxy), necessary-condition PASS/FAIL checklist, physics proxies, on-demand analysis + BibTeX
2. **Landscape probe** — (h11,h21) scatter vs Hall of Fame neighbors + nearest links
3. **Student lab** — ELI5 blurbs, derive-χ widget, quiz

## Honesty

Page states Hodge numbers do not uniquely determine a manifold. Proxies are not vacuum proofs.

## Files

- `physics_dossier.py` — pure compute
- `templates/candidate.html` — UI
- `app.py` — wire dossier + neighbors into page / analysis
- `Dockerfile` — COPY `physics_dossier.py`
