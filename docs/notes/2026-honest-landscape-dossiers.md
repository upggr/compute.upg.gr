# Honest ML, topological no-go certificates, and open Calabi–Yau dossiers

**Draft note** (not an arXiv upload). Intended length ~2–3 pages.  
**Author:** Ioannis Kokkinis / [upg.gr](https://upg.gr)  
**Contact:** ioannis.kokkinis@upg.gr  
**Live site:** [https://compute.upg.gr](https://compute.upg.gr)  
**Date:** 2026-08-09

---

## Abstract

We describe a small, public research tool that ranks synthetic Calabi–Yau-like Hodge targets, attaches shareable **dossiers** (topological identities, necessary-condition certificates, literature model cards), and refuses common overclaims: label leakage in ML marketing, invented period integrals, and spectra without citations. The product surface is the Hall of Fame and permalinked candidate pages on [compute.upg.gr](https://compute.upg.gr). This note is a citable draft for linking; a formal arXiv submission may follow after feedback.

## 1. The problem: label leakage and overclaim

String-phenomenology tooling often mixes three layers that should stay separate:

1. **Search / ranking** over large combinatorial spaces (e.g. Kreuzer–Skarke-inspired Hodge draws).
2. **Geometry backends** (PALP, CYTools, …) that compute triangulations, intersections, periods.
3. **Phenomenology** (flux vacua, bundles, soft terms) that needs vevs, branes, and literature-backed spectra.

Marketing shortcuts are easy: report retrieval lift when the target label (e.g. \(|\chi|\)) was in the feature set; call a Hodge pair a “vacuum”; ship numerical periods that were never computed; quote soft spectra without a reference. An honest stack should **hold out** target-defining labels from the model, state what “Verified” means (dataset target-rule match on synthetic draws — **not** experimental physics), and mark proxies vs exact identities vs unavailable geometry.

## 2. What we built

**upg-strings** on [compute.upg.gr](https://compute.upg.gr) provides:

- **Synthetic retrieval demos** with leakage hold-out and baseline comparisons (API honesty fields).
- A persistent **Hall of Fame** of content-addressed candidates.
- Per-candidate **dossiers**: Hodge diamond, χ / tadpole / generation identities, flux proxies, construction metadata, topological **certificates**, and a **Model-building** tab with necessary-condition exclusions plus literature **model cards** (spectra only when cited).
- An offline **geometry pipeline** stage ladder: vertices → triangulation / ambient → intersections → periods. The web container does **not** run CYTools; stages are filled by seeded packs or offline workers. Literature combinatorial numbers (e.g. quintic \(\kappa(H^3)=5\)) and CdOGP Picard–Fuchs formulas are labeled as such — **not** invented numerical period engines.

**Flagship showcase:** the textbook quintic class \((h^{1,1},h^{2,1})=(1,101)\), permalink  
[https://compute.upg.gr/candidate/kreuzer-skarke-66d611d18a9d](https://compute.upg.gr/candidate/kreuzer-skarke-66d611d18a9d).

## 3. How to cite a candidate

Each dossier has a stable URL `https://compute.upg.gr/candidate/<id>` and a one-click **Cite** panel:

- **BibTeX** `@misc` entry (tool / dataset output, not a journal article).
- **Plain citation** and **Markdown** link snippets.
- Deep links: `#certificates`, `#model-building`.

JSON-LD on the page uses Schema.org `Dataset` with an honest description: software tool output / Hall-of-Fame entry, draft creative status — **not** a peer-reviewed paper.

Suggested wording when linking:

> Topological dossier for Hodge \((h^{1,1},h^{2,1})\) on upg-strings; “Verified” means the dataset target rule passed on synthetic labels.

## 4. Limitations

- Demo corpora are **synthetic Hodge draws** inspired by published statistics, not a live crawl of the full KS census.
- Hodge numbers do **not** uniquely fix a polytope or vacuum.
- Exclusion certificates are **necessary** conditions under stated assumptions — never sufficiency.
- Model cards quote spectra **only** from cited literature.
- Periods beyond published closed forms (CdOGP special points / PF operator) are **not** computed on the site.
- Geometry stages beyond curated pack data await offline workers.

## 5. Live links

| Resource | URL |
|----------|-----|
| Home | https://compute.upg.gr |
| Hall of Fame | https://compute.upg.gr/candidates.html |
| Quintic showcase | https://compute.upg.gr/candidate/kreuzer-skarke-66d611d18a9d |
| Geometry lookup | https://compute.upg.gr/lookup.html |
| Exclusions API | https://compute.upg.gr/api/exclusions?dataset_id=kreuzer-skarke&h11=1&h21=101 |
| Model cards API | https://compute.upg.gr/api/model-cards?dataset_id=kreuzer-skarke&h11=1&h21=101 |
| This note (HTML) | https://compute.upg.gr/note/honest-landscape |
| This note (Markdown) | https://compute.upg.gr/note/honest-landscape.md |
| About | https://compute.upg.gr/about.html |

## Acknowledgments / status

Draft for community feedback (string phenomenology lists, GitHub, social). Not yet submitted to arXiv. Please cite the live dossier URL when referring to a specific Hodge class on the site.

---

*Ioannis Kokkinis · upg.gr · 2026*
