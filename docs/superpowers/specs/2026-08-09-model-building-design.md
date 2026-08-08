# Model-building layer (honest) — design

**Date:** 2026-08-09  
**Site:** compute.upg.gr  
**Status:** Approved to implement (user: start model-building contributions)

## Goal

Contribute to **string model-building workflows** (not proofs of string theory) by:

1. Real geometry stages for selected Hall of Fame hits: vertices → triangulation → intersections / periods  
2. Concrete flux / heterotic / F-theory **model cards** tied to published literature (spectra only when cited)  
3. Clear **negative results**: topological necessary conditions that rule out classes of models under stated assumptions  

## Non-goals

- Proving string theory  
- Inventing soft spectra, Yukawas, or unique polytopes from Hodge numbers alone  
- Running CYTools inside the web image  

## Approaches considered

| | Approach | Pros | Cons |
|---|----------|------|------|
| A | Literature JSON model cards only | Fast, citable, zero fake physics | No new geometry depth |
| B | Full CYTools in web/CI | Real intersections/periods | Image size, ops, Coolify risk |
| **C (chosen)** | **Hybrid:** cards + exclusion certs + geometry **pipeline stages** in SQLite; CYTools offline worker fills stages | Honest now; path to real geometry | Periods empty until offline runs |

## Architecture

```
Offline CYTools/PALP host          Web (Coolify)
─────────────────────────          ─────────────────────────────
compute verts/FRST/                  geometry.sqlite (query)
  intersections/periods    ──JSON──►  model_building.sqlite
                                     (or tables in geometry DB)
literature model cards (git)  ──────► ingest on boot
topological exclusions (code) ──────► Certificates / Model-building tab
```

### Data

**Geometry pipeline** (extend `geometry_store` / records):

- `stage`: `vertices | triangulated | intersections | periods`  
- Fields already present: `vertex_matrix`, `triangulation`, `periods`, `extra`  
- New optional JSON: `intersections` (triple intersection summary when offline-provided)  
- `pipeline_note`: what is filled vs pending  

**Model cards** (`data/model_cards.json` → SQLite `model_cards`):

- `id`, `dataset_id`, `h11`, `h21`, `h31?`, optional `candidate_id`  
- `framework`: `iib-flux | heterotic | f-theory | other`  
- `title`, `reference`, `reference_url`, `arxiv`  
- `assumptions` (list)  
- `spectrum_summary` (text; only literature-backed)  
- `geometry_status`: `hodge-only | representative-polytope | full-offline`  
- `honesty`: short label  

**Exclusions** (computed in `physics_dossier` / new `model_exclusions.py`):

Machine-checkable **necessary** conditions, each with `ok`, `rules_out`, `assumptions`, `detail`. Examples:

- Heterotic net chirality \(n_{\mathrm{gen}}=|\chi|/2\): if \(|\chi|\neq 6\), rules out “standard embedding 3-generation” under that assumption  
- KS tadpole \(L=|\chi|/24\): if \(L < L_{\min}\) (stated), rules out flux+D3 models needing larger budget  
- Mirror: \(\chi=0\) self-mirror locus notes (not an exclusion, a landmark)  
- CY5: incomplete Hodge diamond → cannot claim full CY5 phenomenology  

Never claim sufficiency (“can host”) from Hodge alone — only **cannot** under Y.

### UI

New analysis tab or Construction subsection: **Model-building**

- Pipeline stage checklist for this Hodge/candidate  
- Linked model cards (literature)  
- Exclusion certificates (PASS = does not rule out; FAIL = rules out under assumptions)  

### Offline worker

Extend `scripts/geometry_worker_stub.py` + schema:

- Accept CYTools dump with triangulation, intersection numbers, period metadata  
- Upsert into geometry DB with `stage` advanced  

### API

- `GET /api/model-cards?dataset_id=&h11=&h21=`  
- `GET /api/exclusions?dataset_id=&h11=&h21=&h31=`  
- Geometry lookup already exists; include `stage` / `intersections` when present  

## Phased delivery

**Phase 1 (this sprint):** exclusions engine + model cards seed (quintic flux sketch refs, heterotic 3-gen literature pointers, F-theory seeds) + Model-building tab + APIs + tests + deploy  

**Phase 2:** Offline worker contract + 1–2 HoF textbook geometries advanced to `triangulated` if we can obtain published FRST notes; else leave pending with clear stage  

**Phase 3:** Ingest published spectra tables only with arXiv DOIs; expand cards  

## Success criteria

- Live candidate pages show exclusions that are mathematically correct under stated assumptions  
- At least 3 literature model cards with working citation links  
- Geometry records expose pipeline stage; no fake periods/spectra  
- Coolify green; pytest green  

## Honesty banner (required on UI)

“Model-building aids: topological exclusions and literature cards. Not a proof of string theory; spectra only when cited.”
