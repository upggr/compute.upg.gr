# Model-building Phase 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ship topological exclusion certificates, literature model cards, geometry pipeline stages, and a Model-building tab — without inventing spectra or periods.

**Architecture:** Hybrid — compute exclusions in-process; seed `data/model_cards.json`; extend geometry records with `stage` / `intersections`; APIs + candidate tab; offline worker accepts richer dumps later.

**Tech Stack:** Flask, SQLite (`geometry_store` + optional model card ingest), Jinja `candidate.html`, pytest.

**Spec:** `docs/superpowers/specs/2026-08-09-model-building-design.md`

---

## File map

| File | Responsibility |
|------|----------------|
| `model_exclusions.py` | Pure functions → list of exclusion certificate dicts |
| `model_cards.py` | Load/ingest/lookup literature cards from JSON → optional sqlite or in-memory |
| `data/model_cards.json` | Seed cards (quintic/IIB refs, heterotic 3-gen, F-theory seeds) |
| `geometry_store.py` | Add `stage`, `intersections` columns / JSON in `extra`; helpers |
| `physics_dossier.py` / `build_tabs` | Wire `model_building` tab payload |
| `templates/candidate.html` | Model-building tab UI |
| `app.py` | `/api/exclusions`, `/api/model-cards` |
| `scripts/geometry_worker_stub.py` | Accept stage/intersections/periods in upsert JSON |
| `tests/test_model_building.py` | Exclusions correctness + API + cards |
| `Dockerfile` | COPY new modules |

---

### Task 1: Exclusion engine (TDD)

**Files:** `tests/test_model_building.py`, `model_exclusions.py`

1. Write failing tests:
   - Heterotic \(|\chi|=6\) → does **not** rule out standard-embedding 3-gen; \(|\chi|=8\) → **rules out** under that assumption
   - Tadpole \(L=|\chi|/24\) exclusion when \(L <\) stated minimum
   - Every result has `assumptions`, `rules_out`, `ok`
2. Implement `model_exclusions.evaluate(dataset_id, h11, h21, h31=None, euler=None) -> list[dict]`
3. Run pytest until green
4. Commit

### Task 2: Model cards seed + loader

**Files:** `data/model_cards.json`, `model_cards.py`, tests

1. Seed ≥3 cards with real arXiv/URLs (CdOGP quintic periods literature; heterotic 3-gen reviews; F-theory elliptic seed refs already on site)
2. `lookup(dataset_id, h11, h21, h31=None)`, `list_for_hodge(...)`
3. Tests for load + match
4. Commit

### Task 3: Geometry stage fields

**Files:** `geometry_store.py`, ingest, worker stub, tests

1. Persist `stage` (default from data richness: vertices→`vertices`, triangulation text→`triangulated`, intersections JSON→`intersections`, periods JSON→`periods`)
2. Store `intersections` JSON column or inside `extra`
3. Worker stub documents fields
4. Commit

### Task 4: Wire dossier tab + APIs + UI

**Files:** `physics_dossier.py`, `app.py`, `templates/candidate.html`, `Dockerfile`, CSS lightly

1. `tabs.model_building` = exclusions + cards + geometry stage summary + honesty banner
2. Routes: `GET /api/exclusions`, `GET /api/model-cards`
3. New tab button on candidate page
4. Commit

### Task 5: Ship

1. Full pytest subset
2. Push main; Coolify poll app id 8; verify live quintic + heterotic χ=6 pages show Model-building tab
3. Commit README one-liner under roadmap

---

## Parallel note

Other agents may be editing `app.py` / templates for honesty/security. Rebase carefully; prefer additive tab + new modules.
