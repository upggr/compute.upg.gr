# Museum-glass 3D polish Implementation Plan

> **For agentic workers:** Execute inline; user requested no approval gates.

**Goal:** Dark plinth + metallic specular look for CY manifold WebGL, shared across render/candidates/results.

**Architecture:** Centralize look in `static/js/cy-shape.js` (`configureScene`, `materialFor`, vertex colors). Pages only build mesh + animate.

**Tech Stack:** three@0.160, vanilla JS templates, Playwright `grid.js` for visual verify.

## Global Constraints

- No HDR env maps, bloom, or EffectComposer
- Keep invariant→shape/hue mapping
- Dark clear confined to WebGL canvas

---

### Task 1: Polish `cy-shape.js`

**Files:** Modify `static/js/cy-shape.js`

- [x] Lift palette lightness; metalness 0.55–0.85; roughness 0.18–0.40
- [x] Add `configureScene(THREE, { renderer, scene })` — clear `#0c1016`, warm key, cool rim, soft fill
- [x] Export `configureScene`

### Task 2: Wire pages

**Files:** `templates/render.html`, `templates/candidates.html`, `templates/results.html`, `static/css/style.css`

- [x] `/render`: use `configureScene`; opaque canvas; drop bobbing; keep fit framing
- [x] candidates/results: replace modulo params with `CYShape.paramsFor` + `geometryFor` + `materialFor` + `configureScene`; fixed rotation; auto-fit
- [x] Dark shell CSS for `.renderer-shell` / webgl canvases

### Task 3: Verify

- [x] Update `grid.js` clear to museum dark; regenerate collage
- [x] Commit + push
