# Museum-glass Calabi–Yau 3D polish

Date: 2026-08-08  
Status: approved for planning

## Goal

Make manifold WebGL previews look like museum glassware on a dark plinth: crisp specular highlights, subtle metal sheen, readable topology — without HDR env maps or post-processing bloom/SSAO.

## Context

- Shape/invariant mapping already lives in `static/js/cy-shape.js` (`paramsFor`, parametric surface, vertex colors).
- `/render` uses `CYShape`; candidates/results pages still use leftover modulo/random-hue paths.
- Current look reads muddy: light gray clear color, soft matte materials, weak contrast on folds.

## Approach

Lighting + material retune only (no formula rewrite, no EffectComposer, no env maps).

## Visual design

### Scene

- Clear color: near-black charcoal ≈ `#0c1016` (opaque).
- Lights:
  - Warm key, high intensity, elevated front-right.
  - Cool rim opposite, lower intensity, for edge separation.
  - Soft low fill so fold interiors do not crush to pure black.
- Camera: keep auto-fit to bounding sphere; fixed mesh presentation angle for fair comparisons.
- Animation: keep slow pivot rotation; avoid bobbing that fights specular read.

### Material & color

- Keep χ → hue (cool for negative, warm for positive) and lobe banding tied to `f1`.
- Lift midtone value range so ridges catch the key light (avoid muddy slate).
- Slightly desaturate midtones; brighter lobe bands for glass-on-metal speculars.
- `MeshStandardMaterial`: metalness ≈ 0.55–0.85 (scales with `|χ|` / skew), roughness ≈ 0.18–0.40 (inverse), `flatShading: false`, emissive ≈ 0.

### Shape

- Keep current torus/lobe parametric formula.
- Optional small warp/tube nudge only if needed so ridges catch the new key light without self-intersection spikes.

## Architecture

Centralize look in `cy-shape.js`:

| Export | Responsibility |
|--------|----------------|
| `paramsFor` / `surfaceFor` / `geometryFor` | Unchanged topology mapping (minor warp tweak only if needed) |
| `paletteFor` / `applyVertexColors` | Museum-friendly value/saturation |
| `materialFor` | Metalness/roughness ranges above |
| `configureScene(THREE, { renderer, scene, camera })` (new) | Clear color + the three lights |

Pages (`render.html`, `candidates.html`, `results.html`) call `configureScene` and `CYShape.geometryFor` / `materialFor`; delete modulo param builders and random-hue materials on candidates/results.

## Out of scope

- HDR / environment maps
- Bloom, SSAO, EffectComposer
- Rewriting the parametric surface into a different topology family
- Changing analysis APIs or share URLs

## Verification

- Regenerate comparison sheet via existing `grid.js` → labeled collage (e.g. `grid_v3.png` or `grid_v4.png`).
- Success criteria (visual):
  - Dark plinth backdrop
  - Visible specular ridges (not chalky matte)
  - Distinct silhouettes / hue across the 8 sample manifolds
  - No spike-y self-intersections
- Smoke: open `/render` with sample `h11`/`h21`/`chi` query params; open one candidate detail WebGL preview.

## Risks

- Dark clear color may clash with light page chrome around the canvas — confine dark clear to the WebGL canvas; page layout stays as-is.
- High metalness can wash hue — keep vertex colors saturated enough that χ encoding remains readable under the key light.
