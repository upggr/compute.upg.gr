# upg-strings → Lean: quintic calibration handoff

**A proposed input contract, not a Lean-certified geometry.** This is one
explicit textbook example for testing a data-to-proof bridge, not a rare
ML discovery or a top-k export. No Lean build has been performed for this kit.

## The example

We choose the **Fermat quintic** `x0^5 + x1^5 + x2^5 + x3^5 + x4^5 = 0`
in `P^4` over `C`. The curated inputs are `(h11, h21) = (1, 101)` and
`H^3 = 5`, where `H` is the restricted ambient hyperplane class. The Euler
value `-200` is derived as `2 * (h11 - h21)`, not independently computed.
This example does **not** pass the demo's `|chi| < 100` target.

The exporter selects `polytope_id = simplex-P4` directly from
`data/geometry_pack.json`, rather than matching an arbitrary search row by
Hodge numbers. The selected source record and its literature references are
included verbatim in meaning as `source_geometry.json` (JSON reserialized).
Both that snapshot and the original source file have recorded SHA-256 hashes.
The snapshot is provenance, not an independently audited geometry certificate.

The source's Fermat shorthand is expanded into an explicit equation. Its
`vertex_matrix` is labeled here as **ambient P^4 fan rays in N = Z^4**:
`e1, e2, e3, e4, -(e1+e2+e3+e4)`. All four-ray subsets give the specified
maximal cones. These are **not** Newton-polytope vertices in the dual lattice
M, a CYTools FRST dump, or the differently based KS sidecar matrix. Do not
feed them to a tool without checking that tool's lattice/polar convention.

## What is in the ZIP?

| File | Purpose |
|---|---|
| `quintic.json` | Positive fixture, with explicit construction and provenance |
| `negative_euler.json` | Same input except `euler_char = -198`; deliberately invalid |
| `record.schema.json` | Proposed JSON Schema, not an agreed SocrateAI interface |
| `source_geometry.json` | Snapshot of the selected local curated record |
| `check.py` | Standalone, standard-library exact-integer checker |
| `check_report.json` | Positive and negative check results bound to input hashes |
| `SHA256SUMS` | Hashes of all other files, including this README |

**Do not ingest the negative fixture into a gallery or geometry database.**
It keeps the same example ID so that only one numeric field changes. Use the
filename and input hash to distinguish the fixtures; the ID is not a content hash.

## Reproduce and check

From a repository checkout (Python 3.11+; no Flask, CYTools or Lean needed):

```sh
python3 scripts/prepare_lean_handoff.py
# Creates dist/upg-quintic-lean-pilot.zip; add --force to replace an old ZIP.
```

After receiving the ZIP:

```sh
unzip upg-quintic-lean-pilot.zip
cd upg-quintic-lean-pilot
shasum -a 256 -c SHA256SUMS  # or: sha256sum -c SHA256SUMS
python3 check.py quintic.json          # expected exit 0, all five checks pass
python3 check.py negative_euler.json   # expected exit 1, only euler_identity fails
```

Exit 2 means malformed/unsupported input, **not** a successful mathematical
negative control. Integers must not be floats, strings, or booleans; duplicate
JSON keys are rejected. Checksums detect changed bytes, not scientific truth
or the identity of a sender. ZIP bytes are deterministic for unchanged inputs
and exporter code (fixed entry timestamps/order; no compression).

## Precisely scoped proof targets

| ID | Statement to check on the supplied integers |
|---|---|
| `euler_identity` | `chi = 2 * (h11 - h21)` |
| `anticanonical_degree` | `degree = sum(weights) = 5` |
| `hyperplane_intersection` | Declared `H^3 = degree = 5` |
| `fan_ray_sum` | The five ambient fan ray vectors sum to zero |
| `ambient_cone_determinants` | Each of the five 4×4 ray matrices has determinant `+1` or `-1` |

The CY3 Euler formula, hypersurface adjunction, and the intersection formula
are **literature-level geometric inputs**; the script checks their arithmetic
instances only. It does not derive Hodge numbers, formalize smoothness or
adjunction, identify a unique manifold from Hodge data, or prove physical
viability. Even a successful Lean translation of these five arithmetic
statements would have that same limited scope.

Not supplied: numerical periods/monodromy, K3 fibrations or CM fibers, a Mukai
lattice, orientifold/brane/flux data, tadpole cancellation, or a Ricci-flat
metric/cymetric training dataset. In particular, do not interpret `|chi|/24`
of this CY3 as an F-theory fourfold's D3-tadpole budget.

## Requested return artifact from a formal-verification collaborator

1. Agree on this schema or send a minimal adapter and the exact supported claims.
2. Generate Lean definitions from the actual input, not replacement hard-coded
   values or axioms asserting the external Python result.
3. Return the input SHA-256, generated `.lean` files, pinned Lean/Mathlib versions,
   exact build command/output, theorem statements/assumptions, and `#print axioms`
   output for the exported theorems. No `sorry`/`admit` or unreviewed extra axioms.
4. The positive fixture should prove the agreed statements. With the same
   environment, the negative fixture must fail the requested Euler equality
   specifically, not fail because of syntax, missing files, or dependencies.
5. Label any result **“specified arithmetic invariants checked”**, not
   “Lean-certified geometry” or “certified vacuum.”

No SymPy/Arb monodromy or T-duality-filter experiment is implied by this kit.
Those require additional geometric/analytic inputs and a separate specification.
