# Finding real CI5F configuration matrices

Featured CY5 Hodge triples on compute.upg.gr currently ship **without**
invented configuration matrices:

| (h¹¹, h²¹, h³¹) | Status |
|---|---|
| (140, 62, 18) | `configuration_matrix: null` |
| (151, 41, 22) | `configuration_matrix: null` |
| (131, 55, 29) | `configuration_matrix: null` |
| (112, 48, 33) | `configuration_matrix: null` |

## Where to look

1. **Primary dataset** (27 068 CICY5 configuration matrices + partial Hodge):
   - Paper: [arXiv:2310.15966](https://arxiv.org/abs/2310.15966)
     (*Constructing and Machine Learning Calabi-Yau Five-folds*)
   - Dropbox dump (checked 2026-08-09):
     <https://www.dropbox.com/scl/fo/z7ii5idt6qxu36e0b8azq/h?rlkey=0qfhx3tykytduobpld510gsfy&dl=0>
   - File: `cicy5f.json` (27 068 rows). Fields: `matrix`, `h11`, `h12`, `h13`,
     `h14`, `h22`, `h23`, `eta`. Complete Hodge only for ~12 433 non-product cases.
2. **cymetric**: <https://github.com/pythoncymetric/cymetric>
3. Do **not** invent a multi-degree matrix that merely reproduces the Hodge
   numbers — many configurations can share a Hodge diamond, and false
   matrices would mislead model builders.

## Honest check (2026-08-09)

Downloaded the Dropbox `cicy5f.json` and searched for the four featured triples
as `(h11,h12,h13)` and as any permutation across Hodge fields.

**Result: BLOCKED — no matches.**

Reasons:

- That constructive CI5F census only covers complete intersections in a product
  of **≤4 projective spaces** with ≤4 constraints, so `h11` in the dump only
  runs through `{0…10,15}` (max 15). Featured HoF values like `h11=140` cannot
  appear there.
- No row contains all three numbers of any featured triple in its Hodge fields.

So leaving `configuration_matrix: null` remains the honest state until a
different literature / database source quotes a verified matrix for those
class keys.

## How to ship a real hit

When a literature / database row matches a featured triple **and** quotes the
configuration matrix:

1. Add the numeric matrix to `data/geometry_pack.json` and
   `data/known_constructions.json` for `dataset_id: cy5-folds`.
2. Cite the paper / row id in `reference` (+ `reference_url`).
3. Set `note` to class-level honesty (“one known CI5F with these Hodge numbers;
   not proven unique”).
4. Add a regression test that `is_real_configuration_matrix` accepts it.

Until then, leaving `null` is the honest state.
