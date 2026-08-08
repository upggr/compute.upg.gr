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
   - Companion data linked from that paper / related cymetric resources
2. **cymetric**: <https://github.com/pythoncymetric/cymetric>
3. Do **not** invent a multi-degree matrix that merely reproduces the Hodge
   numbers — many configurations can share a Hodge diamond, and false
   matrices would mislead model builders.

## How to ship a real hit

When a literature / database row matches a featured triple **and** quotes the
configuration matrix:

1. Add the numeric matrix to `data/geometry_pack.json` and
   `data/known_constructions.json` for `dataset_id: cy5-folds`.
2. Cite the paper / row id in `reference`.
3. Set `note` to class-level honesty (“one known CI5F with these Hodge numbers;
   not proven unique”).
4. Add a regression test that `is_real_configuration_matrix` accepts it.

Until then, leaving `null` is the honest state.
