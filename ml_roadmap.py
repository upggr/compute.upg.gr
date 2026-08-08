"""ML extension points — stubs only, no fake trained GNNs.

These hooks document where graph / transfer / active-learning / ensemble
pipelines would plug into the existing MLP / RandomForest search stack.
Nothing here claims a trained GNN or published accuracy numbers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def geometry_featurizer(
    h11: int,
    h21: int,
    *,
    h31: Optional[int] = None,
    vertex_matrix: Optional[List[List[int]]] = None,
) -> Dict[str, Any]:
    """Trivial baseline featurizer used by the classical MLP path.

    When a vertex matrix is present we expose combinatorial sizes only
    (vertex/facet counts proxies via row count) — not a GNN embedding.
    """
    features = {
        'h11': int(h11),
        'h21': int(h21),
        'euler_abs': abs(2 * (int(h11) - int(h21))) if h31 is None else None,
        'has_vertex_matrix': bool(vertex_matrix),
        'vertex_count_proxy': len(vertex_matrix) if vertex_matrix else None,
    }
    if h31 is not None:
        features['h31'] = int(h31)
    return {
        'status': 'baseline_vector',
        'features': features,
        'honesty': (
            'Dense Hodge / size features only. Not a graph neural embedding of '
            'the toric fan or triangulation.'
        ),
    }


def gnn_extension_point(vertex_matrix: Optional[List[List[int]]] = None) -> Dict[str, Any]:
    """Unchecked research stub for a future toric-graph GNN."""
    return {
        'status': 'not_implemented',
        'checked': False,
        'requires': [
            'Triangulation / fan graph construction from polytope vertices',
            'Labeled training set beyond synthetic Hodge draws',
            'Offline CYTools or equivalent geometry backend',
        ],
        'input_hint': {
            'has_vertices': bool(vertex_matrix),
            'proposed_graph': 'vertices→edges of toric skeleton (research)',
        },
        'note': 'Do not treat this stub as a trained model.',
    }


def transfer_learning_stub() -> Dict[str, Any]:
    return {
        'status': 'not_implemented',
        'checked': False,
        'idea': 'Pretrain on KS Hodge ranking, fine-tune on CI5F / F-theory proxies',
    }


def active_learning_stub() -> Dict[str, Any]:
    return {
        'status': 'not_implemented',
        'checked': False,
        'idea': 'Query uncertain candidates for offline geometry verification labels',
    }


def ensemble_stub(scores: Optional[List[float]] = None) -> Dict[str, Any]:
    if not scores:
        return {
            'status': 'not_implemented',
            'checked': False,
            'idea': 'Average RF / MLP / future-GNN ranking scores',
        }
    arr = np.asarray(scores, dtype=float)
    return {
        'status': 'toy_mean',
        'checked': False,
        'mean_score': float(arr.mean()),
        'note': 'Illustrative mean of caller-supplied scores — not a trained ensemble.',
    }


def roadmap_checklist() -> List[Dict[str, Any]]:
    return [
        {'id': 'gnn', 'label': 'Graph neural nets for geometric learning', 'done': False},
        {'id': 'transfer', 'label': 'Transfer learning across datasets', 'done': False},
        {'id': 'active', 'label': 'Active learning for efficient labeling', 'done': False},
        {'id': 'ensemble', 'label': 'Ensemble methods', 'done': False},
        {
            'id': 'featurizer_hook',
            'label': 'Baseline geometry featurizer hook (shipped stub)',
            'done': True,
        },
    ]
