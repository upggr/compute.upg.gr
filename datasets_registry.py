#!/usr/bin/env python3
"""
Dataset Registry for CY-Search
Provides a unified interface for multiple Calabi-Yau and string theory datasets
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class DatasetMetadata:
    """Metadata for a dataset"""
    name: str
    description: str
    total_count: int  # Total manifolds in dataset
    feature_dim: int  # Number of features
    target_description: str  # What we're searching for
    typical_runtime_5k: float  # Expected runtime for 5K candidates (seconds)
    source_url: Optional[str] = None


class BaseDataset(ABC):
    """Base class for all datasets"""

    @abstractmethod
    def get_metadata(self) -> DatasetMetadata:
        """Return dataset metadata"""
        pass

    @abstractmethod
    def generate_candidates(self, n_candidates: int, seed: int) -> np.ndarray:
        """
        Generate synthetic candidates based on dataset statistics
        Returns: numpy array of shape (n_candidates, feature_dim)
        """
        pass

    @abstractmethod
    def generate_labels(self, candidates: np.ndarray, seed: int) -> np.ndarray:
        """
        Generate ground truth labels for candidates
        Returns: numpy array of shape (n_candidates,) with binary labels
        """
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Return list of feature names"""
        pass

    @abstractmethod
    def format_result(self, candidate: np.ndarray, score: float, verified: bool, rank: int) -> Dict[str, Any]:
        """Format a single result for display"""
        pass


class KreuzerSkarkeDataset(BaseDataset):
    """Kreuzer-Skarke database of reflexive polytopes (CY 3-folds)"""

    def get_metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            name="Kreuzer-Skarke Database",
            description="Reflexive polytopes describing Calabi-Yau threefolds. Searching for manifolds with small Euler characteristic (|χ| < 100) suitable for particle physics phenomenology.",
            total_count=473800776,  # ~474 million
            feature_dim=5,
            target_description="Manifolds with |χ| < 100",
            typical_runtime_5k=5.2,
            source_url="http://hep.itp.tuwien.ac.at/~kreuzer/CY/"
        )

    def generate_candidates(self, n_candidates: int, seed: int) -> np.ndarray:
        """Generate Calabi-Yau threefold candidates"""
        np.random.seed(seed)

        # Hodge numbers h11, h21 (positive integers)
        # Statistics from KS database: h11 ∈ [1, 491], h21 ∈ [1, 491]
        h11 = np.random.randint(1, 300, size=n_candidates)
        h21 = np.random.randint(1, 300, size=n_candidates)

        # Euler characteristic: χ = 2(h11 - h21)
        euler = 2 * (h11 - h21)

        # Derived features
        euler_abs = np.abs(euler)
        hodge_ratio = h21 / (h11 + 1e-10)

        # Second Chern class number c2·h11 (approximate)
        c2_h11 = 12 * h11 + 6 * h21 + np.random.randint(-50, 50, size=n_candidates)

        candidates = np.column_stack([h11, h21, euler_abs, hodge_ratio, c2_h11])
        return candidates.astype(np.float32)

    def generate_labels(self, candidates: np.ndarray, seed: int) -> np.ndarray:
        """Target: manifolds with |χ| < 100"""
        euler_abs = candidates[:, 2]
        labels = (euler_abs < 100).astype(int)
        return labels

    def get_feature_names(self) -> List[str]:
        return ['h11', 'h21', 'euler_abs', 'hodge_ratio', 'c2_h11']

    def format_result(self, candidate: np.ndarray, score: float, verified: bool, rank: int) -> Dict[str, Any]:
        h11, h21, euler_abs, hodge_ratio, c2_h11 = candidate
        euler_char = int(2 * (h11 - h21))
        h11_i, h21_i = int(h11), int(h21)

        return {
            'rank': rank,
            'h11': h11_i,
            'h21': h21_i,
            'euler_char': euler_char,
            'score': float(score),
            'verified_target': bool(verified),
            'tadpole_L': round(abs(euler_char) / 24.0, 4),
            'n_generations': abs(euler_char) // 2,
            'c2_J_proxy': int(12 * h11_i + 6 * h21_i),
            'c2_h11_feature': float(c2_h11),
            'hodge_ratio': float(hodge_ratio),
            'picard_fuchs_order': h21_i + 1,
        }


class CY5FoldsDataset(BaseDataset):
    """Complete Intersection Calabi-Yau Five-folds dataset"""

    def get_metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            name="CY5-Folds (CI5F)",
            description="Complete intersection Calabi-Yau five-folds. Searching for manifolds with h^{1,1} > 100 (many Kähler moduli) suitable for large volume scenarios.",
            total_count=27068,
            feature_dim=6,
            target_description="Manifolds with h^{1,1} > 100",
            typical_runtime_5k=4.8,
            source_url="https://github.com/pythoncymetric/cymetric"
        )

    def generate_candidates(self, n_candidates: int, seed: int) -> np.ndarray:
        """Generate CY5-fold candidates"""
        np.random.seed(seed)

        # Hodge numbers for CY5: h^{1,1}, h^{2,1}, h^{3,1}, h^{4,1}
        # Simplify to h11, h21, h31 for this demo
        h11 = np.random.randint(1, 250, size=n_candidates)
        h21 = np.random.randint(1, 150, size=n_candidates)
        h31 = np.random.randint(1, 100, size=n_candidates)

        # Euler characteristic for CY5: χ = 6 + 6(h11 - h21 + h31)
        euler = 6 + 6 * (h11 - h21 + h31)

        # Derived features
        euler_abs = np.abs(euler)
        hodge_sum = h11 + h21 + h31

        candidates = np.column_stack([h11, h21, h31, euler, euler_abs, hodge_sum])
        return candidates.astype(np.float32)

    def generate_labels(self, candidates: np.ndarray, seed: int) -> np.ndarray:
        """Target: manifolds with h11 > 100 (large volume scenarios)"""
        h11 = candidates[:, 0]
        labels = (h11 > 100).astype(int)
        return labels

    def get_feature_names(self) -> List[str]:
        return ['h11', 'h21', 'h31', 'euler', 'euler_abs', 'hodge_sum']

    def format_result(self, candidate: np.ndarray, score: float, verified: bool, rank: int) -> Dict[str, Any]:
        h11, h21, h31, euler, euler_abs, hodge_sum = candidate
        h11_i, h21_i, h31_i = int(h11), int(h21), int(h31)
        euler_i = int(euler)

        return {
            'rank': rank,
            'h11': h11_i,
            'h21': h21_i,
            'h31': h31_i,
            'euler_char': euler_i,
            'score': float(score),
            'verified_target': bool(verified),
            'tadpole_L': round(abs(euler_i) / 24.0, 4),
            'hodge_sum': int(hodge_sum),
        }


class InformationDensityDataset(BaseDataset):
    """
    Information Density ranking for Calabi-Yau manifolds.

    Treats CY manifolds as "information compression" structures and ranks them
    by how efficiently they encode topological complexity. This is a proxy metric
    inspired by the idea that phenomenologically viable geometries may act as
    optimal "filters" between UV and IR scales.

    The information density metric combines:
    - Hodge entropy: Shannon entropy over normalized Hodge numbers
    - Topological efficiency: |χ| / (h11 + h21) ratio
    - Moduli compactness: inverse of total moduli count
    - Flux vacua density: based on tadpole constraint and Bousso-Polchinski
    - Vacuum stability proxy: likelihood of stable dS/AdS vacuum

    Target: High information density candidates (top 10% by composite score)

    Supports custom weights via set_weights() for tuning the composite score.
    """

    # Default weights for composite score
    DEFAULT_WEIGHTS = {
        'entropy': 0.20,
        'efficiency': 0.20,
        'compactness': 0.15,
        'balance': 0.10,
        'flux_density': 0.20,
        'vacuum_stability': 0.15
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        # Copy defensively: storing the caller's dict by reference means
        # set_weights() would mutate an object owned by someone else.
        self.weights = dict(weights) if weights else self.DEFAULT_WEIGHTS.copy()

    def set_weights(self, weights: Dict[str, float]):
        """Set custom weights for the composite score components.

        NOTE: this mutates process-global state shared by every request served
        by this worker. Prefer passing `weights` to generate_candidates().
        """
        self.weights.update(weights)

    def resolve_weights(self, weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Merge caller-supplied weights over the defaults without mutating self."""
        resolved = self.DEFAULT_WEIGHTS.copy()
        resolved.update(self.weights)
        if weights:
            resolved.update(weights)
        return resolved

    def get_metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            name="Information Density Ranking",
            description="Ranks CY manifolds by information-theoretic complexity. Searches for geometries with high 'information density' - efficient topological encoding that may correlate with phenomenological viability and vacuum stability.",
            total_count=473800776,  # Same underlying KS dataset
            feature_dim=10,
            target_description="High information density manifolds (top 10%)",
            typical_runtime_5k=5.5,
            source_url="http://hep.itp.tuwien.ac.at/~kreuzer/CY/"
        )

    def generate_candidates(self, n_candidates: int, seed: int,
                            weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Generate candidates with information density features.

        `weights` overrides the instance weights for this call only, so a
        request can tune the composite score without affecting other requests.
        """
        np.random.seed(seed)

        # Base Hodge numbers (from KS-like distribution)
        h11 = np.random.randint(1, 300, size=n_candidates)
        h21 = np.random.randint(1, 300, size=n_candidates)

        # Euler characteristic: χ = 2(h11 - h21) for CY3
        euler = 2 * (h11 - h21)
        euler_abs = np.abs(euler)

        # === Information Density Features ===
        h_total = h11 + h21 + 1e-10

        # 1. Hodge entropy: Shannon entropy of normalized Hodge numbers
        p11 = h11 / h_total
        p21 = h21 / h_total
        hodge_entropy = -(p11 * np.log(p11 + 1e-10) + p21 * np.log(p21 + 1e-10))
        hodge_entropy_norm = hodge_entropy / np.log(2)  # Normalize to [0, 1]

        # 2. Topological efficiency: curvature info per modulus
        topo_efficiency = euler_abs / (h_total + 1e-10)

        # 3. Moduli compactness: prefer fewer moduli
        moduli_compactness = 1.0 / (1 + np.log1p(h_total))

        # 4. Hodge balance: symmetry of the Hodge diamond
        hodge_balance = 1.0 - np.abs(h11 - h21) / (h_total + 1e-10)

        # === Physics-grounded Vacuum Metrics ===

        # 5. Flux vacua density (Bousso-Polchinski / KKLT inspired)
        #    Number of flux vacua ~ (2πL)^(2K) / K! where:
        #    - K = h21 + 1 (number of 3-cycles for flux, complex structure moduli + dilaton)
        #    - L = tadpole charge = χ/24 (D3 tadpole cancellation condition)
        #    Higher density = more vacua to search = harder to find SM, but more "information"
        #
        #    We compute log of density normalized to [0, 1]:
        #    log(N_flux) ~ 2K * log(2π * χ/24) - log(K!)
        #    Using Stirling: log(K!) ~ K*log(K) - K
        K = h21 + 1
        L = np.maximum(euler_abs / 24.0, 1.0)  # Tadpole bound
        log_flux_density = 2 * K * np.log(2 * np.pi * L + 1e-10) - (K * np.log(K + 1e-10) - K)
        # Normalize to [0, 1] using sigmoid
        flux_density_norm = 1.0 / (1.0 + np.exp(-log_flux_density / 100))

        # 6. Vacuum stability proxy
        #    Likelihood of finding stable (dS or AdS) vacuum based on:
        #    - Small |χ| relative to moduli count (easier tadpole cancellation)
        #    - Balanced h11/h21 (both Kähler and complex structure moduli for stabilization)
        #    - Not too many moduli (easier to stabilize all of them)
        #
        #    Inspired by KKLT/LVS scenarios where stability requires:
        #    - Enough flux (χ) to stabilize complex structure moduli
        #    - Not so much that tadpole is violated
        #    - h11 > 0 for Kähler moduli stabilization
        tadpole_headroom = np.maximum(1.0 - euler_abs / (24 * h11 + 1e-10), 0)  # Room under tadpole
        stabilization_ratio = np.minimum(h11, h21) / (np.maximum(h11, h21) + 1e-10)  # Balance
        moduli_penalty = np.exp(-h_total / 200)  # Fewer moduli = easier stabilization
        vacuum_stability = (0.4 * tadpole_headroom + 0.4 * stabilization_ratio + 0.2 * moduli_penalty)

        # 7. Information density composite score (customizable weights)
        w = self.resolve_weights(weights)
        info_density = (
            w['entropy'] * hodge_entropy_norm +
            w['efficiency'] * np.tanh(topo_efficiency) +
            w['compactness'] * moduli_compactness +
            w['balance'] * hodge_balance +
            w['flux_density'] * flux_density_norm +
            w['vacuum_stability'] * vacuum_stability
        )

        candidates = np.column_stack([
            h11, h21, euler_abs,
            hodge_entropy_norm, topo_efficiency,
            moduli_compactness, hodge_balance,
            flux_density_norm, vacuum_stability, info_density
        ])
        return candidates.astype(np.float32)

    def generate_labels(self, candidates: np.ndarray, seed: int) -> np.ndarray:
        """Target: top 10% by information density score"""
        info_density = candidates[:, 9]  # Last column is composite score
        threshold = np.percentile(info_density, 90)
        labels = (info_density >= threshold).astype(int)
        return labels

    def get_feature_names(self) -> List[str]:
        return [
            'h11', 'h21', 'euler_abs',
            'hodge_entropy', 'topo_efficiency',
            'moduli_compactness', 'hodge_balance',
            'flux_density', 'vacuum_stability', 'info_density'
        ]

    def format_result(self, candidate: np.ndarray, score: float, verified: bool, rank: int) -> Dict[str, Any]:
        (h11, h21, euler_abs, hodge_entropy, topo_efficiency,
         moduli_compactness, hodge_balance, flux_density,
         vacuum_stability, info_density) = candidate

        return {
            'rank': rank,
            'h11': int(h11),
            'h21': int(h21),
            'euler_char': int(2 * (h11 - h21)),
            'tadpole_charge': round(float(euler_abs) / 24, 2),  # χ/24 D3 tadpole
            'hodge_entropy': round(float(hodge_entropy), 4),
            'topo_efficiency': round(float(topo_efficiency), 4),
            'moduli_compactness': round(float(moduli_compactness), 4),
            'hodge_balance': round(float(hodge_balance), 4),
            'flux_density': round(float(flux_density), 4),
            'vacuum_stability': round(float(vacuum_stability), 4),
            'info_density': round(float(info_density), 4),
            'score': float(score),
            'verified_target': bool(verified)
        }


class HeteroticDataset(BaseDataset):
    """Heterotic string compactifications on CY3-manifolds"""

    def get_metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            name="Heterotic Compactifications",
            description="Heterotic string theory compactifications. Searching for manifolds with favorable Yukawa coupling structures (h^{1,1} ≈ h^{2,1} for balanced moduli).",
            total_count=10000000,  # Estimated
            feature_dim=7,
            target_description="Balanced manifolds with h^{1,1} ≈ h^{2,1}",
            typical_runtime_5k=5.5,
            source_url="https://arxiv.org/abs/hep-th/0507229"
        )

    def generate_candidates(self, n_candidates: int, seed: int) -> np.ndarray:
        """Generate heterotic compactification candidates"""
        np.random.seed(seed)

        # Hodge numbers
        h11 = np.random.randint(1, 200, size=n_candidates)
        h21 = np.random.randint(1, 200, size=n_candidates)

        # Euler characteristic
        euler = 2 * (h11 - h21)
        euler_abs = np.abs(euler)

        # Hodge ratio and balance metric
        hodge_ratio = h21 / (h11 + 1e-10)
        hodge_balance = np.abs(h11 - h21) / (h11 + h21 + 1e-10)  # Close to 0 is balanced

        # Number of generations (related to |χ|/2)
        n_gen = np.abs(euler) // 2

        candidates = np.column_stack([h11, h21, euler, euler_abs, hodge_ratio, hodge_balance, n_gen])
        return candidates.astype(np.float32)

    def generate_labels(self, candidates: np.ndarray, seed: int) -> np.ndarray:
        """Target: balanced manifolds with |h11 - h21| < 20"""
        h11 = candidates[:, 0]
        h21 = candidates[:, 1]
        labels = (np.abs(h11 - h21) < 20).astype(int)
        return labels

    def get_feature_names(self) -> List[str]:
        return ['h11', 'h21', 'euler', 'euler_abs', 'hodge_ratio', 'hodge_balance', 'n_gen']

    def format_result(self, candidate: np.ndarray, score: float, verified: bool, rank: int) -> Dict[str, Any]:
        h11, h21, euler, euler_abs, hodge_ratio, hodge_balance, n_gen = candidate
        h11_i, h21_i = int(h11), int(h21)
        euler_i = int(euler)

        return {
            'rank': rank,
            'h11': h11_i,
            'h21': h21_i,
            'euler_char': euler_i,
            'hodge_balance': float(hodge_balance),
            'n_generations': int(n_gen),
            'score': float(score),
            'verified_target': bool(verified),
            'tadpole_L': round(abs(euler_i) / 24.0, 4),
            'three_generation_target': abs(euler_i) == 6,
            'c2_J_proxy': int(12 * h11_i + 6 * h21_i),
            'picard_fuchs_order': h21_i + 1,
        }


class FTheoryEllipticDataset(BaseDataset):
    """Honest F-theory *proxy* dataset — not a Weierstrass-model database.

    Features combine CY3 Hodge numbers with pedagogical elliptic-fibration
    flags. Featured seeds use literature elliptic CY3 Hodge pairs
    (e.g. WP[1,1,1,6,9] degree-18 with (h11,h21)=(2,272)).
    """

    # Literature / textbook elliptic CY3 Hodge pairs (class-level).
    FEATURED_SEEDS = [
        # Famous elliptic CY3: hypersurface of degree 18 in WP[1,1,1,6,9]
        {'h11': 2, 'h21': 272, 'base_label': 'P2', 'tag': 'literature_elliptic'},
        {'h11': 272, 'h21': 2, 'base_label': 'mirror', 'tag': 'literature_elliptic_mirror'},
        # Self-mirror / balanced elliptic-friendly Hodge (proxy)
        {'h11': 19, 'h21': 19, 'base_label': 'self-mirror-proxy', 'tag': 'pedagogical_proxy'},
    ]

    def get_metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            name="F-theory elliptic proxies",
            description=(
                "Pedagogical F-theory search surface using elliptic-fibration "
                "proxy features on CY3 Hodge data. NOT a full Weierstrass / "
                "Tate-model database — targets geometries with elliptic-friendly "
                "Hodge patterns suitable for F-theory model-building sketches."
            ),
            total_count=50000,  # proxy corpus size, not a published Weierstrass census
            feature_dim=7,
            target_description="Elliptic-friendly proxies (low h11 base-like or literature seeds)",
            typical_runtime_5k=5.0,
            source_url="https://arxiv.org/abs/hep-th/9603161",
        )

    def generate_candidates(self, n_candidates: int, seed: int) -> np.ndarray:
        np.random.seed(seed)
        # Mix literature seeds with synthetic elliptic-like Hodge draws.
        h11 = np.random.randint(1, 120, size=n_candidates)
        h21 = np.random.randint(1, 400, size=n_candidates)
        # Force-inject a few literature seeds at the front for reproducibility.
        for i, seed_row in enumerate(self.FEATURED_SEEDS):
            if i < n_candidates:
                h11[i] = seed_row['h11']
                h21[i] = seed_row['h21']
        euler = 2 * (h11 - h21)
        euler_abs = np.abs(euler)
        # Proxy: "base-like" when h11 is small (few Kähler classes — common in
        # elliptic CY3 with rigid fiber + base contributions).
        elliptic_flag = (h11 <= 10).astype(np.float32)
        # Mordell–Weil rank is unknown; use a bounded random proxy in [0, 3].
        mw_rank_proxy = np.random.randint(0, 4, size=n_candidates).astype(np.float32)
        base_h11_proxy = np.minimum(h11, 10).astype(np.float32)
        candidates = np.column_stack([
            h11, h21, euler_abs, elliptic_flag, mw_rank_proxy, base_h11_proxy,
            euler_abs / (h11 + h21 + 1e-10),
        ])
        return candidates.astype(np.float32)

    def generate_labels(self, candidates: np.ndarray, seed: int) -> np.ndarray:
        """Target: elliptic-friendly proxy — small h11 or literature seed Hodge."""
        h11 = candidates[:, 0]
        h21 = candidates[:, 1]
        elliptic_flag = candidates[:, 3]
        lit = np.zeros(len(candidates), dtype=bool)
        for seed_row in self.FEATURED_SEEDS:
            lit |= (h11 == seed_row['h11']) & (h21 == seed_row['h21'])
        labels = ((elliptic_flag > 0.5) | lit).astype(int)
        return labels

    def get_feature_names(self) -> List[str]:
        return [
            'h11', 'h21', 'euler_abs', 'elliptic_flag',
            'mw_rank_proxy', 'base_h11_proxy', 'topo_efficiency',
        ]

    def format_result(self, candidate: np.ndarray, score: float, verified: bool, rank: int) -> Dict[str, Any]:
        h11, h21, euler_abs, elliptic_flag, mw_rank, base_h11, topo_eff = candidate
        h11_i, h21_i = int(h11), int(h21)
        euler_i = int(2 * (h11_i - h21_i))
        return {
            'rank': rank,
            'h11': h11_i,
            'h21': h21_i,
            'euler_char': euler_i,
            'score': float(score),
            'verified_target': bool(verified),
            'tadpole_L': round(abs(euler_i) / 24.0, 4),
            'elliptic_fibration_flag': bool(elliptic_flag > 0.5),
            'mordell_weil_rank_proxy': int(mw_rank),
            'base_h11_proxy': int(base_h11),
            'topo_efficiency': float(topo_eff),
            'honesty': (
                'F-theory elliptic PROXY features — not a Weierstrass equation '
                'or resolved elliptic fibration database.'
            ),
            'picard_fuchs_order': h21_i + 1,
        }


class DatasetRegistry:
    """Central registry for all available datasets"""

    _datasets: Dict[str, BaseDataset] = {}

    @classmethod
    def register(cls, dataset_id: str, dataset: BaseDataset):
        """Register a new dataset"""
        cls._datasets[dataset_id] = dataset

    @classmethod
    def get_dataset(cls, dataset_id: str) -> BaseDataset:
        """Get a dataset by ID"""
        if dataset_id not in cls._datasets:
            raise ValueError(f"Unknown dataset: {dataset_id}. Available: {list(cls._datasets.keys())}")
        return cls._datasets[dataset_id]

    @classmethod
    def list_datasets(cls) -> List[Dict[str, Any]]:
        """List all available datasets with metadata"""
        result = []
        for dataset_id, dataset in cls._datasets.items():
            metadata = dataset.get_metadata()
            result.append({
                'id': dataset_id,
                'name': metadata.name,
                'description': metadata.description,
                'total_count': metadata.total_count,
                'target_description': metadata.target_description,
                'typical_runtime_5k': metadata.typical_runtime_5k
            })
        return result


# Register all available datasets
DatasetRegistry.register('kreuzer-skarke', KreuzerSkarkeDataset())
DatasetRegistry.register('cy5-folds', CY5FoldsDataset())
DatasetRegistry.register('heterotic', HeteroticDataset())
DatasetRegistry.register('f-theory-elliptic', FTheoryEllipticDataset())

# Info-density dataset with default weights (can be customized via API)
_info_density_dataset = InformationDensityDataset()
DatasetRegistry.register('info-density', _info_density_dataset)


def get_info_density_dataset() -> InformationDensityDataset:
    """Get the info-density dataset instance for weight customization"""
    return _info_density_dataset
