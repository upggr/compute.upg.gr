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

        return {
            'rank': rank,
            'h11': int(h11),
            'h21': int(h21),
            'euler_char': euler_char,
            'score': float(score),
            'verified_target': bool(verified)
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

        return {
            'rank': rank,
            'h11': int(h11),
            'h21': int(h21),
            'h31': int(h31),
            'euler_char': int(euler),
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

        return {
            'rank': rank,
            'h11': int(h11),
            'h21': int(h21),
            'euler_char': int(euler),
            'hodge_balance': float(hodge_balance),
            'n_generations': int(n_gen),
            'score': float(score),
            'verified_target': bool(verified)
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
