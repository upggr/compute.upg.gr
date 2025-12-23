# Reproducibility Report

## Run Information

- **Timestamp**: 2025-01-15 14:32:17 UTC
- **Hostname**: compute-node-01
- **Git Commit**: `a1b2c3d4e5f6g7h8i9j0`
- **Total Runtime**: 8m 34s

## Environment

- **Python Version**: 3.11.5
- **Platform**: Linux (x86_64)
- **CPU Count**: 8 cores
- **Memory**: 16GB

## Configuration

```yaml
dataset:
  url: "https://example.com/cy_dataset.csv"
  checksum: "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

model:
  type: "random_forest"
  n_estimators: 100
  max_depth: 10

search:
  top_k: 100
  verification: true

reproducibility:
  seed: 42
  export_metadata: true
```

## Dataset Verification

- **Dataset**: cy_landscape_v1.2.csv
- **Checksum**: PASSED (e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855)
- **Total Candidates**: 50,000

## Random Seeds

- **Global Seed**: 42
- **NumPy Seed**: 42
- **Model Seed**: 42

## Results

- **Precision@100**: 0.847
- **Recall@100**: 0.623
- **Verified Targets**: 84/100
- **Time to First Hit**: 12 seconds

## Dependencies

All dependencies pinned in `requirements.txt`:

```
numpy==1.24.3
pandas==2.0.2
scikit-learn==1.3.0
PyYAML==6.0
```

## Verification

This run can be reproduced exactly by:

1. Checking out commit `a1b2c3d4e5f6g7h8i9j0`
2. Installing pinned dependencies: `pip install -r requirements.txt`
3. Running with same config: `python run_cy_search.py --config default.yml`

All random operations used seed=42. Dataset checksum verified before processing.
