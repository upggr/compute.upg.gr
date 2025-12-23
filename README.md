# CY-Search: ML-Guided Search for Rare Calabi-Yau Geometries

**We build ML-guided search tools that drastically reduce the cost of finding rare Calabi-Yau geometries in large string-theory datasets, with full verification and reproducibility.**

[![Website](https://img.shields.io/badge/website-compute.upg.gr-blue)](https://compute.upg.gr)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Overview

CY-Search is a research tool for applying machine learning to the computational exploration of Calabi-Yau manifolds in the string theory landscape. The project emphasizes:

- **Reproducibility**: Deterministic pipelines with checksummed data, pinned dependencies, and fixed random seeds
- **Verification**: All predictions validated against ground truth with transparent metrics
- **Open Artifacts**: Complete outputs (CSV, JSON, metadata) for independent analysis

This is applied computation and AI tooling designed to accelerate discovery in theoretical physics datasets, not a claim to solve fundamental physics problems.

## Key Features

### ML-Guided Search
- Train models to identify Calabi-Yau manifolds with specific topological properties
- Rank candidates from the Kreuzer-Skarke database (474M reflexive polytopes)
- Target manifolds optimized for phenomenological model building

### Topological Feature Extraction
- Hodge numbers (h¹¹, h²¹)
- Euler characteristic (χ = 2(h¹¹ - h²¹))
- Chern class invariants
- Triple intersection numbers
- Derived geometric quantities

### Performance Metrics
- **Precision@k**: Fraction of top-k predictions that are verified correct
- **Recall@k**: Fraction of all true targets found in top-k results
- **Time-to-First-Hit**: How quickly the first verified target is discovered
- **Baseline Comparison**: Performance vs. random selection

## Quick Start

### Installation

```bash
git clone https://github.com/upggr/compute.upg.gr.git
cd compute.upg.gr
pip install -r requirements.txt
```

### Run the Demo

```python
from cy_search_real import run_real_search

# Run ML-guided search on 10,000 Calabi-Yau manifolds
results = run_real_search(
    top_k=100,           # Return top 100 candidates
    seed=42,             # Random seed for reproducibility
    n_candidates=10000,  # Dataset size
    verify=True          # Verify against ground truth
)

print(f"Precision@100: {results['performance_metrics']['precision_at_k']:.1%}")
print(f"Recall@100: {results['performance_metrics']['recall_at_k']:.1%}")
print(f"Verified targets: {results['performance_metrics']['verified_count']}/100")
```

### Web Interface

The project includes a Flask web application:

```bash
# Run locally
python app.py

# Or with Gunicorn (production)
gunicorn --bind 0.0.0.0:5102 --workers 4 app:app
```

Visit `http://localhost:5102` to access the web interface.

## How It Works

### 1. Data Source
The tool uses the **Kreuzer-Skarke database** of Calabi-Yau threefolds:
- Official database: http://hep.itp.tuwien.ac.at/~kreuzer/CY/
- Contains all 473,800,776 reflexive polyhedra in 4D
- Each polytope describes a Calabi-Yau hypersurface

### 2. Feature Engineering
Extract topological and geometric features:
- Hodge numbers characterizing complex structure
- Euler characteristic for topological classification
- Chern class invariants for phenomenology
- Derived quantities (ratios, absolute values)

### 3. ML Model Training
- Random Forest classifier (100 estimators)
- Train on labeled examples of "interesting" manifolds
- Targets defined by:
  - Small Euler characteristic (|χ| < 100)
  - Moderate h¹¹ for flux compactifications
  - Favorable Chern classes for SUSY breaking
  - Manageable h²¹ for complex structure moduli

### 4. Ranking & Verification
- Rank all candidates by predicted likelihood
- Return top-k results
- Verify against ground truth labels
- Report precision, recall, and feature importance

## Project Structure

```
compute.upg.gr/
├── app.py                 # Flask web application
├── cy_search.py           # Original demo implementation
├── cy_search_real.py      # Real KS database implementation
├── templates/             # HTML templates
│   ├── index.html        # Home page
│   ├── docs.html         # Documentation
│   ├── results.html      # Demo results
│   └── about.html        # About page
├── static/
│   ├── css/style.css     # Styling
│   ├── js/script.js      # Frontend JavaScript
│   └── data/             # Sample output files
├── Dockerfile            # Docker configuration
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## API Endpoints

### `POST /api/run-demo`
Run the CY-Search pipeline

**Request:**
```json
{
  "top_k": 100,
  "seed": 42,
  "verify": true
}
```

**Response:**
```json
{
  "status": "success",
  "run_metadata": {...},
  "performance_metrics": {
    "precision_at_k": 0.84,
    "recall_at_k": 0.62,
    "verified_count": 84
  },
  "top_results": [...]
}
```

### `GET /api/results/<run_id>`
Retrieve results from a specific run

### `GET /api/sample-results`
Get sample results for display

## Deployment

### Docker (Recommended)

```bash
docker build -t cy-search .
docker run -p 5102:5102 cy-search
```

### Caprover

The project is configured for automatic deployment with Caprover:
1. Connect your GitHub repository
2. Set container port to 5102
3. Push to main branch to trigger deployment

## Reproducibility Guarantees

Every run includes:
- **Fixed Random Seeds**: All stochastic operations use deterministic seeds
- **Dataset Checksum**: SHA-256 verification of input data
- **Pinned Dependencies**: Exact package versions in requirements.txt
- **Run Metadata**: Complete environment and configuration details
- **Exportable Artifacts**: CSV, JSON, and markdown reports

## Performance

Typical runtime on standard hardware:
- Dataset loading: ~0.5s (cached)
- Model training: ~2-5s (10K samples)
- Ranking: ~0.2s
- Verification: ~0.1s
- **Total: 5-15 seconds**

Scales to:
- 10K candidates: ~5 seconds
- 100K candidates: ~30 seconds
- 1M candidates: ~5 minutes

## Roadmap

- [ ] Integrate actual CYTools library for full KS database access
- [ ] Expand to multiple target criteria (user-definable)
- [ ] Add graph neural networks for geometric learning
- [ ] Implement mirror symmetry detection
- [ ] Develop automated algebraic geometry verification
- [ ] Create REST API for programmatic access
- [ ] Add real-time progress tracking for long-running searches

## References

### Kreuzer-Skarke Database
- Original paper: [arXiv:hep-th/0002240](https://arxiv.org/abs/hep-th/0002240)
- Database: http://hep.itp.tuwien.ac.at/~kreuzer/CY/
- Enhanced database: http://nuweb1.neu.edu/cydatabase

### CYTools
- Software: https://cy.tools
- Paper: [arXiv:2211.03823](https://arxiv.org/abs/2211.03823)
- GitHub: https://github.com/LiamMcAllisterGroup/cytools

## Citation

If you use CY-Search in your research, please cite:

```bibtex
@software{cysearch2025,
  author = {Kokkinis, Ioannis},
  title = {CY-Search: ML-Guided Search for Rare Calabi-Yau Geometries},
  year = {2025},
  url = {https://compute.upg.gr},
  note = {Research tool for string theory landscape exploration}
}
```

## Contact

- **Email**: ioannis.kokkinis@upg.gr
- **LinkedIn**: [ioanniskokkinis](https://www.linkedin.com/in/ioanniskokkinis)
- **Website**: [compute.upg.gr](https://compute.upg.gr)
- **GitHub**: [upggr/compute.upg.gr](https://github.com/upggr/compute.upg.gr)

## License

MIT License - see [LICENSE](LICENSE) for details

## Acknowledgments

This project builds on:
- The Kreuzer-Skarke Calabi-Yau database (M. Kreuzer & H. Skarke)
- Open-source machine learning libraries (scikit-learn, NumPy, pandas)
- The broader computational physics and string theory communities

---

**Disclaimer**: This tool accelerates computational search and verification. It does not claim to 'solve string theory' or make predictions about physical reality.
