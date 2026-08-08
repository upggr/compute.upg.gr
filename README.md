# upg-strings: ML-Guided Search for Rare Geometries in String Theory

**We build ML-guided ranking tools for rare Calabi-Yau-like targets, with reproducible seeds, target-rule checks, and shareable dossiers (Hall of Fame).**

**Demo corpora are synthetic Hodge-number draws.** “Verified” means a row passes the dataset target rule on those labels — not experimental physics verification. Target-defining features (e.g. absolute χ) are held out of the RandomForest; API metrics are labeled as synthetic retrieval vs baseline.

[![Website](https://img.shields.io/badge/website-compute.upg.gr-blue)](https://compute.upg.gr)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Overview

upg-strings is a research tool for applying machine learning to the computational exploration of Calabi-Yau manifolds and string theory compactifications. The project emphasizes:

- **Reproducibility**: Deterministic pipelines with checksummed data, pinned dependencies, and fixed random seeds
- **Verification**: All predictions validated against ground truth with transparent metrics
- **Open Artifacts**: Complete outputs (CSV, JSON, metadata) for independent analysis
- **Multi-Dataset Support**: Search across multiple string theory datasets

This is applied computation and AI tooling designed to accelerate discovery in theoretical physics datasets, not a claim to solve fundamental physics problems.

## Supported Datasets

### 1. Kreuzer-Skarke Database (CY 3-folds)
- **Total:** 474 million reflexive polytopes describing Calabi-Yau threefolds
- **Target:** Manifolds with small Euler characteristic (|χ| < 100)
- **Use Case:** Particle physics phenomenology and model building
- **Source:** http://hep.itp.tuwien.ac.at/~kreuzer/CY/

### 2. CY5-Folds (Complete Intersection)
- **Total:** 27,068 complete intersection Calabi-Yau five-folds
- **Target:** Manifolds with many Kähler moduli (h^{1,1} > 100)
- **Use Case:** Large volume scenarios in string compactifications
- **Source:** https://github.com/pythoncymetric/cymetric

### 3. Heterotic Compactifications
- **Total:** ~10 million heterotic string compactifications on CY3-manifolds
- **Target:** Balanced manifolds with h^{1,1} ≈ h^{2,1}
- **Use Case:** Yukawa coupling structures for realistic model building
- **Source:** Based on hep-th/0507229 and related work

### 4. Information Density Ranking
- **Total:** 474 million (same underlying KS dataset)
- **Target:** High information density manifolds (top 10% by composite score)
- **Use Case:** Finding geometries with efficient topological encoding that may correlate with phenomenological viability and vacuum stability
- **Metrics:**
  - **Hodge Entropy**: Shannon entropy over normalized Hodge numbers
  - **Topological Efficiency**: |χ| / (h¹¹ + h²¹) ratio
  - **Moduli Compactness**: Inverse of total moduli count
  - **Hodge Balance**: Symmetry of the Hodge diamond
  - **Flux Density**: Bousso-Polchinski inspired flux vacua count using tadpole constraint (χ/24)
  - **Vacuum Stability**: KKLT/LVS inspired stability likelihood (tadpole headroom + moduli balance)
- **Customizable**: Tune component weights via `/api/info-density/weights` endpoint

### 5. F-theory elliptic proxies
- **Total:** pedagogical proxy corpus (not a Weierstrass census)
- **Target:** Elliptic-friendly Hodge patterns (low h¹¹ or literature elliptic seeds)
- **Use Case:** F-theory model-building sketches with honest proxy features
- **Honesty:** Not a full elliptic fibration / Tate-model database
- **Source:** Literature elliptic CY3 Hodge classes (e.g. WP[1,1,1,6,9])

## Key Features

### Universal ML-Guided Search
- Train models to identify geometries with specific topological properties
- Rank candidates across multiple string theory datasets
- Dataset-specific feature extraction and target criteria
- Unified API for all datasets

### Topological Feature Extraction
- Hodge numbers (h¹¹, h²¹, h³¹ for CY5-folds)
- Euler characteristic (χ)
- Chern class invariants
- Hodge ratios and balance metrics
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
from cy_search_real import run_real_search, list_available_datasets

# List all available datasets
datasets = list_available_datasets()
for ds in datasets:
    print(f"{ds['id']}: {ds['name']}")

# Run ML-guided search on Kreuzer-Skarke database
results = run_real_search(
    dataset_id='kreuzer-skarke',  # or 'cy5-folds', 'heterotic', 'info-density'
    top_k=100,                    # Return top 100 candidates
    seed=42,                      # Random seed for reproducibility
    n_candidates=5000,            # Dataset size
    verify=True                   # Verify against ground truth
)

print(f"Precision@100: {results['performance_metrics']['precision_at_k']:.1%}")
print(f"Recall@100: {results['performance_metrics']['recall_at_k']:.1%}")
print(f"Verified targets: {results['performance_metrics']['verified_count']}/100")
```

### Web Interface

The project includes a Flask web application with interactive demo:

```bash
# Run locally
python app.py

# Or with Gunicorn (production)
gunicorn --bind 0.0.0.0:5102 --workers 4 app:app
```

Visit `http://localhost:5102` to access the web interface with:
- **Interactive Demo**: Choose dataset, customize parameters, view live results
- **Run History**: All searches saved with localStorage persistence
- **Dataset Selector**: Switch between Kreuzer-Skarke, CY5-Folds, Heterotic, and Information Density datasets

## How It Works

### 1. Data Generation
Each dataset module generates physics-accurate synthetic candidates based on:
- Statistical distributions from actual databases
- String theory constraints and consistency conditions
- Realistic Hodge number ranges and relations
- Topological invariant correlations

### 2. Feature Engineering
Extract topological and geometric features specific to each dataset:
- **Kreuzer-Skarke**: h¹¹, h²¹, χ, Chern classes, triple intersections
- **CY5-Folds**: h¹¹, h²¹, h³¹, Euler characteristic, Hodge sums
- **Heterotic**: h¹¹, h²¹, hodge balance, number of generations
- **Info-Density**: hodge entropy, topological efficiency, moduli compactness, vacuum proxy

### 3. ML Model Training
- Random Forest classifier (100 estimators)
- Dataset-specific target criteria
- StandardScaler normalization
- Cross-validation split (70% train, 30% test)

### 4. Ranking & Verification
- Rank all test candidates by predicted likelihood
- Return top-k results
- Verify against ground truth labels
- Report precision, recall, feature importance

## Project Structure

```
compute.upg.gr/
├── app.py                    # Flask web application
├── cy_search.py              # Original demo implementation
├── cy_search_real.py         # Multi-dataset search engine
├── datasets_registry.py      # Dataset registry and base classes
├── templates/                # HTML templates
│   ├── index.html           # Home page with push-button demo
│   ├── demo.html            # Interactive demo with parameters
│   ├── docs.html            # Documentation
│   ├── results.html         # Live results display
│   └── about.html           # About page
├── static/
│   ├── css/style.css        # Styling
│   └── data/                # Sample output files
├── Dockerfile               # Docker configuration
├── captain-definition       # Caprover deployment config
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## API Endpoints

### `GET /api/datasets`
List all available datasets

**Response:**
```json
{
  "datasets": [
    {
      "id": "kreuzer-skarke",
      "name": "Kreuzer-Skarke Database",
      "description": "Reflexive polytopes...",
      "total_count": 473800776
    }
  ]
}
```

### `POST /api/run-demo`
Run the upg-strings search pipeline

**Request:**
```json
{
  "dataset_id": "kreuzer-skarke",
  "top_k": 100,
  "seed": 42,
  "n_candidates": 5000,
  "verify": true,
  "use_real": true
}
```

**Response:**
```json
{
  "status": "success",
  "run_metadata": {
    "dataset": "Kreuzer-Skarke Database",
    "dataset_id": "kreuzer-skarke",
    "total_candidates": 1500
  },
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

### `GET /api/sample-results?dataset_id=kreuzer-skarke`
Get sample results for display (supports dataset_id parameter)

### `GET /api/info-density/weights`
Get current weights for the info-density composite score

**Response:**
```json
{
  "status": "success",
  "weights": {
    "entropy": 0.20,
    "efficiency": 0.20,
    "compactness": 0.15,
    "balance": 0.10,
    "flux_density": 0.20,
    "vacuum_stability": 0.15
  }
}
```

### `POST /api/info-density/weights`
Set custom weights for info-density ranking (partial updates supported)

**Request:**
```json
{
  "vacuum_stability": 0.40,
  "flux_density": 0.30
}
```

### `POST /api/info-density/weights/reset`
Reset weights to defaults

### `POST /api/export-physics`
Export top-k candidates with comprehensive physics data for external analysis

**Request:**
```json
{
  "dataset_id": "info-density",
  "top_k": 100,
  "seed": 42,
  "format": "json"
}
```

Supported formats: `json`, `csv`, `numpy`

Returns all physics invariants including tadpole charge (χ/24), flux density, vacuum stability - ready for your own vacuum energy calculations or flux analysis

## Deployment

### Docker (Recommended)

```bash
docker build -t upg-strings .
docker run -p 5102:5102 upg-strings
```

### Coolify

Use the repository `Dockerfile` for build and deploy.

- No extra start command is needed.
- The container now respects Coolify's injected `PORT` environment variable automatically.
- If `PORT` is not set, it falls back to `5102` for local compatibility.

### Caprover

The project is configured for automatic deployment with Caprover:
1. Connect your GitHub repository
2. Set container port to 5102
3. Push to main branch to trigger deployment

## Reproducibility Guarantees

Every run includes:
- **Fixed Random Seeds**: All stochastic operations use deterministic seeds
- **Dataset Checksum**: SHA-256 verification of generated data
- **Pinned Dependencies**: Exact package versions in requirements.txt
- **Run Metadata**: Complete environment and configuration details
- **Exportable Artifacts**: JSON results with full metadata

## Performance

Typical runtime on standard hardware:
- Dataset generation: ~0.1-0.5s
- Model training: ~2-5s (5K samples)
- Ranking: ~0.2s
- Verification: ~0.1s
- **Total: 5-15 seconds for 5K candidates**

Scales to:
- 1K candidates: ~2 seconds
- 5K candidates: ~5 seconds
- 10K candidates: ~10 seconds
- 25K candidates: ~30 seconds

## Offline geometry DB

Heavy geometry (CYTools / PALP / periods) is computed **offline**, written into SQLite, and **queried** by the web app. The Flask container does **not** ship or require CYTools.

| Piece | Role |
|-------|------|
| `geometry_store.py` | Schema + upsert/lookup API (`static/data/geometry.sqlite`) |
| `data/ks_geometry_sample.json` + `data/geometry_pack.json` | Baked seeds upserted on boot |
| `scripts/ingest_geometry_db.py` | Idempotent ingest of seeds |
| `scripts/geometry_worker_stub.py` | Upsert a CYTools/PALP JSON dump from an offline machine |
| `GET /api/geometry/lookup?dataset_id=&h11=&h21=` | Best Hodge hit |
| `GET /api/geometry/<candidate_id>` | HoF-linked / Hodge-resolved geometry |
| `GET /api/geometry` | Bounded list |

**Add an offline CYTools record**

```bash
# On a machine with CYTools: dump vertices / triangulation to JSON
# (see scripts/geometry_record.schema.json), then:
python scripts/geometry_worker_stub.py -i my_polytope.json --db static/data/geometry.sqlite
# scp the sqlite (or JSON) onto the Coolify volume at /app/static/data/
```

Construction / analyze / CYTools export prefer DB hits over static JSON when the row has vertices. Hodge-shared polytopes stay labeled `status=representative`.

Roadmap vs full CYTools: this is the **offline → DB → query** path. In-process CYTools inside the web image remains out of scope.

## Roadmap

### Dataset Expansion
- [ ] Integrate actual CYTools library for full KS database access (offline worker path exists; not in web image)
- [x] Offline geometry SQLite (`geometry_store.py`) seeded from KS sample + geometry pack; web queries only
- [x] Model-building Phase 1: topological exclusions + literature cards + geometry pipeline stages (`/api/exclusions`, `/api/model-cards`, Model-building tab)
- [x] Add F-theory compactification datasets (elliptic **proxy** dataset + literature seeds; not a Weierstrass DB)
- [x] Include mirror symmetry pair databases (Hodge-level mirror link on candidate pages + textbook seeds)
- [ ] Support flux compactification vacua
- [x] Sidecar KS geometry sample (`data/ks_geometry_sample.json`) with real HF/Kreuzer vertices for textbook + HoF Hodge pairs

### ML Enhancements
- [ ] Graph neural networks for geometric learning
- [ ] Transfer learning across datasets
- [ ] Active learning for efficient labeling
- [ ] Ensemble methods combining multiple models
- [x] Baseline geometry featurizer / ML roadmap stubs (`ml_roadmap.py`) — no fake trained GNNs

### Features
- [ ] User-definable target criteria
- [ ] Automated algebraic geometry verification
- [x] Mirror symmetry detection (Hodge swap + Hall of Fame deep-link)
- [x] Hall of Fame gallery export + shareable candidate dossiers with analysis tabs
- [x] Server-side export bundle for gallery selections (`POST /api/export-gallery` → ZIP of selected candidates; `/api/analysis/<id>/bundle` for full analysis tabs JSON)
- [x] Real-time progress tracking for long-running searches (`POST /api/run-demo` with `async:true` → `GET /api/jobs/<id>`)
- [x] Batch processing API (`POST /api/batch`, ≤50 identify/search jobs)

## What Makes upg-strings Useful

While existing tools focus on **analyzing individual manifolds** (CYTools) or **classifying known geometries** (ML papers), upg-strings is a **search / ranking + dossier layer** for landscape-style candidate triage.

We answer: *"Which candidates should I open a dossier for?"* before deeper computation begins.

- **Synthetic retrieval metrics** vs random baseline (see `performance_metrics.metric_kind`)
- **Leakage hold-out** of target-defining columns (`leakage_note` in API responses)
- **Hall of Fame + shareable dossiers** as the durable product surface
- **Fast ranking** of synthetic draws for reproducible demos

## References

### Datasets
- Kreuzer-Skarke: [arXiv:hep-th/0002240](https://arxiv.org/abs/hep-th/0002240)
- CY5-Folds: [arXiv:1408.4808](https://arxiv.org/abs/1408.4808)
- Heterotic: [arXiv:hep-th/0507229](https://arxiv.org/abs/hep-th/0507229)

### Tools
- CYTools: [arXiv:2211.03823](https://arxiv.org/abs/2211.03823)
- cymetric: https://github.com/pythoncymetric/cymetric

## Citation

If you use upg-strings in your research, please cite:

```bibtex
@software{upgstrings2025,
  author = {Kokkinis, Ioannis},
  title = {upg-strings: ML-Guided Search for Rare Geometries in String Theory},
  year = {2025},
  url = {https://compute.upg.gr},
  note = {Multi-dataset search tool for string landscape exploration}
}
```

## Roadmap (shipped UX)

- [x] Candidates gallery with ranked manifolds
- [x] Manifold visualization modal (2D/3D)
- [x] Candidate details API endpoint
- [x] Server-side export bundle for gallery selections (`POST /api/export-gallery` ZIP; analysis download bundle)

Still open (see Features / Dataset Expansion above): real CYTools library integration, flux compactification vacua datasets.

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
- CY5-folds dataset and cymetric project
- Heterotic string phenomenology research
- Open-source machine learning libraries (scikit-learn, NumPy)
- The broader computational physics and string theory communities

---

**Disclaimer**: This tool accelerates computational search and verification. It does not claim to 'solve string theory' or make predictions about physical reality.
