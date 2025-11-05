# RAG Charts Project - Complete File Structure

This document outlines all Python files needed for the complete implementation of the RAG for Charts & Tables project.

## ✅ Files Already Created

### Configuration & Utilities
1. `requirements.txt` - All dependencies
2. `config/config.yaml` - Main configuration file
3. `src/utils/config.py` - Configuration management
4. `src/utils/logger.py` - Logging setup using loguru
5. `src/utils/tracking.py` - Experiment tracking (W&B/MLflow)

### Core Modules
6. `src/ingestion/preprocessor.py` - Image preprocessing and figure detection
7. `src/ocr/engines.py` - OCR engines (Tesseract, TrOCR, Donut)
8. `src/derender/extractors.py` - Chart derendering and table extraction
9. `src/encoders/embedders.py` - Image, text, and table embedders
10. `src/index/vector_store.py` - FAISS and Milvus vector stores
11. `src/rag_pipelines/orchestrator.py` - RAG pipeline implementations

## 📝 Additional Files Needed

### Main Scripts
12. **`run_ingest.py`** - Main ingestion script
```python
# Orchestrates: image loading, preprocessing, figure detection, saving metadata
# CLI interface for batch processing documents
```

13. **`run_ocr.py`** - OCR processing script
```python
# Runs OCR engines on images, saves results with confidence scores
# Supports parallel processing, error handling, and logging
```

14. **`run_derender.py`** - Chart derendering script
```python
# Runs chart-to-table conversion and table extraction
# Saves structured CSV/JSON outputs
```

15. **`index_build.py`** - Index building script
```python
# Builds FAISS/Milvus indexes from embeddings
# Handles image, text, and table embeddings separately
```

16. **`query_rag.py`** - Query interface script
```python
# CLI/API for querying RAG pipelines
# Supports all pipeline variants via flags
```

### Evaluation Module
17. **`src/eval/metrics.py`** - Evaluation metrics
```python
# Functions for: Recall@k, MRR, Exact Match, F1, ROUGE-L, CER, WER
# Statistical significance testing
```

18. **`src/eval/evaluator.py`** - Main evaluator class
```python
# Orchestrates evaluation across pipelines and datasets
# Generates comparison tables and visualizations
```

19. **`eval.py`** - Evaluation runner script
```python
# Runs full evaluation suite on test sets
# Generates reports and saves results
```

20. **`run_ablation.py`** - Ablation study runner
```python
# Runs matrix of experiments (engines × encoders × settings)
# Logs all results to experiment tracking
```

### Data Generation
21. **`data/synthetic/generate_charts.py`** - Synthetic chart generator
```python
# Uses matplotlib/plotly to generate synthetic charts
# Creates controlled variants (fonts, noise, rotations, etc.)
# Saves ground-truth CSV data
```

22. **`src/ingestion/dataset_loader.py`** - Dataset loaders
```python
# Loaders for ChartQA, PlotQA, PubTabNet, DocVQA
# Standardized interface for all datasets
```

### LLM Integration
23. **`src/rag_pipelines/llm_client.py`** - LLM client abstraction
```python
# Unified interface for OpenAI, Anthropic, local models
# Handles rate limiting, retries, token counting
```

24. **`src/rag_pipelines/prompts.py`** - Prompt templates
```python
# Templates for different pipeline types
# Includes few-shot examples and formatting
```

### Analysis & Reporting
25. **`notebooks/01_exploratory_analysis.ipynb`** - EDA notebook
```python
# Dataset statistics, visualization examples
# OCR quality analysis, embedding space visualization
```

26. **`notebooks/02_results_analysis.ipynb`** - Results analysis
```python
# Comparative analysis of pipelines
# Statistical testing, error analysis
# Publication-ready plots
```

27. **`src/eval/report_generator.py`** - Report generator
```python
# Generates final report (markdown/PDF)
# Creates tables, plots, failure analysis
```

### Utilities
28. **`src/utils/data_utils.py`** - Data utilities
```python
# Common data processing functions
# File I/O, format conversions, batch processing
```

29. **`src/utils/visualization.py`** - Visualization utilities
```python
# Plotting functions for embeddings, attention, results
# Chart overlay visualization with bboxes
```

30. **`src/utils/text_processing.py`** - Text processing utilities
```python
# Text normalization, numeric extraction
# Table flattening helpers
```

### Testing
31. **`tests/test_ocr.py`** - OCR engine tests
32. **`tests/test_embedders.py`** - Embedder tests
33. **`tests/test_vector_store.py`** - Vector store tests
34. **`tests/test_pipelines.py`** - Pipeline tests
35. **`tests/test_evaluation.py`** - Evaluation tests

### Docker & Deployment
36. **`Dockerfile`** - Main Dockerfile
```dockerfile
# Multi-stage build for production
# Includes all dependencies and models
```

37. **`docker-compose.yml`** - Docker compose for services
```yaml
# Services: app, milvus, postgres (for metadata)
```

38. **`scripts/download_models.py`** - Model downloader
```python
# Downloads all required HuggingFace models
# Sets up cache directories
```

### API (Optional)
39. **`api/app.py`** - FastAPI application
```python
# REST API for RAG queries
# Endpoints for each pipeline type
```

40. **`api/schemas.py`** - API schemas
```python
# Pydantic models for requests/responses
```

### Documentation
41. **`README.md`** - Main README
42. **`docs/INSTALLATION.md`** - Installation guide
43. **`docs/USAGE.md`** - Usage guide
44. **`docs/EVALUATION.md`** - Evaluation methodology
45. **`docs/API.md`** - API documentation

### Configuration
46. **`.env.example`** - Environment variables template
47. **`.gitignore`** - Git ignore file
48. **`setup.py`** - Package setup script
49. **`pyproject.toml`** - Modern Python project config

### Experiment Configs
50. **`experiments/exp_001_baseline.yaml`** - Baseline experiment
51. **`experiments/exp_002_ocr_comparison.yaml`** - OCR comparison
52. **`experiments/exp_003_encoder_ablation.yaml`** - Encoder ablation

## 🏗️ Directory Structure

```
rag_charts_project/
├── config/
│   ├── config.yaml
│   └── experiment_configs/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── labels/
│   └── synthetic/
│       └── generate_charts.py
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── preprocessor.py ✅
│   │   └── dataset_loader.py
│   ├── ocr/
│   │   ├── __init__.py
│   │   └── engines.py ✅
│   ├── derender/
│   │   ├── __init__.py
│   │   └── extractors.py ✅
│   ├── encoders/
│   │   ├── __init__.py
│   │   └── embedders.py ✅
│   ├── index/
│   │   ├── __init__.py
│   │   └── vector_store.py ✅
│   ├── rag_pipelines/
│   │   ├── __init__.py
│   │   ├── orchestrator.py ✅
│   │   ├── llm_client.py
│   │   └── prompts.py
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── evaluator.py
│   │   └── report_generator.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py ✅
│       ├── logger.py ✅
│       ├── tracking.py ✅
│       ├── data_utils.py
│       ├── visualization.py
│       └── text_processing.py
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   └── 02_results_analysis.ipynb
├── tests/
│   ├── __init__.py
│   ├── test_ocr.py
│   ├── test_embedders.py
│   ├── test_vector_store.py
│   ├── test_pipelines.py
│   └── test_evaluation.py
├── api/
│   ├── __init__.py
│   ├── app.py
│   └── schemas.py
├── scripts/
│   ├── download_models.py
│   └── setup_env.sh
├── experiments/
│   ├── exp_001_baseline.yaml
│   ├── exp_002_ocr_comparison.yaml
│   └── exp_003_encoder_ablation.yaml
├── docs/
│   ├── INSTALLATION.md
│   ├── USAGE.md
│   ├── EVALUATION.md
│   └── API.md
├── run_ingest.py ✅
├── run_ocr.py ✅
├── run_derender.py ✅
├── index_build.py ✅
├── query_rag.py ✅
├── eval.py ✅
├── run_ablation.py ✅
├── requirements.txt ✅
├── Dockerfile
├── docker-compose.yml
├── setup.py
├── pyproject.toml
├── .env.example
├── .gitignore
├── README.md
└── PROJECT_STRUCTURE.md ✅
```

## 🚀 Quick Start Commands

```bash
# 1. Setup environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Download models
python scripts/download_models.py

# 3. Process data
python run_ingest.py --input data/raw --output data/processed

# 4. Run OCR
python run_ocr.py --input data/processed --engine tesseract,trocr

# 5. Build indexes
python index_build.py --config config/config.yaml

# 6. Query
python query_rag.py --query "What was the GDP in 2020?" --pipeline ocr-text-vec

# 7. Evaluate
python eval.py --config config/config.yaml --output results/

# 8. Run ablations
python run_ablation.py --config experiments/exp_002_ocr_comparison.yaml
```

## 📊 Key Implementation Notes

### Priority Order for Remaining Files:
1. **High Priority** (Core functionality):
   - run_ingest.py, run_ocr.py, index_build.py, query_rag.py
   - src/eval/metrics.py, src/eval/evaluator.py
   - src/rag_pipelines/llm_client.py

2. **Medium Priority** (Analysis & Testing):
   - eval.py, run_ablation.py
   - src/utils/data_utils.py, text_processing.py
   - tests/* files

3. **Low Priority** (Nice to have):
   - API files, notebooks, docs
   - Deployment files (Docker, etc.)

### Design Principles Applied:
- ✅ Modular architecture with clear separation of concerns
- ✅ Consistent interfaces across all components
- ✅ Comprehensive error handling and logging
- ✅ Configuration-driven design (no hardcoded values)
- ✅ Type hints throughout
- ✅ Experiment tracking integration
- ✅ Statistical rigor in evaluation

### Next Steps:
1. Implement remaining high-priority files
2. Create minimal test suite
3. Generate synthetic dataset for testing
4. Run end-to-end smoke test
5. Begin baseline experiments