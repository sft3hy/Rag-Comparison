# RAG for Charts & Tables: OCR vs Image-Indexing

A rigorous experimental framework for comparing Retrieval-Augmented Generation (RAG) approaches for querying charts and tables. This project implements and evaluates four main pipelines:

1. **ImageVec**: Direct image embedding using CLIP
2. **OCR-TextVec**: OCR extraction + text embedding
3. **Derender-TableVec**: Chart-to-table conversion + table embedding
4. **E2E-VisionLM**: End-to-end vision-language models

## 🎯 Project Goals

- Determine which RAG approach provides the best accuracy, robustness, and explainability for chart/table QA
- Conduct statistically rigorous comparisons across multiple benchmarks
- Provide reproducible code and comprehensive evaluation analysis

## 📋 Requirements

- Python 3.9-3.11
- CUDA-capable GPU (recommended, 16GB+ VRAM)
- 32GB+ RAM
- 50GB+ disk space for models and data

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd rag_charts_project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For GPU support, install FAISS-GPU
pip install faiss-gpu
```

### 2. Configuration

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your API keys
# OPENAI_API_KEY=your_key_here
# WANDB_API_KEY=your_key_here (optional)
```

### 3. Download Models & Data

```bash
# Download pretrained models
python scripts/download_models.py

# Download benchmark datasets (ChartQA, PlotQA, etc.)
python scripts/download_datasets.py
```

### 4. Process Data

```bash
# Preprocess images and detect figures
python run_ingest.py \
    --input-dir data/raw \
    --output-dir data/processed

# Run OCR on all images
python run_ocr.py \
    --input-dir data/processed \
    --output-file data/ocr_results/results.json \
    --engine tesseract

# Run chart derendering
python run_derender.py \
    --input-dir data/processed \
    --output-dir data/derendered
```

### 5. Build Indexes

```bash
# Build vector indexes for all modalities
python index_build.py \
    --config config/config.yaml \
    --data-dir data/processed \
    --output-dir indexes/
```

### 6. Query the System

```bash
# Query using OCR-TextVec pipeline
python query_rag.py \
    --query "What was the GDP growth in 2020?" \
    --pipeline ocr-text-vec \
    --k 5

# Query using all pipelines
python query_rag.py \
    --query "What was the GDP growth in 2020?" \
    --pipeline all \
    --output results/query_results.json
```

### 7. Run Evaluation

```bash
# Run full evaluation on test set
python eval.py \
    --config config/config.yaml \
    --test-data data/test \
    --output results/evaluation/

# Run ablation studies
python run_ablation.py \
    --config experiments/exp_002_ocr_comparison.yaml \
    --output results/ablations/
```

## 📁 Project Structure

```
rag_charts_project/
├── config/                 # Configuration files
├── data/                   # Data directory
│   ├── raw/               # Raw input data
│   ├── processed/         # Preprocessed images
│   ├── ocr_results/       # OCR outputs
│   └── derendered/        # Derendered tables
├── indexes/               # Vector store indexes
├── models/                # Downloaded models
├── src/                   # Source code
│   ├── ingestion/         # Data ingestion
│   ├── ocr/              # OCR engines
│   ├── derender/         # Derendering
│   ├── encoders/         # Embedding models
│   ├── index/            # Vector stores
│   ├── rag_pipelines/    # RAG implementations
│   ├── eval/             # Evaluation code
│   └── utils/            # Utilities
├── results/              # Evaluation results
├── notebooks/            # Analysis notebooks
└── tests/               # Unit tests
```

## 🧪 Running Experiments

### Baseline Comparison

```bash
python eval.py \
    --config experiments/exp_001_baseline.yaml \
    --datasets chartqa,plotqa \
    --pipelines all \
    --output results/baseline/
```

### OCR Engine Comparison

```bash
python run_ablation.py \
    --config experiments/exp_002_ocr_comparison.yaml \
    --engines tesseract,trocr,donut \
    --metrics recall@k,mrr,f1 \
    --output results/ocr_comparison/
```

### Encoder Ablation

```bash
python run_ablation.py \
    --config experiments/exp_003_encoder_ablation.yaml \
    --encoders clip,openclip,blip \
    --output results/encoder_ablation/
```

## 📊 Evaluation Metrics

- **Retrieval**: Recall@k, Mean Reciprocal Rank (MRR)
- **QA**: Exact Match (EM), Token F1, ROUGE-L, Numeric Accuracy
- **OCR Quality**: Character Error Rate (CER), Word Error Rate (WER)
- **Table Extraction**: Cell-level F1, Structure Accuracy
- **Statistical**: Paired t-tests, Wilcoxon tests, Bootstrap CIs

## 🔧 Configuration

Edit `config/config.yaml` to customize:

- OCR engines and parameters
- Embedding models
- Vector store settings (FAISS/Milvus)
- LLM configuration
- Evaluation metrics and thresholds
- Compute resources (GPU IDs, batch sizes)

## 📈 Experiment Tracking

Results are automatically logged to:
- **Weights & Biases** (W&B): Set `WANDB_API_KEY` in `.env`
- **MLflow**: Set `tracking.backend: mlflow` in config

View results:
```bash
# W&B
wandb login
# Then view at https://wandb.ai/<your-entity>/rag-charts

# MLflow
mlflow ui --port 5000
# Then view at http://localhost:5000
```

## 🧩 Using Individual Components

### OCR Only

```python
from src.ocr.engines import OCRManager
import cv2

config = {'engines': {'tesseract': {'enabled': True}}}
ocr = OCRManager(config)

image = cv2.imread('chart.png')
result = ocr.run_ocr(image, engine_name='tesseract')
print(result['text'])
```

### Image Embedding

```python
from src.encoders.embedders import ImageEmbedder
import numpy as np

embedder = ImageEmbedder(model_name='openai/clip-vit-base-patch32')
image = np.random.rand(224, 224, 3).astype(np.uint8)
embedding = embedder.embed_single(image)
print(embedding.shape)  # (512,)
```

### Vector Store

```python
from src.index.vector_store import FAISSVectorStore

store = FAISSVectorStore(dimension=512, index_type='HNSW')
embeddings = np.random.rand(100, 512)
ids = store.add(embeddings, metadata=[{'id': i} for i in range(100)])
store.save('my_index/')
```

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_ocr.py -v

# Run with coverage
pytest --cov=src tests/
```

## 📝 Generating Reports

```bash
# Generate comprehensive evaluation report
python src/eval/report_generator.py \
    --results results/evaluation/ \
    --output reports/final_report.pdf \
    --format pdf

# Generate comparison tables
python notebooks/02_results_analysis.ipynb
```

## 🐳 Docker Deployment

```bash
# Build image
docker build -t rag-charts .

# Run container
docker run -it --gpus all \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/results:/app/results \
    rag-charts

# Using docker-compose (includes Milvus)
docker-compose up -d
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{rag_charts_2024,
  title = {RAG for Charts and Tables: A Comparative Study},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/rag_charts_project}
}
```

## 🔗 References

- **ChartQA**: [arXiv:2203.10244](https://arxiv.org/abs/2203.10244)
- **DEPLOT**: [arXiv:2212.10505](https://arxiv.org/abs/2212.10505)
- **TrOCR**: [microsoft/trocr](https://huggingface.co/microsoft/trocr-base-printed)
- **TAPAS**: [google/tapas](https://github.com/google-research/tapas)

## 💬 Support

- 📧 Email: your.email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/rag_charts_project/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/rag_charts_project/discussions)

## 🗺️ Roadmap

- [ ] Add support for more OCR engines (PaddleOCR, EasyOCR)
- [ ] Implement multimodal fusion strategies
- [ ] Add real-time API endpoint
- [ ] Support for video/animated charts
- [ ] Interactive evaluation dashboard
- [ ] Cloud deployment guides (AWS, GCP, Azure)

## ⚠️ Known Issues

- TrOCR may require 16GB+ VRAM for batch processing
- Milvus requires separate installation for full functionality
- Some PDF table extractors may fail on scanned documents

## 🎓 Learning Resources

- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [Vector DB Comparison](https://github.com/erikbern/ann-benchmarks)
- [Chart QA Benchmark](https://github.com/vis-nlp/ChartQA)

---

**Built with ❤️ for rigorous ML research**