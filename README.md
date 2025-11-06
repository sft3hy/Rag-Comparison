# RAG for Charts & Tables

An end-to-end Python framework for building, comparing, and evaluating Retrieval-Augmented Generation (RAG) pipelines on visual data like charts and tables.

[Installation](docs/INSTALLATION.md) • [Usage Guide](docs/USAGE.md) • [Evaluation Details](docs/EVALUATION.md) • [API Docs](docs/API.md)

---

![Streamlit App Demo](https://i.imgur.com/your-demo-gif-url.gif)
*(Placeholder: A GIF showing the interactive Streamlit app in action)*

## ✨ Key Features

*   **Compare Four RAG Pipelines:** Systematically evaluate and compare distinct strategies for querying charts:
    1.  **ImageVec**: Direct visual search using CLIP embeddings.
    2.  **OCR-TextVec**: OCR text extraction with semantic text search.
    3.  **Derender-TableVec**: Chart-to-table conversion for structured data retrieval.
    4.  **E2E-VisionLM**: End-to-end querying with multimodal models.
*   **Interactive Streamlit UI:** A user-friendly web app (`app.py`) to run live queries and trigger benchmark comparisons directly from your browser.
*   **Rigorous Evaluation Suite:** Built-in support for comprehensive metrics (Recall@K, MRR, F1, Numeric Accuracy) and statistical significance testing.
*   **Extensible & Modular:** Easily add new models, OCR engines, or vector stores thanks to a clean, object-oriented design.
*   **Production-Ready:** Includes Docker support, a FastAPI endpoint, and experiment tracking with W&B or MLflow.

## 🚀 Quick Start: Interactive Demo in 5 Minutes

The fastest way to see the project in action is to run the Streamlit web application.

### 1. Clone & Install
First, set up the repository and install the required dependencies.
```bash
# Clone the repository
git clone https://github.com/yourusername/rag_charts_project.git
cd rag_charts_project

# Create a virtual environment and install packages
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment
Copy the example environment file and add your OpenAI API key.
```bash
cp .env.example .env
# Now, edit the .env file with your API key
```

### 3. Run the App!
Launch the Streamlit application.
```bash
streamlit run app.py
```
Your browser will open with the interactive UI, where you can upload charts and run benchmark comparisons.

---

## ⚙️ Core Workflow: From Data to Answers

To process your own collection of documents, follow this four-step pipeline.

#### 1. Ingest Documents (`run_ingest.py`)
Preprocess images and automatically extract chart/figure regions.
```bash
python run_ingest.py --input-dir data/raw --output-dir data/processed
```

#### 2. Extract Information (`run_ocr.py`)
Run an OCR engine (e.g., Tesseract) on the extracted figures.
```bash
python run_ocr.py \
    --input-dir data/processed/figures \
    --output-file data/ocr_results/tesseract.jsonl \
    --engine tesseract
```

#### 3. Build Search Index (`index_build.py`)
Create a searchable vector store index from the OCR text.
```bash
python index_build.py \
    --index-type text \
    --input-path data/ocr_results/tesseract.jsonl \
    --output-dir outputs/indexes/my_text_index
```

#### 4. Ask a Question (`query_rag.py`)
Query the system using the pipeline and index you built.
```bash
python query_rag.py \
    --pipeline-name "ocr-text-vec" \
    --index-dir outputs/indexes/my_text_index \
    --query "What was the revenue in the final quarter?"
```

## 🔬 Running Experiments & Evaluation

This framework is built for rigorous benchmarking. For complete details, see the [**Evaluation Methodology**](docs/EVALUATION.md).

*   **Run a Full Benchmark:** Use `eval.py` to run a pipeline against a labeled dataset (like ChartQA) and compute all performance metrics.
    ```bash
    python eval.py \
        --pipeline-name "ocr-text-vec" \
        --index-dir outputs/indexes/chartqa_index \
        --eval-dataset data/raw/chartqa/test.jsonl \
        --output-dir outputs/eval_results/baseline_run
    ```
*   **Metrics Measured:** Includes Retrieval (Recall@K, MRR), QA (F1, Numeric Accuracy), and Component (CER, WER) metrics.
*   **Experiment Tracking:** Results are automatically logged to **Weights & Biases** or **MLflow** for easy comparison and visualization.

## 📁 Project Structure

The repository is organized into distinct modules for clarity and extensibility.

```
rag_charts_project/
├── config/                # Project and experiment configurations
├── data/                  # Datasets (raw, processed, synthetic)
├── docs/                  # Detailed documentation
├── outputs/               # Saved indexes, evaluation results, etc.
├── scripts/               # Helper scripts (downloading, etc.)
├── src/                   # All core source code
│   ├── ingestion/         # Data loading and preprocessing
│   ├── ocr/               # OCR engine implementations
│   ├── derender/          # Chart-to-table logic
│   ├── encoders/          # Embedding models
│   ├── index/             # Vector store logic (FAISS, Milvus)
│   ├── rag_pipelines/     # The four RAG pipeline implementations
│   └── eval/              # Evaluation metrics and runners
├── tests/                 # Unit and integration tests
├── app.py                 # The Streamlit interactive application
└── requirements.txt       # Project dependencies
```

## 🤝 Contributing

Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request with your changes.