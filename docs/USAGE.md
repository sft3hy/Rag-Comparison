# Usage Guide

This guide details the end-to-end workflow for using the RAG for Charts & Tables system, from processing raw documents to asking questions and evaluating performance. All commands should be run from the root directory of the project.

## Core Workflow: From Documents to Answers

The primary workflow involves four main steps: Ingestion, Information Extraction, Indexing, and Querying.

### Step 1: Ingestion (`run_ingest.py`)
This script processes your source documents (e.g., PNG or JPG files containing charts), cleans them, and automatically extracts the chart/figure regions.

**Usage:**
```bash
python run_ingest.py \
    --input-dir path/to/your/raw/images \
    --output-dir path/to/save/processed/data
```
*   `--input-dir`: A folder containing your source images.
*   `--output-dir`: A folder where the cleaned images and extracted figures will be saved.

**Example:**
```bash
python run_ingest.py --input-dir data/raw --output-dir data/processed
```
This will create `data/processed/figures` containing the cropped-out charts.

### Step 2: Information Extraction (`run_ocr.py`)
This script runs an OCR engine on the extracted figures to convert the visual text into machine-readable text.

**Usage:**
```bash
python run_ocr.py \
    --input-dir data/processed/figures \
    --output-file data/ocr_results/ocr_results.jsonl \
    --engine tesseract
```
*   `--input-dir`: The `figures` directory created in the previous step.
*   `--output-file`: The output `.jsonl` file where each line contains the OCR text for one figure.
*   `--engine`: The OCR engine to use.

**Example:**```bash
python run_ocr.py \
    --input-dir data/processed/figures \
    --output-file data/ocr_results/tesseract.jsonl \
    --engine tesseract
```

### Step 3: Building the Search Index (`index_build.py`)
This script takes the extracted information (text, images, or tables) and builds a searchable vector store index.

**Usage:**
```bash
python index_build.py \
    --index-type [text|image|table] \
    --input-path data/processed/ingestion_metadata.jsonl \
    --output-dir data/FAISS_index
```
*   `--index-type`: The modality to index. `text` uses the OCR output, `image` uses the figure images directly.
*   `--input-path`: Path to the source data (e.g., the `.jsonl` from OCR).
*   `--output-dir`: Folder to save the FAISS index files.

**Example (building a text index):**
```bash
python index_build.py \
    --index-type text \
    --input-path data/ocr_results/tesseract.jsonl \
    --output-dir outputs/indexes/text_index
```

### Step 4: Querying the System (`query_rag.py`)
You can now ask questions! This script uses a specified pipeline and index to answer your natural language query.

**Usage:**
```bash
python query_rag.py \
    --pipeline-name [ocr-text-vec|image-vec|...] \
    --index-dir /path/to/your/index \
    --query "Your question about the charts"
```
*   `--pipeline-name`: The RAG strategy to use. This must match the type of index you built.
*   `--index-dir`: The path to the index created in Step 3.
*   `--query`: The question you want to ask.

**Example:**
```bash
python query_rag.py \
    --pipeline-name "ocr-text-vec" \
    --index-dir outputs/indexes/text_index \
    --query "What was the revenue in the final quarter?"
```

---

## Benchmarking and Evaluation

The repository includes powerful tools for systematically evaluating the performance of different pipelines.

### Running a Single Evaluation (`eval.py`)
This script runs a pipeline against a labeled dataset (like ChartQA) and computes all performance metrics (Recall, MRR, F1, etc.).

**Example:**
```bash
python eval.py \
    --pipeline-name "ocr-text-vec" \
    --index-dir outputs/indexes/text_index \
    --eval-dataset data/raw/chartqa/test_human.jsonl \
    --output-dir outputs/eval_results/baseline_run \
    --experiment-name "baseline_tesseract_on_chartqa"
```
This will save a `summary.json` and a detailed `evaluation_results.csv` in the output directory, and log the run to W&B if configured.

### Running Ablation Studies (`run_ablation.py`)
This script orchestrates a series of evaluation runs based on a configuration file, making it easy to compare different models, engines, and settings.

**Example:**
```bash
python run_ablation.py \
    --ablation-config experiments/exp_002_ocr_comparison.yaml \
    --eval-dataset data/raw/chartqa/test_human.jsonl \
    --base-output-dir outputs/ablation_studies/ocr_comparison
```
This will execute all the experimental runs defined in the YAML config file, saving the results for each in a separate subdirectory.