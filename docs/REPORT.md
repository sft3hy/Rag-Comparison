## Report & Guide: RAG System for Charts & Tables

### 1. Executive Summary

**Problem:** Our organization possesses a vast number of documents containing charts, graphs, and tables. The valuable, structured data within these visualizations is "locked," making it impossible to query or analyze at scale. Answering a simple question like "What was the revenue trend in Q4 across all project reports?" requires manually finding and interpreting hundreds of charts, a process that is slow, expensive, and error-prone.

**Solution:** This project implements a sophisticated **Retrieval-Augmented Generation (RAG)** system specifically designed to unlock the data within charts and tables. The system allows users to ask questions in natural language and receive precise, data-driven answers by intelligently finding the most relevant chart and interpreting its content. We have developed four distinct AI pipelines to compare different methods, from simple image-based retrieval to advanced chart-to-table conversion, ensuring we can identify the most effective approach for our needs.

**Impact:**
*   **Drastic Time Reduction:** Reduces the time required to find specific data points from hours or days to mere seconds.
*   **Unlocks Hidden Insights:** Enables large-scale analysis across thousands of documents, revealing trends and correlations that are currently undiscoverable.
*   **Improves Data-Driven Decisions:** Empowers all team members, not just data analysts, to get immediate answers from visual data, fostering a more agile and informed decision-making culture.
*   **Future-Proof & Scalable:** The system is built on a modern, modular, and scalable architecture using Docker and industry-standard AI models, ready for integration into internal dashboards or applications.

---

### 2. How It Works: System Architecture

The system operates in two main phases: **Offline Indexing** (preparing the data) and **Online Querying** (answering questions).

#### **Phase 1: Offline Indexing (Ingestion & Processing)**

This is the one-time process of preparing your documents.
1.  **Ingestion (`run_ingest.py`):** The system scans a directory of documents or images. An `ImagePreprocessor` cleans the images (resizing, denoising), and a `FigureDetector` automatically identifies and crops out the charts and tables from the surrounding page content.
2.  **Information Extraction (`run_ocr.py`, `run_derender.py`):** Each extracted chart is processed to extract its information in different ways:
    *   **OCR:** Optical Character Recognition (Tesseract, TrOCR) reads all the text on the chart (titles, labels, numbers).
    *   **Derendering:** A vision model (DePlot) "derenders" the chart, attempting to reconstruct the original data table that was used to create the visualization.
3.  **Embedding & Indexing (`index_build.py`):** The extracted information (the image itself, the OCR text, or the derendered table) is converted into a numerical representation called an **embedding** using advanced AI models (e.g., CLIP, Sentence-Transformers). These embeddings are stored in a specialized **Vector Store** (FAISS or Milvus), which acts like a high-speed search index for finding similar charts.

#### **Phase 2: Online Querying (`query_rag.py`)**

This is what happens when a user asks a question.
1.  **Query Embedding:** The user's question (e.g., "What was the GDP growth in 2020?") is converted into an embedding using the same AI model.
2.  **Retrieval:** The system searches the Vector Store to find the chart embeddings that are most semantically similar to the question's embedding. This efficiently retrieves the top-K most relevant charts from the entire collection.
3.  **Augmentation & Generation:** The retrieved information (the context) is passed to a powerful Large Language Model (LLM), along with the original question. The LLM receives a carefully crafted prompt, such as: *"Based on the following OCR text from a chart, answer the question: 'What was the GDP growth in 2020?' Context: ..."*
4.  **Answer Synthesis:** The LLM analyzes the provided context and synthesizes a precise, natural language answer.

---

### 3. Core RAG Pipelines Explained

The project implements four different strategies to find the best performance trade-offs.

1.  **ImageVec (Image Embedding):**
    *   **How it works:** Directly embeds the chart image using a vision model like CLIP. Retrieval is based on visual similarity.
    *   **Pros:** Very fast, good for finding visually similar charts.
    *   **Cons:** Poor at understanding specific numbers or text within the chart.

2.  **OCR-TextVec (OCR + Text Embedding):**
    *   **How it works:** Extracts all text from the chart via OCR, then embeds this text. Retrieval is based on textual similarity.
    *   **Pros:** Excellent for questions involving text, titles, or labels present on the chart.
    *   **Cons:** Its performance is entirely dependent on the quality of the OCR. Errors in number recognition can lead to wrong answers.

3.  **Derender-TableVec (Chart-to-Table + Table Embedding):**
    *   **How it works:** Converts the chart image back into a structured data table, which is then embedded.
    *   **Pros:** The most robust method for answering precise, numerical questions, as it understands the chart's underlying structure.
    *   **Cons:** Technologically challenging and can fail on complex or unusual chart types.

4.  **E2E-VisionLM (End-to-End Vision Model):**
    *   **How it works:** Uses a multimodal model (like Donut) that can directly answer questions from an image without separate OCR or embedding steps.
    *   **Pros:** Simpler pipeline.
    *   **Cons:** A newer technology that may be less accurate for complex reasoning compared to the multi-step RAG approach.

---

### 4. Technology Stack & Key Dependencies

*   **Programming Language:** Python 3.9+
*   **Deep Learning:** PyTorch, Hugging Face Transformers (for accessing state-of-the-art models).
*   **Vector Search:**
    *   **FAISS:** A highly efficient similarity search library from Facebook AI, ideal for fast, in-memory indexing.
    *   **Milvus:** A production-grade, open-source vector database for large-scale, distributed deployments.
*   **Core AI Models & Repositories:**
    *   **Image Embeddings:** `openai/clip-vit-base-patch32`
    *   **Text Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
    *   **OCR:** `pytesseract` (Tesseract), `microsoft/trocr-base-printed` (TrOCR)
    *   **Chart Derendering:** `google/deplot`
    *   **LLMs for Generation:** `meta-llama/llama-4-scout-17b-16e-instruct`
*   **Datasets for Benchmarking:** ChartQA, PlotQA, PubTabNet.
*   **Deployment & API:** Docker, Docker Compose, FastAPI.
*   **Experiment Tracking:** Weights & Biases (Wandb) / MLflow.

---

### 5. How to Use the System: A Step-by-Step Guide

Follow these steps to set up the project and run your first query.

#### **Step 1: Setup the Environment**
Clone the repository, create a virtual environment, and install all dependencies.
```bash
# 1. Clone the repo
git clone <your-repo-url>
cd rag_charts_project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup API Keys
cp .env.example .env
# Now, edit the .env file and add your OpenAI/Wandb API keys
```

#### **Step 2: Download Models and Datasets**
Run the provided scripts to download all necessary AI models and the datasets for benchmarking.
```bash
# Download pre-trained models from Hugging Face
python scripts/download_models.py

# Download the ChartQA and other datasets
python scripts/download_datasets.py --dataset all
```

#### **Step 3: Run the Ingestion Pipeline**
Process your raw images/documents to extract the charts. Place your documents (e.g., PNG files) in a directory like `data/raw/my_reports`.
```bash
# This command will find charts in 'my_reports' and save them to 'processed/my_reports_figures'
python run_ingest.py \
    --input-dir data/raw/my_reports \
    --output-dir data/processed/my_reports_processed
```

#### **Step 4: Extract Information (OCR or Derender)**
Run OCR on the extracted figures. This creates a text representation of each chart.
```bash
# Run Tesseract OCR on the extracted figures
python run_ocr.py \
    --input-dir data/processed/my_reports_processed/figures \
    --output-file data/processed/ocr_results.jsonl \
    --engine tesseract
```

#### **Step 5: Build the Search Index**
Create the vector store index from the extracted OCR text.
```bash
# Create a FAISS index from the OCR results
python index_build.py \
    --index-type text \
    --input-path data/processed/ocr_results.jsonl \
    --output-dir outputs/indexes/my_reports_text_index
```

#### **Step 6: Ask a Question!**
You are now ready to query the system.
```bash
# Run a query against the OCR-TextVec pipeline using the index you just built
python query_rag.py \
    --pipeline-name "ocr-text-vec" \
    --index-dir outputs/indexes/my_reports_text_index \
    --query "What were the sales figures for the Alpha project in June?"
```

---

### 6. How to Benchmark and Compare Pipelines

This system is built for rigorous evaluation. The key is to run the standardized evaluation script against a dataset with known question-answer pairs.

#### **Performance Metrics**

We measure performance in two key areas:
1.  **Retrieval Metrics (Did we find the right chart?):**
    *   **Recall@K:** What percentage of the time was the correct chart in the top K retrieved results?
    *   **MRR (Mean Reciprocal Rank):** How high up in the ranking was the correct chart, on average?
2.  **QA Metrics (Did we answer the question correctly?):**
    *   **Exact Match (EM):** Did the generated answer exactly match the ground truth?
    *   **F1 Score:** A score that balances precision and recall, good for text-based answers.
    *   **Numeric Accuracy:** For numerical answers, is the predicted number close to the actual number?

#### **Running a Benchmark**

The `eval.py` script automates this process. It runs every question from a dataset against a pipeline and calculates the average metrics.

```bash
# Example: Evaluate the ocr-text-vec pipeline on the synthetic dataset
python eval.py \
    --pipeline-name "ocr-text-vec" \
    --index-dir outputs/indexes/synthetic_text_index \
    --eval-dataset data/synthetic/labels.jsonl \
    --output-dir outputs/eval_results/baseline_tesseract \
    --experiment-name "baseline_tesseract_on_synthetic"
```
The results, including a detailed CSV and a summary JSON, will be saved to the output directory. If configured, they will also be logged to W&B for easy comparison.

#### **Running Ablation Studies**

To systematically compare different configurations (e.g., Tesseract vs. TrOCR), use the `run_ablation.py` script with an experiment configuration file. This allows you to automatically run a matrix of benchmarks to find the optimal system setup.