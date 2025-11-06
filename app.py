"""
Streamlit Frontend for the RAG for Charts & Tables Project.

This application provides an interactive interface to:
1.  Upload a chart and ask questions using a pre-built RAG pipeline.
2.  Run a small-scale benchmark evaluation to compare the performance of
    different OCR engines (Tesseract, TrOCR, Donut) on a sample dataset.

How to Run:
1. Make sure all dependencies from requirements.txt are installed.
2. Ensure you have run `scripts/download_models.py`.
3. From the project root, run: `streamlit run app.py`
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import cv2
import numpy as np
import json
import tempfile
import time

# --- Core Codebase Imports ---
import sys

sys.path.insert(0, str(Path(__file__).parent.absolute()))

from src.utils.config import load_config, Config, OCRConfig
from src.utils.logger import setup_logger
from src.ocr.engines import OCRManager
from src.encoders.embedders.embedder import EncoderManager
from src.index.vector_store import VectorStoreManager
from src.rag_pipelines.orchestrator import PipelineOrchestrator, OCRTextVecPipeline
from src.eval.metrics import MetricsCalculator

# --- App Configuration & Initialization ---

st.set_page_config(
    page_title="RAG for Charts & Tables",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

setup_logger(log_dir="logs", log_file="streamlit_app.log")


# --- Caching for Expensive Operations ---
@st.cache_resource
def get_config():
    return load_config()


@st.cache_resource
def get_encoder_manager(_config: Config):
    # st.write("`[Cache]` Loading Encoder Manager (CLIP, Sentence-Transformers)...")
    return EncoderManager(_config)


@st.cache_resource
def get_ocr_manager(_config: OCRConfig):  # <-- FIX: Expects the specific OCRConfig
    # st.write("`[Cache]` Loading OCR Manager (Tesseract, TrOCR, Donut)...")
    return OCRManager(_config)


# --- Helper Functions for Backend Logic (Unchanged) ---


def create_dummy_benchmark_assets(temp_dir: Path):
    """Creates a few dummy chart images and a labels file for benchmarking."""
    charts_dir = temp_dir / "charts"
    charts_dir.mkdir(parents=True)
    img1 = np.full((200, 600, 3), 255, dtype=np.uint8)
    cv2.putText(
        img1,
        "Annual Revenue is $500K",
        (50, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 0, 0),
        3,
    )
    cv2.imwrite(str(charts_dir / "chart1.png"), img1)
    img2 = np.full((200, 600, 3), 255, dtype=np.uint8)
    cv2.putText(
        img2,
        "User Growth: 25 percent",
        (50, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 0, 0),
        3,
    )
    cv2.imwrite(str(charts_dir / "chart2.png"), img2)
    labels = [
        {
            "question": "What is the annual revenue?",
            "answer": "500K",
            "relevant_doc_ids": ["chart1.png"],
        },
        {
            "question": "What was the growth for users?",
            "answer": "25%",
            "relevant_doc_ids": ["chart2.png"],
        },
    ]
    with open(temp_dir / "labels.jsonl", "w") as f:
        for label in labels:
            f.write(json.dumps(label) + "\n")
    return charts_dir, temp_dir / "labels.jsonl"


def run_ocr_process(
    ocr_manager: OCRManager, engine: str, input_dir: Path, output_file: Path
):
    """Replicates the logic of run_ocr.py."""
    image_paths = list(input_dir.glob("*.png"))
    with open(output_file, "w") as f_out:
        for image_path in image_paths:
            image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
            result = ocr_manager.run_ocr(image, engine_name=engine)
            result["image_path"] = image_path.name
            f_out.write(json.dumps(result) + "\n")


def run_indexing_process(
    encoder_manager: EncoderManager, input_path: Path, output_dir: Path
):
    """Replicates the logic of index_build.py for text."""
    vs_manager = VectorStoreManager(get_config())
    embedder = encoder_manager.text_embedder
    dimension = embedder.model.get_sentence_embedding_dimension()
    store = vs_manager.get_store(name=f"temp_index_{time.time()}", dimension=dimension)
    with open(input_path, "r") as f:
        lines = f.readlines()
    texts, metadatas = [], []
    for line in lines:
        data = json.loads(line)
        if data.get("text"):
            texts.append(data["text"])
            metadatas.append({"chunk_id": data["image_path"], "text": data["text"]})
    embeddings = embedder.embed(texts)
    store.add(embeddings, metadatas)
    store.save(str(output_dir))


def run_evaluation_process(
    encoder_manager: EncoderManager, index_dir: Path, labels_file: Path
) -> dict:
    """Replicates the logic of eval.py."""
    config = get_config()
    vs_manager = VectorStoreManager(config)
    orchestrator = PipelineOrchestrator(config)
    metrics_calculator = MetricsCalculator()
    dimension = encoder_manager.text_embedder.model.get_sentence_embedding_dimension()
    store = vs_manager.get_store(name=f"eval_store_{time.time()}", dimension=dimension)
    store.load(str(index_dir))
    pipeline = OCRTextVecPipeline(config, encoder_manager.text_embedder, store, {})
    orchestrator.register_pipeline("ocr-text-vec", pipeline)
    eval_data = []
    with open(labels_file, "r") as f:
        for line in f:
            eval_data.append(json.loads(line))
    results = []
    for item in eval_data:
        result = orchestrator.run_pipeline("ocr-text-vec", item["question"], k=1)
        qa_metrics = metrics_calculator.calculate_qa_metrics(
            result.get("answer", ""), item["answer"]
        )
        retrieved_ids = [
            doc["metadata"].get("chunk_id", "")
            for doc in result.get("retrieved_docs", [])
        ]
        retrieval_metrics = metrics_calculator.calculate_retrieval_metrics(
            item["relevant_doc_ids"], retrieved_ids
        )
        results.append({**qa_metrics, **retrieval_metrics})
    df_results = pd.DataFrame(results)
    return df_results.mean(numeric_only=True).to_dict()


# --- Main Application UI ---

st.title("📊 RAG for Charts & Tables: Interactive Demo")
st.markdown(
    "Welcome! This application allows you to interact with the RAG system in two ways: "
    "**Live Querying** on an uploaded image, or **Benchmark Evaluation** to compare OCR engines."
)

# --- Load shared resources ---
config = get_config()
encoder_manager = get_encoder_manager(config)

# --- FIX IS HERE ---
# Pass the correct sub-configuration (config.ocr) to the OCR manager loader.
ocr_manager = get_ocr_manager(config.ocr)
# --- END OF FIX ---

st.sidebar.title("App Mode")
app_mode = st.sidebar.radio(
    "Choose an operation mode:", ("Live Querying", "Benchmark Evaluation")
)

# --- LIVE QUERYING MODE ---
if app_mode == "Live Querying":
    st.header("💬 Live Querying")
    st.info(
        "Upload a chart image, select an OCR engine and a query to get a live answer. "
        "This demonstrates a simplified end-to-end RAG pipeline."
    )
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader(
            "Upload your chart image", type=["png", "jpg", "jpeg"]
        )
        if uploaded_file:
            bytes_data = uploaded_file.getvalue()
            cv2_img = cv2.imdecode(
                np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR
            )
            st.image(
                cv2_img,
                channels="BGR",
                caption="Uploaded Chart",
                width="stretch",
            )
    with col2:
        if uploaded_file:
            engine_choice = st.selectbox(
                "1. Choose OCR Engine", ("tesseract", "trocr", "donut")
            )
            query = st.text_input(
                "2. Enter your question", "What is the annual revenue?"
            )
            if st.button("🚀 Get Answer", use_container_width=True):
                with st.spinner(f"Processing with {engine_choice}..."):
                    ocr_result = ocr_manager.run_ocr(cv2_img, engine_name=engine_choice)
                    st.write("**Extracted Text:**")
                    st.text(ocr_result["text"][:500] + "...")
                    with tempfile.TemporaryDirectory() as temp_dir:
                        index_dir = Path(temp_dir) / "live_index"
                        index_dir.mkdir()
                        vs_manager = VectorStoreManager(config)
                        embedder = encoder_manager.text_embedder
                        dimension = embedder.model.get_sentence_embedding_dimension()
                        store = vs_manager.get_store(
                            name="live_query", dimension=dimension
                        )
                        embedding = embedder.embed_single(ocr_result["text"])
                        store.add(
                            np.array([embedding]),
                            [
                                {
                                    "chunk_id": uploaded_file.name,
                                    "text": ocr_result["text"],
                                }
                            ],
                        )
                        orchestrator = PipelineOrchestrator(config)
                        pipeline = OCRTextVecPipeline(config, embedder, store, {})
                        orchestrator.register_pipeline("live-pipeline", pipeline)
                        answer_result = orchestrator.run_pipeline(
                            "live-pipeline", query, k=1
                        )
                        st.divider()
                        st.write("### 💡 Answer")
                        st.success(
                            answer_result.get("answer", "Could not generate an answer.")
                        )
                        with st.expander("Show retrieved context"):
                            st.json(answer_result.get("retrieved_docs", []))

# --- BENCHMARK EVALUATION MODE ---
elif app_mode == "Benchmark Evaluation":
    st.header("🔬 Benchmark Evaluation")
    st.info(
        "This mode runs a small, predefined benchmark to compare the performance of the "
        "available OCR engines on a consistent set of questions. This is a CPU-intensive process!"
    )
    engine_to_eval = st.selectbox(
        "Choose an OCR engine to benchmark:", ("tesseract", "trocr", "donut")
    )
    if st.button(f"🚀 Run Benchmark for `{engine_to_eval}`", use_container_width=True):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with st.status(
                f"Running benchmark for **{engine_to_eval}**...", expanded=True
            ) as status:
                st.write("Step 1: Creating dummy benchmark assets...")
                charts_dir, labels_file = create_dummy_benchmark_assets(temp_path)
                st.write(
                    f"Step 2: Running **{engine_to_eval}** OCR on sample charts..."
                )
                ocr_output_file = temp_path / "ocr_results.jsonl"
                run_ocr_process(
                    ocr_manager, engine_to_eval, charts_dir, ocr_output_file
                )
                st.write("Step 3: Building vector index from OCR results...")
                index_dir = temp_path / "index"
                index_dir.mkdir()
                run_indexing_process(encoder_manager, ocr_output_file, index_dir)
                st.write("Step 4: Running evaluation against ground truth...")
                summary_metrics = run_evaluation_process(
                    encoder_manager, index_dir, labels_file
                )
                st.session_state[f"results_{engine_to_eval}"] = summary_metrics
                status.update(
                    label=f"Benchmark for **{engine_to_eval}** complete!",
                    state="complete",
                    expanded=False,
                )
    st.divider()
    st.subheader("📊 Evaluation Results")
    st.markdown(
        "Results from benchmark runs will appear here. Run for multiple engines to compare them."
    )
    results_data = []
    for engine in ("tesseract", "trocr", "donut"):
        if f"results_{engine}" in st.session_state:
            res = st.session_state[f"results_{engine}"]
            res["engine"] = engine
            results_data.append(res)
    if results_data:
        df_results = pd.DataFrame(results_data).set_index("engine")
        st.dataframe(df_results.style.format("{:.3f}"))
        last_engine = results_data[-1]["engine"]
        st.write(f"#### Metrics for `{last_engine}`:")
        cols = st.columns(len(results_data[-1]) - 1)
        i = 0
        for metric, value in results_data[-1].items():
            if metric != "engine":
                cols[i].metric(metric.replace("_", " ").title(), f"{value:.3f}")
                i += 1
