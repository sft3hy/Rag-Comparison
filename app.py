"""
Streamlit Frontend for the RAG for Charts & Tables Project.

This application provides an interactive interface to:
1.  Upload a document and ask questions using a live RAG pipeline.
2.  Run a benchmark evaluation on a pre-defined set of real processed images
    to compare the performance of different OCR engines.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import cv2
import numpy as np
import json
import tempfile
import time
from typing import List, Dict

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
from pdf2image import convert_from_path

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
    return EncoderManager(_config)


@st.cache_resource
def get_ocr_manager(_config: OCRConfig):
    return OCRManager(_config)


# --- NEW: Define a Static Benchmark Using Real Images ---
BENCHMARK_IMAGE_DIR = Path("data/processed/processed")
BENCHMARK_SET = {
    "images": ["34.png", "43.png", "46.png", "74.png", "86.png"],
    "labels": [
        # NOTE: You will need to manually create the correct questions and answers for your images.
        # These are illustrative placeholders.
        {
            "question": "What is the value for the 'East' region?",
            "answer": "2500",
            "relevant_doc_ids": ["34.png"],
        },
        {
            "question": "Which category had the highest value in the chart?",
            "answer": "Category C",
            "relevant_doc_ids": ["43.png"],
        },
        {
            "question": "What was the percentage in 2022?",
            "answer": "45%",
            "relevant_doc_ids": ["46.png"],
        },
        {
            "question": "How many units were sold in Q3?",
            "answer": "75",
            "relevant_doc_ids": ["74.png"],
        },
        {
            "question": "What is the trend for Product A?",
            "answer": "increasing",
            "relevant_doc_ids": ["86.png"],
        },
    ],
}

# --- Helper Functions (Updated) ---


def run_ocr_process(
    ocr_manager: OCRManager, engine: str, image_paths: List[Path], output_file: Path
):
    """Runs OCR on a specific list of real image files."""
    with open(output_file, "w") as f_out:
        for image_path in image_paths:
            if not image_path.exists():
                st.warning(f"Benchmark image not found: {image_path}. Skipping.")
                continue
            image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
            result = ocr_manager.run_ocr(image, engine_name=engine)
            result["image_path"] = image_path.name
            f_out.write(json.dumps(result) + "\n")


# ... (run_indexing_process and run_evaluation_process remain the same) ...
def run_indexing_process(
    encoder_manager: EncoderManager, input_path: Path, output_dir: Path
):
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
    encoder_manager: EncoderManager, index_dir: Path, labels: List[Dict]
) -> dict:
    config = get_config()
    vs_manager = VectorStoreManager(config)
    orchestrator = PipelineOrchestrator(config)
    metrics_calculator = MetricsCalculator()
    dimension = encoder_manager.text_embedder.model.get_sentence_embedding_dimension()
    store = vs_manager.get_store(name=f"eval_store_{time.time()}", dimension=dimension)
    store.load(str(index_dir))
    pipeline = OCRTextVecPipeline(config, encoder_manager.text_embedder, store, {})
    orchestrator.register_pipeline("ocr-text-vec", pipeline)
    results = []
    for item in labels:  # Use the labels list directly
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
    "Welcome! Interact with the RAG system via **Live Querying** or compare OCR engines with **Benchmark Evaluation**."
)

# --- Load shared resources ---
config = get_config()
encoder_manager = get_encoder_manager(config)
ocr_manager = get_ocr_manager(config.ocr)

# --- Sidebar for Mode Selection ---
st.sidebar.title("App Mode")
app_mode = st.sidebar.radio(
    "Choose an operation mode:", ("Live Querying", "Benchmark Evaluation")
)

# --- LIVE QUERYING MODE (Unchanged) ---
if app_mode == "Live Querying":
    st.header("💬 Live Querying")
    st.info(
        "Upload a chart image or a PDF document, select a page, choose an OCR engine, and ask a question."
    )
    col1, col2 = st.columns([2, 3])
    with col1:
        uploaded_file = st.file_uploader(
            "Upload your document", type=["png", "jpg", "jpeg", "pdf"]
        )
        if "pdf_pages" not in st.session_state:
            st.session_state.pdf_pages = []
        if "current_file" not in st.session_state:
            st.session_state.current_file = None
        cv2_img = None
        if uploaded_file:
            if uploaded_file.name != st.session_state.current_file:
                st.session_state.current_file = uploaded_file.name
                st.session_state.pdf_pages = []
                if uploaded_file.type == "application/pdf":
                    with st.spinner("Converting PDF..."):
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".pdf"
                        ) as tf:
                            tf.write(uploaded_file.getvalue())
                            pil_images = convert_from_path(tf.name)
                        st.session_state.pdf_pages = [np.array(p) for p in pil_images]
                else:
                    bytes_data = uploaded_file.getvalue()
                    st.session_state.pdf_pages = [
                        cv2.imdecode(
                            np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR
                        )
                    ]
            if len(st.session_state.pdf_pages) > 1:
                page_selection = st.selectbox(
                    f"Select a page (Total: {len(st.session_state.pdf_pages)})",
                    options=[
                        f"Page {i+1}" for i in range(len(st.session_state.pdf_pages))
                    ],
                )
                page_index = int(page_selection.split(" ")[1]) - 1
                cv2_img = st.session_state.pdf_pages[page_index]
            elif len(st.session_state.pdf_pages) == 1:
                cv2_img = st.session_state.pdf_pages[0]
            if cv2_img is not None:
                st.image(
                    cv2_img,
                    channels=(
                        "RGB" if uploaded_file.type != "application/pdf" else "BGR"
                    ),
                    caption="Selected Page/Image",
                    use_column_width=True,
                )
    with col2:
        if cv2_img is not None:
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
        else:
            st.info("Please upload a document to begin.")

# --- BENCHMARK EVALUATION MODE (UPDATED) ---
elif app_mode == "Benchmark Evaluation":
    st.header("🔬 Benchmark Evaluation")
    st.info(
        "This mode runs a benchmark on a predefined set of real images from `data/processed/processed/` to compare OCR engines."
    )

    # Display the benchmark set
    with st.expander("Show Benchmark Images and Questions"):
        st.write("The following images and questions will be used for the evaluation:")
        for img_name, label in zip(BENCHMARK_SET["images"], BENCHMARK_SET["labels"]):
            st.image(str(BENCHMARK_IMAGE_DIR / img_name), width=300, caption=img_name)
            # st.markdown(f"- **Q:** {label['question']}")
            # st.markdown(f"- **A:** {label['answer']}")
            st.divider()

    engine_to_eval = st.selectbox(
        "Choose an OCR engine to benchmark:", ("tesseract", "trocr", "donut")
    )

    if st.button(f"🚀 Run Benchmark for `{engine_to_eval}`", use_container_width=True):
        image_paths_to_process = [
            BENCHMARK_IMAGE_DIR / fname for fname in BENCHMARK_SET["images"]
        ]

        # Check if benchmark directory exists
        if not BENCHMARK_IMAGE_DIR.exists() or not any(
            p.exists() for p in image_paths_to_process
        ):
            st.error(
                f"Benchmark image directory not found or is empty: `{BENCHMARK_IMAGE_DIR}`. Please run `run_ingest.py` first."
            )
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                with st.status(
                    f"Running benchmark for **{engine_to_eval}**...", expanded=True
                ) as status:
                    st.write(
                        f"Step 1: Running **{engine_to_eval}** OCR on {len(image_paths_to_process)} real images..."
                    )
                    ocr_output_file = temp_path / "ocr_results.jsonl"
                    run_ocr_process(
                        ocr_manager,
                        engine_to_eval,
                        image_paths_to_process,
                        ocr_output_file,
                    )

                    st.write("Step 2: Building vector index from OCR results...")
                    index_dir = temp_path / "index"
                    index_dir.mkdir()
                    run_indexing_process(encoder_manager, ocr_output_file, index_dir)

                    st.write("Step 3: Running evaluation against ground truth...")
                    summary_metrics = run_evaluation_process(
                        encoder_manager, index_dir, BENCHMARK_SET["labels"]
                    )

                    st.session_state[f"results_{engine_to_eval}"] = summary_metrics
                    status.update(
                        label=f"Benchmark for **{engine_to_eval}** complete!",
                        state="complete",
                        expanded=False,
                    )

    st.divider()
    st.subheader("📊 Evaluation Results")
    st.markdown("Results from benchmark runs in this session will appear here.")
    results_data = []
    for engine in ("tesseract", "trocr", "donut"):
        if f"results_{engine}" in st.session_state:
            res = st.session_state[f"results_{engine}"]
            res["engine"] = engine
            results_data.append(res)
    if results_data:
        df_results = pd.DataFrame(results_data).set_index("engine")
        st.dataframe(df_results.style.format("{:.3f}"))
