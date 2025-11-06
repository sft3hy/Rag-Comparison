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
from typing import List, Dict, Tuple
import os

# --- Core Codebase Imports ---
import sys

sys.path.insert(0, str(Path(__file__).parent.absolute()))

from src.utils.config import load_config, Config, OCRConfig
from src.utils.logger import setup_logger
from src.ocr.engines import OCRManager
from src.encoders.embedders.embedder import EncoderManager
from src.index.vector_store import VectorStoreManager, FAISSVectorStore
from src.rag_pipelines.orchestrator import PipelineOrchestrator, OCRTextVecPipeline
from src.eval.metrics import MetricsCalculator
from pdf2image import convert_from_path

os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
            "question": "What is the value for 2004 on the top line?",
            "answer": "67",
            "relevant_doc_ids": ["34.png"],
        },
        {
            "question": "What percentage have not heard of latinx?",
            "answer": "76%",
            "relevant_doc_ids": ["43.png"],
        },
        {
            "question": "What percentage voted for 'stay about the same'?",
            "answer": "42%",
            "relevant_doc_ids": ["46.png"],
        },
        {
            "question": "What percent gave no answer?",
            "answer": "1%",
            "relevant_doc_ids": ["74.png"],
        },
        {
            "question": "What percent thought the worst is behind us in June?",
            "answer": "40&",
            "relevant_doc_ids": ["86.png"],
        },
    ],
}

# --- Helper Functions (Updated) ---


def run_ocr_process(
    ocr_manager: OCRManager, engine: str, image_paths: List[Path], output_file: Path
):
    with open(output_file, "w") as f_out:
        all_ocr_results = {}
        for image_path in image_paths:
            if not image_path.exists():
                continue
            image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
            result = ocr_manager.run_ocr(image, engine_name=engine)
            result["image_path"] = image_path.name
            f_out.write(json.dumps(result) + "\n")
            all_ocr_results[image_path.name] = result["text"]
        return all_ocr_results


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
    if not texts:
        return False  # Indicate failure if no text was indexed
    embeddings = embedder.embed(texts)
    store.add(embeddings, metadatas)
    store.save(str(output_dir))
    return True


# --- ENHANCED EVALUATION PROCESS WITH DEBUG OUTPUT & FIX ---
def run_evaluation_process(
    encoder_manager: EncoderManager, index_dir: Path, labels: List[Dict]
) -> Tuple[Dict, List[Dict]]:
    config = get_config()
    orchestrator = PipelineOrchestrator(config)
    metrics_calculator = MetricsCalculator()

    # --- FIX: Directly load the store and capture the returned object ---
    try:
        # The load method is a staticmethod that returns a new, populated store instance.
        store = FAISSVectorStore.load(str(index_dir))
        if store.index.ntotal == 0:
            st.error(
                "Evaluation Error: The vector index is empty. OCR may have failed."
            )
            return {}, []
    except Exception as e:
        st.error(f"Failed to load the vector index from path: {index_dir}. Error: {e}")
        return {}, []
    # --- END FIX ---

    pipeline = OCRTextVecPipeline(config, encoder_manager.text_embedder, store, {})
    orchestrator.register_pipeline("ocr-text-vec", pipeline)
    all_results, debug_steps = [], []

    for item in labels:
        with st.container():
            st.markdown(f"--- \n#### ❓ Evaluating Question: `{item['question']}`")
            st.markdown(
                f"> **Expected Answer:** `{item['answer']}` | **Correct Document:** `{item['relevant_doc_ids'][0]}`"
            )
            rag_result = orchestrator.run_pipeline(
                "ocr-text-vec", item["question"], k=1
            )
            retrieved_docs = rag_result.get("retrieved_docs", [])
            retrieved_id, retrieval_status = ("None", "❌ **Failed**")
            if retrieved_docs:
                retrieved_id = retrieved_docs[0]["metadata"].get(
                    "chunk_id", "Unknown ID"
                )
                if retrieved_id in item["relevant_doc_ids"]:
                    retrieval_status = "✅ **Correct**"
            st.markdown(
                f"**1. Retrieval:** The system retrieved `{retrieved_id}`. Status: {retrieval_status}"
            )
            with st.expander("Show Context Sent to LLM"):
                st.text(rag_result.get("context_text", "No context sent."))
            predicted_answer = rag_result.get("answer", "")
            st.markdown("**2. Generation:** LLM generated:")
            st.info(predicted_answer)
            qa_metrics = metrics_calculator.calculate_qa_metrics(
                predicted_answer, item["answer"]
            )
            retrieved_ids = [
                doc["metadata"].get("chunk_id", "") for doc in retrieved_docs
            ]
            retrieval_metrics = metrics_calculator.calculate_retrieval_metrics(
                item["relevant_doc_ids"], retrieved_ids
            )
            final_metrics = {**qa_metrics, **retrieval_metrics}
            st.markdown("**3. Metrics:**")
            st.json(final_metrics)
            all_results.append(final_metrics)

    df_results = pd.DataFrame(all_results)
    return df_results.mean(numeric_only=True).to_dict(), debug_steps


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

# --- BENCHMARK EVALUATION MODE (UPDATED with DEBUG UI) ---
elif app_mode == "Benchmark Evaluation":
    st.header("🔬 Benchmark Evaluation")
    st.info(
        "This mode runs a benchmark on real images from `data/processed/processed/` to compare OCR engines."
    )
    with st.expander("Show Benchmark Set"):
        st.write("The following images and questions will be used:")
        count = 0
        for i in BENCHMARK_SET["images"]:
            question = BENCHMARK_SET["labels"][count]["question"]
            answer = BENCHMARK_SET["labels"][count]["answer"]
            formatted = f"Q/A: {question} {answer}"
            st.image(str(BENCHMARK_IMAGE_DIR / i), width=200, caption=formatted)
            count += 1

    engine_to_eval = st.selectbox(
        "Choose an OCR engine to benchmark:", ("tesseract", "trocr", "donut")
    )

    if st.button(f"🚀 Run Benchmark for `{engine_to_eval}`", use_container_width=True):
        image_paths_to_process = [
            BENCHMARK_IMAGE_DIR / fname for fname in BENCHMARK_SET["images"]
        ]
        if not BENCHMARK_IMAGE_DIR.exists() or not any(
            p.exists() for p in image_paths_to_process
        ):
            st.error(
                f"Benchmark image directory not found: `{BENCHMARK_IMAGE_DIR}`. Please run `run_ingest.py` first."
            )
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                st.subheader(f"Detailed Log for `{engine_to_eval}` Benchmark")
                # --- Step 1: OCR ---
                st.markdown("--- \n### Step 1: OCR Extraction")
                ocr_output_file = temp_path / "ocr_results.jsonl"
                with st.spinner(
                    f"Running **{engine_to_eval}** OCR on {len(image_paths_to_process)} images..."
                ):
                    ocr_texts = run_ocr_process(
                        ocr_manager,
                        engine_to_eval,
                        image_paths_to_process,
                        ocr_output_file,
                    )
                st.success("OCR step complete.")
                with st.expander("Show Extracted OCR Text for Each Image"):
                    st.json(ocr_texts)

                # --- Step 2: Indexing ---
                st.markdown("--- \n### Step 2: Indexing")
                index_dir = temp_path / "index"
                index_dir.mkdir()
                with st.spinner("Building vector index from OCR results..."):
                    success = run_indexing_process(
                        encoder_manager, ocr_output_file, index_dir
                    )
                if success:
                    st.success("Indexing complete.")
                else:
                    st.error("Indexing failed: No text was extracted from any image.")
                    st.stop()  # Stop the execution if indexing failed

                # --- Step 3: Evaluation ---
                st.markdown("--- \n### Step 3: Evaluation per Question")
                with st.spinner("Running evaluation against ground truth..."):
                    summary_metrics, debug_info = run_evaluation_process(
                        encoder_manager, index_dir, BENCHMARK_SET["labels"]
                    )
                st.success("Evaluation step complete.")
                st.session_state[f"results_{engine_to_eval}"] = summary_metrics
                st.session_state[f"debug_{engine_to_eval}"] = debug_info

    st.divider()
    st.subheader("📊 Final Evaluation Results")
    st.markdown(
        "Aggregated results from benchmark runs in this session will appear here."
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
