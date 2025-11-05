"""
Simple usage example for the RAG Charts project.
This script demonstrates the complete workflow from ingestion to querying.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.ingestion.preprocessor import ImagePreprocessor, FigureDetector
from src.ocr.engines import OCRManager
from src.derender.extractors import DerenderingManager
from src.encoders.embedders import EncoderManager
from src.index.vector_store import VectorStoreManager
from src.rag_pipelines.orchestrator import (
    ImageVecPipeline,
    OCRTextVecPipeline,
    DerenderTableVecPipeline,
    PipelineOrchestrator,
)
from loguru import logger
import numpy as np
import pandas as pd


def example_1_basic_ocr():
    """Example 1: Basic OCR on a chart image."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Basic OCR")
    print("=" * 60 + "\n")

    # Setup
    setup_logger(log_dir="logs", log_file="example.log")

    # Load config
    config = load_config()

    # Initialize OCR manager
    ocr_manager = OCRManager(config["ocr"])

    # Create a dummy image (in real use, load from file)
    dummy_image = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)

    # Run OCR with Tesseract
    logger.info("Running OCR on image...")
    result = ocr_manager.run_ocr(dummy_image, engine_name="tesseract")

    print(f"Extracted Text: {result['text'][:100]}...")
    print(f"Number of tokens: {len(result['tokens'])}")
    print(f"Average confidence: {np.mean(result['confidences']):.2f}")

    return result


def example_2_image_preprocessing():
    """Example 2: Image preprocessing and figure detection."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Image Preprocessing")
    print("=" * 60 + "\n")

    setup_logger(log_dir="logs", log_file="example.log")

    # Initialize preprocessor
    preprocessor = ImagePreprocessor(
        target_dpi=300, max_dimension=2048, normalize=True, denoise=True
    )

    # Create dummy image
    dummy_image = np.random.randint(0, 255, (1200, 1600, 3), dtype=np.uint8)

    # Preprocess
    logger.info("Preprocessing image...")
    processed = preprocessor.preprocess(dummy_image)

    print(f"Original shape: {dummy_image.shape}")
    print(f"Processed shape: {processed.shape}")

    # Detect figures
    detector = FigureDetector(min_area=10000)
    figures = detector.detect_figures(processed)

    print(f"Detected {len(figures)} figures")
    for i, fig in enumerate(figures[:3]):
        print(f"  Figure {i+1}: bbox={fig['bbox']}, area={fig['area']}")

    return processed, figures


def example_3_embeddings():
    """Example 3: Generate embeddings for different modalities."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Generating Embeddings")
    print("=" * 60 + "\n")

    setup_logger(log_dir="logs", log_file="example.log")
    config = load_config()

    # Initialize encoder manager
    encoder_manager = EncoderManager(config)

    # 1. Image embedding
    logger.info("Generating image embedding...")
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img_embedding = encoder_manager.image_embedder.embed_single(dummy_image)
    print(f"Image embedding shape: {img_embedding.shape}")

    # 2. Text embedding
    logger.info("Generating text embedding...")
    text = "What was the GDP growth rate in 2020?"
    text_embedding = encoder_manager.text_embedder.embed_single(text)
    print(f"Text embedding shape: {text_embedding.shape}")

    # 3. Table embedding
    logger.info("Generating table embedding...")
    dummy_table = pd.DataFrame(
        {"Year": [2018, 2019, 2020], "GDP": [100, 105, 103], "Growth": [5.0, 5.0, -1.9]}
    )
    table_embedding = encoder_manager.table_embedder.embed_single(dummy_table)
    print(f"Table embedding shape: {table_embedding.shape}")

    return img_embedding, text_embedding, table_embedding


def example_4_vector_store():
    """Example 4: Build and query a vector store."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Vector Store Operations")
    print("=" * 60 + "\n")

    setup_logger(log_dir="logs", log_file="example.log")
    config = load_config()

    # Initialize vector store manager
    vs_manager = VectorStoreManager(config)

    # Create a store for image embeddings
    dimension = 512
    store = vs_manager.get_store(name="images", dimension=dimension)

    # Add some embeddings
    logger.info("Adding embeddings to store...")
    embeddings = np.random.rand(100, dimension).astype(np.float32)
    metadata = [{"image_id": f"img_{i}", "caption": f"Chart {i}"} for i in range(100)]
    ids = store.add(embeddings, metadata)
    print(f"Added {len(ids)} embeddings to store")

    # Query the store
    logger.info("Querying vector store...")
    query_embedding = np.random.rand(dimension).astype(np.float32)
    distances, indices, retrieved_metadata = store.search(query_embedding, k=5)

    print(f"\nTop 5 results:")
    for i, (dist, idx, meta) in enumerate(zip(distances, indices, retrieved_metadata)):
        print(f"  {i+1}. Distance: {dist:.4f}, ID: {meta.get('image_id', 'N/A')}")

    # Save store
    logger.info("Saving vector store...")
    store.save("temp_index/")
    print("Vector store saved to temp_index/")

    return store


def example_5_end_to_end_pipeline():
    """Example 5: Complete end-to-end RAG pipeline."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: End-to-End RAG Pipeline")
    print("=" * 60 + "\n")

    setup_logger(log_dir="logs", log_file="example.log")
    config = load_config()

    # Initialize components
    encoder_manager = EncoderManager(config)
    vs_manager = VectorStoreManager(config)

    # Create vector stores
    text_store = vs_manager.get_store(name="text_chunks", dimension=384)

    # Simulate indexing some OCR text
    logger.info("Indexing OCR text...")
    texts = [
        "GDP growth was 5.2% in 2019",
        "Unemployment rate decreased to 3.5%",
        "Inflation remained at 2.1%",
        "Stock market reached all-time high",
        "Trade deficit narrowed by 10%",
    ]

    text_embeddings = encoder_manager.text_embedder.embed(texts)
    metadata = [
        {"chunk_id": f"chunk_{i}", "text": text} for i, text in enumerate(texts)
    ]
    text_store.add(text_embeddings, metadata)

    # Create mapping
    text_to_doc_map = {f"chunk_{i}": {"source": f"doc_{i}"} for i in range(len(texts))}

    # Initialize OCR-TextVec pipeline
    logger.info("Initializing OCR-TextVec pipeline...")
    ocr_pipeline = OCRTextVecPipeline(
        config=config,
        text_embedder=encoder_manager.text_embedder,
        vector_store=text_store,
        text_to_doc_map=text_to_doc_map,
    )

    # Create orchestrator
    orchestrator = PipelineOrchestrator(config)
    orchestrator.register_pipeline("ocr-text-vec", ocr_pipeline)

    # Query
    query = "What was the GDP growth?"
    logger.info(f"Querying: {query}")
    result = orchestrator.run_pipeline("ocr-text-vec", query, k=3)

    print(f"\nQuery: {query}")
    print(f"Answer: {result.get('answer', 'N/A')}")
    print(f"\nRetrieved contexts:")
    for i, doc in enumerate(result.get("retrieved_docs", [])[:3]):
        print(f"  {i+1}. {doc.get('text', 'N/A')}")

    return result


def example_6_evaluation():
    """Example 6: Evaluate with metrics."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Evaluation Metrics")
    print("=" * 60 + "\n")

    from src.eval.metrics import MetricsCalculator

    setup_logger(log_dir="logs", log_file="example.log")

    calculator = MetricsCalculator()

    # Example retrieval evaluation
    relevant_ids = ["doc_1", "doc_3", "doc_5"]
    retrieved_ids = ["doc_3", "doc_7", "doc_1", "doc_9", "doc_2"]

    retrieval_metrics = calculator.calculate_retrieval_metrics(
        relevant_ids, retrieved_ids, k_values=[1, 3, 5]
    )

    print("Retrieval Metrics:")
    for metric, value in retrieval_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Example QA evaluation
    prediction = "The GDP growth was 5.2 percent in 2019"
    ground_truth = "GDP growth was 5.2% in 2019"

    qa_metrics = calculator.calculate_qa_metrics(prediction, ground_truth)

    print("\nQA Metrics:")
    for metric, value in qa_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Statistical significance test
    import numpy as np

    scores1 = np.random.rand(50) * 0.8 + 0.1  # Method 1: mean ~0.5
    scores2 = np.random.rand(50) * 0.7 + 0.2  # Method 2: mean ~0.55

    sig_results = calculator.calculate_statistical_significance(
        scores1, scores2, test="paired_t"
    )

    print("\nStatistical Significance Test:")
    print(f"  Test: {sig_results['test']}")
    print(f"  p-value: {sig_results['p_value']:.4f}")
    print(f"  Significant: {sig_results['significant']}")
    print(f"  Mean difference: {sig_results['mean_diff']:.4f}")

    return retrieval_metrics, qa_metrics, sig_results


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("RAG CHARTS PROJECT - USAGE EXAMPLES")
    print("=" * 60)

    try:
        # Run examples
        example_1_basic_ocr()
        example_2_image_preprocessing()
        example_3_embeddings()
        example_4_vector_store()
        example_5_end_to_end_pipeline()
        example_6_evaluation()

        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60 + "\n")

    except Exception as e:
        logger.error(f"Error running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
