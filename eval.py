"""
Evaluation runner.

Runs a dataset of questions against a RAG pipeline and computes metrics.
"""

import click
from pathlib import Path
import json
from tqdm import tqdm
from loguru import logger
import pandas as pd
import numpy as np

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.eval.metrics import MetricsCalculator
from src.utils.tracking import ExperimentTracker

# Import necessary pipeline components as in query_rag.py
from src.encoders.embedders import EncoderManager
from src.index.vector_store import VectorStoreManager
from src.rag_pipelines.orchestrator import *


@click.command()
@click.option("--pipeline-name", type=str, required=True)
@click.option(
    "--index-dir", type=click.Path(exists=True, path_type=Path), required=True
)
@click.option(
    "--eval-dataset", type=click.Path(exists=True, path_type=Path), required=True
)
@click.option(
    "--output-dir", type=click.Path(writable=True, path_type=Path), required=True
)
@click.option("--experiment-name", type=str, default="rag_eval")
def main(
    pipeline_name: str,
    index_dir: Path,
    eval_dataset: Path,
    output_dir: Path,
    experiment_name: str,
):
    """
    Run evaluation for a RAG pipeline.
    """
    # Setup
    config = load_config()
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(log_dir=str(output_dir), log_file="eval.log")
    logger.info(f"Starting evaluation for pipeline: {pipeline_name}")

    # Initialize components
    encoder_manager = EncoderManager(config)
    vs_manager = VectorStoreManager(config)
    orchestrator = PipelineOrchestrator(config)
    metrics_calculator = MetricsCalculator()

    # Load index and create pipeline (similar to query_rag.py)
    # This part should be refactored into a helper function in a real project
    if pipeline_name == "ocr-text-vec":
        dimension = (
            encoder_manager.text_embedder.model.get_sentence_embedding_dimension()
        )
        store = vs_manager.get_store("text_eval", dimension)
        store.load(str(index_dir))
        pipeline = OCRTextVecPipeline(config, encoder_manager.text_embedder, store, {})
        orchestrator.register_pipeline(pipeline_name, pipeline)
    else:
        logger.error(f"Pipeline {pipeline_name} setup for eval not implemented yet.")
        return

    # Load dataset
    eval_data = []
    with open(eval_dataset, "r") as f:
        for line in f:
            eval_data.append(json.loads(line))

    # Run evaluation
    results = []
    with ExperimentTracker(
        backend=config.tracking.get("backend", "wandb"),
        project_name=config.tracking.get("project_name", "rag-charts"),
        experiment_name=experiment_name,
        config=config.model_dump(),
    ) as tracker:
        for item in tqdm(eval_data, desc="Evaluating"):
            query = item["question"]
            ground_truth = item["answer"]
            relevant_ids = item["relevant_doc_ids"]

            result = orchestrator.run_pipeline(
                pipeline_name, query, k=max(config.evaluation.k_values)
            )
            prediction = result.get("answer", "")
            retrieved_ids = [
                doc["metadata"].get("chunk_id", "")
                for doc in result.get("retrieved_docs", [])
            ]

            # Calculate metrics
            qa_metrics = metrics_calculator.calculate_qa_metrics(
                prediction, ground_truth
            )
            retrieval_metrics = metrics_calculator.calculate_retrieval_metrics(
                relevant_ids, retrieved_ids, k_values=config.evaluation.k_values
            )

            all_metrics = {**qa_metrics, **retrieval_metrics}
            results.append(
                {
                    "query": query,
                    "prediction": prediction,
                    "ground_truth": ground_truth,
                    **all_metrics,
                }
            )

        # Aggregate and log results
        df_results = pd.DataFrame(results)
        avg_metrics = df_results.mean(numeric_only=True).to_dict()
        logger.info(f"Average Metrics: {avg_metrics}")

        tracker.log_metrics(avg_metrics)
        tracker.log_table("evaluation_results", df_results)

        # Save results locally
        df_results.to_csv(output_dir / "evaluation_results.csv", index=False)
        with open(output_dir / "summary.json", "w") as f:
            json.dump(avg_metrics, f, indent=4)

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
