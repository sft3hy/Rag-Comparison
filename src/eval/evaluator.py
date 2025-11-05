"""
Main evaluator for orchestrating RAG pipeline evaluations.
"""

from pathlib import Path
import pandas as pd
from tqdm import tqdm
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
from src.utils.config import Config
from src.eval.metrics import MetricsCalculator
from src.utils.tracking import ExperimentTracker
from src.rag_pipelines.orchestrator import PipelineOrchestrator
from src.ingestion.dataset_loader import DatasetLoader


class Evaluator:
    """Orchestrates evaluation of RAG pipelines across multiple datasets."""

    def __init__(self, config: Config, orchestrator: PipelineOrchestrator):
        """
        Initialize the evaluator.

        Args:
            config: The main configuration object.
            orchestrator: The pipeline orchestrator with registered pipelines.
        """
        self.config = config
        self.orchestrator = orchestrator
        self.dataset_loader = DatasetLoader(config.paths.data_dir)
        self.metrics_calculator = MetricsCalculator()
        logger.info("Evaluator initialized.")

    def run_evaluation(
        self, experiment_name: str, pipeline_name: str, dataset_name: str
    ) -> pd.DataFrame:
        """
        Run a full evaluation for a given pipeline on a specific dataset.

        Args:
            experiment_name: A name for the experiment run.
            pipeline_name: The name of the pipeline to evaluate.
            dataset_name: The name of the dataset to use (e.g., 'chartqa').

        Returns:
            A pandas DataFrame containing the detailed results.
        """
        logger.info(
            f"Starting evaluation for pipeline '{pipeline_name}' on dataset '{dataset_name}'."
        )

        # Load dataset
        eval_data = self.dataset_loader.load_dataset(dataset_name)
        if not eval_data:
            logger.error(f"Could not load dataset: {dataset_name}")
            return pd.DataFrame()

        results = []
        with ExperimentTracker(
            backend=self.config.tracking.get("backend", "wandb"),
            project_name=self.config.tracking.get("project_name", "rag-charts"),
            experiment_name=experiment_name,
            config=self.config.model_dump(),
        ) as tracker:
            for item in tqdm(
                eval_data, desc=f"Evaluating {pipeline_name} on {dataset_name}"
            ):
                query = item["question"]
                ground_truth_answer = item["answer"]
                relevant_doc_ids = item.get("relevant_doc_ids", [])

                # Run pipeline
                result = self.orchestrator.run_pipeline(
                    pipeline_name, query, k=max(self.config.evaluation.k_values)
                )
                predicted_answer = result.get("answer", "")
                retrieved_docs = result.get("retrieved_docs", [])
                retrieved_ids = [
                    doc["metadata"].get("chunk_id") for doc in retrieved_docs
                ]

                # Calculate metrics
                qa_metrics = self.metrics_calculator.calculate_qa_metrics(
                    predicted_answer, ground_truth_answer
                )
                retrieval_metrics = self.metrics_calculator.calculate_retrieval_metrics(
                    relevant_doc_ids,
                    retrieved_ids,
                    k_values=self.config.evaluation.k_values,
                )

                # Store result
                all_metrics = {**qa_metrics, **retrieval_metrics}
                results.append(
                    {
                        "query": query,
                        "prediction": predicted_answer,
                        "ground_truth": ground_truth_answer,
                        "retrieved_ids": retrieved_ids,
                        "relevant_ids": relevant_doc_ids,
                        **all_metrics,
                    }
                )

            # Aggregate and log results
            if not results:
                logger.warning("No results were generated during evaluation.")
                return pd.DataFrame()

            df_results = pd.DataFrame(results)
            avg_metrics = df_results.mean(numeric_only=True).to_dict()

            logger.info("--- Evaluation Summary ---")
            for key, value in avg_metrics.items():
                logger.info(f"{key}: {value:.4f}")
            logger.info("-------------------------")

            tracker.log_metrics(avg_metrics)
            tracker.log_table("evaluation_results", df_results)

        return df_results

    def generate_comparison_report(
        self, results: Dict[str, pd.DataFrame], output_dir: Path
    ):
        """
        Generates a comparison report from multiple evaluation results.

        Args:
            results: A dictionary mapping experiment names to result DataFrames.
            output_dir: The directory to save the report and plots.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_data = []
        for name, df in results.items():
            summary = df.mean(numeric_only=True).to_dict()
            summary["experiment"] = name
            summary_data.append(summary)

        df_summary = pd.DataFrame(summary_data).set_index("experiment")

        # Save summary table
        df_summary.to_csv(output_dir / "comparison_summary.csv")
        logger.info(
            f"Comparison summary saved to {output_dir / 'comparison_summary.csv'}"
        )

        # Generate and save plots
        self._plot_results(df_summary, output_dir)

    def _plot_results(self, df_summary: pd.DataFrame, output_dir: Path):
        """Helper to generate and save plots."""
        for metric in ["exact_match", "f1", "recall@5", "mrr"]:
            if metric in df_summary.columns:
                plt.figure(figsize=(10, 6))
                sns.barplot(x=df_summary.index, y=df_summary[metric])
                plt.title(f"Comparison of {metric}")
                plt.ylabel(metric.replace("_", " ").title())
                plt.xlabel("Experiment")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                plot_path = output_dir / f"comparison_{metric}.png"
                plt.savefig(plot_path)
                plt.close()
                logger.info(f"Saved plot: {plot_path}")
