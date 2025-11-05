"""Experiment tracking utilities for W&B and MLflow."""

from typing import Any, Dict, Optional

import wandb
import mlflow
from loguru import logger


class ExperimentTracker:
    """Unified interface for experiment tracking."""

    def __init__(
        self,
        backend: str = "wandb",
        project_name: str = "rag-charts",
        experiment_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Initialize experiment tracker.

        Args:
            backend: 'wandb' or 'mlflow'
            project_name: Name of project
            experiment_name: Name of specific experiment/run
            config: Configuration dict to log
            **kwargs: Additional backend-specific arguments
        """
        self.backend = backend.lower()
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.run = None

        if self.backend == "wandb":
            self._init_wandb(config, **kwargs)
        elif self.backend == "mlflow":
            self._init_mlflow(config, **kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _init_wandb(self, config: Optional[Dict[str, Any]], **kwargs):
        """Initialize Weights & Biases."""
        self.run = wandb.init(
            project=self.project_name,
            name=self.experiment_name,
            config=config,
            **kwargs,
        )
        logger.info(f"W&B run initialized: {self.run.name}")

    def _init_mlflow(self, config: Optional[Dict[str, Any]], **kwargs):
        """Initialize MLflow."""
        mlflow.set_experiment(self.project_name)
        self.run = mlflow.start_run(run_name=self.experiment_name)

        if config:
            for key, value in config.items():
                mlflow.log_param(key, value)

        logger.info(f"MLflow run initialized: {self.run.info.run_id}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics.

        Args:
            metrics: Dictionary of metric names and values
            step: Optional step/iteration number
        """
        if self.backend == "wandb":
            wandb.log(metrics, step=step)
        elif self.backend == "mlflow":
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)

    def log_artifact(self, file_path: str, artifact_type: Optional[str] = None):
        """Log artifact file.

        Args:
            file_path: Path to artifact file
            artifact_type: Type of artifact (for W&B)
        """
        if self.backend == "wandb":
            artifact = wandb.Artifact(
                name=self.experiment_name or "artifact", type=artifact_type or "dataset"
            )
            artifact.add_file(file_path)
            wandb.log_artifact(artifact)
        elif self.backend == "mlflow":
            mlflow.log_artifact(file_path)

    def log_model(self, model_path: str, model_name: str):
        """Log model.

        Args:
            model_path: Path to model file/directory
            model_name: Name for the model
        """
        if self.backend == "wandb":
            artifact = wandb.Artifact(name=model_name, type="model")
            artifact.add_dir(model_path)
            wandb.log_artifact(artifact)
        elif self.backend == "mlflow":
            mlflow.log_artifacts(model_path, artifact_path=model_name)

    def log_table(self, table_name: str, data: Any):
        """Log table data.

        Args:
            table_name: Name of table
            data: Table data (pandas DataFrame or W&B Table)
        """
        if self.backend == "wandb":
            if hasattr(data, "to_dict"):  # pandas DataFrame
                wandb.log({table_name: wandb.Table(dataframe=data)})
            else:
                wandb.log({table_name: data})
        elif self.backend == "mlflow":
            # MLflow doesn't have direct table logging, save as artifact
            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as f:
                if hasattr(data, "to_csv"):
                    data.to_csv(f.name, index=False)
                    mlflow.log_artifact(f.name, artifact_path=table_name)

    def finish(self):
        """Finish experiment run."""
        if self.backend == "wandb":
            wandb.finish()
        elif self.backend == "mlflow":
            mlflow.end_run()

        logger.info("Experiment tracking finished")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()
