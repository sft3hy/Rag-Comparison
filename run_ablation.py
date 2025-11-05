"""
Ablation study runner.

Systematically runs evaluations across different configurations
to compare performance.
"""

import click
from pathlib import Path
import yaml
import subprocess
from loguru import logger

from src.utils.logger import setup_logger
from src.utils.config import load_config, save_config


@click.command()
@click.option(
    "--ablation-config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="YAML file defining the ablation studies to run.",
)
@click.option(
    "--eval-dataset",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the evaluation dataset.",
)
@click.option(
    "--base-output-dir",
    type=click.Path(writable=True, path_type=Path),
    required=True,
    help="Base directory to store outputs for all ablation runs.",
)
def main(ablation_config: Path, eval_dataset: Path, base_output_dir: Path):
    """
    Orchestrate and run a series of ablation studies based on a config file.
    """
    setup_logger(log_dir=str(base_output_dir), log_file="ablation_runner.log")
    logger.info("Starting ablation studies...")

    with open(ablation_config, "r") as f:
        studies = yaml.safe_load(f).get("ablation_studies", [])

    if not studies:
        logger.warning("No ablation studies defined in the config file.")
        return

    for study in studies:
        experiment_name = study.get("name")
        params = study.get("params", {})
        logger.info(f"\n{'='*20}\nRunning Ablation: {experiment_name}\n{'='*20}")

        # Create a temporary config for this run
        temp_config_obj = load_config()
        for key, value in params.items():
            # Simple key-value update, for nested use a proper update util
            # Example: "ocr.engines.trocr.enabled" -> needs parsing
            keys = key.split(".")
            d = temp_config_obj
            for k in keys[:-1]:
                d = getattr(d, k)
            setattr(d, keys[-1], value)

        temp_config_path = base_output_dir / f"temp_config_{experiment_name}.yaml"
        save_config(temp_config_obj, str(temp_config_path))

        # --- This is a simplified runner using subprocess ---
        # In a real MLOps pipeline, this would trigger a job (e.g., on K8s, Vertex AI)

        # 1. Re-run OCR with different engine if specified
        # 2. Re-build index with different embedder if specified
        # 3. Run evaluation with the specified pipeline

        # For this example, we assume indexes are pre-built and just run eval
        pipeline_name = params.get("pipeline.name", "ocr-text-vec")  # Example
        index_dir = Path(
            params.get("index.path", "outputs/indexes/text_default")
        )  # Example

        run_output_dir = base_output_dir / experiment_name
        run_output_dir.mkdir(parents=True, exist_ok=True)

        command = [
            "python",
            "eval.py",
            "--pipeline-name",
            pipeline_name,
            "--index-dir",
            str(index_dir),
            "--eval-dataset",
            str(eval_dataset),
            "--output-dir",
            str(run_output_dir),
            "--experiment-name",
            experiment_name,
            # In a real scenario, you'd pass the --config-path to eval.py
            # and have it load the temporary config.
        ]

        try:
            logger.info(f"Executing command: {' '.join(command)}")
            subprocess.run(command, check=True)
            logger.success(f"Ablation run '{experiment_name}' completed successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Ablation run '{experiment_name}' failed: {e}")
        finally:
            # Clean up temp config
            temp_config_path.unlink(missing_ok=True)

    logger.info("All ablation studies have been executed.")


if __name__ == "__main__":
    main()
