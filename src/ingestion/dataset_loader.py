"""
Loaders for standard chart and document QA datasets.
"""

from pathlib import Path
import json
from typing import List, Dict, Any
from loguru import logger


class DatasetLoader:
    """Loads and standardizes different datasets."""

    def __init__(self, base_data_dir: str):
        self.base_dir = Path(base_data_dir)

    def load_dataset(self, name: str) -> List[Dict[str, Any]]:
        """
        Loads a dataset by name.

        Args:
            name: The name of the dataset (e.g., 'chartqa', 'synthetic').

        Returns:
            A list of standardized data samples.
        """
        loader_fn = getattr(self, f"_load_{name}", None)
        if loader_fn:
            logger.info(f"Loading dataset: {name}")
            return loader_fn()
        else:
            logger.error(f"No loader found for dataset: {name}")
            return []

    def _load_synthetic(self) -> List[Dict[str, Any]]:
        """Loads the locally generated synthetic dataset."""
        dataset_path = self.base_dir / "synthetic" / "labels.jsonl"
        if not dataset_path.exists():
            logger.warning(
                f"Synthetic dataset not found at {dataset_path}. Please generate it first."
            )
            return []

        data = []
        with open(dataset_path, "r") as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def _load_chartqa(self) -> List[Dict[str, Any]]:
        """
        Placeholder for loading the ChartQA dataset.
        Assumes a specific file structure and format.
        """
        # This is a simplified placeholder. The actual implementation would
        # need to handle the specific format of the ChartQA dataset.
        logger.warning(
            "ChartQA loader is a placeholder. Please implement based on actual dataset format."
        )
        return [
            {
                "question": "What was the value in 2020?",
                "answer": "500",
                "image_path": "chartqa_sample.png",
                "relevant_doc_ids": ["chartqa_sample"],
            }
        ]
