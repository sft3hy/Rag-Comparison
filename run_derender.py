"""
Chart derendering script.

Processes chart images to extract underlying data tables.
"""

import click
from pathlib import Path
import json
from tqdm import tqdm
from loguru import logger
import cv2
import pandas as pd

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.derender.extractors import DerenderingManager


@click.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory containing chart images.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, writable=True, path_type=Path),
    required=True,
    help="Directory to save derendered tables (as CSVs) and metadata.",
)
def main(input_dir: Path, output_dir: Path):
    """
    Run the chart derendering pipeline.

    This script takes a directory of chart images, applies a chart-to-table
    model to extract the underlying data, and saves the resulting tables
    as CSV files. A metadata file is also created.
    """
    # Setup
    config = load_config()
    setup_logger(log_dir=config.paths.logs_dir, log_file="derender.log")
    logger.info("Starting chart derendering process...")

    # Initialize manager
    derender_manager = DerenderingManager(config.derendering)
    if not derender_manager.chart_derenderer:
        logger.error("Chart derenderer is not enabled in the configuration.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "derender_metadata.jsonl"
    image_paths = list(input_dir.glob("*.[pP][nN][gG]"))

    with open(metadata_path, "w") as f_meta:
        for image_path in tqdm(image_paths, desc="Derendering charts"):
            try:
                # Load image
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Derender chart
                result = derender_manager.process_chart(image)

                if result["success"] and not result["table"].empty:
                    # Save table
                    table_filename = f"{image_path.stem}_table.csv"
                    table_path = output_dir / table_filename
                    result["table"].to_csv(table_path, index=False)

                    # Write metadata
                    metadata = {
                        "source_image": image_path.name,
                        "table_path": table_filename,
                        "raw_output": result["raw_output"],
                    }
                    f_meta.write(json.dumps(metadata) + "\n")
                else:
                    logger.warning(f"Failed to derender chart: {image_path.name}")

            except Exception as e:
                logger.error(f"Error processing {image_path.name}: {e}")

    logger.info(f"Derendering complete. Tables saved to: {output_dir}")
    logger.info(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
