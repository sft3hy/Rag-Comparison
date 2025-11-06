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
    help="Directory containing source files (images for charts, PDFs for tables).",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, writable=True, path_type=Path),
    required=True,
    help="Directory to save derendered tables (as CSVs) and metadata.",
)
@click.option(
    "--mode",
    type=click.Choice(["chart-to-table", "pdf-table-extraction"]),
    required=True,
    help="Specify the extraction mode to run.",
)
def main(input_dir: Path, output_dir: Path, mode: str):
    """
    Run the data extraction pipeline.
    - 'chart-to-table': Derenders chart images into data tables.
    - 'pdf-table-extraction': Extracts native tables directly from PDF files.
    """
    # Setup
    config = load_config()
    setup_logger(log_dir=config.paths.logs_dir, log_file="derender.log")
    logger.info(f"Starting data extraction process in mode: '{mode}'...")

    # Initialize manager
    derender_manager = DerenderingManager(config.derendering)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / f"{mode}_metadata.jsonl"

    with open(metadata_path, "w") as f_meta:
        # --- UPDATED SECTION: Handle different modes ---
        if mode == "chart-to-table":
            if not derender_manager.chart_derenderer:
                logger.error("Chart derenderer is not enabled/available.")
                return

            image_paths = list(input_dir.glob("*.[pP][nN][gG]"))
            for image_path in tqdm(image_paths, desc="Derendering charts"):
                try:
                    image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
                    result = derender_manager.process_chart(image)
                    if result["success"] and not result["table"].empty:
                        table_filename = f"{image_path.stem}_table.csv"
                        result["table"].to_csv(output_dir / table_filename, index=False)
                        f_meta.write(
                            json.dumps(
                                {
                                    "source_image": image_path.name,
                                    "table_path": table_filename,
                                    "raw_output": result["raw_output"],
                                }
                            )
                            + "\n"
                        )
                except Exception as e:
                    logger.error(f"Error processing {image_path.name}: {e}")

        elif mode == "pdf-table-extraction":
            pdf_paths = list(input_dir.glob("*.[pP][dD][fF]"))
            for pdf_path in tqdm(pdf_paths, desc="Extracting tables from PDFs"):
                try:
                    # This extracts from all pages
                    # In a real scenario, you might iterate page by page
                    extracted_tables = derender_manager.process_pdf_table(
                        str(pdf_path), page="all"
                    )
                    for i, table_df in enumerate(extracted_tables):
                        if not table_df.empty:
                            table_filename = f"{pdf_path.stem}_table_{i+1}.csv"
                            table_df.to_csv(output_dir / table_filename, index=False)
                            f_meta.write(
                                json.dumps(
                                    {
                                        "source_pdf": pdf_path.name,
                                        "table_index": i + 1,
                                        "table_path": table_filename,
                                    }
                                )
                                + "\n"
                            )
                except Exception as e:
                    logger.error(f"Error processing {pdf_path.name}: {e}")
        # --- END UPDATED SECTION ---

    logger.info(f"Extraction complete. Tables saved to: {output_dir}")
    logger.info(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
