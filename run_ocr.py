"""
OCR processing script.

Runs OCR engines on a directory of figure images and saves the
structured output.
"""

import click
from pathlib import Path
import json
from tqdm import tqdm
from loguru import logger
import cv2

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.ocr.engines import OCRManager


@click.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory containing figure images from the ingestion step.",
)
@click.option(
    "--output-file",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    required=True,
    help="JSONL file to save OCR results.",
)
@click.option(
    "--engine",
    type=click.Choice(["tesseract", "trocr", "donut"]),
    default="tesseract",
    help="OCR engine to use.",
)
def main(input_dir: Path, output_file: Path, engine: str):
    """
    Run the OCR pipeline on a directory of images.

    This script processes each image in the input directory using the
    specified OCR engine and saves the extracted text and metadata
    to a JSONL file.
    """
    # Setup
    config = load_config()
    setup_logger(log_dir=config.paths.logs_dir, log_file="ocr.log")
    logger.info(f"Starting OCR process with engine: {engine}...")

    # Initialize OCR manager
    ocr_manager = OCRManager(config.ocr)
    if engine not in ocr_manager.engines:
        logger.error(f"Engine '{engine}' is not enabled in the configuration.")
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)
    image_paths = list(input_dir.glob("*.[pP][nN][gG]"))

    with open(output_file, "w") as f_out:
        for image_path in tqdm(image_paths, desc=f"Running OCR ({engine})"):
            try:
                # Load image
                image = cv2.imread(str(image_path))
                if image is None:
                    logger.warning(f"Could not read image: {image_path}")
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Run OCR
                result = ocr_manager.run_ocr(image, engine_name=engine)

                # Add image info to result
                result["image_path"] = image_path.name
                f_out.write(json.dumps(result) + "\n")

            except Exception as e:
                logger.error(f"Failed to process {image_path.name}: {e}")

    logger.info(f"OCR process complete. Results saved to: {output_file}")


if __name__ == "__main__":
    main()
