"""
Main ingestion script.

Processes a directory of raw images/documents, performs preprocessing,
detects figures, and saves the extracted figures and metadata.
"""

import click
from pathlib import Path
import json
from tqdm import tqdm
from loguru import logger

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.ingestion.preprocessor import ImagePreprocessor, FigureDetector


@click.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory containing raw images or documents.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, writable=True, path_type=Path),
    required=True,
    help="Directory to save processed images, figures, and metadata.",
)
def main(input_dir: Path, output_dir: Path):
    """
    Run the ingestion pipeline.

    This script takes an input directory of images, preprocesses them,
    detects figures within each image, and saves the extracted figures
    to the specified output directory. It also creates a metadata file
    (`metadata.jsonl`) containing information about the detected figures.
    """
    # Setup
    config = load_config()
    setup_logger(log_dir=config.paths.logs_dir, log_file="ingest.log")
    logger.info("Starting ingestion process...")

    # Create output directories
    figures_dir = output_dir / "figures"
    processed_dir = output_dir / "processed"
    figures_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # --- FIX STARTS HERE ---
    # Filter the config to only include arguments expected by ImagePreprocessor
    img_proc_config = config.image_processing
    valid_preprocessor_keys = ["target_dpi", "max_dimension", "normalize", "denoise"]
    preprocessor_args = {
        key: img_proc_config[key]
        for key in valid_preprocessor_keys
        if key in img_proc_config
    }

    # Initialize components with the correct arguments
    preprocessor = ImagePreprocessor(**preprocessor_args)
    detector = FigureDetector()
    # --- FIX ENDS HERE ---

    metadata_path = output_dir / "ingestion_metadata.jsonl"
    logger.info(f"Recursively searching for images in {input_dir}...")
    image_paths = (
        list(input_dir.glob("**/*.[pP][nN][gG]"))
        + list(input_dir.glob("**/*.[jJ][pP][gG]"))
        + list(input_dir.glob("**/*.[jJ][pP][eE][gG]"))
    )
    logger.info(f"Found {len(image_paths)} images to process.")

    with open(metadata_path, "w") as f_meta:
        for image_path in tqdm(image_paths, desc="Processing images"):
            try:
                # 1. Preprocess image
                processed_image = preprocessor.process_file(
                    str(image_path), str(processed_dir / image_path.name)
                )

                # 2. Detect figures (conditionally based on config)
                if img_proc_config.get("detect_figures", True):
                    figures = detector.detect_figures(processed_image)
                    if not figures:
                        logger.warning(f"No figures detected in {image_path.name}")
                        continue

                    # 3. Save figures and metadata
                    base_name = image_path.stem
                    saved_paths = detector.save_figures(
                        processed_image, figures, str(figures_dir), base_name
                    )

                    # 4. Write metadata to file
                    for i, fig in enumerate(figures):
                        metadata = {
                            "source_image": image_path.name,
                            "figure_path": Path(saved_paths[i]).name,
                            "bbox": fig["bbox"],
                            "area": fig["area"],
                        }
                        f_meta.write(json.dumps(metadata) + "\n")
                else:
                    logger.info("Skipping figure detection as per config.")
                    # If not detecting figures, the whole image is the "figure"
                    metadata = {
                        "source_image": image_path.name,
                        "figure_path": image_path.name,  # The processed image is the figure
                        "bbox": [
                            0,
                            0,
                            processed_image.shape[1],
                            processed_image.shape[0],
                        ],
                        "area": processed_image.shape[0] * processed_image.shape[1],
                    }
                    f_meta.write(json.dumps(metadata) + "\n")

            except Exception as e:
                logger.error(f"Failed to process {image_path.name}: {e}")

    logger.info(f"Ingestion complete. Processed data saved to: {output_dir}")
    logger.info(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
