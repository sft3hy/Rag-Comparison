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
from pdf2image import convert_from_path
import numpy as np
import cv2

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
    file_paths = (
        list(input_dir.glob("**/*.[pP][nN][gG]"))
        + list(input_dir.glob("**/*.[jJ][pP][gG]"))
        + list(input_dir.glob("**/*.[jJ][pP][eE][gG]"))
        + list(input_dir.glob("**/*.[pP][dD][fF]"))
    )
    logger.info(f"Found {len(file_paths)} images to process.")

    with open(metadata_path, "w") as f_meta:
        for file_path in tqdm(file_paths, desc="Processing files"):
            try:
                # --- UPDATED SECTION: Handle PDF and Image files differently ---
                images_to_process = []
                if file_path.suffix.lower() == ".pdf":
                    logger.debug(f"Converting PDF: {file_path.name}")
                    # Convert PDF pages to a list of PIL Images
                    pil_images = convert_from_path(
                        file_path, dpi=config.image_processing.get("target_dpi", 300)
                    )
                    for i, pil_image in enumerate(pil_images):
                        # Convert PIL image to numpy array for processing
                        image_np = np.array(pil_image)
                        # Create a unique name for the page
                        page_base_name = f"{file_path.stem}_page_{i+1}"
                        images_to_process.append((image_np, page_base_name, i + 1))
                else:
                    # For regular images, process as before
                    image_np = preprocessor.load_image(str(file_path))
                    images_to_process.append((image_np, file_path.stem, None))

                for image, base_name, page_num in images_to_process:
                    # 1. Preprocess image
                    processed_image = preprocessor.preprocess(image)

                    # Save the processed full page/image for reference
                    processed_img_path = processed_dir / f"{base_name}.png"
                    cv2.imwrite(
                        str(processed_img_path),
                        cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR),
                    )

                    # 2. Detect figures
                    figures = detector.detect_figures(processed_image)
                    if not figures:
                        logger.warning(f"No figures detected in {base_name}")
                        continue

                    # 3. Save figures
                    saved_paths = detector.save_figures(
                        processed_image, figures, str(figures_dir), base_name
                    )

                    # 4. Write metadata for each figure
                    for i, fig in enumerate(figures):
                        metadata = {
                            "source_file": file_path.name,
                            "source_page": page_num,
                            "figure_path": Path(saved_paths[i]).name,
                            "bbox": fig["bbox"],
                            "area": fig["area"],
                        }
                        f_meta.write(json.dumps(metadata) + "\n")
                # --- END UPDATED SECTION ---

            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}")

    logger.info(f"Ingestion complete. Figures saved to: {figures_dir}")
    logger.info(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
