"""
Script to download common datasets for the RAG-Charts project.

This script handles downloading and extracting datasets like ChartQA, PlotQA,
and PubTabNet into the appropriate data/raw directory.
"""

import click
from pathlib import Path
import requests
import zipfile
import tarfile
from tqdm import tqdm
from loguru import logger
import sys

# --- Dataset Registry ---
# NOTE: The direct download links for these datasets can be unstable or require
# authentication. These URLs are placeholders and may need to be updated.
# Please find the official sources for these datasets to get the correct links.
DATASET_REGISTRY = {
    "chartqa": {
        "url": "https://github.com/vis-nlp/ChartQA/archive/refs/heads/main.zip",
        "filename": "ChartQA-main.zip",
        "type": "zip",
        "description": "The ChartQA dataset repository.",
    },
    "plotqa": {
        "url": "https://github.com/NiteshN-verma/PlotQA/archive/refs/heads/master.zip",  # Example, find official link
        "filename": "PlotQA-master.zip",
        "type": "zip",
        "description": "The PlotQA dataset repository.",
    },
    "pubtabnet": {
        "url": "https://dax-cdn.cdn.appdomain.cloud/dax-pubtabnet/2.0.0/pubtabnet.tar.gz",  # Official link, but large
        "filename": "pubtabnet.tar.gz",
        "type": "tar.gz",
        "description": "PubTabNet dataset for table recognition.",
    },
}


def _setup_logger():
    """Configure Loguru logger."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
        colorize=True,
    )


def _download_file(url: str, destination: Path):
    """Downloads a file with a progress bar."""
    logger.info(f"Downloading from: {url}")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            with open(destination, "wb") as f, tqdm(
                total=total_size, unit="B", unit_scale=True, desc=destination.name
            ) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        logger.success(f"Successfully downloaded {destination.name}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download file: {e}")
        raise


def _extract_file(source: Path, destination: Path):
    """Extracts a compressed file and removes the archive."""
    logger.info(f"Extracting {source.name} to {destination}...")
    try:
        if source.suffix == ".zip":
            with zipfile.ZipFile(source, "r") as zip_ref:
                zip_ref.extractall(destination)
        elif ".tar.gz" in source.name:
            with tarfile.open(source, "r:gz") as tar_ref:
                tar_ref.extractall(destination)
        else:
            raise ValueError(f"Unsupported file type: {source.suffix}")

        # Clean up the archive file
        source.unlink()
        logger.success(f"Successfully extracted and removed {source.name}")
    except Exception as e:
        logger.error(f"Failed to extract file: {e}")
        raise


@click.command()
@click.option(
    "--dataset",
    type=click.Choice(["all"] + list(DATASET_REGISTRY.keys())),
    default="all",
    help="Name of the dataset to download.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, writable=True, path_type=Path),
    default="data/raw",
    help="Directory to save the raw datasets.",
)
def main(dataset: str, output_dir: Path):
    """
    Downloads and extracts specified datasets for the RAG-Charts project.
    """
    _setup_logger()
    logger.info("Starting dataset download process...")

    output_dir.mkdir(parents=True, exist_ok=True)

    datasets_to_download = DATASET_REGISTRY.keys() if dataset == "all" else [dataset]

    for name in datasets_to_download:
        info = DATASET_REGISTRY[name]
        logger.info(f"--- Processing dataset: {name} ---")

        dataset_path = output_dir / name
        if dataset_path.exists():
            logger.warning(
                f"Directory '{dataset_path}' already exists. Skipping download."
            )
            continue

        archive_path = output_dir / info["filename"]

        try:
            # Download
            _download_file(info["url"], archive_path)

            # Extract
            # Create a temporary extraction dir to handle nested structures
            temp_extract_path = output_dir / f"{name}_temp"
            _extract_file(archive_path, temp_extract_path)

            # Often archives extract into a single sub-folder. We move contents up.
            subfolders = [d for d in temp_extract_path.iterdir() if d.is_dir()]
            if len(subfolders) == 1:
                subfolders[0].rename(dataset_path)
                temp_extract_path.rmdir()
            else:
                temp_extract_path.rename(dataset_path)

            logger.info(f"Dataset '{name}' is ready at '{dataset_path}'")

        except Exception as e:
            logger.error(f"An error occurred while processing '{name}': {e}")
            # Clean up partial downloads/extractions
            if archive_path.exists():
                archive_path.unlink()

    logger.info("Dataset download process finished.")


if __name__ == "__main__":
    main()
