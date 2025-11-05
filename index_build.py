"""
Index building script.

Creates vector store indexes for different data modalities
(text, image, table).
"""

import click
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm
from loguru import logger
import numpy as np
from PIL import Image

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.encoders.embedders import EncoderManager
from src.index.vector_store import VectorStoreManager


@click.command()
@click.option(
    "--index-type",
    type=click.Choice(["text", "image", "table"]),
    required=True,
    help="Type of index to build.",
)
@click.option(
    "--input-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to input data (JSONL for text/table, directory for images).",
)
@click.option(
    "--image-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory with images (required for image index).",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, writable=True, path_type=Path),
    required=True,
    help="Directory to save the built index.",
)
def main(index_type: str, input_path: Path, image_dir: Path, output_dir: Path):
    """
    Build a vector store index for the specified data type.
    """
    # Setup
    config = load_config()
    setup_logger(log_dir=config.paths.logs_dir, log_file="index.log")
    logger.info(f"Starting to build '{index_type}' index...")

    # Initialize managers
    encoder_manager = EncoderManager(config)
    vs_manager = VectorStoreManager(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    if index_type == "text":
        build_text_index(config, encoder_manager, vs_manager, input_path, output_dir)
    elif index_type == "image":
        if not image_dir:
            raise click.UsageError("--image-dir is required for index-type 'image'")
        build_image_index(config, encoder_manager, vs_manager, image_dir, output_dir)
    elif index_type == "table":
        build_table_index(config, encoder_manager, vs_manager, input_path, output_dir)

    logger.info(f"Successfully built and saved '{index_type}' index to {output_dir}")


def build_text_index(config, encoder, vs_manager, input_path, output_dir):
    """Build index from OCR text."""
    embedder = encoder.text_embedder
    dimension = embedder.model.get_sentence_embedding_dimension()
    store = vs_manager.get_store(name="text_index", dimension=dimension)

    with open(input_path, "r") as f:
        lines = f.readlines()

    texts, metadatas = [], []
    for line in tqdm(lines, desc="Loading text data"):
        data = json.loads(line)
        if data.get("text"):
            texts.append(data["text"])
            metadatas.append({"chunk_id": data["image_path"], "text": data["text"]})

    logger.info(f"Generating embeddings for {len(texts)} text chunks...")
    embeddings = embedder.embed(texts)
    store.add(embeddings, metadatas)
    store.save(str(output_dir))


def build_image_index(config, encoder, vs_manager, image_dir, output_dir):
    """Build index from images."""
    embedder = encoder.image_embedder
    # Dimension for CLIP vit-base-patch32 is 512
    dimension = 512
    store = vs_manager.get_store(name="image_index", dimension=dimension)

    image_paths = list(image_dir.glob("*.[pP][nN][gG]"))
    images, metadatas = [], []

    for path in tqdm(image_paths, desc="Loading images"):
        images.append(Image.open(path).convert("RGB"))
        metadatas.append({"image_id": path.name})

    logger.info(f"Generating embeddings for {len(images)} images...")
    embeddings = embedder.embed(images)
    store.add(embeddings, metadatas)
    store.save(str(output_dir))


def build_table_index(config, encoder, vs_manager, input_path, output_dir):
    """Build index from derendered tables."""
    embedder = encoder.table_embedder
    # Use text embedder dimension for flattened tables
    dimension = encoder.text_embedder.model.get_sentence_embedding_dimension()
    store = vs_manager.get_store(name="table_index", dimension=dimension)

    with open(input_path, "r") as f:
        lines = f.readlines()

    tables, metadatas = [], []
    table_dir = input_path.parent
    for line in tqdm(lines, desc="Loading table data"):
        data = json.loads(line)
        try:
            df = pd.read_csv(table_dir / data["table_path"])
            tables.append(df)
            metadatas.append(
                {
                    "table_id": data["source_image"],
                    "table_data": df.to_json(orient="split"),
                }
            )
        except Exception as e:
            logger.warning(f"Could not process table {data['table_path']}: {e}")

    logger.info(f"Generating embeddings for {len(tables)} tables...")
    embeddings = embedder.embed(tables)
    store.add(embeddings, metadatas)
    store.save(str(output_dir))


if __name__ == "__main__":
    main()
