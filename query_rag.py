"""
Query interface for the RAG system.

Allows running queries against a specified RAG pipeline.
"""

import click
from pathlib import Path
from loguru import logger
import pprint

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.encoders.embedders import EncoderManager
from src.index.vector_store import VectorStoreManager
from src.rag_pipelines.orchestrator import (
    ImageVecPipeline,
    OCRTextVecPipeline,
    DerenderTableVecPipeline,
    PipelineOrchestrator,
)


@click.command()
@click.option(
    "--pipeline-name",
    type=click.Choice(["image-vec", "ocr-text-vec", "derender-table-vec"]),
    required=True,
)
@click.option(
    "--index-dir", type=click.Path(exists=True, path_type=Path), required=True
)
@click.option("--query", type=str, required=True)
@click.option("--top-k", type=int, default=5)
def main(pipeline_name: str, index_dir: Path, query: str, top_k: int):
    """
    Query the RAG system with a specific pipeline.
    """
    # Setup
    config = load_config()
    setup_logger(log_dir=config.paths.logs_dir, log_file="query.log")
    logger.info(f"Initializing query with pipeline: {pipeline_name}")

    # Initialize components
    encoder_manager = EncoderManager(config)
    vs_manager = VectorStoreManager(config)
    orchestrator = PipelineOrchestrator(config)

    # Load the required index and create the pipeline
    if pipeline_name == "ocr-text-vec":
        dimension = (
            encoder_manager.text_embedder.model.get_sentence_embedding_dimension()
        )
        store = vs_manager.get_store("text_query", dimension)
        store.load(str(index_dir))
        pipeline = OCRTextVecPipeline(config, encoder_manager.text_embedder, store, {})
    elif pipeline_name == "image-vec":
        dimension = 512  # CLIP dimension
        store = vs_manager.get_store("image_query", dimension)
        store.load(str(index_dir))
        pipeline = ImageVecPipeline(config, encoder_manager.image_embedder, store, {})
    elif pipeline_name == "derender-table-vec":
        dimension = (
            encoder_manager.text_embedder.model.get_sentence_embedding_dimension()
        )
        store = vs_manager.get_store("table_query", dimension)
        store.load(str(index_dir))
        pipeline = DerenderTableVecPipeline(
            config, encoder_manager.table_embedder, store, {}
        )
    else:
        raise NotImplementedError(
            f"Pipeline {pipeline_name} not implemented in this script."
        )

    orchestrator.register_pipeline(pipeline_name, pipeline)

    # Run query
    logger.info(f"Running query: '{query}'")
    result = orchestrator.run_pipeline(pipeline_name, query, k=top_k)

    # Print results
    click.echo("\n" + "=" * 20 + " QUERY RESULT " + "=" * 20)
    click.echo(f"Pipeline: {result.get('pipeline')}")
    click.echo(f"Query: {query}\n")
    click.secho("Answer:", fg="green", bold=True)
    click.echo(result.get("answer", "No answer generated."))
    click.echo("\n" + "-" * 54)
    click.secho(
        f"Retrieved Top {len(result.get('retrieved_docs', []))} Documents:", fg="yellow"
    )
    for doc in result.get("retrieved_docs", []):
        pprint.pprint(doc, indent=2)
    click.echo("=" * 54 + "\n")


if __name__ == "__main__":
    main()
