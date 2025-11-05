"""RAG pipeline orchestrator for different approaches."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
from pathlib import Path
import openai
import anthropic


class RAGPipeline:
    """Base class for RAG pipelines."""

    def __init__(self, config: Dict[str, Any], name: str):
        """Initialize RAG pipeline.

        Args:
            config: Configuration dict
            name: Pipeline name
        """
        self.config = config
        self.name = name
        self.llm_config = config.get("llm", {})

        # Initialize LLM client
        if self.llm_config.get("provider") == "openai":
            self.llm_client = openai.OpenAI()
        elif self.llm_config.get("provider") == "anthropic":
            self.llm_client = anthropic.Anthropic()
        else:
            self.llm_client = None

        logger.info(f"Initialized {name} pipeline")

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents.

        Args:
            query: Query string
            k: Number of results

        Returns:
            List of retrieved documents with metadata
        """
        raise NotImplementedError

    def generate_answer(
        self, query: str, context: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate answer using LLM.

        Args:
            query: User query
            context: Retrieved context

        Returns:
            Generated answer and metadata
        """
        raise NotImplementedError

    def query(self, query: str, k: int = 5) -> Dict[str, Any]:
        """End-to-end query pipeline.

        Args:
            query: User query
            k: Number of docs to retrieve

        Returns:
            Answer and metadata
        """
        logger.info(f"Processing query with {self.name}: {query}")

        # Retrieve
        retrieved = self.retrieve(query, k=k)

        # Generate
        result = self.generate_answer(query, retrieved)

        # Add retrieval info
        result["retrieved_docs"] = retrieved
        result["pipeline"] = self.name

        return result


class ImageVecPipeline(RAGPipeline):
    """Image embedding RAG pipeline."""

    def __init__(
        self,
        config: Dict[str, Any],
        image_embedder,
        vector_store,
        image_to_doc_map: Dict[str, Any],
    ):
        """Initialize ImageVec pipeline.

        Args:
            config: Configuration
            image_embedder: Image embedder instance
            vector_store: Vector store instance
            image_to_doc_map: Mapping from image IDs to document info
        """
        super().__init__(config, "ImageVec")
        self.image_embedder = image_embedder
        self.vector_store = vector_store
        self.image_to_doc_map = image_to_doc_map

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve using image embeddings.

        We embed the text query using CLIP's text encoder and search
        in the image embedding space.
        """
        # For CLIP, we can encode text queries directly
        from transformers import CLIPProcessor, CLIPModel
        import torch

        processor = CLIPProcessor.from_pretrained(self.image_embedder.model_name)
        model = CLIPModel.from_pretrained(self.image_embedder.model_name)
        model.to(self.image_embedder.device)

        # Encode query as text
        inputs = processor(text=[query], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.image_embedder.device) for k, v in inputs.items()}

        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(
                p=2, dim=-1, keepdim=True
            )

        query_embedding = text_features.cpu().numpy()[0]

        # Search
        distances, indices, metadata_list = self.vector_store.search(
            query_embedding, k=k
        )

        results = []
        for i, (dist, idx, meta) in enumerate(zip(distances, indices, metadata_list)):
            doc_info = self.image_to_doc_map.get(meta.get("image_id", ""), {})
            results.append(
                {
                    "rank": i + 1,
                    "distance": float(dist),
                    "metadata": meta,
                    "doc_info": doc_info,
                }
            )

        return results

    def generate_answer(
        self, query: str, context: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate answer using multimodal LLM."""
        # For image context, we would pass images to a vision-capable LLM
        # This is a simplified implementation

        prompt = f"""Based on the retrieved chart images, answer the following question:

Question: {query}

Retrieved charts:
{len(context)} relevant charts were found.
"""

        for i, ctx in enumerate(context[:3]):  # Limit to top 3
            prompt += (
                f"\nChart {i+1}: {ctx.get('metadata', {}).get('caption', 'No caption')}"
            )

        prompt += "\n\nAnswer:"

        # Call LLM
        if self.llm_config.get("provider") == "openai":
            response = self.llm_client.chat.completions.create(
                model=self.llm_config.get("model", "gpt-4"),
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions about charts and data.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.llm_config.get("temperature", 0.0),
                max_tokens=self.llm_config.get("max_tokens", 1000),
            )
            answer = response.choices[0].message.content
        else:
            answer = "LLM not configured"

        return {"answer": answer, "prompt": prompt}


class OCRTextVecPipeline(RAGPipeline):
    """OCR + Text embedding RAG pipeline."""

    def __init__(
        self,
        config: Dict[str, Any],
        text_embedder,
        vector_store,
        text_to_doc_map: Dict[str, Any],
    ):
        """Initialize OCR TextVec pipeline.

        Args:
            config: Configuration
            text_embedder: Text embedder instance
            vector_store: Vector store instance
            text_to_doc_map: Mapping from text chunk IDs to document info
        """
        super().__init__(config, "OCR-TextVec")
        self.text_embedder = text_embedder
        self.vector_store = vector_store
        self.text_to_doc_map = text_to_doc_map

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve using text embeddings."""
        # Embed query
        query_embedding = self.text_embedder.embed_single(query)

        # Search
        distances, indices, metadata_list = self.vector_store.search(
            query_embedding, k=k
        )

        results = []
        for i, (dist, idx, meta) in enumerate(zip(distances, indices, metadata_list)):
            doc_info = self.text_to_doc_map.get(meta.get("chunk_id", ""), {})
            results.append(
                {
                    "rank": i + 1,
                    "distance": float(dist),
                    "text": meta.get("text", ""),
                    "metadata": meta,
                    "doc_info": doc_info,
                }
            )

        return results

    def generate_answer(
        self, query: str, context: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate answer using text LLM."""
        # Build context from retrieved text
        context_text = "\n\n".join(
            [f"[Context {i+1}]\n{ctx.get('text', '')}" for i, ctx in enumerate(context)]
        )

        prompt = f"""You are a helpful assistant that answers questions using provided context from OCR-extracted chart text.

Context:
{context_text}

Question: {query}

Answer:"""

        # Call LLM
        if self.llm_config.get("provider") == "openai":
            response = self.llm_client.chat.completions.create(
                model=self.llm_config.get("model", "gpt-4"),
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions about charts using OCR-extracted text.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.llm_config.get("temperature", 0.0),
                max_tokens=self.llm_config.get("max_tokens", 1000),
            )
            answer = response.choices[0].message.content
        else:
            answer = "LLM not configured"

        return {"answer": answer, "prompt": prompt, "context_text": context_text}


class DerenderTableVecPipeline(RAGPipeline):
    """Derendered table RAG pipeline."""

    def __init__(
        self,
        config: Dict[str, Any],
        table_embedder,
        vector_store,
        table_to_doc_map: Dict[str, Any],
    ):
        """Initialize Derender TableVec pipeline.

        Args:
            config: Configuration
            table_embedder: Table embedder instance
            vector_store: Vector store instance
            table_to_doc_map: Mapping from table IDs to document info
        """
        super().__init__(config, "Derender-TableVec")
        self.table_embedder = table_embedder
        self.vector_store = vector_store
        self.table_to_doc_map = table_to_doc_map

    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve using table embeddings."""
        # For table retrieval, we use flattened text representation
        query_embedding = self.table_embedder.text_embedder.embed_single(query)

        # Search
        distances, indices, metadata_list = self.vector_store.search(
            query_embedding, k=k
        )

        results = []
        for i, (dist, idx, meta) in enumerate(zip(distances, indices, metadata_list)):
            doc_info = self.table_to_doc_map.get(meta.get("table_id", ""), {})
            results.append(
                {
                    "rank": i + 1,
                    "distance": float(dist),
                    "table": meta.get("table_data", None),
                    "metadata": meta,
                    "doc_info": doc_info,
                }
            )

        return results

    def generate_answer(
        self, query: str, context: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate answer using table-aware approach."""
        # Build context from tables
        context_text = ""

        for i, ctx in enumerate(context):
            table_data = ctx.get("table")
            if isinstance(table_data, pd.DataFrame):
                context_text += f"\n\n[Table {i+1}]\n{table_data.to_string()}"
            elif isinstance(table_data, str):
                context_text += f"\n\n[Table {i+1}]\n{table_data}"

        prompt = f"""You are a helpful assistant that answers questions using structured table data extracted from charts.

Tables:
{context_text}

Question: {query}

Please provide a precise answer based on the table data.

Answer:"""

        # Call LLM
        if self.llm_config.get("provider") == "openai":
            response = self.llm_client.chat.completions.create(
                model=self.llm_config.get("model", "gpt-4"),
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that analyzes structured table data to answer questions.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.llm_config.get("temperature", 0.0),
                max_tokens=self.llm_config.get("max_tokens", 1000),
            )
            answer = response.choices[0].message.content
        else:
            answer = "LLM not configured"

        return {"answer": answer, "prompt": prompt, "context_text": context_text}


class E2EVisionLMPipeline(RAGPipeline):
    """End-to-end vision LM pipeline (Donut-style)."""

    def __init__(
        self, config: Dict[str, Any], vision_model, image_store: Dict[str, Any]
    ):
        """Initialize E2E Vision LM pipeline.

        Args:
            config: Configuration
            vision_model: Vision-language model
            image_store: Store of images
        """
        super().__init__(config, "E2E-VisionLM")
        self.vision_model = vision_model
        self.image_store = image_store

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """For E2E, we might still do retrieval or process all images."""
        # Simplified: return all available images (or use a simple retrieval)
        return [
            {"image_id": img_id, "image": img_data}
            for img_id, img_data in list(self.image_store.items())[:k]
        ]

    def generate_answer(
        self, query: str, context: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate answer directly from images using vision LM."""
        # This would use the vision model to answer directly
        # Simplified implementation

        answer = f"E2E Vision LM processing {len(context)} images for query: {query}"

        return {"answer": answer, "prompt": query, "num_images_processed": len(context)}


class PipelineOrchestrator:
    """Orchestrate multiple RAG pipelines."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize orchestrator.

        Args:
            config: Configuration dict
        """
        self.config = config
        self.pipelines: Dict[str, RAGPipeline] = {}
        logger.info("Pipeline orchestrator initialized")

    def register_pipeline(self, name: str, pipeline: RAGPipeline):
        """Register a pipeline.

        Args:
            name: Pipeline name
            pipeline: Pipeline instance
        """
        self.pipelines[name] = pipeline
        logger.info(f"Registered pipeline: {name}")

    def run_pipeline(
        self, pipeline_name: str, query: str, k: int = 5
    ) -> Dict[str, Any]:
        """Run specific pipeline.

        Args:
            pipeline_name: Name of pipeline to run
            query: Query string
            k: Number of results

        Returns:
            Pipeline results
        """
        if pipeline_name not in self.pipelines:
            raise ValueError(
                f"Pipeline {pipeline_name} not found. Available: {list(self.pipelines.keys())}"
            )

        return self.pipelines[pipeline_name].query(query, k=k)

    def run_all_pipelines(self, query: str, k: int = 5) -> Dict[str, Dict[str, Any]]:
        """Run all registered pipelines.

        Args:
            query: Query string
            k: Number of results

        Returns:
            Dict mapping pipeline names to results
        """
        results = {}

        for name, pipeline in self.pipelines.items():
            try:
                logger.info(f"Running pipeline: {name}")
                results[name] = pipeline.query(query, k=k)
            except Exception as e:
                logger.error(f"Error in pipeline {name}: {e}")
                results[name] = {"error": str(e), "pipeline": name}

        return results
