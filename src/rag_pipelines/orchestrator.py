"""RAG pipeline orchestrator for different approaches."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
import os
import openai
import anthropic
import groq

# Correctly import the Pydantic Config model
from src.utils.config import Config


class RAGPipeline:
    """Base class for RAG pipelines."""

    def __init__(self, config: Config, name: str):
        """
        Initialize RAG pipeline.

        Args:
            config: The main Pydantic Config object.
            name: Pipeline name.
        """
        self.config = config
        self.name = name
        self.llm_config = config.llm

        # Initialize LLM client using environment variables for API keys
        if self.llm_config.provider == "openai":
            self.llm_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.llm_config.provider == "anthropic":
            self.llm_client = anthropic.Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        elif self.llm_config.provider == "groq":
            self.llm_client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))
        else:
            self.llm_client = None

        logger.info(
            f"Initialized '{name}' pipeline with LLM provider: '{self.llm_config.provider}'"
        )

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents."""
        raise NotImplementedError

    def _generate_chat_completion(self, prompt: str, system_prompt: str = "") -> str:
        """Centralized method for making chat completion calls."""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = self.llm_client.chat.completions.create(
                model=self.llm_config.model,
                messages=messages,
                temperature=self.llm_config.temperature,
                max_tokens=self.llm_config.max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error during LLM call to '{self.llm_config.provider}': {e}")
            return f"Error: Could not generate answer from LLM provider '{self.llm_config.provider}'."

    def generate_answer(
        self, query: str, context: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate answer using LLM."""
        raise NotImplementedError

    def query(self, query: str, k: int = 5) -> Dict[str, Any]:
        """End-to-end query pipeline."""
        logger.info(f"Processing query with '{self.name}': \"{query}\"")
        retrieved = self.retrieve(query, k=k)
        result = self.generate_answer(query, retrieved)
        result["retrieved_docs"] = retrieved
        result["pipeline"] = self.name
        return result


class ImageVecPipeline(RAGPipeline):
    """Image embedding RAG pipeline."""

    def __init__(
        self,
        config: Config,
        image_embedder,
        vector_store,
        image_to_doc_map: Dict[str, Any],
    ):
        super().__init__(config, "ImageVec")
        self.image_embedder = image_embedder
        self.vector_store = vector_store
        self.image_to_doc_map = image_to_doc_map

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        # This implementation remains specific to CLIP-style models for now
        from transformers import CLIPProcessor, CLIPModel
        import torch

        processor = CLIPProcessor.from_pretrained(self.image_embedder.model_name)
        model = CLIPModel.from_pretrained(self.image_embedder.model_name).to(
            self.image_embedder.device
        )
        inputs = processor(text=[query], return_tensors="pt", padding=True).to(
            self.image_embedder.device
        )
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
            text_features /= text_features.norm(p=2, dim=-1, keepdim=True)
        query_embedding = text_features.cpu().numpy()[0]
        distances, _, metadata_list = self.vector_store.search(query_embedding, k=k)

        results = []
        for i, (dist, meta) in enumerate(zip(distances, metadata_list)):
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
        system_prompt = "You are a helpful assistant that answers questions about charts and data based on image captions."
        context_str = "\n".join(
            [
                f"Chart {i+1}: {ctx.get('metadata', {}).get('caption', 'No caption')}"
                for i, ctx in enumerate(context[:3])
            ]
        )
        prompt = f"Based on the following retrieved chart captions, answer the question.\n\nContext:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"

        if not self.llm_client:
            return {"answer": "LLM client not configured.", "prompt": prompt}

        answer = self._generate_chat_completion(prompt, system_prompt)
        return {"answer": answer, "prompt": prompt}


class OCRTextVecPipeline(RAGPipeline):
    """OCR + Text embedding RAG pipeline."""

    def __init__(
        self,
        config: Config,
        text_embedder,
        vector_store,
        text_to_doc_map: Dict[str, Any],
    ):
        super().__init__(config, "OCR-TextVec")
        self.text_embedder = text_embedder
        self.vector_store = vector_store
        self.text_to_doc_map = text_to_doc_map

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.text_embedder.embed_single(query)
        distances, _, metadata_list = self.vector_store.search(query_embedding, k=k)
        results = []
        for i, (dist, meta) in enumerate(zip(distances, metadata_list)):
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
        system_prompt = "You are a helpful assistant that answers questions using provided context from OCR-extracted chart text. Answer only based on the context provided."
        context_text = "\n\n".join(
            [
                f"[Context {i+1} from document {ctx['metadata'].get('chunk_id', 'N/A')}]\n{ctx.get('text', '')}"
                for i, ctx in enumerate(context)
            ]
        )
        prompt = f"Context:\n---\n{context_text}\n---\n\nQuestion: {query}\n\nAnswer:"

        if not self.llm_client:
            return {
                "answer": "LLM client not configured.",
                "prompt": prompt,
                "context_text": context_text,
            }

        answer = self._generate_chat_completion(prompt, system_prompt)
        return {"answer": answer, "prompt": prompt, "context_text": context_text}


class DerenderTableVecPipeline(RAGPipeline):
    """Derendered table RAG pipeline."""

    def __init__(
        self,
        config: Config,
        table_embedder,
        vector_store,
        table_to_doc_map: Dict[str, Any],
    ):
        super().__init__(config, "Derender-TableVec")
        self.table_embedder = table_embedder
        self.vector_store = vector_store
        self.table_to_doc_map = table_to_doc_map

    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        query_embedding = self.table_embedder.text_embedder.embed_single(query)
        distances, _, metadata_list = self.vector_store.search(query_embedding, k=k)
        results = []
        for i, (dist, meta) in enumerate(zip(distances, metadata_list)):
            doc_info = self.table_to_doc_map.get(meta.get("table_id", ""), {})
            results.append(
                {
                    "rank": i + 1,
                    "distance": float(dist),
                    "table": meta.get("table_data"),
                    "metadata": meta,
                    "doc_info": doc_info,
                }
            )
        return results

    def generate_answer(
        self, query: str, context: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        system_prompt = "You are a helpful assistant that answers questions using structured table data extracted from charts. Provide a precise answer based only on the table data."
        context_text = ""
        for i, ctx in enumerate(context):
            table_data = ctx.get("table")
            if isinstance(
                table_data, str
            ):  # Assuming table data is stored as a stringified DataFrame
                try:
                    df = pd.read_json(table_data, orient="split")
                    context_text += f"\n\n[Table {i+1} from document {ctx['metadata'].get('table_id', 'N/A')}]\n{df.to_string()}"
                except Exception:
                    context_text += f"\n\n[Table {i+1}]\n{table_data}"
            elif isinstance(table_data, pd.DataFrame):
                context_text += f"\n\n[Table {i+1}]\n{table_data.to_string()}"

        prompt = f"Tables:\n---\n{context_text}\n---\n\nQuestion: {query}\n\nPlease provide a precise answer based on the table data.\n\nAnswer:"

        if not self.llm_client:
            return {
                "answer": "LLM client not configured.",
                "prompt": prompt,
                "context_text": context_text,
            }

        answer = self._generate_chat_completion(prompt, system_prompt)
        return {"answer": answer, "prompt": prompt, "context_text": context_text}


class PipelineOrchestrator:
    """Orchestrate multiple RAG pipelines."""

    def __init__(self, config: Config):
        self.config = config
        self.pipelines: Dict[str, RAGPipeline] = {}
        logger.info("Pipeline orchestrator initialized")

    def register_pipeline(self, name: str, pipeline: RAGPipeline):
        self.pipelines[name] = pipeline
        logger.info(f"Registered pipeline: '{name}'")

    def run_pipeline(
        self, pipeline_name: str, query: str, k: int = 5
    ) -> Dict[str, Any]:
        if pipeline_name not in self.pipelines:
            raise ValueError(
                f"Pipeline '{pipeline_name}' not found. Available: {list(self.pipelines.keys())}"
            )
        return self.pipelines[pipeline_name].query(query, k=k)
