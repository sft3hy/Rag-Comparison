"""Embedding models for images, text, and tables."""

import numpy as np
import pandas as pd
import torch
from PIL import Image
from typing import List, Union, Dict, Any
from abc import ABC, abstractmethod
from loguru import logger
from typing import Optional
from transformers import (
    CLIPProcessor,
    CLIPModel,
    TapasTokenizer,
    TapasModel,
    AutoTokenizer,
    AutoModel,
)
from sentence_transformers import SentenceTransformer
from src.utils.config import Config


class Embedder(ABC):
    """Abstract base class for embedders."""

    @abstractmethod
    def embed(self, inputs: Any) -> np.ndarray:
        """Generate embeddings.

        Args:
            inputs: Input data

        Returns:
            Embedding array
        """
        pass


class ImageEmbedder(Embedder):
    """Embed images using CLIP or similar models."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 32,
    ):
        """Initialize image embedder.

        Args:
            model_name: Name of model
            device: Device to run on
            batch_size: Batch size for inference
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

        logger.info(f"Loading image encoder: {model_name}")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        logger.info(f"Image encoder loaded on {device}")

    def embed(
        self, images: Union[np.ndarray, List[np.ndarray], List[Image.Image]]
    ) -> np.ndarray:
        """Generate image embeddings.

        Args:
            images: Single image or list of images

        Returns:
            Embedding array of shape (n_images, embedding_dim)
        """
        # Normalize input
        if isinstance(images, np.ndarray):
            if images.ndim == 3:
                images = [Image.fromarray(images)]
            else:
                images = [Image.fromarray(img) for img in images]
        elif isinstance(images, list) and isinstance(images[0], np.ndarray):
            images = [Image.fromarray(img) for img in images]

        # Process in batches
        all_embeddings = []

        for i in range(0, len(images), self.batch_size):
            batch = images[i : i + self.batch_size]

            # Process
            inputs = self.processor(images=batch, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Encode
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                # Normalize embeddings
                image_features = image_features / image_features.norm(
                    p=2, dim=-1, keepdim=True
                )

            all_embeddings.append(image_features.cpu().numpy())

        return np.vstack(all_embeddings)

    def embed_single(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Embed single image.

        Args:
            image: Single image

        Returns:
            1D embedding array
        """
        return self.embed([image])[0]


class TextEmbedder(Embedder):
    """Embed text using sentence transformers."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 64,
    ):
        """Initialize text embedder.

        Args:
            model_name: Name of model
            device: Device to run on
            batch_size: Batch size for inference
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

        logger.info(f"Loading text encoder: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        logger.info(f"Text encoder loaded on {device}")

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate text embeddings.

        Args:
            texts: Single text or list of texts

        Returns:
            Embedding array of shape (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        # Encode
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        return embeddings

    def embed_single(self, text: str) -> np.ndarray:
        """Embed single text.

        Args:
            text: Single text string

        Returns:
            1D embedding array
        """
        return self.embed([text])[0]


class TableEmbedder(Embedder):
    """Embed tables using TAPAS or similar models."""

    def __init__(
        self,
        model_name: str = "google/tapas-base-finetuned-wtq",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_flattened: bool = False,
    ):
        """Initialize table embedder.

        Args:
            model_name: Name of model
            device: Device to run on
            use_flattened: If True, flatten table to text and use text embedder
        """
        self.model_name = model_name
        self.device = device
        self.use_flattened = use_flattened

        if use_flattened:
            # Use text embedder for flattened tables
            logger.info("Using flattened text representation for tables")
            self.text_embedder = TextEmbedder(device=device)
            self.tokenizer = None
            self.model = None
        else:
            # Use TAPAS
            logger.info(f"Loading table encoder: {model_name}")
            self.tokenizer = TapasTokenizer.from_pretrained(model_name)
            self.model = TapasModel.from_pretrained(model_name)
            self.model.to(device)
            self.model.eval()
            logger.info(f"Table encoder loaded on {device}")

    def flatten_table(self, df: pd.DataFrame) -> str:
        """Flatten table to text representation.

        Args:
            df: Input DataFrame

        Returns:
            Flattened text
        """
        # Create readable text representation
        parts = []

        # Add column headers
        parts.append("Columns: " + ", ".join(df.columns.astype(str)))

        # Add rows
        for idx, row in df.iterrows():
            row_text = " | ".join([f"{col}: {val}" for col, val in row.items()])
            parts.append(row_text)

        return "\n".join(parts)

    def embed(
        self,
        tables: Union[pd.DataFrame, List[pd.DataFrame]],
        queries: Optional[Union[str, List[str]]] = None,
    ) -> np.ndarray:
        """Generate table embeddings.

        Args:
            tables: Single table or list of tables
            queries: Optional queries for each table (for TAPAS)

        Returns:
            Embedding array
        """
        if isinstance(tables, pd.DataFrame):
            tables = [tables]

        if self.use_flattened:
            # Flatten and use text embedder
            flattened = [self.flatten_table(df) for df in tables]
            return self.text_embedder.embed(flattened)

        # Use TAPAS (requires queries)
        if queries is None:
            queries = [""] * len(tables)
        elif isinstance(queries, str):
            queries = [queries] * len(tables)

        all_embeddings = []

        for table, query in zip(tables, queries):
            # Tokenize
            inputs = self.tokenizer(
                table=table,
                queries=query,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Encode
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use pooler output as embedding
                embedding = outputs.pooler_output

            all_embeddings.append(embedding.cpu().numpy())

        return np.vstack(all_embeddings)

    def embed_single(
        self, table: pd.DataFrame, query: Optional[str] = None
    ) -> np.ndarray:
        """Embed single table.

        Args:
            table: Single DataFrame
            query: Optional query

        Returns:
            1D embedding array
        """
        return self.embed([table], [query] if query else None)[0]


class HybridEmbedder:
    """Combine multiple embedding types."""

    def __init__(
        self,
        image_embedder: Optional[ImageEmbedder] = None,
        text_embedder: Optional[TextEmbedder] = None,
        table_embedder: Optional[TableEmbedder] = None,
        fusion_method: str = "concat",
    ):
        """Initialize hybrid embedder.

        Args:
            image_embedder: Image embedder instance
            text_embedder: Text embedder instance
            table_embedder: Table embedder instance
            fusion_method: How to combine embeddings ('concat', 'mean', 'weighted')
        """
        self.image_embedder = image_embedder
        self.text_embedder = text_embedder
        self.table_embedder = table_embedder
        self.fusion_method = fusion_method

        logger.info(f"Initialized HybridEmbedder with fusion={fusion_method}")

    def embed_multimodal(
        self,
        image: Optional[np.ndarray] = None,
        text: Optional[str] = None,
        table: Optional[pd.DataFrame] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Generate hybrid embedding from multiple modalities.

        Args:
            image: Optional image
            text: Optional text
            table: Optional table
            weights: Optional weights for weighted fusion

        Returns:
            Combined embedding
        """
        embeddings = []
        modality_weights = []

        if image is not None and self.image_embedder is not None:
            img_emb = self.image_embedder.embed_single(image)
            embeddings.append(img_emb)
            modality_weights.append(weights.get("image", 1.0) if weights else 1.0)

        if text is not None and self.text_embedder is not None:
            txt_emb = self.text_embedder.embed_single(text)
            embeddings.append(txt_emb)
            modality_weights.append(weights.get("text", 1.0) if weights else 1.0)

        if table is not None and self.table_embedder is not None:
            tbl_emb = self.table_embedder.embed_single(table)
            embeddings.append(tbl_emb)
            modality_weights.append(weights.get("table", 1.0) if weights else 1.0)

        if not embeddings:
            raise ValueError("No valid embeddings generated")

        # Combine embeddings
        if self.fusion_method == "concat":
            return np.concatenate(embeddings)
        elif self.fusion_method == "mean":
            # Pad to same length if needed
            max_len = max(emb.shape[0] for emb in embeddings)
            padded = [np.pad(emb, (0, max_len - len(emb))) for emb in embeddings]
            return np.mean(padded, axis=0)
        elif self.fusion_method == "weighted":
            # Weighted average
            max_len = max(emb.shape[0] for emb in embeddings)
            padded = [np.pad(emb, (0, max_len - len(emb))) for emb in embeddings]
            weighted = np.average(padded, axis=0, weights=modality_weights)
            return weighted
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")


class EncoderManager:
    """Manager for all encoders."""

    # --- FIX STARTS HERE ---
    def __init__(self, config: Config):
        """
        Initialize encoder manager.

        Args:
            config: The main Pydantic Config object.
        """
        self.config = config
        # Access the 'encoders' field as a direct attribute of the Pydantic object.
        encoder_config = config.encoders

        # The rest of the code works with .get() because encoder_config.image, etc.,
        # are dictionaries defined in the Pydantic model.
        image_cfg = encoder_config.image
        self.image_embedder = ImageEmbedder(
            model_name=image_cfg.get("model", "openai/clip-vit-base-patch32"),
            device=image_cfg.get("device", "cuda"),
            batch_size=image_cfg.get("batch_size", 32),
        )

        text_cfg = encoder_config.text
        self.text_embedder = TextEmbedder(
            model_name=text_cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2"),
            device=text_cfg.get("device", "cuda"),
            batch_size=text_cfg.get("batch_size", 64),
        )

        table_cfg = encoder_config.table
        self.table_embedder = TableEmbedder(
            model_name=table_cfg.get("model", "google/tapas-base-finetuned-wtq"),
            device=table_cfg.get("device", "cuda"),
            use_flattened=table_cfg.get("use_flattened", False),
        )

        # HybridEmbedder is not used in the Streamlit app, but we keep it for completeness
        # self.hybrid_embedder = HybridEmbedder(...)

        logger.info("Encoder manager initialized")
