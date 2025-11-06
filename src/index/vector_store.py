"""Vector store implementations using FAISS and Milvus."""

import numpy as np
import faiss
import pickle
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from loguru import logger
from abc import ABC, abstractmethod

from src.utils.config import Config


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def add(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> List[str]:
        """Add embeddings with metadata."""
        pass

    @abstractmethod
    def search(
        self, query_embedding: np.ndarray, k: int = 5
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Search for similar embeddings."""
        pass

    @abstractmethod
    def save(self, path: str):
        """Save index to disk."""
        pass

    @abstractmethod
    def load(self, path: str):
        """Load index from disk."""
        pass


class FAISSVectorStore(VectorStore):
    def __init__(
        self, dimension: int, index_type: str = "HNSW", metric: str = "cosine", **kwargs
    ):
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.metadata_store = []
        self.id_counter = 0

        if index_type == "Flat":
            self.index = (
                faiss.IndexFlatIP(dimension)
                if metric == "cosine"
                else faiss.IndexFlatL2(dimension)
            )
        elif index_type == "IVF":
            nlist = kwargs.get("nlist", 100)
            quantizer = faiss.IndexFlatL2(dimension)
            metric_type = (
                faiss.METRIC_INNER_PRODUCT if metric == "cosine" else faiss.METRIC_L2
            )
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist, metric_type)
            self.index.nprobe = kwargs.get("nprobe", 10)
        elif index_type == "HNSW":
            m = kwargs.get("m", 32)
            self.index = faiss.IndexHNSWFlat(dimension, m)
            self.index.hnsw.efConstruction = kwargs.get("ef_construction", 200)
            self.index.hnsw.efSearch = kwargs.get("ef_search", 50)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        logger.info(f"Initialized FAISS {index_type} index with dimension {dimension}")

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        if self.metric == "cosine":
            faiss.normalize_L2(embeddings)
        return embeddings

    def add(
        self, embeddings: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Add embeddings to the index, robustly handling 1D or empty inputs.
        """
        # --- FIX STARTS HERE: Handle empty and 1D arrays ---
        if embeddings.size == 0:
            logger.warning("Attempted to add an empty list of embeddings. Skipping.")
            return []

        # Ensure embeddings are 2D for batch processing
        if embeddings.ndim == 1:
            logger.debug("Received a 1D embedding array. Reshaping to (1, D).")
            embeddings = embeddings.reshape(1, -1)
        # --- FIX ENDS HERE ---

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} != index dimension {self.dimension}"
            )

        embeddings_np = embeddings.astype("float32")
        self._normalize_embeddings(embeddings_np)
        if self.index_type == "IVF" and not self.index.is_trained:
            self.index.train(embeddings_np)
        self.index.add(embeddings_np)

        if metadata is None:
            metadata = [{}] * embeddings.shape[0]

        ids = []
        for i, meta in enumerate(metadata):
            doc_id = f"doc_{self.id_counter + i}"
            meta["_id"] = doc_id
            self.metadata_store.append(meta)
            ids.append(doc_id)
        self.id_counter += len(metadata)
        logger.info(
            f"Added {len(metadata)} embeddings to index (total: {self.index.ntotal})"
        )
        return ids

    def search(
        self, query_embedding: np.ndarray, k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        query_embedding = query_embedding.reshape(1, -1).astype("float32")
        self._normalize_embeddings(query_embedding)
        k = min(k, self.index.ntotal)
        if k == 0:
            return np.array([]), np.array([]), []
        distances, indices = self.index.search(query_embedding, k)
        metadata_list = [
            self.metadata_store[i] for i in indices[0] if i < len(self.metadata_store)
        ]
        return distances[0], indices[0], metadata_list

    def save(self, path: str):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(p / "index.faiss"))
        with open(p / "metadata.pkl", "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Saved index to {p}")

    @staticmethod
    def load(path: str):
        p = Path(path)
        with open(p / "metadata.pkl", "rb") as f:
            store = pickle.load(f)
        store.index = faiss.read_index(str(p / "index.faiss"))
        logger.info(f"Loaded index from {p} ({store.index.ntotal} vectors)")
        return store


class MilvusVectorStore(VectorStore):
    """Milvus-based vector store."""

    def __init__(
        self,
        collection_name: str,
        dimension: int,
        host: str = "localhost",
        port: int = 19530,
        metric_type: str = "COSINE",
    ):
        """Initialize Milvus vector store.

        Args:
            collection_name: Name of collection
            dimension: Embedding dimension
            host: Milvus host
            port: Milvus port
            metric_type: Distance metric ('COSINE', 'L2', 'IP')
        """
        try:
            from pymilvus import (
                connections,
                Collection,
                CollectionSchema,
                FieldSchema,
                DataType,
                utility,
            )

            self.collection_name = collection_name
            self.dimension = dimension
            self.metric_type = metric_type

            # Connect to Milvus
            connections.connect(host=host, port=port)
            logger.info(f"Connected to Milvus at {host}:{port}")

            # Create collection if it doesn't exist
            if not utility.has_collection(collection_name):
                fields = [
                    FieldSchema(
                        name="id", dtype=DataType.INT64, is_primary=True, auto_id=True
                    ),
                    FieldSchema(
                        name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension
                    ),
                    FieldSchema(
                        name="metadata", dtype=DataType.VARCHAR, max_length=65535
                    ),
                ]
                schema = CollectionSchema(fields=fields, description="Chart embeddings")
                self.collection = Collection(name=collection_name, schema=schema)

                # Create index
                index_params = {
                    "metric_type": metric_type,
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 1024},
                }
                self.collection.create_index(
                    field_name="embedding", index_params=index_params
                )
                logger.info(f"Created collection {collection_name}")
            else:
                self.collection = Collection(collection_name)
                logger.info(f"Loaded existing collection {collection_name}")

            self.collection.load()

        except ImportError:
            logger.error("pymilvus not installed. Install with: pip install pymilvus")
            raise

    def add(
        self, embeddings: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """Add embeddings to Milvus collection.

        Args:
            embeddings: Array of embeddings
            metadata: List of metadata dicts

        Returns:
            List of assigned IDs
        """
        import json

        n_samples = embeddings.shape[0]

        if metadata is None:
            metadata = [{}] * n_samples

        # Convert metadata to JSON strings
        metadata_strs = [json.dumps(meta) for meta in metadata]

        # Insert
        entities = [embeddings.tolist(), metadata_strs]

        insert_result = self.collection.insert(entities)
        ids = [str(id_) for id_ in insert_result.primary_keys]

        self.collection.flush()
        logger.info(f"Added {n_samples} embeddings to Milvus")

        return ids

    def search(
        self, query_embedding: np.ndarray, k: int = 5
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Search Milvus collection.

        Args:
            query_embedding: Query embedding
            k: Number of results

        Returns:
            Tuple of (distances, metadata_list)
        """
        import json

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        search_params = {"metric_type": self.metric_type, "params": {"nprobe": 10}}

        results = self.collection.search(
            data=query_embedding.tolist(),
            anns_field="embedding",
            param=search_params,
            limit=k,
            output_fields=["metadata"],
        )

        distances = []
        metadata_list = []

        for hits in results:
            for hit in hits:
                distances.append(hit.distance)
                metadata_list.append(json.loads(hit.entity.get("metadata", "{}")))

        return np.array(distances), metadata_list

    def save(self, path: str):
        """Milvus persists automatically."""
        logger.info("Milvus collection persists automatically")

    def load(self, path: str):
        """Milvus loads automatically."""
        logger.info("Milvus collection loaded automatically")


class VectorStoreManager:
    """Manager for vector stores."""

    def __init__(self, config: Config):
        """
        Initialize vector store manager.

        Args:
            config: The main Pydantic Config object.
        """
        self.config = config
        # Access the 'vector_store' field as a direct attribute.
        vs_config = config.vector_store
        backend = vs_config.backend

        self.stores: Dict[str, VectorStore] = {}

        if backend == "faiss":
            self.backend = "faiss"
            # vs_config.faiss is a dictionary from the Pydantic model
            self.faiss_config = vs_config.faiss
        elif backend == "milvus":
            self.backend = "milvus"
            self.milvus_config = vs_config.milvus
        else:
            raise ValueError(f"Unknown backend: {backend}")

        logger.info(f"Vector store manager initialized with backend: {backend}")

    def get_store(self, name: str, dimension: int) -> VectorStore:
        """Get or create a vector store.

        Args:
            name: Store name
            dimension: Embedding dimension

        Returns:
            VectorStore instance
        """
        if name in self.stores:
            return self.stores[name]

        if self.backend == "faiss":
            # --- FIX STARTS HERE ---
            # Create a copy of the config to safely modify it.
            faiss_kwargs = self.faiss_config.copy()

            # Pop the arguments that we are defining explicitly to avoid duplication.
            index_type = faiss_kwargs.pop("index_type", "HNSW")
            metric = faiss_kwargs.pop("metric", "cosine")

            # Now, faiss_kwargs only contains the remaining specific params (e.g., m, nlist).
            store = FAISSVectorStore(
                dimension=dimension,
                index_type=index_type,
                metric=metric,
                **faiss_kwargs,  # Unpack the cleaned dictionary
            )
        else:  # milvus
            store = MilvusVectorStore(
                collection_name=name,
                dimension=dimension,
                host=self.milvus_config.get("host", "localhost"),
                port=self.milvus_config.get("port", 19530),
                metric_type=self.milvus_config.get("metric_type", "COSINE"),
            )

        self.stores[name] = store
        return store
