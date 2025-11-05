"""Vector store implementations using FAISS and Milvus."""

import numpy as np
import faiss
import pickle
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from loguru import logger
from abc import ABC, abstractmethod


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
    """FAISS-based vector store."""

    def __init__(
        self, dimension: int, index_type: str = "HNSW", metric: str = "cosine", **kwargs
    ):
        """Initialize FAISS vector store.

        Args:
            dimension: Embedding dimension
            index_type: Type of index ('Flat', 'IVF', 'HNSW')
            metric: Distance metric ('cosine', 'l2')
            **kwargs: Additional parameters for specific index types
        """
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.metadata_store = []
        self.id_counter = 0

        # Create index based on type
        if index_type == "Flat":
            if metric == "cosine":
                self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine
            else:
                self.index = faiss.IndexFlatL2(dimension)

        elif index_type == "IVF":
            nlist = kwargs.get("nlist", 100)
            quantizer = faiss.IndexFlatL2(dimension)
            if metric == "cosine":
                self.index = faiss.IndexIVFFlat(
                    quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT
                )
            else:
                self.index = faiss.IndexIVFFlat(
                    quantizer, dimension, nlist, faiss.METRIC_L2
                )
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
        """Normalize embeddings for cosine similarity.

        Args:
            embeddings: Input embeddings

        Returns:
            Normalized embeddings
        """
        if self.metric == "cosine":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            return embeddings / norms
        return embeddings

    def add(
        self, embeddings: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """Add embeddings to index.

        Args:
            embeddings: Array of embeddings (n_samples, dimension)
            metadata: List of metadata dicts for each embedding

        Returns:
            List of assigned IDs
        """
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} != index dimension {self.dimension}"
            )

        # Normalize if using cosine
        embeddings = self._normalize_embeddings(embeddings)

        # Train index if needed (for IVF)
        if self.index_type == "IVF" and not self.index.is_trained:
            logger.info("Training IVF index...")
            self.index.train(embeddings)

        # Add to index
        n_samples = embeddings.shape[0]
        self.index.add(embeddings.astype("float32"))

        # Store metadata
        if metadata is None:
            metadata = [{}] * n_samples

        ids = []
        for i, meta in enumerate(metadata):
            doc_id = f"doc_{self.id_counter + i}"
            meta["_id"] = doc_id
            self.metadata_store.append(meta)
            ids.append(doc_id)

        self.id_counter += n_samples
        logger.info(
            f"Added {n_samples} embeddings to index (total: {self.index.ntotal})"
        )

        return ids

    def search(
        self, query_embedding: np.ndarray, k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Search for k nearest neighbors.

        Args:
            query_embedding: Query embedding (1D or 2D array)
            k: Number of results to return

        Returns:
            Tuple of (distances, indices, metadata_list)
        """
        # Ensure 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Normalize if using cosine
        query_embedding = self._normalize_embeddings(query_embedding)

        # Search
        k = min(k, self.index.ntotal)  # Can't return more than we have
        distances, indices = self.index.search(query_embedding.astype("float32"), k)

        # Get metadata for results
        metadata_list = []
        for idx in indices[0]:
            if idx < len(self.metadata_store):
                metadata_list.append(self.metadata_store[idx])
            else:
                metadata_list.append({})

        return distances[0], indices[0], metadata_list

    def save(self, path: str):
        """Save index and metadata to disk.

        Args:
            path: Directory path to save to
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_path = path / "index.faiss"
        faiss.write_index(self.index, str(index_path))

        # Save metadata
        metadata_path = path / "metadata.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump(
                {
                    "metadata_store": self.metadata_store,
                    "id_counter": self.id_counter,
                    "dimension": self.dimension,
                    "index_type": self.index_type,
                    "metric": self.metric,
                },
                f,
            )

        logger.info(f"Saved index to {path}")

    def load(self, path: str):
        """Load index and metadata from disk.

        Args:
            path: Directory path to load from
        """
        path = Path(path)

        # Load FAISS index
        index_path = path / "index.faiss"
        self.index = faiss.read_index(str(index_path))

        # Load metadata
        metadata_path = path / "metadata.pkl"
        with open(metadata_path, "rb") as f:
            data = pickle.load(f)
            self.metadata_store = data["metadata_store"]
            self.id_counter = data["id_counter"]
            self.dimension = data["dimension"]
            self.index_type = data["index_type"]
            self.metric = data["metric"]

        logger.info(f"Loaded index from {path} ({self.index.ntotal} vectors)")


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

    def __init__(self, config: Dict[str, Any]):
        """Initialize vector store manager.

        Args:
            config: Configuration dict
        """
        self.config = config
        vs_config = config.get("vector_store", {})
        backend = vs_config.get("backend", "faiss")

        self.stores: Dict[str, VectorStore] = {}

        if backend == "faiss":
            faiss_config = vs_config.get("faiss", {})
            # We'll create stores on-demand with specific dimensions
            self.backend = "faiss"
            self.faiss_config = faiss_config
        elif backend == "milvus":
            self.backend = "milvus"
            self.milvus_config = vs_config.get("milvus", {})
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
            store = FAISSVectorStore(
                dimension=dimension,
                index_type=self.faiss_config.get("index_type", "HNSW"),
                metric=self.faiss_config.get("metric", "cosine"),
                **self.faiss_config,
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
