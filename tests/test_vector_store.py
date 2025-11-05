import numpy as np
import shutil
from pathlib import Path
from src.index.vector_store import FAISSVectorStore


def test_faiss_vector_store():
    dim = 16
    store = FAISSVectorStore(dimension=dim, index_type="Flat")
    embeddings = np.random.rand(10, dim).astype(np.float32)
    metadata = [{"id": i} for i in range(10)]

    store.add(embeddings, metadata)
    assert store.index.ntotal == 10

    query = embeddings[0]
    distances, indices, metas = store.search(query, k=1)

    assert indices[0] == 0
    assert metas[0]["id"] == 0

    # Test save/load
    save_path = Path("./temp_test_index")
    store.save(str(save_path))

    new_store = FAISSVectorStore(dimension=dim)
    new_store.load(str(save_path))
    assert new_store.index.ntotal == 10

    shutil.rmtree(save_path)  # cleanup
