import numpy as np
from src.encoders.embedders import TextEmbedder


def test_text_embedder():
    embedder = TextEmbedder()
    text = "This is a test sentence."
    embedding = embedder.embed_single(text)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (384,)  # for all-MiniLM-L6-v2
