"""
Pydantic schemas for API requests and responses.
"""

from pydantic import BaseModel
from typing import List, Dict, Any


class QueryRequest(BaseModel):
    query: str
    pipeline_name: str
    top_k: int = 5


class Document(BaseModel):
    rank: int
    text: str = None
    metadata: Dict[str, Any] = {}


class QueryResponse(BaseModel):
    answer: str
    retrieved_docs: List[Document]
    pipeline: str
