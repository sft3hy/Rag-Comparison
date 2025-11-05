"""
FastAPI application for the RAG system.
"""

from fastapi import FastAPI, HTTPException
from .schemas import QueryRequest, QueryResponse
from src.utils.config import load_config
from src.rag_pipelines.orchestrator import PipelineOrchestrator

# Import other necessary components (managers, pipelines) here...

app = FastAPI(title="RAG for Charts API")

# --- Globals (initialize on startup) ---
# This is a simplified setup. In production, use dependency injection.
config = load_config()
# orchestrator = PipelineOrchestrator(config)
# ... load indexes and register pipelines ...
# ----------------------------------------


@app.post("/query", response_model=QueryResponse)
async def query_rag_system(request: QueryRequest):
    """
    Main endpoint to query the RAG system.
    """
    # This is a placeholder for the full pipeline logic.
    # if request.pipeline_name not in orchestrator.pipelines:
    #     raise HTTPException(status_code=404, detail="Pipeline not found")

    # result = orchestrator.run_pipeline(request.pipeline_name, request.query, k=request.top_k)

    # Mock response
    mock_result = {
        "answer": f"This is a mock answer for the query '{request.query}' using pipeline '{request.pipeline_name}'.",
        "retrieved_docs": [
            {"rank": 1, "text": "Mock retrieved context 1."},
            {"rank": 2, "text": "Mock retrieved context 2."},
        ],
        "pipeline": request.pipeline_name,
    }

    return QueryResponse(**mock_result)


@app.get("/health")
def health_check():
    return {"status": "ok"}
