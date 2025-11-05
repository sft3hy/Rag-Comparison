# API Documentation

This document provides details for interacting with the RAG for Charts & Tables system via its REST API. The API is built using **FastAPI** and is the recommended way to integrate the system into other applications.

## Getting Started

To run the API server, it is easiest to use the Docker Compose setup as described in the [Installation Guide](./INSTALLATION.md).

```bash
# Start the API and its dependencies
docker-compose up -d
```
The API will be available at `http://localhost:8000`.

### Interactive Documentation (Swagger UI)

FastAPI automatically generates interactive API documentation. Once the server is running, you can access it in your browser at:

*   **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
*   **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

This is the best place to explore and test the API endpoints live.

---

## Endpoints

### Health Check

A simple endpoint to verify that the API server is running and healthy.

*   **Endpoint:** `GET /health`
*   **Description:** Returns the operational status of the service.
*   **Request Body:** None.
*   **Success Response (`200 OK`):**
    ```json
    {
      "status": "ok"
    }
    ```

### Query the RAG System

This is the main endpoint for submitting a natural language query and receiving an answer from a specific RAG pipeline.

*   **Endpoint:** `POST /query`
*   **Description:** Processes a query using the specified pipeline and returns a synthesized answer along with the retrieved source documents.

#### Request Body

The request must be a JSON object with the following fields:

| Field           | Type   | Required | Description                                                              |
| --------------- | ------ | -------- | ------------------------------------------------------------------------ |
| `query`         | string | Yes      | The natural language question you want to ask.                           |
| `pipeline_name` | string | Yes      | The name of the RAG pipeline to use (e.g., `ocr-text-vec`, `image-vec`). |
| `top_k`         | integer| No       | The number of documents to retrieve. Defaults to `5`.                    |

**Example Request Body:**
```json
{
  "query": "What were the sales figures for the Alpha project in June?",
  "pipeline_name": "ocr-text-vec",
  "top_k": 3
}
```

#### Success Response (`200 OK`)

A successful response will be a JSON object containing the answer and context.

| Field            | Type        | Description                                                          |
| ---------------- | ----------- | -------------------------------------------------------------------- |
| `answer`         | string      | The final, synthesized answer from the LLM.                          |
| `retrieved_docs` | array[object] | A list of the source documents retrieved to generate the answer.     |
| `pipeline`       | string      | The name of the pipeline that processed the request.                 |

**Example Success Response:**
```json
{
  "answer": "The sales figures for the Alpha project in June were approximately $15,200.",
  "retrieved_docs": [
    {
      "rank": 1,
      "text": "Source Text: ...project Alpha sales reached $15.2k in June...",
      "metadata": {
        "chunk_id": "report_q2_fig_3.png",
        "distance": 0.89
      }
    },
    {
      "rank": 2,
      "text": "Source Text: ...May sales were $12.5k, June was $15.2k...",
      "metadata": {
        "chunk_id": "summary_page_fig_1.png",
        "distance": 0.85
      }
    }
  ],
  "pipeline": "ocr-text-vec"
}
```

#### Error Response

If an issue occurs (e.g., the specified pipeline does not exist), the API will return an error.

*   **Status Code:** `404 Not Found`
*   **Response Body:**
    ```json
    {
      "detail": "Pipeline 'non-existent-pipeline' not found."
    }
    ```

### Example `curl` Request

You can test the API from your terminal using `curl`.

```bash
curl -X 'POST' \
  'http://localhost:8000/query' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "What is the average value for Category B?",
    "pipeline_name": "ocr-text-vec",
    "top_k": 3
  }'
```