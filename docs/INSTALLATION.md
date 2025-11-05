# Installation Guide

This guide provides step-by-step instructions for setting up the RAG for Charts & Tables project on your local machine. Two installation methods are provided: using a Python virtual environment, and using Docker.

## Method 1: Python Virtual Environment (Recommended for Development)

This method gives you direct control over the environment and is ideal for development and experimentation.

### Prerequisites
*   **Python 3.9+**
*   **Git**
*   **Tesseract OCR Engine:** You must have Tesseract installed on your system.
    *   **macOS (via Homebrew):** `brew install tesseract`
    *   **Ubuntu/Debian:** `sudo apt-get update && sudo apt-get install tesseract-ocr`

### Step 1: Clone the Repository
First, clone the project repository from GitHub to your local machine.

```bash
git clone https://github.com/yourusername/rag_charts_project.git
cd rag_charts_project
```

### Step 2: Create and Activate a Virtual Environment
It is highly recommended to use a virtual environment to manage project dependencies and avoid conflicts.

```bash
# Create a virtual environment named 'venv'
python3 -m venv venv

# Activate the environment
source venv/bin/activate
```
Your terminal prompt should now be prefixed with `(venv)`.

### Step 3: Install Dependencies
Install all the required Python packages using the `requirements.txt` file.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables
The project uses a `.env` file to manage sensitive API keys. Copy the example file and then add your keys.

```bash
# Copy the example file
cp .env.example .env
```
Now, open the `.env` file in a text editor and add your API keys for services like OpenAI and Weights & Biases.
```
OPENAI_API_KEY="sk-..."
WANDB_API_KEY="..."
WANDB_ENTITY="..."
```

### Step 5: Download AI Models and Datasets
The system relies on several pre-trained models from Hugging Face Hub and standard datasets for evaluation. Run the provided scripts to download them.

```bash
# Download all required AI models
python scripts/download_models.py

# Download all datasets (e.g., ChartQA, PubTabNet)
python scripts/download_datasets.py --dataset all
```
The project is now fully set up and ready to use!

---

## Method 2: Docker & Docker Compose (Recommended for Production & Easy Setup)

This method uses containers to run the application and its dependencies (like the Milvus vector database), ensuring a consistent environment.

### Prerequisites
*   **Docker Desktop** installed and running on your system.

### Step 1: Clone the Repository and Set Up Environment
Follow steps 1 and 4 from the previous method to clone the repository and create your `.env` file with the necessary API keys.

### Step 2: Build and Run the Services
Use Docker Compose to build the application image and start all the services defined in `docker-compose.yml`.

```bash
# Build the Docker image for the application
docker-compose build

# Start the application, Milvus, and other services in the background
docker-compose up -d
```

### Step 3: Verify the Setup
You can check the logs to ensure all services started correctly. The API should now be running and accessible.

```bash
# View the logs of the running services
docker-compose logs -f

# Check the health of the API
curl http://localhost:8000/health
# Expected output: {"status":"ok"}
```
The entire application stack is now running inside Docker containers.