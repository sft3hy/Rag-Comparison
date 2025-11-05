"""Configuration management utilities."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field


class PathConfig(BaseModel):
    """Path configuration."""

    data_dir: str = "data"
    raw_data: str = "data/raw"
    labels_dir: str = "data/labels"
    synthetic_dir: str = "data/synthetic"
    models_dir: str = "models"
    cache_dir: str = "cache"
    output_dir: str = "outputs"
    logs_dir: str = "logs"


class OCRConfig(BaseModel):
    """OCR engine configuration."""

    engines: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class EncoderConfig(BaseModel):
    """Encoder configuration."""

    image: Dict[str, Any] = Field(default_factory=dict)
    text: Dict[str, Any] = Field(default_factory=dict)
    table: Dict[str, Any] = Field(default_factory=dict)


class VectorStoreConfig(BaseModel):
    """Vector store configuration."""

    backend: str = "faiss"
    faiss: Dict[str, Any] = Field(default_factory=dict)
    milvus: Dict[str, Any] = Field(default_factory=dict)


class LLMConfig(BaseModel):
    """LLM configuration."""

    provider: str = "openai"
    model: str = "gpt-4"
    temperature: float = 0.0
    max_tokens: int = 1000
    timeout: int = 60


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""

    metrics: list = Field(default_factory=list)
    k_values: list = Field(default_factory=lambda: [1, 3, 5, 10])
    statistical_tests: bool = True
    confidence_level: float = 0.95


class Config(BaseModel):
    """Main configuration class."""

    project: Dict[str, Any] = Field(default_factory=dict)
    paths: PathConfig = Field(default_factory=PathConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    image_processing: Dict[str, Any] = Field(default_factory=dict)
    encoders: EncoderConfig = Field(default_factory=EncoderConfig)
    derendering: Dict[str, Any] = Field(default_factory=dict)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    pipelines: Dict[str, Any] = Field(default_factory=dict)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    datasets: Dict[str, Any] = Field(default_factory=dict)
    tracking: Dict[str, Any] = Field(default_factory=dict)
    compute: Dict[str, Any] = Field(default_factory=dict)


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file. Defaults to config/config.yaml

    Returns:
        Config object
    """
    if config_path is None:
        config_path = os.path.join(
            Path(__file__).parent.parent.parent, "config", "config.yaml"
        )

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return Config(**config_dict)


def save_config(config: Config, output_path: str) -> None:
    """Save configuration to YAML file.

    Args:
        config: Config object
        output_path: Path to save config
    """
    with open(output_path, "w") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False)


def get_env_var(key: str, default: Optional[str] = None) -> str:
    """Get environment variable with optional default.

    Args:
        key: Environment variable key
        default: Default value if not found

    Returns:
        Environment variable value

    Raises:
        ValueError: If key not found and no default provided
    """
    value = os.getenv(key, default)
    if value is None:
        raise ValueError(f"Environment variable {key} not set")
    return value
