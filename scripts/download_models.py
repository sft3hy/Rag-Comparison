"""
Downloads and caches all required Hugging Face models.
"""

from transformers import (
    AutoProcessor,
    AutoModel,
    AutoModelForVision2Seq,
    VisionEncoderDecoderModel,
    Pix2StructForConditionalGeneration,
    DonutProcessor,
    Pix2StructProcessor,
)
from sentence_transformers import SentenceTransformer
from loguru import logger

MODELS = [
    # OCR
    "microsoft/trocr-base-printed",
    "naver-clova-ix/donut-base",
    # Encoders
    "openai/clip-vit-base-patch32",
    "sentence-transformers/all-MiniLM-L6-v2",
    "google/tapas-base-finetuned-wtq",
    # Derender
    "google/deplot",
    "microsoft/table-transformer-detection",
]


def main():
    logger.info("Starting model download process...")
    for model_name in MODELS:
        try:
            logger.info(f"Downloading {model_name}...")
            if "sentence-transformers" in model_name:
                SentenceTransformer(model_name)
            elif model_name == "microsoft/trocr-base-printed":
                AutoProcessor.from_pretrained(model_name)
                AutoModelForVision2Seq.from_pretrained(model_name)
            elif model_name == "naver-clova-ix/donut-base":
                DonutProcessor.from_pretrained(model_name)
                VisionEncoderDecoderModel.from_pretrained(model_name)
            elif model_name == "google/deplot":
                Pix2StructProcessor.from_pretrained(model_name)
                Pix2StructForConditionalGeneration.from_pretrained(model_name)
            else:
                AutoProcessor.from_pretrained(model_name)
                AutoModel.from_pretrained(model_name)
            logger.success(f"Successfully downloaded {model_name}")
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {e}")


if __name__ == "__main__":
    main()
