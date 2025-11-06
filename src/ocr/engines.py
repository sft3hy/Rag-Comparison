"""OCR engine implementations."""

import numpy as np
import pytesseract
from PIL import Image
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod
from loguru import logger
import torch
from src.utils.config import OCRConfig
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, DonutProcessor


class OCREngine(ABC):
    """Abstract base class for OCR engines."""

    @abstractmethod
    def extract_text(self, image: np.ndarray) -> Dict[str, any]:
        """Extract text from image.

        Args:
            image: Input image as numpy array

        Returns:
            Dict with text, tokens, bboxes, and confidences
        """
        pass


class TesseractOCR(OCREngine):
    """Tesseract OCR engine."""

    def __init__(self, lang: str = "eng", config: str = "--oem 3 --psm 6"):
        """Initialize Tesseract OCR.

        Args:
            lang: Language code
            config: Tesseract configuration string
        """
        self.lang = lang
        self.config = config
        logger.info(f"Initialized TesseractOCR with lang={lang}")

    def extract_text(self, image: np.ndarray) -> Dict[str, any]:
        """Extract text using Tesseract.

        Args:
            image: Input image

        Returns:
            OCR results dict
        """
        # Convert to PIL Image
        pil_image = Image.fromarray(image)

        # Get detailed data
        data = pytesseract.image_to_data(
            pil_image,
            lang=self.lang,
            config=self.config,
            output_type=pytesseract.Output.DICT,
        )

        # Extract text
        full_text = pytesseract.image_to_string(
            pil_image, lang=self.lang, config=self.config
        )

        # Process tokens and bboxes
        tokens = []
        bboxes = []
        confidences = []

        n_boxes = len(data["text"])
        for i in range(n_boxes):
            if int(data["conf"][i]) > 0:  # Filter out low confidence
                tokens.append(data["text"][i])
                bboxes.append(
                    (
                        data["left"][i],
                        data["top"][i],
                        data["width"][i],
                        data["height"][i],
                    )
                )
                confidences.append(float(data["conf"][i]) / 100.0)

        return {
            "text": full_text,
            "tokens": tokens,
            "bboxes": bboxes,
            "confidences": confidences,
            "engine": "tesseract",
        }


class TrOCREngine(OCREngine):
    """Microsoft TrOCR engine."""

    def __init__(
        self,
        model_name: str = "microsoft/trocr-base-printed",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize TrOCR.

        Args:
            model_name: HuggingFace model name
            device: Device to run model on
        """
        self.model_name = model_name
        self.device = device

        logger.info(f"Loading TrOCR model: {model_name}")
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        logger.info(f"TrOCR loaded on {device}")

    def extract_text(self, image: np.ndarray) -> Dict[str, any]:
        """Extract text using TrOCR.

        Args:
            image: Input image

        Returns:
            OCR results dict
        """
        # Convert to PIL
        pil_image = Image.fromarray(image)

        # Process image
        pixel_values = self.processor(
            images=pil_image, return_tensors="pt"
        ).pixel_values.to(self.device)

        # Generate text
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)

        # Decode
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        # TrOCR doesn't provide token-level bboxes or confidences
        # We return the full text and mark confidence as N/A
        return {
            "text": generated_text,
            "tokens": [generated_text],  # Single token for simplicity
            "bboxes": [(0, 0, image.shape[1], image.shape[0])],  # Full image bbox
            "confidences": [1.0],  # Placeholder confidence
            "engine": "trocr",
        }


class DonutOCREngine(OCREngine):
    def __init__(
        self, model_name: str = "naver-clova-ix/donut-base", device: str = "cuda"
    ):
        self.model_name = model_name
        device_preference = device
        if device_preference == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available. Falling back to MPS or CPU.")
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device_preference
        try:
            logger.info(f"Loading Donut model: {model_name} onto device: {self.device}")
            self.processor = DonutProcessor.from_pretrained(model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(
                self.device
            )
            self.model.eval()
            logger.info(f"Donut loaded on {self.device}")
        except ImportError:
            logger.warning(
                "Donut model not available, install with: pip install donut-python"
            )
            self.processor = None
            self.model = None

    def extract_text(self, image: np.ndarray) -> Dict[str, any]:
        if self.model is None or self.processor is None:
            return {
                "text": "",
                "tokens": [],
                "bboxes": [],
                "confidences": [],
                "engine": "donut",
            }

        pil_image = Image.fromarray(image).convert("RGB")
        pixel_values = self.processor(pil_image, return_tensors="pt").pixel_values.to(
            self.device
        )

        # --- FIX STARTS HERE: Definitive fallback for special tokens ---
        task_prompt = "<s_doc-parser>"
        decoder_input_ids = self.processor.tokenizer(
            task_prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(self.device)

        # Get special token IDs from the tokenizer, which is the most reliable source.
        pad_token_id = self.processor.tokenizer.pad_token_id
        eos_token_id = self.processor.tokenizer.eos_token_id
        unk_token_id = self.processor.tokenizer.unk_token_id  # Fallback token

        # If pad token is missing, try to use eos token.
        if pad_token_id is None:
            pad_token_id = eos_token_id

        # If both pad and eos tokens are missing, use the unk token as a last resort.
        # This prevents the NoneType error.
        if pad_token_id is None and eos_token_id is None:
            if unk_token_id is not None:
                logger.warning(
                    "PAD and EOS tokens not found, falling back to UNK token."
                )
                pad_token_id = unk_token_id
                eos_token_id = unk_token_id
            else:
                # This case is extremely unlikely but handles a completely broken tokenizer.
                logger.error(
                    "Critical error: No PAD, EOS, or UNK tokens found in tokenizer."
                )
                return {
                    "text": "Error: Tokenizer is misconfigured",
                    "tokens": [],
                    "bboxes": [],
                    "confidences": [],
                    "engine": "donut",
                }

        # Generate text
        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=self.model.decoder.config.max_position_embeddings,
                early_stopping=True,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
        # --- FIX ENDS HERE ---

        sequence = self.processor.batch_decode(outputs)[0]
        sequence = sequence.replace(
            self.processor.tokenizer.eos_token or "", ""
        ).replace(self.processor.tokenizer.pad_token or "", "")
        sequence = sequence.replace(task_prompt, "").strip()

        return {
            "text": sequence,
            "tokens": [sequence],
            "bboxes": [(0, 0, image.shape[1], image.shape[0])],
            "confidences": [1.0],
            "engine": "donut",
        }


class OCRManager:
    """Manager for multiple OCR engines."""

    # --- FIX STARTS HERE ---
    def __init__(self, config: OCRConfig):
        """
        Initialize OCR manager.

        Args:
            config: Pydantic OCRConfig object.
        """
        self.engines = {}
        self.config = config

        # Access the 'engines' field as a direct attribute of the Pydantic object.
        # This is the corrected line.
        engines_config = config.engines

        # The rest of the code works as expected because engines_config is a dictionary.
        if engines_config.get("tesseract", {}).get("enabled", False):
            self.engines["tesseract"] = TesseractOCR(
                lang=engines_config["tesseract"].get("lang", "eng"),
                config=engines_config["tesseract"].get("config", "--oem 3 --psm 6"),
            )

        if engines_config.get("trocr", {}).get("enabled", False):
            self.engines["trocr"] = TrOCREngine(
                model_name=engines_config["trocr"].get(
                    "model", "microsoft/trocr-base-printed"
                ),
                device=engines_config["trocr"].get("device", "cuda"),
            )

        if engines_config.get("donut", {}).get("enabled", False):
            self.engines["donut"] = DonutOCREngine(
                model_name=engines_config["donut"].get(
                    "model", "naver-clova-ix/donut-base"
                ),
                device=engines_config["donut"].get("device", "cuda"),
            )

        logger.info(
            f"OCR Manager initialized with engines: {list(self.engines.keys())}"
        )

    # --- FIX ENDS HERE ---

    def run_ocr(
        self, image: np.ndarray, engine_name: str = "tesseract"
    ) -> Dict[str, any]:
        if engine_name not in self.engines:
            raise ValueError(
                f"Engine {engine_name} not available. Available: {list(self.engines.keys())}"
            )
        logger.info(f"Running OCR with {engine_name}")
        return self.engines[engine_name].extract_text(image)

    def run_all_engines(self, image: np.ndarray) -> Dict[str, Dict[str, any]]:
        results = {}
        for engine_name in self.engines:
            try:
                results[engine_name] = self.run_ocr(image, engine_name)
            except Exception as e:
                logger.error(f"Error running {engine_name}: {e}")
                results[engine_name] = {
                    "text": "",
                    "tokens": [],
                    "bboxes": [],
                    "confidences": [],
                    "engine": engine_name,
                    "error": str(e),
                }
        return results
