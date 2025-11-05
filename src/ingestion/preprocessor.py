"""Image preprocessing and figure detection."""

import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from loguru import logger


class ImagePreprocessor:
    """Preprocess images for OCR and analysis."""

    def __init__(
        self,
        target_dpi: int = 300,
        max_dimension: int = 2048,
        normalize: bool = True,
        denoise: bool = True,
    ):
        """Initialize preprocessor.

        Args:
            target_dpi: Target DPI for images
            max_dimension: Maximum dimension for resizing
            normalize: Whether to normalize image
            denoise: Whether to apply denoising
        """
        self.target_dpi = target_dpi
        self.max_dimension = max_dimension
        self.normalize = normalize
        self.denoise = denoise

    def load_image(self, image_path: str) -> np.ndarray:
        """Load image from file.

        Args:
            image_path: Path to image

        Returns:
            Image as numpy array
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to max dimension while maintaining aspect ratio.

        Args:
            image: Input image

        Returns:
            Resized image
        """
        h, w = image.shape[:2]
        if max(h, w) <= self.max_dimension:
            return image

        scale = self.max_dimension / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)

        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image intensity.

        Args:
            image: Input image

        Returns:
            Normalized image
        """
        if len(image.shape) == 3:
            # Convert to grayscale for normalization
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            normalized = clahe.apply(gray)
            # Convert back to RGB
            return cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
        return image

    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        """Apply denoising to image.

        Args:
            image: Input image

        Returns:
            Denoised image
        """
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Apply full preprocessing pipeline.

        Args:
            image: Input image

        Returns:
            Preprocessed image
        """
        # Resize
        image = self.resize_image(image)

        # Denoise
        if self.denoise:
            image = self.denoise_image(image)

        # Normalize
        if self.normalize:
            image = self.normalize_image(image)

        return image

    def process_file(
        self, input_path: str, output_path: Optional[str] = None
    ) -> np.ndarray:
        """Process image file.

        Args:
            input_path: Path to input image
            output_path: Optional path to save processed image

        Returns:
            Processed image
        """
        logger.info(f"Processing image: {input_path}")

        # Load and preprocess
        image = self.load_image(input_path)
        processed = self.preprocess(image)

        # Save if output path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            pil_image = Image.fromarray(processed)
            pil_image.save(output_path)
            logger.info(f"Saved processed image to: {output_path}")

        return processed


class FigureDetector:
    """Detect and extract figure regions from documents."""

    def __init__(self, min_area: int = 10000, confidence_threshold: float = 0.5):
        """Initialize figure detector.

        Args:
            min_area: Minimum area for detected figures
            confidence_threshold: Confidence threshold for detection
        """
        self.min_area = min_area
        self.confidence_threshold = confidence_threshold

    def detect_figures(self, image: np.ndarray) -> List[Dict[str, any]]:
        """Detect figure regions in image using contour detection.

        Args:
            image: Input image

        Returns:
            List of detected figures with bounding boxes
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        figures = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            figures.append(
                {
                    "id": i,
                    "bbox": (x, y, w, h),
                    "area": area,
                    "confidence": 1.0,  # Placeholder for contour-based detection
                }
            )

        logger.info(f"Detected {len(figures)} figures")
        return figures

    def extract_figure(
        self, image: np.ndarray, bbox: Tuple[int, int, int, int], padding: int = 10
    ) -> np.ndarray:
        """Extract figure region from image.

        Args:
            image: Input image
            bbox: Bounding box (x, y, w, h)
            padding: Padding around bbox

        Returns:
            Extracted figure region
        """
        x, y, w, h = bbox
        h_img, w_img = image.shape[:2]

        # Add padding
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w_img, x + w + padding)
        y2 = min(h_img, y + h + padding)

        return image[y1:y2, x1:x2]

    def save_figures(
        self,
        image: np.ndarray,
        figures: List[Dict[str, any]],
        output_dir: str,
        base_name: str,
    ) -> List[str]:
        """Save extracted figures to disk.

        Args:
            image: Source image
            figures: List of detected figures
            output_dir: Output directory
            base_name: Base name for saved files

        Returns:
            List of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for fig in figures:
            fig_img = self.extract_figure(image, fig["bbox"])

            output_path = output_dir / f"{base_name}_fig_{fig['id']}.png"
            pil_img = Image.fromarray(fig_img)
            pil_img.save(output_path)

            saved_paths.append(str(output_path))
            logger.debug(f"Saved figure to: {output_path}")

        return saved_paths
