import numpy as np
import pytest
from src.ocr.engines import TesseractOCR


@pytest.fixture
def sample_image():
    # Create a simple white image with black text
    img = np.full((100, 400, 3), 255, dtype=np.uint8)
    # Note: cv2 needs to be installed for this to work
    import cv2

    cv2.putText(img, "Hello World", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return img


def test_tesseract_ocr(sample_image):
    ocr = TesseractOCR()
    result = ocr.extract_text(sample_image)
    assert "Hello" in result["text"]
    assert "World" in result["text"]
