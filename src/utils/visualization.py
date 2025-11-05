"""
Plotting and visualization utilities.
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path


def plot_retrieval_results(
    query_text: str, retrieved_images: list, scores: np.ndarray, output_path: Path
):
    """Plots retrieved images for a given query."""
    n = len(retrieved_images)
    fig, axes = plt.subplots(1, n, figsize=(n * 4, 4))
    fig.suptitle(f"Query: '{query_text}'")
    for i, (img, score) in enumerate(zip(retrieved_images, scores)):
        axes[i].imshow(img)
        axes[i].set_title(f"Score: {score:.4f}")
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def draw_bboxes(image: np.ndarray, bboxes: list) -> np.ndarray:
    """Draws bounding boxes on an image."""
    vis_image = image.copy()
    for bbox in bboxes:
        x, y, w, h = bbox
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return vis_image
