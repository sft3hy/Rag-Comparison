"""
Utilities for text normalization and processing.
"""

import re
import pandas as pd
from typing import Optional


def normalize_text(text: str) -> str:
    """Lowercase, remove articles and punctuation."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def extract_first_number(text: str) -> Optional[float]:
    """Extracts the first floating-point number from a string."""
    match = re.search(r"[-+]?\d*\.?\d+", text)
    return float(match.group()) if match else None


def flatten_table_for_prompt(df: pd.DataFrame) -> str:
    """Converts a DataFrame to a string representation for LLM prompts."""
    return df.to_string(index=False)
