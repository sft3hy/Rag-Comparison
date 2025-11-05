"""
Common data processing utilities.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Generator


def read_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Reads a .jsonl file into a list of dictionaries."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def write_jsonl(data: List[Dict[str, Any]], file_path: Path):
    """Writes a list of dictionaries to a .jsonl file."""
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def batch_generator(data: list, batch_size: int) -> Generator[list, None, None]:
    """Yields successive n-sized chunks from a list."""
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]
