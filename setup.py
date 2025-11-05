"""Setup script for RAG Charts project."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = (
    (this_directory / "requirements.txt").read_text(encoding="utf-8").splitlines()
)
requirements = [
    req.strip() for req in requirements if req.strip() and not req.startswith("#")
]

setup(
    name="rag-charts",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="RAG for Charts & Tables: OCR vs Image-Indexing Comparison",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/rag_charts_project",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9,<3.12",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.11.0",
            "black>=23.9.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
            "isort>=5.12.0",
        ],
        "gpu": [
            "faiss-gpu>=1.7.4",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "sphinx-autodoc-typehints>=1.24.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rag-ingest=run_ingest:main",
            "rag-ocr=run_ocr:main",
            "rag-derender=run_derender:main",
            "rag-index=index_build:main",
            "rag-query=query_rag:main",
            "rag-eval=eval:main",
            "rag-ablation=run_ablation:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    zip_safe=False,
)
