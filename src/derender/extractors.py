"""Chart derendering and table extraction."""

import numpy as np
import pandas as pd
from PIL import Image
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
from loguru import logger
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
import camelot
import tabula


class TableExtractor(ABC):
    """Abstract base class for table extractors."""
    
    @abstractmethod
    def extract_table(self, image: np.ndarray) -> pd.DataFrame:
        """Extract table from image.
        
        Args:
            image: Input image
            
        Returns:
            Extracted table as DataFrame
        """
        pass


class CamelotExtractor(TableExtractor):
    """Extract tables using Camelot (PDF-based)."""
    
    def __init__(self, flavor: str = "lattice"):
        """Initialize Camelot extractor.
        
        Args:
            flavor: 'lattice' or 'stream'
        """
        self.flavor = flavor
        logger.info(f"Initialized CamelotExtractor with flavor={flavor}")
    
    def extract_table(self, pdf_path: str, page: int = 1) -> List[pd.DataFrame]:
        """Extract tables from PDF.
        
        Args:
            pdf_path: Path to PDF file
            page: Page number (1-indexed)
            
        Returns:
            List of extracted tables
        """
        try:
            tables = camelot.read_pdf(
                pdf_path,
                pages=str(page),
                flavor=self.flavor
            )
            
            logger.info(f"Extracted {len(tables)} tables from page {page}")
            return [table.df for table in tables]
        except Exception as e:
            logger.error(f"Camelot extraction failed: {e}")
            return []


class TabulaExtractor(TableExtractor):
    """Extract tables using Tabula."""
    
    def __init__(self, lattice: bool = True):
        """Initialize Tabula extractor.
        
        Args:
            lattice: Use lattice mode (for tables with lines)
        """
        self.lattice = lattice
        logger.info(f"Initialized TabulaExtractor with lattice={lattice}")
    
    def extract_table(self, pdf_path: str, page: int = 1) -> List[pd.DataFrame]:
        """Extract tables from PDF.
        
        Args:
            pdf_path: Path to PDF file
            page: Page number (1-indexed)
            
        Returns:
            List of extracted tables
        """
        try:
            tables = tabula.read_pdf(
                pdf_path,
                pages=page,
                lattice=self.lattice,
                multiple_tables=True
            )
            
            logger.info(f"Extracted {len(tables)} tables from page {page}")
            return tables if isinstance(tables, list) else [tables]
        except Exception as e:
            logger.error(f"Tabula extraction failed: {e}")
            return []


class ChartDerenderer:
    """Derender charts to extract underlying data tables."""
    
    def __init__(
        self,
        model_name: str = "google/deplot",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize chart derenderer.
        
        Args:
            model_name: Model name for derendering
            device: Device to run on
        """
        self.model_name = model_name
        self.device = device
        
        try:
            logger.info(f"Loading derendering model: {model_name}")
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForVision2Seq.from_pretrained(model_name)
            self.model.to(device)
            self.model.eval()
            logger.info(f"Derendering model loaded on {device}")
        except Exception as e:
            logger.warning(f"Could not load derendering model: {e}")
            self.processor = None
            self.model = None
    
    def derender_chart(self, image: np.ndarray) -> Dict[str, any]:
        """Derender chart to extract data table.
        
        Args:
            image: Input chart image
            
        Returns:
            Dict with extracted table data
        """
        if self.model is None:
            logger.warning("Derendering model not available")
            return {
                'table': pd.DataFrame(),
                'raw_output': '',
                'success': False
            }
        
        # Convert to PIL
        pil_image = Image.fromarray(image)
        
        # Process
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=512)
        
        # Decode
        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Parse output to DataFrame
        table_df = self._parse_deplot_output(generated_text)
        
        return {
            'table': table_df,
            'raw_output': generated_text,
            'success': not table_df.empty
        }
    
    def _parse_deplot_output(self, text: str) -> pd.DataFrame:
        """Parse DePlot output text to DataFrame.
        
        Args:
            text: Raw model output
            
        Returns:
            Parsed DataFrame
        """
        try:
            # DePlot outputs in format: "TITLE | x1 | y1 <0x0A> x2 | y2 ..."
            # This is a simplified parser - actual implementation may vary
            lines = text.split('<0x0A>')
            
            if not lines:
                return pd.DataFrame()
            
            # First line might be title
            data_lines = [line.strip() for line in lines if '|' in line]
            
            rows = []
            for line in data_lines:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 2:
                    rows.append(parts)
            
            if not rows:
                return pd.DataFrame()
            
            # Create DataFrame
            df = pd.DataFrame(rows[1:], columns=rows[0] if len(rows) > 1 else None)
            return df
            
        except Exception as e:
            logger.error(f"Failed to parse DePlot output: {e}")
            return pd.DataFrame()


class TableStructureExtractor:
    """Extract table structure from images using deep learning."""
    
    def __init__(
        self,
        model_name: str = "microsoft/table-transformer-detection",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize table structure extractor.
        
        Args:
            model_name: Model for table detection
            device: Device to run on
        """
        self.model_name = model_name
        self.device = device
        
        try:
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
            
            logger.info(f"Loading table detection model: {model_name}")
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForObjectDetection.from_pretrained(model_name)
            self.model.to(device)
            self.model.eval()
            logger.info(f"Table detection model loaded on {device}")
        except Exception as e:
            logger.warning(f"Could not load table detection model: {e}")
            self.processor = None
            self.model = None
    
    def detect_table_structure(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.5
    ) -> Dict[str, any]:
        """Detect table structure (rows, columns, cells).
        
        Args:
            image: Input table image
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            Dict with detected structure
        """
        if self.model is None:
            logger.warning("Table detection model not available")
            return {
                'rows': [],
                'columns': [],
                'cells': [],
                'success': False
            }
        
        # Convert to PIL
        pil_image = Image.fromarray(image)
        
        # Process
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Detect
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process
        target_sizes = torch.tensor([pil_image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=confidence_threshold
        )[0]
        
        # Extract components
        rows = []
        columns = []
        cells = []
        
        for score, label, box in zip(
            results["scores"],
            results["labels"],
            results["boxes"]
        ):
            label_name = self.model.config.id2label[label.item()]
            bbox = box.tolist()
            
            if label_name == "table row":
                rows.append({'bbox': bbox, 'confidence': score.item()})
            elif label_name == "table column":
                columns.append({'bbox': bbox, 'confidence': score.item()})
            elif label_name == "table":
                cells.append({'bbox': bbox, 'confidence': score.item()})
        
        return {
            'rows': rows,
            'columns': columns,
            'cells': cells,
            'success': True
        }


class DerenderingManager:
    """Manager for derendering and table extraction."""
    
    def __init__(self, config: Dict[str, any]):
        """Initialize derendering manager.
        
        Args:
            config: Configuration dict
        """
        self.config = config
        
        # Initialize chart derenderer
        if config.get('chart_to_table', {}).get('enabled', False):
            self.chart_derenderer = ChartDerenderer()
        else:
            self.chart_derenderer = None
        
        # Initialize table extractors
        self.camelot_extractor = CamelotExtractor()
        self.tabula_extractor = TabulaExtractor()
        
        # Initialize structure extractor
        if config.get('table_extraction', {}).get('enabled', False):
            self.structure_extractor = TableStructureExtractor()
        else:
            self.structure_extractor = None
        
        logger.info("Derendering manager initialized")
    
    def process_chart(self, image: np.ndarray) -> Dict[str, any]:
        """Process chart image to extract table.
        
        Args:
            image: Chart image
            
        Returns:
            Extraction results
        """
        if self.chart_derenderer is None:
            logger.warning("Chart derenderer not available")
            return {'success': False, 'table': pd.DataFrame()}
        
        return self.chart_derenderer.derender_chart(image)
    
    def process_table_image(self, image: np.ndarray) -> Dict[str, any]:
        """Process table image to extract structure and data.
        
        Args:
            image: Table image
            
        Returns:
            Extraction results
        """
        result = {
            'structure': None,
            'table': pd.DataFrame(),
            'success': False
        }
        
        # Detect structure
        if self.structure_extractor:
            result['structure'] = self.structure_extractor.detect_table_structure(image)
        
        # Note: For actual table extraction from images, you would combine
        # structure detection with OCR to populate cells
        # This is a simplified implementation
        
        return result
    
    def process_pdf_table(self, pdf_path: str, page: int = 1) -> List[pd.DataFrame]:
        """Extract tables from PDF.
        
        Args:
            pdf_path: Path to PDF
            page: Page number
            
        Returns:
            List of extracted tables
        """
        tables = []
        
        # Try Camelot first (better for bordered tables)
        try:
            tables = self.camelot_extractor.extract_table(pdf_path, page)
        except Exception as e:
            logger.warning(f"Camelot failed: {e}, trying Tabula")
        
        # Fallback to Tabula
        if not tables:
            try:
                tables = self.tabula_extractor.extract_table(pdf_path, page)
            except Exception as e:
                logger.error(f"Tabula also failed: {e}")
        
        return tables