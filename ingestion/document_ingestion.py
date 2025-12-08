"""
Document Ingestion Pipeline
Handles PDF and DOCX processing with intelligent chunking.
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from loguru import logger


@dataclass
class DocumentChunk:
    """Represents a chunk of document text."""
    chunk_id: str
    content: str
    page_number: int
    source_file: str
    char_start: int
    char_end: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DocumentIngestionPipeline:
    """
    Production-grade document ingestion pipeline.
    Supports PDF and DOCX files with intelligent chunking.
    """
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        """
        Initialize document ingestion pipeline.
        
        Args:
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", "200"))
        
        # Check for PDF libraries
        self.pdf_library = None
        try:
            import pdfplumber
            self.pdf_library = "pdfplumber"
            logger.info("Using pdfplumber for PDF extraction")
        except ImportError:
            try:
                from pypdf import PdfReader
                self.pdf_library = "pypdf"
                logger.info("Using pypdf for PDF extraction")
            except ImportError:
                try:
                    from PyPDF2 import PdfReader
                    self.pdf_library = "PyPDF2"
                    logger.info("Using PyPDF2 for PDF extraction")
                except ImportError:
                    logger.warning("No PDF library available. Install pdfplumber, pypdf, or PyPDF2")
        
        logger.info(f"Document ingestion pipeline initialized (chunk_size={self.chunk_size}, overlap={self.chunk_overlap})")
    
    def ingest_document(
        self,
        file_path: str,
        output_dir: Optional[str] = None
    ) -> Tuple[List[DocumentChunk], Dict[str, Any]]:
        """
        Ingest a document file.
        
        Args:
            file_path: Path to the document file
            output_dir: Directory to save processed chunks
            
        Returns:
            Tuple of (list of chunks, metadata dict)
        """
        file_path = Path(file_path)
        logger.info(f"Ingesting document: {file_path.name}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        # Extract text based on file type
        suffix = file_path.suffix.lower()
        
        if suffix == ".pdf":
            pages, metadata = self.extract_text_from_pdf(str(file_path))
        elif suffix in [".docx", ".doc"]:
            pages, metadata = self.extract_text_from_docx(str(file_path))
        elif suffix == ".txt":
            pages, metadata = self.extract_text_from_txt(str(file_path))
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
        
        # Create chunks
        chunks = self.create_chunks(pages, str(file_path), metadata)
        
        # Update metadata
        metadata["num_chunks"] = len(chunks)
        metadata["source_file"] = file_path.name
        
        # Save chunks if output directory specified
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = output_dir / f"{file_path.stem}_chunks.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump([chunk.to_dict() for chunk in chunks], f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(chunks)} chunks to {output_file}")
        
        return chunks, metadata
    
    def extract_text_from_pdf(self, file_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Extract text from PDF file with metadata.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Tuple of (list of page dicts, metadata dict)
        """
        pages = []
        metadata = {
            "title": "",
            "author": "",
            "num_pages": 0,
            "file_type": "pdf"
        }
        
        if self.pdf_library == "pdfplumber":
            import pdfplumber
            
            with pdfplumber.open(file_path) as pdf:
                metadata["num_pages"] = len(pdf.pages)
                
                # Extract metadata from PDF info
                if pdf.metadata:
                    metadata["title"] = pdf.metadata.get("Title", "") or ""
                    metadata["author"] = pdf.metadata.get("Author", "") or ""
                    metadata["creator"] = pdf.metadata.get("Creator", "") or ""
                    metadata["producer"] = pdf.metadata.get("Producer", "") or ""
                
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    
                    # Try to extract title from first page if not in metadata
                    if i == 0 and not metadata["title"]:
                        metadata["title"] = self._extract_title_from_text(text)
                    
                    pages.append({
                        "page_number": i + 1,
                        "text": text,
                        "char_count": len(text)
                    })
        
        elif self.pdf_library in ["pypdf", "PyPDF2"]:
            if self.pdf_library == "pypdf":
                from pypdf import PdfReader
            else:
                from PyPDF2 import PdfReader
            
            reader = PdfReader(file_path)
            metadata["num_pages"] = len(reader.pages)
            
            # Extract metadata
            if reader.metadata:
                metadata["title"] = reader.metadata.get("/Title", "") or ""
                metadata["author"] = reader.metadata.get("/Author", "") or ""
                metadata["creator"] = reader.metadata.get("/Creator", "") or ""
            
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                
                # Try to extract title from first page if not in metadata
                if i == 0 and not metadata["title"]:
                    metadata["title"] = self._extract_title_from_text(text)
                
                pages.append({
                    "page_number": i + 1,
                    "text": text,
                    "char_count": len(text)
                })
        
        else:
            raise RuntimeError("No PDF library available")
        
        total_chars = sum(p["char_count"] for p in pages)
        logger.info(f"Extracted {len(pages)} pages, {total_chars} total characters")
        
        # If still no title, use filename
        if not metadata["title"]:
            metadata["title"] = Path(file_path).stem.replace("_", " ").replace("-", " ")
        
        return pages, metadata
    
    def _extract_title_from_text(self, text: str) -> str:
        """
        Try to extract title from first page text.
        
        Args:
            text: First page text
            
        Returns:
            Extracted title or empty string
        """
        if not text:
            return ""
        
        lines = text.strip().split("\n")
        
        # Filter and clean lines
        clean_lines = []
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if len(line) > 5 and len(line) < 200:  # Reasonable title length
                # Skip lines that look like headers/footers
                if not re.match(r'^(page|chapter|\d+|www\.|http|©)', line.lower()):
                    clean_lines.append(line)
        
        if clean_lines:
            # Return first substantial line as title
            return clean_lines[0]
        
        return ""
    
    def extract_text_from_docx(self, file_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Extract text from DOCX file.
        
        Args:
            file_path: Path to DOCX file
            
        Returns:
            Tuple of (list of page dicts, metadata dict)
        """
        try:
            from docx import Document
        except ImportError:
            raise RuntimeError("python-docx not installed. Run: pip install python-docx")
        
        doc = Document(file_path)
        
        # Extract metadata
        core_props = doc.core_properties
        metadata = {
            "title": core_props.title or Path(file_path).stem,
            "author": core_props.author or "",
            "num_pages": 1,  # DOCX doesn't have page concept
            "file_type": "docx"
        }
        
        # Extract all paragraphs
        full_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        
        pages = [{
            "page_number": 1,
            "text": full_text,
            "char_count": len(full_text)
        }]
        
        logger.info(f"Extracted {len(full_text)} characters from DOCX")
        
        return pages, metadata
    
    def extract_text_from_txt(self, file_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Extract text from TXT file.
        
        Args:
            file_path: Path to TXT file
            
        Returns:
            Tuple of (list of page dicts, metadata dict)
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        metadata = {
            "title": Path(file_path).stem,
            "author": "",
            "num_pages": 1,
            "file_type": "txt"
        }
        
        pages = [{
            "page_number": 1,
            "text": text,
            "char_count": len(text)
        }]
        
        return pages, metadata
    
    def create_chunks(
        self,
        pages: List[Dict[str, Any]],
        source_file: str,
        metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """
        Create overlapping chunks from pages.
        
        Args:
            pages: List of page dicts with text
            source_file: Source file path
            metadata: Document metadata
            
        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        chunk_id = 0
        
        for page in pages:
            page_text = page["text"]
            page_number = page["page_number"]
            
            if not page_text.strip():
                continue
            
            # Split into sentences for better chunking
            sentences = self._split_into_sentences(page_text)
            
            current_chunk = ""
            char_start = 0
            
            for sentence in sentences:
                # If adding sentence exceeds chunk size, save current chunk
                if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                    chunk = DocumentChunk(
                        chunk_id=f"chunk_{chunk_id}",
                        content=current_chunk.strip(),
                        page_number=page_number,
                        source_file=Path(source_file).name,
                        char_start=char_start,
                        char_end=char_start + len(current_chunk),
                        metadata={
                            "title": metadata.get("title", ""),
                            "author": metadata.get("author", "")
                        }
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                    
                    # Start new chunk with overlap
                    overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                    current_chunk = current_chunk[overlap_start:] + sentence
                    char_start = char_start + overlap_start
                else:
                    current_chunk += sentence
            
            # Add remaining text as final chunk for this page
            if current_chunk.strip():
                chunk = DocumentChunk(
                    chunk_id=f"chunk_{chunk_id}",
                    content=current_chunk.strip(),
                    page_number=page_number,
                    source_file=Path(source_file).name,
                    char_start=char_start,
                    char_end=char_start + len(current_chunk),
                    metadata={
                        "title": metadata.get("title", ""),
                        "author": metadata.get("author", "")
                    }
                )
                chunks.append(chunk)
                chunk_id += 1
        
        logger.info(f"Created {len(chunks)} chunks from {len(pages)} pages")
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s for s in sentences if s.strip()]