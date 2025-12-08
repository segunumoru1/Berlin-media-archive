"""
Document Ingestion Pipeline
Handles PDF and document processing with intelligent chunking and metadata extraction.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import hashlib

from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger

from utils.config import settings


@dataclass
class DocumentChunk:
    """Represents a single document chunk with metadata."""
    text: str
    page_number: int
    chunk_index: int
    source: str
    metadata: Dict
    chunk_id: Optional[str] = None
    
    def __post_init__(self):
        """Generate unique chunk ID after initialization."""
        if not self.chunk_id:
            self.chunk_id = self._generate_chunk_id()
    
    def _generate_chunk_id(self) -> str:
        """Generate unique chunk ID based on content and metadata."""
        content = f"{self.source}_{self.page_number}_{self.chunk_index}_{self.text[:100]}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def get_citation(self) -> str:
        """Get formatted citation string."""
        filename = Path(self.source).name
        return f"Source: {filename}, Page {self.page_number}"


class DocumentIngestionPipeline:
    """Pipeline for ingesting and processing documents."""
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        """
        Initialize document ingestion pipeline.
        
        Args:
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size or settings.pdf_chunk_size
        self.chunk_overlap = chunk_overlap or settings.pdf_chunk_overlap
        
        logger.info(f"Initializing DocumentIngestionPipeline")
        logger.info(f"Chunk size: {self.chunk_size}, Overlap: {self.chunk_overlap}")
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            is_separator_regex=False
        )
    
    def ingest_document(
        self,
        document_path: str,
        output_dir: Optional[str] = None
    ) -> Tuple[List[DocumentChunk], Dict]:
        """
        Ingest and process a document file.
        
        Args:
            document_path: Path to document file (PDF)
            output_dir: Optional directory to save processing results
            
        Returns:
            Tuple of (document_chunks, metadata)
        """
        try:
            document_path = Path(document_path)
            logger.info(f"Ingesting document: {document_path}")
            
            # Validate file exists
            if not document_path.exists():
                raise FileNotFoundError(f"Document not found: {document_path}")
            
            # Validate file type
            if document_path.suffix.lower() != '.pdf':
                raise ValueError(f"Unsupported file type: {document_path.suffix}")
            
            # Extract text from PDF
            logger.info("Extracting text from PDF...")
            pages_text, metadata = self._extract_pdf_text(document_path)
            
            # Process and chunk the text
            logger.info("Chunking document text...")
            chunks = self._chunk_document(pages_text, document_path, metadata)
            
            logger.info(f"Created {len(chunks)} document chunks")
            
            # Save results if output directory specified
            if output_dir:
                self._save_results(document_path, chunks, metadata, output_dir)
            
            logger.info("Document ingestion completed successfully")
            return chunks, metadata
            
        except Exception as e:
            logger.error(f"Document ingestion failed: {e}", exc_info=True)
            raise
    
    def _extract_pdf_text(self, pdf_path: Path) -> Tuple[List[Dict], Dict]:
        """
        Extract text from PDF file page by page.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (pages_text, metadata)
        """
        try:
            reader = PdfReader(str(pdf_path))
            
            # Extract metadata
            metadata = {
                "filename": pdf_path.name,
                "filepath": str(pdf_path),
                "num_pages": len(reader.pages),
                "file_size_mb": pdf_path.stat().st_size / (1024 * 1024),
            }
            
            # Try to extract PDF metadata
            if reader.metadata:
                pdf_info = reader.metadata
                metadata.update({
                    "title": pdf_info.get("/Title", ""),
                    "author": pdf_info.get("/Author", ""),
                    "subject": pdf_info.get("/Subject", ""),
                    "creator": pdf_info.get("/Creator", ""),
                    "producer": pdf_info.get("/Producer", ""),
                    "creation_date": str(pdf_info.get("/CreationDate", "")),
                })
            
            # Extract text page by page
            pages_text = []
            for page_num, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text()
                    
                    # Clean extracted text
                    text = self._clean_text(text)
                    
                    if text.strip():  # Only add non-empty pages
                        pages_text.append({
                            "page_number": page_num,
                            "text": text,
                            "char_count": len(text)
                        })
                        logger.debug(f"Extracted page {page_num}: {len(text)} characters")
                    else:
                        logger.warning(f"Page {page_num} is empty or could not be extracted")
                        
                except Exception as e:
                    logger.error(f"Failed to extract text from page {page_num}: {e}")
                    continue
            
            if not pages_text:
                raise ValueError("No text could be extracted from PDF")
            
            logger.info(f"Extracted text from {len(pages_text)} pages")
            return pages_text, metadata
            
        except Exception as e:
            logger.error(f"PDF text extraction failed: {e}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers at start/end of lines (common pattern)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _chunk_document(
        self,
        pages_text: List[Dict],
        document_path: Path,
        document_metadata: Dict
    ) -> List[DocumentChunk]:
        """
        Chunk document text intelligently.
        
        Args:
            pages_text: List of page text dictionaries
            document_path: Path to original document
            document_metadata: Document metadata
            
        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        
        for page_data in pages_text:
            page_number = page_data["page_number"]
            page_text = page_data["text"]
            
            # Split page text into chunks
            text_chunks = self.text_splitter.split_text(page_text)
            
            # Create DocumentChunk objects
            for chunk_idx, chunk_text in enumerate(text_chunks):
                # Prepare metadata
                chunk_metadata = {
                    "document_name": document_metadata["filename"],
                    "total_pages": document_metadata["num_pages"],
                    "char_count": len(chunk_text),
                    "word_count": len(chunk_text.split()),
                    "chunks_on_page": len(text_chunks),
                }
                
                # Add optional metadata if available
                if "title" in document_metadata and document_metadata["title"]:
                    chunk_metadata["document_title"] = document_metadata["title"]
                if "author" in document_metadata and document_metadata["author"]:
                    chunk_metadata["document_author"] = document_metadata["author"]
                
                chunk = DocumentChunk(
                    text=chunk_text,
                    page_number=page_number,
                    chunk_index=chunk_idx,
                    source=str(document_path),
                    metadata=chunk_metadata
                )
                
                chunks.append(chunk)
        
        return chunks
    
    def _save_results(
        self,
        document_path: Path,
        chunks: List[DocumentChunk],
        metadata: Dict,
        output_dir: str
    ):
        """
        Save processing results to files.
        
        Args:
            document_path: Original document path
            chunks: Document chunks
            metadata: Document metadata
            output_dir: Output directory
        """
        try:
            import json
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            base_name = document_path.stem
            
            # Save JSON with full details
            json_path = output_path / f"{base_name}_chunks.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "metadata": metadata,
                    "chunks": [chunk.to_dict() for chunk in chunks],
                    "statistics": {
                        "total_chunks": len(chunks),
                        "avg_chunk_size": sum(len(c.text) for c in chunks) / len(chunks),
                        "pages_processed": metadata["num_pages"]
                    }
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved JSON chunks: {json_path}")
            
            # Save human-readable text
            txt_path = output_path / f"{base_name}_chunks.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"Document: {document_path.name}\n")
                f.write(f"Pages: {metadata['num_pages']}\n")
                f.write(f"Total Chunks: {len(chunks)}\n")
                f.write("=" * 80 + "\n\n")
                
                for chunk in chunks:
                    f.write(f"--- Chunk {chunk.chunk_id} ---\n")
                    f.write(f"Page: {chunk.page_number}, Chunk Index: {chunk.chunk_index}\n")
                    f.write(f"Text Length: {len(chunk.text)} characters\n")
                    f.write(f"{chunk.text}\n")
                    f.write("\n" + "=" * 80 + "\n\n")
            
            logger.info(f"Saved text chunks: {txt_path}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def batch_ingest_documents(
        self,
        document_dir: str,
        output_dir: Optional[str] = None,
        file_pattern: str = "*.pdf"
    ) -> Dict[str, Tuple[List[DocumentChunk], Dict]]:
        """
        Batch process multiple documents from a directory.
        
        Args:
            document_dir: Directory containing documents
            output_dir: Optional output directory
            file_pattern: File pattern to match (default: *.pdf)
            
        Returns:
            Dictionary mapping filenames to (chunks, metadata) tuples
        """
        document_dir = Path(document_dir)
        results = {}
        
        # Find all matching documents
        documents = list(document_dir.glob(file_pattern))
        logger.info(f"Found {len(documents)} documents to process")
        
        for doc_path in documents:
            try:
                logger.info(f"Processing document: {doc_path.name}")
                chunks, metadata = self.ingest_document(doc_path, output_dir)
                results[doc_path.name] = (chunks, metadata)
                logger.info(f"Successfully processed: {doc_path.name}")
            except Exception as e:
                logger.error(f"Failed to process {doc_path.name}: {e}")
                continue
        
        logger.info(f"Batch processing complete. Processed {len(results)}/{len(documents)} documents")
        return results


def ingest_document_file(
    document_path: str,
    output_dir: Optional[str] = None
) -> Tuple[List[DocumentChunk], Dict]:
    """
    Convenience function to ingest a single document file.
    
    Args:
        document_path: Path to document file
        output_dir: Optional output directory
        
    Returns:
        Tuple of (chunks, metadata)
    """
    pipeline = DocumentIngestionPipeline()
    return pipeline.ingest_document(document_path, output_dir)


def batch_ingest_documents(
    document_dir: str,
    output_dir: Optional[str] = None
) -> Dict[str, Tuple[List[DocumentChunk], Dict]]:
    """
    Convenience function to batch ingest documents from a directory.
    
    Args:
        document_dir: Directory containing documents
        output_dir: Optional output directory
        
    Returns:
        Dictionary mapping filenames to (chunks, metadata) tuples
    """
    pipeline = DocumentIngestionPipeline()
    return pipeline.batch_ingest_documents(document_dir, output_dir)