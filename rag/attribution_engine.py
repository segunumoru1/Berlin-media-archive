"""
Attribution Engine
Core RAG system with precise citation and source attribution.
"""

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import json
from openai import OpenAI
from vectorstore.chroma_store import UnifiedVectorStore
from utils.config import settings

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass
class Citation:
    """Represents a single citation."""
    source: str
    source_type: str  # "audio" or "text"
    content: str
    location: str  # Timestamp for audio, Page number for text
    metadata: Dict
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def format_citation(self) -> str:
        """Format citation for display."""
        if self.source_type == "audio":
            return f"(Source: {self.source} at {self.location})"
        else:
            return f"(Source: {self.source}, {self.location})"


@dataclass
class QueryResponse:
    """Represents the response to a user query."""
    query: str
    answer: str
    citations: List[Citation]
    retrieved_chunks: List[Dict]
    metadata: Dict
    timestamp: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "retrieved_chunks": self.retrieved_chunks,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class AttributionEngine:
    """
    Core RAG engine with attribution and citation tracking.
    Ensures all answers are grounded in retrieved context.
    """
    
    def __init__(
        self,
        vector_store: UnifiedVectorStore,
        llm_model: Optional[str] = None,
        temperature: float = None,
        use_hybrid_search: bool = None
    ):
        """
        Initialize attribution engine.
        
        Args:
            vector_store: Vector store instance
            llm_model: LLM model name
            temperature: LLM temperature (lower = more deterministic)
            use_hybrid_search: Whether to use hybrid search
        """
        self.vector_store = vector_store
        self.llm_model = llm_model or settings.llm_model
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.use_hybrid_search = use_hybrid_search if use_hybrid_search is not None else settings.enable_hybrid_search
        
        logger.info(f"Initializing AttributionEngine")
        logger.info(f"LLM Model: {self.llm_model}")
        logger.info(f"Temperature: {self.temperature}")
        logger.info(f"Hybrid Search: {self.use_hybrid_search}")
        
        # Initialize OpenAI client
        try:
            self.client = OpenAI(api_key=settings.openai_api_key)
            logger.info("OpenAI client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
        
        # System prompt for the LLM
        self.system_prompt = """You are an expert archivist and researcher for the Berlin Media Archive. 

Your role is to answer questions based ONLY on the provided context from historical documents and audio transcripts.

CRITICAL RULES:
1. Use ONLY information from the provided context
2. If the answer is not in the context, say "I cannot find information about this in the archive"
3. Always cite your sources with EXACT locations:
   - For audio: Use timestamps like "at 04:22"
   - For documents: Use page numbers like "Page 12"
4. Never make up or infer information beyond what is explicitly stated
5. If multiple sources discuss the same topic, cite all relevant sources
6. Maintain accuracy and precision in all citations

Format your citations inline using this pattern:
- Audio: "According to the interview (Source: interview.mp3 at 04:22), the speaker mentioned..."
- Document: "The document states (Source: history.pdf, Page 4) that..."

Be conversational but precise. Combine information from multiple sources when relevant."""
    
    def query_archive(
        self,
        question: str,
        n_results: int = None,
        filter_metadata: Optional[Dict] = None,
        return_raw_chunks: bool = True
    ) -> QueryResponse:
        """
        Query the archive and get an attributed answer.
        
        Args:
            question: User's question
            n_results: Number of chunks to retrieve
            filter_metadata: Optional metadata filters
            return_raw_chunks: Whether to include raw chunks in response
            
        Returns:
            QueryResponse with answer and citations
        """
        logger.info(f"Processing query: '{question}'")
        
        try:
            # Step 1: Retrieve relevant chunks
            logger.info("Step 1: Retrieving relevant chunks...")
            retrieved_chunks = self._retrieve_chunks(
                question,
                n_results=n_results,
                filter_metadata=filter_metadata
            )
            
            if not retrieved_chunks:
                logger.warning("No relevant chunks found")
                return self._create_no_results_response(question)
            
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks")
            
            # Step 2: Build context for LLM
            logger.info("Step 2: Building context...")
            context = self._build_context(retrieved_chunks)
            
            # Step 3: Generate answer with LLM
            logger.info("Step 3: Generating attributed answer...")
            answer = self._generate_answer(question, context)
            
            # Step 4: Extract citations from answer and chunks
            logger.info("Step 4: Extracting citations...")
            citations = self._extract_citations(answer, retrieved_chunks)
            
            # Step 5: Create response
            response = QueryResponse(
                query=question,
                answer=answer,
                citations=citations,
                retrieved_chunks=retrieved_chunks if return_raw_chunks else [],
                metadata={
                    "num_chunks_retrieved": len(retrieved_chunks),
                    "num_citations": len(citations),
                    "llm_model": self.llm_model,
                    "search_type": "hybrid" if self.use_hybrid_search else "semantic",
                    "filter_metadata": filter_metadata
                },
                timestamp=datetime.now().isoformat()
            )
            
            logger.info("Query processing completed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}", exc_info=True)
            raise
    
    def _retrieve_chunks(
        self,
        query: str,
        n_results: int = None,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks from vector store.
        
        Args:
            query: Search query
            n_results: Number of results
            filter_metadata: Optional filters
            
        Returns:
            List of retrieved chunks with metadata
        """
        n_results = n_results or settings.top_k_results
        
        try:
            if self.use_hybrid_search:
                logger.debug("Using hybrid search")
                results = self.vector_store.hybrid_search(
                    query=query,
                    n_results=n_results,
                    filter_metadata=filter_metadata,
                    semantic_weight=1 - settings.bm25_weight
                )
            else:
                logger.debug("Using semantic search")
                raw_results = self.vector_store.search(
                    query=query,
                    n_results=n_results,
                    filter_metadata=filter_metadata
                )
                results = self.vector_store._format_search_results(raw_results)
            
            return results
            
        except Exception as e:
            logger.error(f"Chunk retrieval failed: {e}")
            raise
    
    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Build context string from retrieved chunks.
        
        Args:
            chunks: Retrieved chunks
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for idx, chunk in enumerate(chunks, 1):
            metadata = chunk["metadata"]
            content = chunk["document"]
            
            # Format based on source type
            if metadata["type"] == "audio":
                source_info = f"[AUDIO SOURCE {idx}]\n"
                source_info += f"File: {metadata['source']}\n"
                source_info += f"Timestamp: {metadata['timestamp']}\n"
                if metadata.get("speaker") and metadata["speaker"] != "UNKNOWN":
                    source_info += f"Speaker: {metadata['speaker']}\n"
                source_info += f"Content: {content}\n"
            else:  # text
                source_info = f"[DOCUMENT SOURCE {idx}]\n"
                source_info += f"File: {metadata['source']}\n"
                source_info += f"Page: {metadata['page_number']}\n"
                if metadata.get("document_title"):
                    source_info += f"Title: {metadata['document_title']}\n"
                source_info += f"Content: {content}\n"
            
            context_parts.append(source_info)
        
        return "\n---\n\n".join(context_parts)
    
    def _generate_answer(self, question: str, context: str) -> str:
        """
        Generate answer using LLM with the provided context.
        
        Args:
            question: User's question
            context: Retrieved context
            
        Returns:
            Generated answer with citations
        """
        try:
            # Create the prompt
            user_prompt = f"""Based on the following sources from the Berlin Media Archive, please answer this question:

QUESTION: {question}

SOURCES:
{context}

Remember to:
1. Only use information from the sources above
2. Cite sources with exact locations (timestamps for audio, page numbers for documents)
3. If the information is not in the sources, clearly state that
4. Combine information from multiple sources when relevant

ANSWER:"""

            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content.strip()
            logger.debug(f"Generated answer: {answer[:100]}...")
            
            return answer
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            raise
    
    def _extract_citations(
        self,
        answer: str,
        chunks: List[Dict[str, Any]]
    ) -> List[Citation]:
        """
        Extract citations from the answer and retrieved chunks.
        
        Args:
            answer: Generated answer
            chunks: Retrieved chunks
            
        Returns:
            List of Citation objects
        """
        citations = []
        
        for chunk in chunks:
            metadata = chunk["metadata"]
            content = chunk["document"]
            
            # Check if this source is mentioned in the answer
            source_name = metadata["source"]
            if source_name.lower() in answer.lower():
                if metadata["type"] == "audio":
                    citation = Citation(
                        source=source_name,
                        source_type="audio",
                        content=content[:200] + "..." if len(content) > 200 else content,
                        location=metadata["timestamp"],
                        metadata={
                            "start_time": metadata["start_time"],
                            "end_time": metadata["end_time"],
                            "speaker": metadata.get("speaker", "UNKNOWN")
                        }
                    )
                else:  # text
                    citation = Citation(
                        source=source_name,
                        source_type="text",
                        content=content[:200] + "..." if len(content) > 200 else content,
                        location=f"Page {metadata['page_number']}",
                        metadata={
                            "page_number": metadata["page_number"],
                            "chunk_index": metadata["chunk_index"],
                            "document_title": metadata.get("document_title", "")
                        }
                    )
                
                citations.append(citation)
        
        return citations
    
    def _create_no_results_response(self, question: str) -> QueryResponse:
        """Create a response when no results are found."""
        return QueryResponse(
            query=question,
            answer="I cannot find information about this in the archive. The available documents and audio files do not contain relevant information to answer your question.",
            citations=[],
            retrieved_chunks=[],
            metadata={
                "num_chunks_retrieved": 0,
                "num_citations": 0,
                "llm_model": self.llm_model
            },
            timestamp=datetime.now().isoformat()
        )
    
    def query_with_filters(
        self,
        question: str,
        source_type: Optional[str] = None,
        speaker: Optional[str] = None,
        source_file: Optional[str] = None
    ) -> QueryResponse:
        """
        Query with specific filters.
        
        Args:
            question: User's question
            source_type: Filter by type ("audio" or "text")
            speaker: Filter by speaker (for audio)
            source_file: Filter by specific source file
            
        Returns:
            QueryResponse
        """
        filter_metadata = {}
        
        if source_type:
            filter_metadata["type"] = source_type
        if speaker:
            filter_metadata["speaker"] = speaker
        if source_file:
            filter_metadata["source"] = source_file
        
        logger.info(f"Query with filters: {filter_metadata}")
        return self.query_archive(question, filter_metadata=filter_metadata)
    
    def save_response(
        self,
        response: QueryResponse,
        output_path: str
    ):
        """
        Save query response to file.
        
        Args:
            response: QueryResponse object
            output_path: Path to save the response
        """
        try:
            from pathlib import Path
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as JSON
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(response.to_json())
            
            logger.info(f"Response saved to: {output_path}")
            
            # Also save a human-readable version
            txt_path = output_path.with_suffix('.txt')
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"QUERY: {response.query}\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"ANSWER:\n{response.answer}\n\n")
                f.write("=" * 80 + "\n\n")
                f.write("CITATIONS:\n")
                for i, citation in enumerate(response.citations, 1):
                    f.write(f"\n{i}. {citation.format_citation()}\n")
                    f.write(f"   Content: {citation.content}\n")
                f.write("\n" + "=" * 80 + "\n\n")
                f.write(f"METADATA:\n")
                f.write(f"Chunks Retrieved: {response.metadata['num_chunks_retrieved']}\n")
                f.write(f"Citations: {response.metadata['num_citations']}\n")
                f.write(f"Timestamp: {response.timestamp}\n")
            
            logger.info(f"Human-readable version saved to: {txt_path}")
            
        except Exception as e:
            logger.error(f"Failed to save response: {e}")
            raise


def query_archive(
    question: str,
    vector_store: Optional[UnifiedVectorStore] = None,
    **kwargs
) -> QueryResponse:
    """
    Convenience function to query the archive.
    
    Args:
        question: User's question
        vector_store: Optional vector store instance
        **kwargs: Additional arguments for AttributionEngine
        
    Returns:
        QueryResponse
    """
    if vector_store is None:
        vector_store = UnifiedVectorStore()
    
    engine = AttributionEngine(vector_store, **kwargs)
    return engine.query_archive(question)