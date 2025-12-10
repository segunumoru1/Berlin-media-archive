"""
Berlin Media Archive - Command Line Interface
Production-grade CLI tool for all operations.
"""

import sys
import json
import click
from pathlib import Path
from typing import Optional
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger


def setup_logging(log_level: str = "INFO", log_file: str = "cli.log"):
    """Setup logging configuration"""
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
    )
    logger.add(
        log_file,
        rotation="10 MB",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}"
    )


def ensure_directories():
    """Ensure all necessary directories exist"""
    dirs = [
        os.getenv("DATA_DIR", "./data"),
        os.getenv("AUDIO_DIR", "./data/audio"),
        os.getenv("DOCUMENTS_DIR", "./data/documents"),
        os.getenv("OUTPUT_DIR", "./output"),
        os.getenv("LOGS_DIR", "./logs"),
        os.getenv("VECTORSTORE_PATH", "./data/vectorstore"),
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured directories exist: {dirs}")


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose):
    """Berlin Media Archive - Multi-Modal RAG System CLI"""
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(log_level=log_level, log_file="cli.log")
    ensure_directories()


@cli.group()
def ingest():
    """Ingest content into the archive"""
    pass


@ingest.command()
@click.argument('audio_file', type=click.Path(exists=True))
@click.option('--output-dir', '-o', type=click.Path(), help='Output directory')
@click.option('--enable-diarization/--no-diarization', default=True, help='Enable speaker diarization')
def audio(audio_file, output_dir, enable_diarization):
    """Ingest an audio file with transcription and diarization"""
    try:
        click.echo(f"🎙️  Ingesting audio file: {audio_file}")
        
        from ingestion.audio_ingestion import AudioIngestionPipeline
        
        pipeline = AudioIngestionPipeline(enable_diarization=enable_diarization)
        
        output_dir = output_dir or os.path.join(os.getenv("OUTPUT_DIR", "./output"), "audio")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        segments, metadata = pipeline.ingest_audio(audio_file, output_dir)
        
        click.echo(f"\n✅ Audio ingestion complete!")
        click.echo(f"   - Duration: {metadata.get('duration_seconds', 'N/A'):.2f}s")
        click.echo(f"   - Segments: {len(segments)}")
        
        if metadata.get('speakers'):
            click.echo(f"   - Speakers: {', '.join(metadata['speakers'])}")
        
        click.echo(f"\n💾 Output saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Audio ingestion failed: {e}", exc_info=True)
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@ingest.command()
@click.argument('document_file', type=click.Path(exists=True))
@click.option('--output-dir', '-o', type=click.Path(), help='Output directory')
@click.option('--chunk-size', '-c', type=int, help='Chunk size (default from env)')
@click.option('--chunk-overlap', type=int, help='Chunk overlap (default from env)')
def document(document_file, output_dir, chunk_size, chunk_overlap):
    """Ingest a PDF document with intelligent chunking"""
    try:
        click.echo(f"📄 Ingesting document: {document_file}")
        
        from ingestion.document_ingestion import DocumentIngestionPipeline
        
        # Use env defaults if not provided
        chunk_size = chunk_size or int(os.getenv("PDF_CHUNK_SIZE", "1000"))
        chunk_overlap = chunk_overlap or int(os.getenv("PDF_CHUNK_OVERLAP", "200"))
        
        pipeline = DocumentIngestionPipeline(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        output_dir = output_dir or os.path.join(os.getenv("OUTPUT_DIR", "./output"), "documents")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        chunks, metadata = pipeline.ingest_document(document_file, output_dir)
        
        click.echo(f"\n✅ Document ingestion complete!")
        click.echo(f"   - Pages: {metadata.get('num_pages', 'N/A')}")
        click.echo(f"   - Chunks: {len(chunks)}")
        click.echo(f"\n💾 Output saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Document ingestion failed: {e}", exc_info=True)
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('question')
@click.option('--top-k', '-k', type=int, help='Number of results (default from env)')
@click.option('--filter-source', '-f', type=str, help='Filter by source type (audio/document)')
@click.option('--filter-speaker', '-s', type=str, help='Filter by speaker')
def query(question, top_k, filter_source, filter_speaker):
    """Query the archive with natural language"""
    try:
        top_k = top_k or int(os.getenv("TOP_K_RESULTS", "5"))
        
        click.echo(f"🔍 Querying: {question}")
        click.echo(f"   Top-K: {top_k}")
        
        from rag.attribution_engine import AttributionEngine
        from embeddings.openai_embeddings import OpenAIEmbeddings
        from pipeline.orchestrator import BerlinArchivePipeline
        from vectorstore.chroma_store import UnifiedVectorStore
        
        # Initialize components
        embeddings = OpenAIEmbeddings()
        vector_store = UnifiedVectorStore()
        orchestrator = BerlinArchivePipeline()
        attribution_engine = AttributionEngine(vector_store=vector_store)
        
        # Build filters
        filters = {}
        if filter_source:
            filters['source_type'] = filter_source
        if filter_speaker:
            filters['speaker'] = filter_speaker
        
        # Process query
        result = orchestrator.process_query(
            query=question,
            top_k=top_k,
            filters=filters
        )
        
        # Add attribution
        attribution = attribution_engine.generate_attribution(
            query=question,
            answer=result['answer'],
            sources=result['sources']
        )
        
        click.echo(f"\n💬 Answer:\n{result['answer']}")
        
        if attribution and attribution.get('citations'):
            click.echo(f"\n📚 Sources ({len(attribution['citations'])}):")
            for i, citation in enumerate(attribution['citations'], 1):
                click.echo(f"   {i}. {citation}")
        
        if attribution and attribution.get('confidence_score'):
            click.echo(f"\n📊 Confidence: {attribution['confidence_score']:.2f}")
        
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.group()
def evaluate():
    """Evaluate RAG system performance"""
    pass


@evaluate.command()
@click.option('--test-cases', '-t', type=click.Path(exists=True), help='Path to test cases JSON')
@click.option('--output', '-o', type=click.Path(), help='Output file for results')
def run(test_cases, output):
    """Run evaluation on test cases"""
    try:
        click.echo("📈 Running RAG evaluation...")
        
        from rag_evaluator.llm_rag_evaluator import LLMRAGEvaluator
        from rag_evaluator.test_cases import EvaluationTestCases
        
        # Check if evaluation is enabled
        if os.getenv("ENABLE_EVALUATION", "true").lower() != "true":
            click.echo("⚠️  Evaluation is disabled in configuration")
            return
        
        evaluator = LLMRAGEvaluator()
        
        # Load test cases
        if test_cases:
            with open(test_cases, 'r') as f:
                cases = json.load(f)
            click.echo(f"   Loaded {len(cases)} test cases from file")
        else:
            click.echo("ℹ️  Using default test cases")
            test_case_generator = EvaluationTestCases()
            cases = test_case_generator.get_sample_test_cases()
        
        click.echo(f"   Test cases: {len(cases)}")
        
        # Run evaluation
        output_path = output or os.path.join(os.getenv("OUTPUT_DIR", "./output"), "evaluation_results.json")
        
        results = evaluator.batch_evaluate(
            test_cases=cases,
            save_results=True,
            output_path=output_path
        )
        
        # Display results
        click.echo(f"\n✅ Evaluation complete!")
        click.echo(f"\n📊 Results:")
        
        if isinstance(results, dict):
            click.echo(f"   - Average Faithfulness: {results.get('avg_faithfulness', 0):.2f}")
            click.echo(f"   - Average Relevance: {results.get('avg_relevance', 0):.2f}")
            click.echo(f"   - Average Citation Quality: {results.get('avg_citation_quality', 0):.2f}")
            click.echo(f"   - Overall Score: {results.get('avg_overall', 0):.2f}")
            
            if 'grade_distribution' in results:
                click.echo(f"\n   - Grade Distribution: {results['grade_distribution']}")
            
            if 'production_ready_percentage' in results:
                click.echo(f"   - Production Ready: {results.get('production_ready_count', 0)}/{results.get('total_cases', 0)} ({results['production_ready_percentage']:.1f}%)")
        
        click.echo(f"\n💾 Detailed results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@evaluate.command()
@click.argument('query')
@click.option('--answer', '-a', required=True, help='System answer')
@click.option('--contexts', '-c', multiple=True, required=True, help='Retrieved contexts')
@click.option('--ground-truth', '-g', help='Ground truth answer (optional)')
def single(query: str, answer: str, contexts: tuple, ground_truth: Optional[str]):
    """Evaluate a single query response."""
    try:
        logger.info(f"Evaluating query: {query}")
        
        # Initialize evaluator
        from rag_evaluator.gemini_rag_evaluator import GeminiRAGEvaluator
        evaluator = GeminiRAGEvaluator()
        
        # Convert contexts tuple to list
        contexts_list = list(contexts)
        
        # Run evaluation
        result = evaluator.evaluate(
            query=query,
            answer=answer,
            retrieved_contexts=contexts_list,
            citations=[],  # Can be added if needed
            ground_truth=ground_truth
        )
        
        # Print results
        click.echo("\n" + "="*80)
        click.echo("EVALUATION RESULTS")
        click.echo("="*80 + "\n")
        
        metrics = result.metrics
        click.echo(f"Overall Score: {metrics.overall_score:.2f}")
        click.echo(f"\nRetrieval Quality:")
        click.echo(f"  Precision: {metrics.retrieval_precision:.2f}")
        click.echo(f"  Recall: {metrics.retrieval_recall:.2f}")
        
        click.echo(f"\nAnswer Quality:")
        click.echo(f"  Relevance: {metrics.answer_relevance:.2f}")
        click.echo(f"  Correctness: {metrics.answer_correctness:.2f}")
        click.echo(f"  Completeness: {metrics.answer_completeness:.2f}")
        
        click.echo(f"\nCitation Quality:")
        click.echo(f"  Accuracy: {metrics.citation_accuracy:.2f}")
        click.echo(f"  Coverage: {metrics.citation_coverage:.2f}")
        
        click.echo(f"\n✅ Strengths:")
        for s in metrics.strengths:
            click.echo(f"  • {s}")
        
        click.echo(f"\n⚠️  Weaknesses:")
        for w in metrics.weaknesses:
            click.echo(f"  • {w}")
        
        click.echo(f"\n💡 Suggestions:")
        for s in metrics.suggestions:
            click.echo(f"  • {s}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@evaluate.command()
@click.argument('test_cases_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file for report')
def batch(test_cases_file: str, output: Optional[str]):
    """
    Evaluate multiple test cases from a JSON file.
    
    JSON format:
    [
        {
            "query": "...",
            "answer": "...",
            "contexts": [...],
            "citations": [...],
            "ground_truth": "..." (optional)
        }
    ]
    """
    try:
        logger.info(f"Loading test cases from: {test_cases_file}")
        
        # Load test cases
        with open(test_cases_file, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
        
        click.echo(f"Loaded {len(test_cases)} test cases")
        
        # Initialize evaluator
        from rag_evaluator.gemini_rag_evaluator import GeminiRAGEvaluator
        evaluator = GeminiRAGEvaluator()
        
        # Run batch evaluation
        click.echo("Evaluating...")
        results = evaluator.evaluate_batch(test_cases)
        
        # Generate report
        output_file = output or f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report = evaluator.generate_report(results, output_path=output_file)
        
        # Print summary
        click.echo("\n" + "="*80)
        click.echo("EVALUATION SUMMARY")
        click.echo("="*80 + "\n")
        
        summary = report['summary']
        click.echo(f"Total Evaluations: {summary['total_evaluations']}")
        click.echo(f"\nAverage Scores:")
        for metric, score in summary['average_scores'].items():
            click.echo(f"  {metric}: {score:.2f}")
        
        click.echo(f"\nScore Distribution:")
        for category, count in summary['score_distribution'].items():
            click.echo(f"  {category}: {count}")
        
        click.echo(f"\n✅ Report saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Batch evaluation failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def server():
    """Start the API server"""
    try:
        api_host = os.getenv("API_HOST", "0.0.0.0")
        api_port = int(os.getenv("API_PORT", "8000"))
        
        click.echo("🚀 Starting Berlin Media Archive API server...")
        click.echo(f"📍 Server: http://{api_host}:{api_port}")
        click.echo(f"📚 API Docs: http://{api_host}:{api_port}/docs")
        click.echo("\nPress Ctrl+C to stop\n")
        
        import uvicorn
        uvicorn.run(
            "main:app",
            host=api_host,
            port=api_port,
            reload=True,
            log_level=os.getenv("LOG_LEVEL", "INFO").lower()
        )
        
    except KeyboardInterrupt:
        click.echo("\n\n👋 Server stopped")
    except Exception as e:
        logger.error(f"Server failed: {e}", exc_info=True)
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def demo():
    """Run interactive demo"""
    try:
        click.echo("🎬 Starting interactive demo...\n")
        
        from demo import BerlinArchiveDemo
        
        demo_instance = BerlinArchiveDemo()
        demo_instance.run_full_demo()
        
    except KeyboardInterrupt:
        click.echo("\n\n👋 Demo stopped")
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def status():
    """Check system status and configuration"""
    try:
        click.echo("🏛️  Berlin Media Archive - System Status")
        click.echo("=" * 60)
        
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        click.echo(f"\n📁 Directories:")
        click.echo(f"   - Data: {os.getenv('DATA_DIR', './data')}")
        click.echo(f"   - Audio: {os.getenv('AUDIO_DIR', './data/audio')}")
        click.echo(f"   - Documents: {os.getenv('DOCUMENTS_DIR', './data/documents')}")
        click.echo(f"   - Output: {os.getenv('OUTPUT_DIR', './output')}")
        click.echo(f"   - Vector Store: {os.getenv('VECTORSTORE_PATH', './data/vectorstore')}")
        
        click.echo(f"\n⚙️  Configuration:")
        click.echo(f"   - LLM Model: {os.getenv('LLM_MODEL', 'gpt-4-turbo-preview')}")
        click.echo(f"   - Embedding Model: {os.getenv('EMBEDDING_MODEL', 'text-embedding-3-large')}")
        click.echo(f"   - Whisper Model: {os.getenv('WHISPER_MODEL', 'base')}")
        click.echo(f"   - Vector Store: {os.getenv('VECTORSTORE_TYPE', 'chroma')}")
        
        enable_hybrid = os.getenv('ENABLE_HYBRID_SEARCH', 'true').lower() == 'true'
        enable_diarization = os.getenv('ENABLE_SPEAKER_DIARIZATION', 'true').lower() == 'true'
        enable_evaluation = os.getenv('ENABLE_EVALUATION', 'true').lower() == 'true'
        
        click.echo(f"   - Hybrid Search: {'✅ Enabled' if enable_hybrid else '❌ Disabled'}")
        click.echo(f"   - Speaker Diarization: {'✅ Enabled' if enable_diarization else '❌ Disabled'}")
        click.echo(f"   - Evaluation: {'✅ Enabled' if enable_evaluation else '❌ Disabled'}")
        
        # Check API keys
        openai_key = os.getenv("OPENAI_API_KEY")
        hf_key = os.getenv("HUGGINGFACE_API_KEY")
        
        click.echo(f"\n🔑 API Keys:")
        click.echo(f"   - OpenAI: {'✅ Set' if openai_key else '❌ Not set'}")
        click.echo(f"   - HuggingFace: {'✅ Set' if hf_key else '❌ Not set'}")
        
        # Check directory existence
        click.echo(f"\n📂 Directory Status:")
        dirs_to_check = [
            os.getenv('DATA_DIR', './data'),
            os.getenv('AUDIO_DIR', './data/audio'),
            os.getenv('DOCUMENTS_DIR', './data/documents'),
            os.getenv('OUTPUT_DIR', './output'),
            os.getenv('VECTORSTORE_PATH', './data/vectorstore'),
        ]
        
        for dir_path in dirs_to_check:
            exists = Path(dir_path).exists()
            status_icon = "✅" if exists else "❌"
            click.echo(f"   {status_icon} {dir_path}")
        
        # Overall status
        all_keys_set = openai_key is not None
        all_dirs_exist = all(Path(d).exists() for d in dirs_to_check)
        
        click.echo()
        if all_keys_set and all_dirs_exist:
            click.echo("✅ System ready!")
        else:
            click.echo("⚠️  System has missing requirements")
            if not all_keys_set:
                click.echo("   - Set required API keys in .env file")
            if not all_dirs_exist:
                click.echo("   - Run 'ensure_directories()' to create missing directories")
        
    except Exception as e:
        logger.error(f"Status check failed: {e}", exc_info=True)
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def init():
    """Initialize the Berlin Media Archive (create directories, check dependencies)"""
    try:
        click.echo("🚀 Initializing Berlin Media Archive...")
        
        # Ensure directories
        click.echo("\n📁 Creating directories...")
        ensure_directories()
        click.echo("   ✅ Directories created")
        
        # Check .env file
        click.echo("\n🔧 Checking configuration...")
        if not Path(".env").exists():
            click.echo("   ⚠️  .env file not found")
            click.echo("   Creating .env from template...")
            
            env_template = """# API Keys
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# LLM Configuration
LLM_MODEL=gpt-4-turbo-preview
LLM_TEMPERATURE=0.1
EMBEDDING_MODEL=text-embedding-3-large

# Audio Processing
WHISPER_MODEL=base
AUDIO_CHUNK_LENGTH=30
ENABLE_SPEAKER_DIARIZATION=true

# Document Processing
PDF_CHUNK_SIZE=1000
PDF_CHUNK_OVERLAP=200

# Vector Store
VECTORSTORE_TYPE=chroma
VECTORSTORE_PATH=./data/vectorstore
COLLECTION_NAME=berlin_archive

# Search Configuration
ENABLE_HYBRID_SEARCH=true
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.7
BM25_WEIGHT=0.3

# Evaluation
ENABLE_EVALUATION=true
EVALUATOR_MODEL=gpt-4-turbo-preview

# File Paths
DATA_DIR=./data
AUDIO_DIR=./data/audio
DOCUMENTS_DIR=./data/documents
OUTPUT_DIR=./output
LOGS_DIR=./logs

# API Server
API_HOST=0.0.0.0
API_PORT=8000

# Logging
LOG_LEVEL=INFO
"""
            with open(".env", "w") as f:
                f.write(env_template)
            
            click.echo("   ✅ .env file created")
            click.echo("   ⚠️  Please edit .env and add your API keys")
        else:
            click.echo("   ✅ .env file found")
        
        click.echo("\n✅ Initialization complete!")
        click.echo("\nNext steps:")
        click.echo("   1. Edit .env and add your API keys")
        click.echo("   2. Run 'python cli.py status' to check system status")
        click.echo("   3. Run 'python cli.py server' to start the API server")
        
    except Exception as e:
        logger.error(f"Initialization failed: {e}", exc_info=True)
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()