"""
Berlin Media Archive - Command Line Interface
Production-grade CLI tool for all operations.
"""

import sys
import json
import click
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.config import settings, ensure_directories
from utils.logger import setup_logging
from loguru import logger


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
        
        output_dir = output_dir or str(Path(settings.output_dir) / "audio")
        
        segments, metadata = pipeline.ingest_audio(audio_file, output_dir)
        
        click.echo(f"\n✅ Audio ingestion complete!")
        click.echo(f"   - Duration: {metadata.get('duration_seconds', 'N/A'):.2f}s")
        click.echo(f"   - Segments: {len(segments)}")
        
        speakers = set(seg.speaker for seg in segments if seg.speaker)
        if speakers:
            click.echo(f"   - Speakers: {', '.join(speakers)}")
        
        click.echo(f"\n💾 Output saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Audio ingestion failed: {e}", exc_info=True)
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@ingest.command()
@click.argument('document_file', type=click.Path(exists=True))
@click.option('--output-dir', '-o', type=click.Path(), help='Output directory')
@click.option('--chunk-size', '-c', type=int, default=1000, help='Chunk size')
@click.option('--chunk-overlap', type=int, default=200, help='Chunk overlap')
def document(document_file, output_dir, chunk_size, chunk_overlap):
    """Ingest a PDF document with intelligent chunking"""
    try:
        click.echo(f"📄 Ingesting document: {document_file}")
        
        from ingestion.document_ingestion import DocumentIngestionPipeline
        
        pipeline = DocumentIngestionPipeline(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        output_dir = output_dir or str(Path(settings.output_dir) / "documents")
        
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
@click.option('--top-k', '-k', type=int, default=5, help='Number of results')
@click.option('--filter-source', '-f', type=str, help='Filter by source type (audio/document)')
@click.option('--filter-speaker', '-s', type=str, help='Filter by speaker')
def query(question, top_k, filter_source, filter_speaker):
    """Query the archive with natural language"""
    try:
        click.echo(f"🔍 Querying: {question}")
        click.echo(f"   Top-K: {top_k}")
        
        from rag.attribution_engine import AttributionEngine
        
        engine = AttributionEngine()
        
        filters = {}
        if filter_source:
            filters['source_type'] = filter_source
        if filter_speaker:
            filters['speaker'] = filter_speaker
        
        result = engine.query(question, top_k=top_k, filters=filters)
        
        click.echo(f"\n💬 Answer:\n{result['answer']}")
        
        click.echo(f"\n📚 Sources ({len(result['sources'])}):")
        for i, source in enumerate(result['sources'], 1):
            click.echo(f"   {i}. {source['citation']}")
        
        if result.get('confidence'):
            click.echo(f"\n📊 Confidence: {result['confidence']:.2f}")
        
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
        
        evaluator = LLMRAGEvaluator()
        
        # Load test cases
        if test_cases:
            with open(test_cases, 'r') as f:
                cases = json.load(f)
        else:
            click.echo("ℹ️  Using default test cases")
            cases = EvaluationTestCases.get_sample_test_cases()
        
        click.echo(f"   Test cases: {len(cases)}")
        
        # Run evaluation
        output_path = output or str(Path(settings.output_dir) / "evaluation_results.json")
        
        batch_result = evaluator.batch_evaluate(
            cases,
            save_results=True,
            output_path=output_path
        )
        
        # Display results
        click.echo(f"\n✅ Evaluation complete!")
        click.echo(f"\n📊 Results:")
        click.echo(f"   - Average Faithfulness: {batch_result.avg_faithfulness:.2f}")
        click.echo(f"   - Average Relevance: {batch_result.avg_relevance:.2f}")
        click.echo(f"   - Average Citation Quality: {batch_result.avg_citation_quality:.2f}")
        click.echo(f"   - Overall Score: {batch_result.avg_overall:.2f}")
        click.echo(f"\n   - Grade Distribution: {batch_result.grade_distribution}")
        click.echo(f"   - Production Ready: {batch_result.production_ready_count}/{batch_result.total_cases} ({batch_result.production_ready_percentage:.1f}%)")
        
        click.echo(f"\n💾 Detailed results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@evaluate.command()
@click.argument('question')
@click.argument('answer')
@click.option('--context', '-c', multiple=True, required=True, help='Context chunks')
@click.option('--ground-truth', '-g', type=str, help='Ground truth answer')
def single(question, answer, context, ground_truth):
    """Evaluate a single question-answer pair"""
    try:
        click.echo("📈 Evaluating single response...")
        
        from rag_evaluator.llm_rag_evaluator import LLMRAGEvaluator
        
        evaluator = LLMRAGEvaluator()
        
        result = evaluator.evaluate_response(
            question=question,
            answer=answer,
            retrieved_context=list(context),
            ground_truth=ground_truth
        )
        
        click.echo(f"\n✅ Evaluation complete!")
        click.echo(f"\n📊 Scores:")
        click.echo(f"   - Faithfulness: {result.faithfulness:.2f}")
        click.echo(f"   - Relevance: {result.relevance:.2f}")
        click.echo(f"   - Citation Quality: {result.citation_quality:.2f}")
        click.echo(f"   - Overall: {result.overall_score:.2f}")
        click.echo(f"   - Grade: {result.grade}")
        
        production_ready = "✅ YES" if result.is_production_ready() else "❌ NO"
        click.echo(f"\n   Production Ready: {production_ready}")
        
        if result.errors:
            click.echo(f"\n⚠️  Errors:")
            for error in result.errors:
                click.echo(f"   - {error}")
        
        if result.warnings:
            click.echo(f"\n⚠️  Warnings:")
            for warning in result.warnings:
                click.echo(f"   - {warning}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def server():
    """Start the API server"""
    try:
        click.echo("🚀 Starting Berlin Media Archive API server...")
        click.echo(f"📍 Server: http://{settings.api_host}:{settings.api_port}")
        click.echo(f"📚 API Docs: http://{settings.api_host}:{settings.api_port}/docs")
        click.echo("\nPress Ctrl+C to stop\n")
        
        import uvicorn
        uvicorn.run(
            "main:app",
            host=settings.api_host,
            port=settings.api_port,
            reload=True,
            log_level="info"
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
        
        click.echo(f"\n📁 Directories:")
        click.echo(f"   - Data: {settings.data_dir}")
        click.echo(f"   - Audio: {settings.audio_dir}")
        click.echo(f"   - Documents: {settings.documents_dir}")
        click.echo(f"   - Output: {settings.output_dir}")
        click.echo(f"   - Vector Store: {settings.vectorstore_path}")
        
        click.echo(f"\n⚙️  Configuration:")
        click.echo(f"   - LLM Model: {settings.llm_model}")
        click.echo(f"   - Embedding Model: {settings.embedding_model}")
        click.echo(f"   - Whisper Model: {settings.whisper_model}")
        click.echo(f"   - Vector Store: {settings.vectorstore_type}")
        click.echo(f"   - Hybrid Search: {'Enabled' if settings.enable_hybrid_search else 'Disabled'}")
        click.echo(f"   - Speaker Diarization: {'Enabled' if settings.enable_speaker_diarization else 'Disabled'}")
        
        # Check API key
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        api_key_status = "✅ Set" if api_key else "❌ Not set"
        click.echo(f"\n🔑 API Keys:")
        click.echo(f"   - OpenAI: {api_key_status}")
        
        click.echo("\n✅ System ready!")
        
    except Exception as e:
        logger.error(f"Status check failed: {e}", exc_info=True)
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()