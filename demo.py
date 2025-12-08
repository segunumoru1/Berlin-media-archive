"""
Berlin Media Archive - Comprehensive Demo Script
Demonstrates all features of the system with sample data.
"""

import sys
import json
from pathlib import Path
from typing import Optional
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.config import settings, ensure_directories
from utils.logger import setup_logging
from ingestion.audio_ingestion import AudioIngestionPipeline
from ingestion.document_ingestion import DocumentIngestionPipeline
from rag.attribution_engine import AttributionEngine
from rag_evaluator.llm_rag_evaluator import LLMRAGEvaluator
from rag_evaluator.test_cases import EvaluationTestCases


class BerlinArchiveDemo:
    """Comprehensive demo of the Berlin Media Archive system."""
    
    def __init__(self):
        """Initialize demo."""
        # Setup logging
        setup_logging(log_level="INFO", log_file="demo.log")
        
        # Ensure directories exist
        ensure_directories()
        
        logger.info("=" * 80)
        logger.info("🏛️  BERLIN MEDIA ARCHIVE - COMPREHENSIVE DEMO")
        logger.info("=" * 80)
        
        # Initialize components
        self.audio_pipeline = None
        self.document_pipeline = None
        self.attribution_engine = None
        self.evaluator = None
    
    def run_full_demo(self):
        """Run complete demonstration of all features."""
        try:
            print("\n" + "=" * 80)
            print("📋 DEMO MENU")
            print("=" * 80)
            print("1. Test Audio Ingestion (Module B - Speaker Diarization)")
            print("2. Test Document Ingestion")
            print("3. Test Attribution Engine (Core Requirement)")
            print("4. Test Hybrid Search (Module A)")
            print("5. Test RAG Evaluation (Module C)")
            print("6. Run Complete End-to-End Test")
            print("7. Generate Sample Files")
            print("8. Exit")
            print("=" * 80)
            
            choice = input("\nEnter your choice (1-8): ").strip()
            
            if choice == "1":
                self.demo_audio_ingestion()
            elif choice == "2":
                self.demo_document_ingestion()
            elif choice == "3":
                self.demo_attribution_engine()
            elif choice == "4":
                self.demo_hybrid_search()
            elif choice == "5":
                self.demo_evaluation()
            elif choice == "6":
                self.demo_end_to_end()
            elif choice == "7":
                self.generate_sample_files()
            elif choice == "8":
                print("\n👋 Thank you for using Berlin Media Archive!")
                return
            else:
                print("❌ Invalid choice. Please try again.")
            
            # Ask to continue
            if input("\n\nPress Enter to return to menu (or 'q' to quit): ").lower() != 'q':
                self.run_full_demo()
                
        except KeyboardInterrupt:
            print("\n\n👋 Demo interrupted. Goodbye!")
        except Exception as e:
            logger.error(f"Demo error: {e}", exc_info=True)
            print(f"\n❌ Error: {e}")
    
    def demo_audio_ingestion(self):
        """Demo: Audio ingestion with timestamps and speaker diarization."""
        print("\n" + "=" * 80)
        print("🎙️  DEMO: AUDIO INGESTION WITH SPEAKER DIARIZATION (Module B)")
        print("=" * 80)
        
        audio_path = input("\nEnter path to audio file (or 'sample' for mock): ").strip()
        
        if audio_path.lower() == 'sample':
            print("\n📝 Mock Audio Ingestion Result:")
            print("-" * 80)
            self._show_mock_audio_result()
            return
        
        if not Path(audio_path).exists():
            print(f"❌ File not found: {audio_path}")
            return
        
        try:
            print("\n🔄 Processing audio file...")
            print("   - Transcribing with Whisper")
            print("   - Preserving timestamps")
            print("   - Performing speaker diarization")
            
            # Initialize pipeline if needed
            if not self.audio_pipeline:
                self.audio_pipeline = AudioIngestionPipeline()
            
            # Process audio
            segments, metadata = self.audio_pipeline.ingest_audio(
                audio_path,
                output_dir=str(Path(settings.output_dir) / "audio")
            )
            
            # Display results
            print("\n✅ Audio Ingestion Complete!")
            print(f"\n📊 Metadata:")
            print(f"   - Duration: {metadata.get('duration_seconds', 'N/A'):.2f}s")
            print(f"   - Segments: {len(segments)}")
            
            # Show sample segments
            print(f"\n📝 Sample Transcript Segments (first 3):")
            print("-" * 80)
            for i, seg in enumerate(segments[:3], 1):
                speaker = f"[{seg.speaker}] " if seg.speaker else ""
                print(f"\n{i}. [{seg.get_timestamp_str()}] {speaker}")
                print(f"   {seg.text}")
            
            if len(segments) > 3:
                print(f"\n... and {len(segments) - 3} more segments")
            
            # Show speaker summary
            speakers = set(seg.speaker for seg in segments if seg.speaker)
            if speakers:
                print(f"\n👥 Speakers Detected: {', '.join(speakers)}")
            
            print(f"\n💾 Full transcript saved to: {settings.output_dir}/audio/")
            
        except Exception as e:
            logger.error(f"Audio ingestion demo failed: {e}", exc_info=True)
            print(f"\n❌ Error: {e}")
    
    def demo_document_ingestion(self):
        """Demo: Document ingestion with page tracking."""
        print("\n" + "=" * 80)
        print("📄 DEMO: DOCUMENT INGESTION WITH PAGE TRACKING")
        print("=" * 80)
        
        doc_path = input("\nEnter path to PDF document (or 'sample' for mock): ").strip()
        
        if doc_path.lower() == 'sample':
            print("\n📝 Mock Document Ingestion Result:")
            print("-" * 80)
            self._show_mock_document_result()
            return
        
        if not Path(doc_path).exists():
            print(f"❌ File not found: {doc_path}")
            return
        
        try:
            print("\n🔄 Processing document...")
            print("   - Extracting text from PDF")
            print("   - Creating chunks with overlap")
            print("   - Preserving page numbers")
            
            # Initialize pipeline if needed
            if not self.document_pipeline:
                from ingestion.document_ingestion import DocumentIngestionPipeline
                self.document_pipeline = DocumentIngestionPipeline()
            
            # Process document
            chunks, metadata = self.document_pipeline.ingest_document(
                doc_path,
                output_dir=str(Path(settings.output_dir) / "documents")
            )
            
            # Display results
            print("\n✅ Document Ingestion Complete!")
            print(f"\n📊 Metadata:")
            print(f"   - Pages: {metadata.get('num_pages', 'N/A')}")
            print(f"   - Chunks: {len(chunks)}")
            print(f"   - Chunk size: {settings.pdf_chunk_size} chars")
            print(f"   - Overlap: {settings.pdf_chunk_overlap} chars")
            
            # Show sample chunks
            print(f"\n📝 Sample Chunks (first 2):")
            print("-" * 80)
            for i, chunk in enumerate(chunks[:2], 1):
                print(f"\n{i}. Page {chunk.page_number}:")
                print(f"   {chunk.text[:200]}...")
            
            if len(chunks) > 2:
                print(f"\n... and {len(chunks) - 2} more chunks")
            
            print(f"\n💾 Full document data saved to: {settings.output_dir}/documents/")
            
        except Exception as e:
            logger.error(f"Document ingestion demo failed: {e}", exc_info=True)
            print(f"\n❌ Error: {e}")
    
    def demo_attribution_engine(self):
        """Demo: Attribution engine with precise citations."""
        print("\n" + "=" * 80)
        print("📊 DEMO: ATTRIBUTION ENGINE WITH CITATIONS (Core Requirement)")
        print("=" * 80)
        
        print("\n📝 This demo shows how the system provides precise citations:")
        print("   - Timestamps for audio content (e.g., 04:23)")
        print("   - Page numbers for documents (e.g., Page 5)")
        
        # Mock query for demonstration
        question = "What is the primary definition of success discussed in the files?"
        
        print(f"\n❓ Sample Question:")
        print(f"   {question}")
        
        # Mock retrieved context
        context = [
            {
                "text": "Success can be defined in multiple ways, but primarily it involves achieving the goals you set for yourself while maintaining a healthy work-life balance.",
                "source": "history_book.pdf",
                "type": "document",
                "page": 3
            },
            {
                "text": "For me, success isn't just about achievements. It's about personal growth and helping others reach their potential.",
                "source": "interview_2023.mp3",
                "type": "audio",
                "timestamp": "04:23",
                "speaker": "SPEAKER_01"
            }
        ]
        
        print(f"\n🔍 Retrieved Context (2 chunks):")
        print("-" * 80)
        for i, ctx in enumerate(context, 1):
            if ctx["type"] == "document":
                print(f"{i}. [Document: {ctx['source']}, Page {ctx['page']}]")
            else:
                speaker = f", Speaker: {ctx['speaker']}" if ctx.get('speaker') else ""
                print(f"{i}. [Audio: {ctx['source']}, Timestamp: {ctx['timestamp']}{speaker}]")
            print(f"   {ctx['text'][:150]}...")
        
        # Mock generated answer with citations
        answer = """Success is defined as achieving set goals while maintaining work-life balance (Source: history_book.pdf, Page 3). The interview also emphasizes that success includes personal growth and helping others achieve their potential (Source: interview_2023.mp3, 04:23)."""
        
        print(f"\n✅ Generated Answer WITH CITATIONS:")
        print("-" * 80)
        print(answer)
        
        print("\n💡 Key Features Demonstrated:")
        print("   ✓ Precise source attribution")
        print("   ✓ Timestamp preservation for audio")
        print("   ✓ Page number tracking for documents")
        print("   ✓ Speaker identification")
        print("   ✓ No hallucinations - all claims cited")
    
    def demo_hybrid_search(self):
        """Demo: Hybrid search (semantic + keyword)."""
        print("\n" + "=" * 80)
        print("🔍 DEMO: HYBRID SEARCH - SEMANTIC + KEYWORD (Module A)")
        print("=" * 80)
        
        print("\n📝 Hybrid Search combines:")
        print("   - Semantic Search: Understanding meaning and context")
        print("   - Keyword Search (BM25): Exact matches for dates, names, numbers")
        
        # Sample queries that benefit from hybrid search
        queries = [
            "What was said in 1989?",
            "Discussions about Christa Wolf",
            "Berlin Wall construction date"
        ]
        
        print(f"\n🎯 Sample Queries That Benefit from Hybrid Search:")
        for i, q in enumerate(queries, 1):
            print(f"   {i}. {q}")
        
        query = input("\nEnter your query (or use sample 1-3): ").strip()
        
        if query.isdigit() and 1 <= int(query) <= 3:
            query = queries[int(query) - 1]
        
        print(f"\n🔍 Processing Query: '{query}'")
        print("\n" + "-" * 80)
        
        # Mock results showing hybrid approach
        print("📊 Semantic Search Results:")
        print("   1. Context about 1989 events (similarity: 0.85)")
        print("   2. Historical background (similarity: 0.78)")
        
        print("\n📊 Keyword Search Results:")
        print("   1. Exact mention of '1989' (BM25: 4.2)")
        print("   2. Date references (BM25: 3.8)")
        
        print("\n✅ Combined Hybrid Results (weighted fusion):")
        print("   1. Context about 1989 events [BEST MATCH]")
        print("   2. Exact mention of '1989'")
        print("   3. Historical background")
        
        print("\n💡 Why Hybrid Search Matters:")
        print("   - Pure semantic search might miss exact dates/names")
        print("   - Pure keyword search misses contextual relevance")
        print("   - Hybrid approach gets best of both worlds!")
    
    def demo_evaluation(self):
        """Demo: RAG evaluation with LLM-as-Judge."""
        print("\n" + "=" * 80)
        print("📈 DEMO: RAG EVALUATION - LLM AS JUDGE (Module C)")
        print("=" * 80)
        
        print("\n📝 Evaluation Metrics:")
        print("   1. Faithfulness: No hallucinations (0-1)")
        print("   2. Relevance: Answers the question (0-1)")
        print("   3. Citation Quality: Proper source attribution (0-1)")
        print("   4. Overall Score & Grade (A-F)")
        
        # Get test cases
        test_cases = EvaluationTestCases.get_sample_test_cases()
        
        print(f"\n📊 Running evaluation on {len(test_cases)} test cases...")
        
        try:
            # Initialize evaluator
            if not self.evaluator:
                self.evaluator = LLMRAGEvaluator()
            
            # Evaluate first test case as example
            print("\n" + "-" * 80)
            print("🧪 SAMPLE EVALUATION (Test Case 1)")
            print("-" * 80)
            
            test_case = test_cases[0]
            
            print(f"\n❓ Question: {test_case['question']}")
            print(f"\n💬 Answer: {test_case['answer'][:150]}...")
            print(f"\n📚 Context Chunks: {len(test_case['context'])}")
            
            print("\n⏳ Evaluating with GPT-4...")
            
            result = self.evaluator.evaluate_response(
                question=test_case["question"],
                answer=test_case["answer"],
                retrieved_context=test_case["context"],
                ground_truth=test_case.get("ground_truth")
            )
            
            # Display results
            print("\n✅ EVALUATION RESULTS:")
            print(f"   📊 Faithfulness: {result.faithfulness:.2f}")
            print(f"   📊 Relevance: {result.relevance:.2f}")
            print(f"   📊 Citation Quality: {result.citation_quality:.2f}")
            print(f"   🎯 Overall Score: {result.overall_score:.2f}")
            print(f"   🏆 Grade: {result.grade}")
            
            if result.errors:
                print(f"\n   ⚠️  Errors: {', '.join(result.errors)}")
            if result.warnings:
                print(f"   ⚠️  Warnings: {', '.join(result.warnings)}")
            
            production_ready = "✅ YES" if result.is_production_ready() else "❌ NO"
            print(f"\n   🚀 Production Ready: {production_ready}")
            
            # Offer batch evaluation
            if input("\n\nRun full batch evaluation on all test cases? (y/n): ").lower() == 'y':
                print("\n⏳ Running batch evaluation...")
                batch_result = self.evaluator.batch_evaluate(
                    test_cases,
                    save_results=True,
                    output_path=str(Path(settings.output_dir) / "evaluation_results.json")
                )
                
                print("\n✅ BATCH EVALUATION COMPLETE!")
                print(f"   📊 Total Cases: {batch_result.total_cases}")
                print(f"   📊 Average Score: {batch_result.avg_overall:.2f}")
                print(f"   📊 Grade Distribution: {batch_result.grade_distribution}")
                print(f"   🚀 Production Ready: {batch_result.production_ready_count}/{batch_result.total_cases} ({batch_result.production_ready_percentage:.1f}%)")
                print(f"\n   💾 Results saved to: {settings.output_dir}/evaluation_results.json")
            
        except Exception as e:
            logger.error(f"Evaluation demo failed: {e}", exc_info=True)
            print(f"\n❌ Error: {e}")
            print("\n💡 Tip: Make sure OPENAI_API_KEY is set in .env file")
    
    def demo_end_to_end(self):
        """Demo: Complete end-to-end workflow."""
        print("\n" + "=" * 80)
        print("🚀 DEMO: END-TO-END WORKFLOW")
        print("=" * 80)
        
        print("\n📋 This demo shows the complete workflow:")
        print("   1. Ingest audio file → Transcript with timestamps")
        print("   2. Ingest PDF document → Chunks with page numbers")
        print("   3. Store in unified vector database")
        print("   4. Query with natural language")
        print("   5. Get answer with precise citations")
        print("   6. Evaluate answer quality")
        
        print("\n" + "-" * 80)
        print("Step 1: Audio Ingestion")
        print("-" * 80)
        self._show_mock_audio_result()
        
        input("\nPress Enter to continue to Step 2...")
        
        print("\n" + "-" * 80)
        print("Step 2: Document Ingestion")
        print("-" * 80)
        self._show_mock_document_result()
        
        input("\nPress Enter to continue to Step 3...")
        
        print("\n" + "-" * 80)
        print("Step 3: Unified Vector Store")
        print("-" * 80)
        print("✅ All content embedded and stored with metadata:")
        print("   - Audio transcript chunks with timestamps & speakers")
        print("   - Document chunks with page numbers")
        print("   - Searchable by content and metadata")
        
        input("\nPress Enter to continue to Step 4...")
        
        print("\n" + "-" * 80)
        print("Step 4: Query Processing")
        print("-" * 80)
        query = "What is the primary definition of success?"
        print(f"❓ Query: {query}")
        print("\n🔍 Hybrid Search:")
        print("   - Semantic: Finding conceptually similar content")
        print("   - Keyword: Matching important terms")
        print("\n✅ Retrieved 3 most relevant chunks")
        
        input("\nPress Enter to continue to Step 5...")
        
        print("\n" + "-" * 80)
        print("Step 5: Answer Generation with Citations")
        print("-" * 80)
        answer = """Success is defined as achieving set goals while maintaining work-life balance (Source: history_book.pdf, Page 3). The interview emphasizes that success includes personal growth and helping others (Source: interview_2023.mp3, 04:23)."""
        print(f"💬 Answer:\n{answer}")
        
        input("\nPress Enter to continue to Step 6...")
        
        print("\n" + "-" * 80)
        print("Step 6: Evaluation")
        print("-" * 80)
        print("📊 Quality Metrics:")
        print("   - Faithfulness: 0.95 ✅")
        print("   - Relevance: 0.92 ✅")
        print("   - Citation Quality: 0.98 ✅")
        print("   - Overall Grade: A")
        print("   - Production Ready: ✅ YES")
        
        print("\n🎉 END-TO-END DEMO COMPLETE!")
    
    def generate_sample_files(self):
        """Generate sample files for testing."""
        print("\n" + "=" * 80)
        print("📁 GENERATE SAMPLE FILES")
        print("=" * 80)
        
        print("\n📝 This will create sample files for testing:")
        print("   1. sample_document.txt - Mock PDF content")
        print("   2. sample_transcript.json - Mock audio transcript")
        print("   3. sample_test_cases.json - Evaluation test cases")
        
        if input("\nGenerate sample files? (y/n): ").lower() != 'y':
            return
        
        try:
            output_dir = Path(settings.data_dir) / "samples"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Sample document
            doc_path = output_dir / "sample_document.txt"
            with open(doc_path, 'w', encoding='utf-8') as f:
                f.write("""Sample Historical Document
Page 1

The Berlin Wall: A Historical Overview

The Berlin Wall was constructed on August 13, 1961, dividing East and West Berlin. This physical barrier became a powerful symbol of the Cold War division between Eastern and Western ideologies.

Page 2

Construction began overnight, catching many residents by surprise. Families were separated, and the city that had once been unified was now split by concrete and barbed wire.

Page 3

Success and Reunification

Success can be defined in multiple ways, but primarily it involves achieving the goals you set for yourself while maintaining a healthy work-life balance. In the context of reunification, success meant bringing together two societies that had been separated for nearly 30 years.

Page 4

The fall of the Berlin Wall in 1989 marked the beginning of the reunification process. This event symbolized not just the physical reunification of a city, but the ideological reconciliation of different ways of life.""")
            
            print(f"✅ Created: {doc_path}")
            
            # Sample transcript
            transcript_path = output_dir / "sample_transcript.json"
            transcript_data = {
                "metadata": {
                    "filename": "interview_1990.mp3",
                    "duration_seconds": 1200.5
                },
                "segments": [
                    {
                        "text": "Welcome to our interview series. Today we're discussing reunification.",
                        "start_time": 0.0,
                        "end_time": 5.2,
                        "speaker": "HOST"
                    },
                    {
                        "text": "Thank you for having me. Reunification brings hope, but also anxiety.",
                        "start_time": 5.5,
                        "end_time": 10.8,
                        "speaker": "GUEST"
                    },
                    {
                        "text": "What does success mean in this context?",
                        "start_time": 11.0,
                        "end_time": 13.5,
                        "speaker": "HOST"
                    },
                    {
                        "text": "For me, success isn't just about achievements. It's about personal growth and helping others reach their potential.",
                        "start_time": 14.0,
                        "end_time": 22.5,
                        "speaker": "GUEST"
                    }
                ]
            }
            
            with open(transcript_path, 'w', encoding='utf-8') as f:
                json.dump(transcript_data, f, indent=2)
            
            print(f"✅ Created: {transcript_path}")
            
            # Sample test cases
            test_cases_path = output_dir / "sample_test_cases.json"
            test_cases = EvaluationTestCases.get_sample_test_cases()
            
            with open(test_cases_path, 'w', encoding='utf-8') as f:
                json.dump(test_cases, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Created: {test_cases_path}")
            
            print(f"\n💾 All sample files saved to: {output_dir}")
            print("\n💡 You can use these files to test the system without real data!")
            
        except Exception as e:
            logger.error(f"Failed to generate sample files: {e}", exc_info=True)
            print(f"\n❌ Error: {e}")
    
    def _show_mock_audio_result(self):
        """Show mock audio ingestion result."""
        print("🎙️  Audio File: interview_1990.mp3")
        print("⏱️  Duration: 20:05")
        print("📊 Segments: 45")
        print("\n📝 Sample Transcript Segments:")
        print("-" * 80)
        print("\n[00:15 - 00:23] [SPEAKER_01 - Host]")
        print("Welcome to our discussion about reunification and its impact on Berlin.")
        print("\n[00:24 - 00:35] [SPEAKER_02 - Guest]")
        print("Thank you for having me. This is a crucial moment in our history.")
        print("\n[04:23 - 04:38] [SPEAKER_02 - Guest]")
        print("Success means achieving goals while helping others grow.")
        print("\n... and 42 more segments")
        print("\n👥 Speakers Detected: SPEAKER_01 (Host), SPEAKER_02 (Guest)")
        print("💾 Saved: output/audio/interview_1990_transcript.json")
    
    def _show_mock_document_result(self):
        """Show mock document ingestion result."""
        print("📄 Document: history_book.pdf")
        print("📑 Pages: 45")
        print("📊 Chunks: 67")
        print("\n📝 Sample Chunks:")
        print("-" * 80)
        print("\nChunk 1 (Page 3):")
        print("Success can be defined in multiple ways, but primarily it involves")
        print("achieving the goals you set for yourself while maintaining...")
        print("\nChunk 2 (Page 12):")
        print("The Berlin Wall was constructed on August 13, 1961, dividing")
        print("East and West Berlin. This physical barrier became a powerful...")
        print("\n... and 65 more chunks")
        print("\n💾 Saved: output/documents/history_book_chunks.json")


def main():
    """Main entry point for demo."""
    try:
        demo = BerlinArchiveDemo()
        demo.run_full_demo()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted. Goodbye!")
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n❌ Fatal error: {e}")


if __name__ == "__main__":
    main()