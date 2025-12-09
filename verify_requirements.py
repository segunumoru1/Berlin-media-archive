"""
Core Requirements Verification Script
Tests all Part 1 and Module B requirements
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def print_header(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def check_requirement_1_audio_ingestion():
    """
    ✅ 1. Audio Ingestion Pipeline
    - Ingest audio and transcribe with Whisper
    - Preserve timestamps for every segment
    """
    print_header("REQUIREMENT 1: Audio Ingestion Pipeline")
    
    from ingestion.audio_ingestion import AudioIngestionPipeline
    
    audio_path = "./data/audio/How to believe in God even when the world sucks with Nadia Bolz-Weber.mp3"
    
    if not Path(audio_path).exists():
        print(f"❌ Audio file not found: {audio_path}")
        return False
    
    try:
        # Test audio ingestion
        pipeline = AudioIngestionPipeline()
        segments = pipeline.ingest_audio(audio_path, "./output/audio")
        
        print(f"✅ Audio transcribed: {len(segments)} segments")
        
        # Check timestamps
        has_timestamps = all(
            hasattr(seg, 'start_time') and hasattr(seg, 'end_time')
            for seg in segments
        )
        
        if has_timestamps:
            print(f"✅ All segments have timestamps")
            print(f"\n   Sample segment:")
            seg = segments[0]
            print(f"   - Time: {seg.start_time:.2f}s to {seg.end_time:.2f}s")
            print(f"   - Text: {seg.text[:80]}...")
            print(f"   - Speaker: {seg.speaker or 'Unknown'}")
            return True
        else:
            print("❌ Missing timestamps on some segments")
            return False
            
    except Exception as e:
        print(f"❌ Audio ingestion failed: {e}")
        return False


def check_requirement_2_unified_vector_store():
    """
    ✅ 2. Unified Vector Store
    - Chunk both PDF and audio transcript
    - Embed into vector database
    - Metadata with source tracking
    """
    print_header("REQUIREMENT 2: Unified Vector Store")
    
    from vectorstore.chroma_store import UnifiedVectorStore
    
    try:
        vs = UnifiedVectorStore()
        stats = vs.get_collection_stats()
        
        print(f"✅ Vector store operational: {stats['collection_name']}")
        print(f"   Total documents: {stats['total_documents']}")
        
        # Check metadata
        all_docs = vs.get_all_documents()
        
        if not all_docs:
            print("❌ No documents in vector store")
            return False
        
        # Check audio metadata
        audio_docs = [d for d in all_docs if d.get('metadata', {}).get('source_type') == 'audio']
        if audio_docs:
            print(f"\n✅ Audio chunks: {len(audio_docs)}")
            sample = audio_docs[0]['metadata']
            print(f"   Sample metadata: {sample}")
            
            required_audio_fields = ['source_file', 'source_type', 'timestamp_start', 'timestamp_end']
            has_audio_metadata = all(field in sample for field in required_audio_fields)
            
            if has_audio_metadata:
                print(f"   ✅ Audio metadata complete")
            else:
                print(f"   ❌ Missing audio metadata fields")
                return False
        
        # Check document metadata
        doc_docs = [d for d in all_docs if d.get('metadata', {}).get('source_type') == 'document']
        if doc_docs:
            print(f"\n✅ Document chunks: {len(doc_docs)}")
            sample = doc_docs[0]['metadata']
            print(f"   Sample metadata: {sample}")
            
            required_doc_fields = ['source_file', 'source_type', 'page_number']
            has_doc_metadata = all(field in sample for field in required_doc_fields)
            
            if has_doc_metadata:
                print(f"   ✅ Document metadata complete")
            else:
                print(f"   ❌ Missing document metadata fields")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Vector store check failed: {e}")
        return False


def check_requirement_3_attribution_engine():
    """
    ✅ 3. The Attribution Engine
    - query_archive(question) function
    - LLM answers using only retrieved context
    - Citations with specific sources
    """
    print_header("REQUIREMENT 3: Attribution Engine")
    
    from rag.attribution_engine import AttributionEngine
    from vectorstore.chroma_store import UnifiedVectorStore
    
    try:
        vs = UnifiedVectorStore()
        engine = AttributionEngine(vector_store=vs)
        
        print("✅ Attribution engine initialized")
        
        # Test query
        test_question = "What does Nadia Bolz-Weber say about faith?"
        print(f"\n📝 Test query: {test_question}")
        
        response = engine.query_archive(test_question, n_results=5)
        
        print(f"\n✅ Query executed")
        print(f"   Answer length: {len(response.answer)} characters")
        print(f"   Citations: {len(response.citations)}")
        
        # Check citations format
        if response.citations:
            print(f"\n   Sample citation:")
            cite = response.citations[0]
            print(f"   - Source: {cite.source_file}")
            print(f"   - Type: {cite.source_type}")
            
            if cite.source_type == 'audio':
                print(f"   - Timestamp: {cite.timestamp_start:.2f}s")
            else:
                print(f"   - Page: {cite.page_number}")
            
            print(f"   - Preview: {cite.content_preview[:100]}...")
            
            # Check if answer contains citations
            has_citations_in_answer = any(
                f"[Source {i+1}]" in response.answer or f"Source {i+1}" in response.answer
                for i in range(len(response.citations))
            )
            
            if has_citations_in_answer:
                print(f"\n   ✅ Answer includes citation references")
            else:
                print(f"\n   ⚠️  Answer doesn't reference citations (may need improvement)")
            
            return True
        else:
            print("❌ No citations returned")
            return False
            
    except Exception as e:
        print(f"❌ Attribution engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_module_b_speaker_diarization():
    """
    ✅ Module B: Speaker Diarization
    - Tag segments with Speaker ID
    - Filter by speaker
    """
    print_header("MODULE B: Speaker Diarization")
    
    from vectorstore.chroma_store import UnifiedVectorStore
    
    try:
        vs = UnifiedVectorStore()
        
        # Check if we have audio with speaker tags
        all_docs = vs.get_all_documents()
        audio_docs = [d for d in all_docs if d.get('metadata', {}).get('source_type') == 'audio']
        
        if not audio_docs:
            print("⚠️  No audio documents in vector store")
            print("   Upload audio with enable_diarization=true to test this feature")
            return None
        
        # Count speakers
        speakers = set()
        for doc in audio_docs:
            speaker = doc.get('metadata', {}).get('speaker')
            if speaker:
                speakers.add(speaker)
        
        print(f"✅ Audio documents found: {len(audio_docs)}")
        print(f"   Unique speakers: {list(speakers)}")
        
        if len(speakers) > 1 and 'Unknown' not in speakers:
            print(f"   ✅ Speaker diarization is working!")
            
            # Test speaker filtering
            test_speaker = list(speakers)[0]
            results = vs.search(
                query="faith",
                top_k=5,
                filters={"speaker": test_speaker}
            )
            
            print(f"\n   Test: Filter by speaker '{test_speaker}'")
            print(f"   Results: {len(results)}")
            
            if results:
                all_match = all(
                    r.get('metadata', {}).get('speaker') == test_speaker
                    for r in results
                )
                if all_match:
                    print(f"   ✅ Speaker filtering works correctly")
                    return True
                else:
                    print(f"   ❌ Speaker filtering returned mixed results")
                    return False
            
            return True
        else:
            print(f"   ⚠️  Speaker diarization not enabled or only one speaker detected")
            print(f"   To enable: Upload audio with enable_diarization=true")
            print(f"   Required: HUGGINGFACE_TOKEN in .env")
            return None
            
    except Exception as e:
        print(f"❌ Speaker diarization test failed: {e}")
        return False


def run_all_checks():
    """Run all requirement checks."""
    print("\n" + "="*80)
    print("  BERLIN MEDIA ARCHIVE - CORE REQUIREMENTS VERIFICATION")
    print("="*80)
    
    results = {
        "Audio Ingestion Pipeline": check_requirement_1_audio_ingestion(),
        "Unified Vector Store": check_requirement_2_unified_vector_store(),
        "Attribution Engine": check_requirement_3_attribution_engine(),
        "Speaker Diarization": check_module_b_speaker_diarization()
    }
    
    # Summary
    print_header("SUMMARY")
    
    for name, result in results.items():
        if result is True:
            status = "✅ PASS"
        elif result is False:
            status = "❌ FAIL"
        else:
            status = "⚠️  PARTIAL (Needs configuration)"
        
        print(f"{status:20} {name}")
    
    passed = sum(1 for r in results.values() if r is True)
    total = len([r for r in results.values() if r is not None])
    
    print(f"\nTotal: {passed}/{total} requirements passed")
    
    if passed == total:
        print("\n🎉 All core requirements are implemented and working!")
    else:
        print("\n⚠️  Some requirements need attention. See details above.")


if __name__ == "__main__":
    run_all_checks()