"""
Reset vector store and re-ingest documents
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def reset_and_test():
    print("=" * 60)
    print("RESET AND TEST")
    print("=" * 60)
    
    # 1. Clear vector store with reset flag
    print("\n1. Resetting vector store...")
    from vectorstore.chroma_store import UnifiedVectorStore
    
    vs = UnifiedVectorStore(reset_collection=True)
    print(f"   Collection count: {vs.collection.count()} documents")
    
    # 2. Re-ingest document
    print("\n2. Re-ingesting document...")
    from ingestion.document_ingestion import DocumentIngestionPipeline
    
    pdf_path = "./data/documents/How to believe in God even when the world sucks with Nadia Bolz-Weber.pdf"
    
    if not Path(pdf_path).exists():
        print(f"   ❌ PDF not found: {pdf_path}")
        return
    
    doc_pipeline = DocumentIngestionPipeline()
    chunks, metadata = doc_pipeline.ingest_document(pdf_path, "./output/documents")
    
    print(f"   Created {len(chunks)} chunks")
    print(f"   Title: {metadata.get('title', 'N/A')}")
    
    # 3. Add to vector store
    print("\n3. Adding to vector store...")
    documents = []
    for chunk in chunks:
        doc = {
            "id": f"nadia_{chunk.chunk_id}",
            "content": chunk.content,
            "page_number": chunk.page_number,
            "source_file": chunk.source_file,
            "title": metadata.get("title", ""),
            "author": metadata.get("author", ""),
            "chunk_id": chunk.chunk_id,
            "source_type": "document"
        }
        documents.append(doc)
    
    ids = vs.add_documents(documents, source_type="document")
    print(f"   Added {len(ids)} documents")
    print(f"   Total in store: {vs.collection.count()}")
    
    # 4. Test search
    print("\n4. Testing search...")
    query = "What does Nadia Bolz-Weber say about faith?"
    results = vs.search(query, top_k=3)
    
    print(f"   Query: {query}")
    print(f"   Results: {len(results)}")
    
    for i, r in enumerate(results):
        print(f"\n   Result {i+1}:")
        print(f"     Score: {r.get('score', 0):.3f}")
        print(f"     Content: {r.get('content', '')[:150]}...")
    
    print("\n" + "=" * 60)
    print("✅ Reset and test complete!")
    print("=" * 60)

if __name__ == "__main__":
    reset_and_test()