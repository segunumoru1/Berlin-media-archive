"""
Force reset vector store with OpenAI embeddings
"""
import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def force_reset():
    print("=" * 60)
    print("FORCE RESET VECTOR STORE")
    print("=" * 60)
    
    # 1. Delete the entire vectorstore directory
    vectorstore_path = Path("./data/vectorstore")
    
    if vectorstore_path.exists():
        print(f"\n1. Deleting vectorstore directory: {vectorstore_path}")
        try:
            shutil.rmtree(vectorstore_path)
            print("   ✅ Deleted successfully")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return
    else:
        print(f"\n1. Vectorstore directory doesn't exist: {vectorstore_path}")
    
    # 2. Verify OpenAI API key
    print("\n2. Checking OpenAI API key...")
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and api_key != "your_openai_api_key_here":
        print(f"   ✅ API key found: {api_key[:10]}...{api_key[-4:]}")
    else:
        print("   ❌ No valid OpenAI API key found in .env")
        return
    
    # 3. Initialize vector store with OpenAI embeddings
    print("\n3. Initializing new vector store with OpenAI embeddings...")
    from vectorstore.chroma_store import UnifiedVectorStore
    
    vs = UnifiedVectorStore()
    print(f"   Collection: {vs.collection_name}")
    print(f"   Documents: {vs.collection.count()}")
    
    # 4. Ingest the PDF
    print("\n4. Ingesting PDF document...")
    from ingestion.document_ingestion import DocumentIngestionPipeline
    
    pdf_path = "./data/documents/How to believe in God even when the world sucks with Nadia Bolz-Weber.pdf"
    
    if not Path(pdf_path).exists():
        print(f"   ❌ PDF not found: {pdf_path}")
        print("\n   Available files:")
        docs_dir = Path("./data/documents")
        if docs_dir.exists():
            for f in docs_dir.glob("*.pdf"):
                print(f"      - {f.name}")
        return
    
    doc_pipeline = DocumentIngestionPipeline()
    chunks, metadata = doc_pipeline.ingest_document(pdf_path, "./output/documents")
    
    print(f"   ✅ Created {len(chunks)} chunks")
    
    # 5. Add to vector store
    print("\n5. Adding chunks to vector store...")
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
    print(f"   ✅ Added {len(ids)} documents")
    print(f"   Total in store: {vs.collection.count()}")
    
    # 6. Test search
    print("\n6. Testing search...")
    test_queries = [
        "What does Nadia Bolz-Weber say about faith?",
        "spiritual leadership",
        "believing in God"
    ]
    
    for query in test_queries:
        results = vs.search(query, top_k=3)
        print(f"\n   Query: '{query}'")
        print(f"   Results: {len(results)}")
        
        if results:
            print(f"   Top result score: {results[0].get('score', 0):.3f}")
            print(f"   Preview: {results[0].get('content', '')[:100]}...")
        else:
            print("   ⚠️  No results found")
    
    print("\n" + "=" * 60)
    print("✅ Force reset complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Restart the server: python main.py")
    print("2. Test the query endpoint")

if __name__ == "__main__":
    force_reset()