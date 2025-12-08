"""
Debug script to check vector store contents
"""
from vectorstore.chroma_store import UnifiedVectorStore

def check_vectorstore():
    print("=" * 60)
    print("VECTOR STORE DEBUG")
    print("=" * 60)
    
    vs = UnifiedVectorStore()
    stats = vs.get_collection_stats()
    
    print(f"\nCollection: {stats['collection_name']}")
    print(f"Total documents: {stats['total_documents']}")
    print(f"Persist directory: {stats['persist_directory']}")
    print(f"Hybrid search: {stats['hybrid_search_enabled']}")
    
    if stats['total_documents'] > 0:
        print("\n--- Sample Documents ---")
        docs = vs.get_all_documents()
        for i, doc in enumerate(docs[:3]):
            print(f"\nDoc {i+1}:")
            print(f"  ID: {doc.get('id', 'N/A')}")
            print(f"  Content: {doc.get('content', '')[:100]}...")
            print(f"  Metadata: {doc.get('metadata', {})}")
        
        # Test search
        print("\n--- Test Search ---")
        results = vs.search("faith", top_k=3)
        print(f"Search for 'faith' returned {len(results)} results")
        for r in results:
            print(f"  - Score: {r.get('score', 0):.3f}, Content: {r.get('content', '')[:80]}...")
    else:
        print("\n⚠️  Vector store is EMPTY!")
        print("Documents were not added to the vector store.")

if __name__ == "__main__":
    check_vectorstore()