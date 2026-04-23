import os
import json
import chromadb
from chromadb.config import Settings
from typing import List

class RAGRetriever:
    """
    Query ChromaDB knowledge base to retrieve relevant geospatial context.
    """
    
    def __init__(self, db_path: str = "data/chroma_db", collection_name: str = "geoevolve_kb"):
        """
        Initialize retriever with ChromaDB connection.
        
        Args:
            db_path: Path to ChromaDB directory
            collection_name: Name of the collection
        """
        os.makedirs(db_path, exist_ok=True)
        
        try:
            self.client = chromadb.PersistentClient(path=db_path)
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            print(f"Warning: ChromaDB initialization failed: {e}")
            self.client = None
            self.collection = None
    
    def index_documents(self, documents: List[dict], batch_size: int = 100):
        """
        Index documents into ChromaDB.
        
        Args:
            documents: List of dicts with 'id', 'content', 'metadata'
            batch_size: Batch size for indexing
        """
        if not self.collection:
            print("Warning: ChromaDB not initialized, skipping indexing")
            return
        
        ids = []
        contents = []
        metadatas = []
        
        for i, doc in enumerate(documents):
            ids.append(f"doc_{i}")
            # Extract text from various document formats
            if isinstance(doc, dict):
                if 'content' in doc:
                    contents.append(doc['content'])
                elif 'summary' in doc:
                    contents.append(doc['summary'])
                else:
                    contents.append(str(doc))
                
                metadatas.append({
                    'source': doc.get('source', 'unknown'),
                    'title': doc.get('title', ''),
                })
            else:
                contents.append(str(doc))
                metadatas.append({'source': 'unknown'})
        
        try:
            self.collection.add(
                ids=ids,
                documents=contents,
                metadatas=metadatas
            )
            print(f"Indexed {len(documents)} documents into ChromaDB")
        except Exception as e:
            print(f"Error indexing documents: {e}")
    
    def query(self, query_text: str, top_k: int = 5) -> str:
        """
        Query the knowledge base and return relevant context.
        
        Args:
            query_text: Query string
            top_k: Number of top results
            
        Returns:
            str: Concatenated context from top-k results
        """
        if not self.collection:
            print("Warning: ChromaDB not initialized, returning empty context")
            return ""
        
        try:
            # Query the collection
            results = self.collection.query(
                query_texts=[query_text],
                n_results=top_k,
                where=None
            )
            
            if not results or not results['documents'][0]:
                return ""
            
            # Combine retrieved documents
            context_parts = []
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                context_parts.append(f"[{metadata.get('source', 'unknown')}] {doc[:500]}")
            
            return "\n\n".join(context_parts)
        
        except Exception as e:
            print(f"Error querying ChromaDB: {e}")
            return ""
    
    def load_and_index_files(self, kb_dir: str = "data/knowledge"):
        """
        Load JSON documents from kb_dir and index them.
        
        Args:
            kb_dir: Directory containing knowledge base files
        """
        if not os.path.exists(kb_dir):
            print(f"Knowledge directory not found: {kb_dir}")
            return
        
        all_documents = []
        
        for filename in os.listdir(kb_dir):
            if filename.endswith('.jsonl'):
                filepath = os.path.join(kb_dir, filename)
                print(f"Loading {filename}...")
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                doc = json.loads(line)
                                all_documents.append(doc)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        if all_documents:
            self.index_documents(all_documents)
            print(f"Loaded and indexed {len(all_documents)} documents")
        else:
            print("No documents found to index")


if __name__ == "__main__":
    # Test the retriever
    retriever = RAGRetriever()
    
    # Try to load documents from knowledge base
    retriever.load_and_index_files()
    
    # Test query
    test_queries = [
        "variogram model selection kriging",
        "kriging parameter optimization",
        "adaptive kriging local neighborhoods"
    ]
    
    for query in test_queries:
        print(f"\n--- Query: {query} ---")
        context = retriever.query(query, top_k=3)
        if context:
            print(context[:500])
        else:
            print("No results found")
