"""
Vector Store Manager for Multi-Agent PDF Analysis System

Handles:
- ChromaDB integration with multi-document support
- Document indexing with metadata
- Retrieval with similarity scores and metadata
- Cross-document search
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from pdf_processor import DocumentMetadata
from config import Config


class VectorStoreManager:
    """Manage vector database for multi-document PDF storage and retrieval"""
    
    def __init__(self, persist_directory: str = None):
        """
        Initialize vector store manager
        
        Args:
            persist_directory: Directory for ChromaDB persistence
        """
        self.persist_directory = persist_directory or Config.CHROMA_DB_DIR
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings (HuggingFace Local)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL
        )
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize Langchain Chroma wrapper
        self.vector_store = None
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize or load existing vector store"""
        self.vector_store = Chroma(
            client=self.client,
            collection_name=Config.COLLECTION_NAME,
            embedding_function=self.embeddings
        )
    
    def index_documents(
        self,
        chunks: List[DocumentMetadata]
    ) -> Dict[str, Any]:
        """
        Index document chunks into vector store
        
        Args:
            chunks: List of DocumentMetadata objects
            
        Returns:
            Dict with indexing stats
        """
        if not chunks:
            return {"status": "error", "message": "No chunks to index"}
        
        # Prepare texts and metadatas
        texts = [chunk.chunk_text for chunk in chunks]
        metadatas = [chunk.to_dict() for chunk in chunks]
        
        # Generate unique IDs for each chunk
        ids = [
            f"{chunk.doc_name}_page{chunk.page_number}_chunk{chunk.chunk_id}"
            for chunk in chunks
        ]
        
        try:
            # Add to vector store
            self.vector_store.add_texts(
                texts=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            return {
                "status": "success",
                "chunks_indexed": len(chunks),
                "doc_name": chunks[0].doc_name if chunks else None
            }
        
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error indexing documents: {str(e)}"
            }
    
    def index_multiple_documents(
        self,
        all_chunks: Dict[str, List[DocumentMetadata]]
    ) -> List[Dict[str, Any]]:
        """
        Index multiple documents
        
        Args:
            all_chunks: Dict mapping doc_name to list of DocumentMetadata
            
        Returns:
            List of indexing results for each document
        """
        results = []
        
        for doc_name, chunks in all_chunks.items():
            result = self.index_documents(chunks)
            results.append(result)
        
        return results
    
    def retrieve_with_metadata(
        self,
        query: str,
        top_k: int = None,
        doc_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks with full metadata and similarity scores
        
        Args:
            query: Search query
            top_k: Number of results to return
            doc_filter: Optional document name to filter results
            
        Returns:
            List of dicts with chunk text, metadata, and similarity score
        """
        top_k = top_k or Config.TOP_K_RETRIEVAL
        
        # Build filter if doc_filter is provided
        where_filter = None
        if doc_filter:
            where_filter = {"doc_name": doc_filter}
        
        try:
            # Perform similarity search with scores
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=top_k,
                filter=where_filter
            )
            
            # Format results
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": float(1 - score),  # Convert distance to similarity
                    "doc_name": doc.metadata.get("doc_name", ""),
                    "page_number": doc.metadata.get("page_number", 0),
                    "chunk_id": doc.metadata.get("chunk_id", 0)
                })
            
            return formatted_results
        
        except Exception as e:
            print(f"Error retrieving documents: {str(e)}")
            return []
    
    def cross_document_search(
        self,
        query: str,
        top_k: int = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search across all documents and group by document
        
        Args:
            query: Search query
            top_k: Number of results per document
            
        Returns:
            Dict mapping doc_name to list of results
        """
        top_k = top_k or Config.TOP_K_RETRIEVAL
        
        # Get all documents
        all_results = self.retrieve_with_metadata(query, top_k * 5)  # Get more results
        
        # Group by document
        grouped_results = {}
        for result in all_results:
            doc_name = result["doc_name"]
            if doc_name not in grouped_results:
                grouped_results[doc_name] = []
            grouped_results[doc_name].append(result)
        
        # Limit results per document
        for doc_name in grouped_results:
            grouped_results[doc_name] = grouped_results[doc_name][:top_k]
        
        return grouped_results
    
    def get_indexed_documents(self) -> List[str]:
        """
        Get list of all indexed document names
        
        Returns:
            List of unique document names
        """
        try:
            collection = self.client.get_collection(Config.COLLECTION_NAME)
            all_data = collection.get()
            
            if not all_data or "metadatas" not in all_data:
                return []
            
            # Extract unique document names
            doc_names = set()
            for metadata in all_data["metadatas"]:
                if "doc_name" in metadata:
                    doc_names.add(metadata["doc_name"])
            
            return sorted(list(doc_names))
        
        except Exception as e:
            print(f"Error getting indexed documents: {str(e)}")
            return []
    
    def delete_document(self, doc_name: str) -> Dict[str, Any]:
        """
        Delete all chunks for a specific document
        
        Args:
            doc_name: Name of document to delete
            
        Returns:
            Dict with deletion status
        """
        try:
            collection = self.client.get_collection(Config.COLLECTION_NAME)
            
            # Get all IDs for this document
            all_data = collection.get(
                where={"doc_name": doc_name}
            )
            
            if all_data and "ids" in all_data and all_data["ids"]:
                collection.delete(ids=all_data["ids"])
                return {
                    "status": "success",
                    "message": f"Deleted {len(all_data['ids'])} chunks from {doc_name}"
                }
            else:
                return {
                    "status": "error",
                    "message": f"No chunks found for {doc_name}"
                }
        
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error deleting document: {str(e)}"
            }
    
    def reset_database(self) -> Dict[str, Any]:
        """
        Reset the entire vector database
        
        Returns:
            Dict with reset status
        """
        try:
            self.client.reset()
            self._initialize_vector_store()
            return {"status": "success", "message": "Database reset successfully"}
        except Exception as e:
            return {"status": "error", "message": f"Error resetting database: {str(e)}"}
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database
        
        Returns:
            Dict with database statistics
        """
        try:
            collection = self.client.get_collection(Config.COLLECTION_NAME)
            count = collection.count()
            indexed_docs = self.get_indexed_documents()
            
            return {
                "total_chunks": count,
                "total_documents": len(indexed_docs),
                "indexed_documents": indexed_docs
            }
        except Exception as e:
            return {
                "total_chunks": 0,
                "total_documents": 0,
                "indexed_documents": [],
                "error": str(e)
            }
