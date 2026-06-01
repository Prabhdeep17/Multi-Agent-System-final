"""
RAG Agent for Multi-Agent PDF Analysis System

Handles:
- Question answering with evidence retrieval
- Context-aware response generation
- Citation formatting with metadata
"""
from typing import List, Dict, Any, Optional
from llm_utils import get_llm
from vector_store_manager import VectorStoreManager
from config import Config


class RAGAgent:
    """Retrieval-Augmented Generation Agent for question answering"""
    
    def __init__(
        self,
        name: str = "RAG_Agent",
        vector_store: VectorStoreManager = None
    ):
        """
        Initialize RAG Agent
        
        Args:
            name: Agent name
            vector_store: Vector store manager instance
        """
        self.name = name
        self.vector_store = vector_store
        
        # Initialize LLM
        self.llm = get_llm()
        
        self.system_prompt = """You are a helpful assistant that answers questions based on provided context from PDF documents.

INSTRUCTIONS:
1. Answer questions using ONLY the information from the provided context
2. If the context doesn't contain enough information, say so explicitly
3. Reference specific evidence by citation numbers [1], [2], etc.
4. Be precise and factual
5. Cite your sources for every claim

Format your response as:
- Clear, concise answer
- Use citation numbers [1], [2] to reference evidence
- Each citation corresponds to a piece of evidence provided"""
    
    def retrieve_evidence(
        self,
        query: str,
        top_k: int = None,
        doc_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant evidence for a query
        
        Args:
            query: User query
            top_k: Number of results to retrieve
            doc_filter: Optional document name filter
            
        Returns:
            List of evidence with metadata and scores
        """
        if not self.vector_store:
            return []
        
        top_k = top_k or Config.TOP_K_RETRIEVAL
        
        evidence = self.vector_store.retrieve_with_metadata(
            query=query,
            top_k=top_k,
            doc_filter=doc_filter
        )
        
        # DEBUG LOGGING
        with open("debug_log.txt", "a") as f:
            f.write(f"Query: {query}\n")
            f.write(f"Top K: {top_k}\n")
            f.write(f"Evidence length: {len(evidence)}\n")
            f.write(f"Vector Store Stats: {self.vector_store.get_stats()}\n")
            f.write(f"Doc filter: {doc_filter}\n\n")
            
        return evidence
    
    def format_citations(self, evidence: List[Dict[str, Any]]) -> str:
        """
        Format evidence as numbered citations
        
        Args:
            evidence: List of evidence dicts
            
        Returns:
            Formatted citation string
        """
        if not evidence:
            return "No relevant evidence found."
        
        citations = []
        for idx, ev in enumerate(evidence, 1):
            citation = f"[{idx}] {ev['doc_name']} (Page {ev['page_number']}, Chunk {ev['chunk_id']}) - Score: {ev['similarity_score']:.3f}\n{ev['text']}"
            citations.append(citation)
        
        return "\n\n".join(citations)
    
    def answer_query(
        self,
        query: str,
        doc_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Answer a query using RAG
        
        Args:
            query: User query
            doc_filter: Optional document name filter
            
        Returns:
            Dict with answer, evidence, and metadata
        """
        # Retrieve evidence
        evidence = self.retrieve_evidence(query, doc_filter=doc_filter)
        
        if not evidence:
            return {
                "answer": "I couldn't find relevant information in the documents to answer your query.",
                "evidence": [],
                "query": query,
                "agent": self.name
            }
        
        # Format context for LLM
        context = self.format_citations(evidence)
        
        from langchain_core.messages import SystemMessage, HumanMessage
        
        # Create messages
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"CONTEXT (Evidence from documents):\n{context}\n\nUSER QUESTION: {query}")
        ]
        
        # Generate answer
        try:
            response = self.llm.invoke(messages)
            content = response.content
            if isinstance(content, list):
                answer = "".join(item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text")
            else:
                answer = content
        except Exception as e:
            answer = f"Error generating answer: {str(e)}"
        
        return {
            "answer": answer,
            "evidence": evidence,
            "query": query,
            "agent": self.name,
            "context_used": len(evidence)
        }
    
    def reply(self, x: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Reply method for compatibility with orchestrator
        
        Args:
            x: Input dict with query
            
        Returns:
            Output dict with answer and metadata
        """
        if x is None:
            return {"name": self.name, "content": "No input provided", "metadata": {}}
        
        # Extract query from input
        query = x.get("content", "") if isinstance(x, dict) else str(x)
        
        # Get doc filter if provided in metadata
        doc_filter = None
        if isinstance(x, dict) and "metadata" in x:
            doc_filter = x["metadata"].get('doc_filter')
        
        # Answer query
        result = self.answer_query(query, doc_filter=doc_filter)
        
        # Return dict with result
        return {
            "name": self.name,
            "content": result["answer"],
            "metadata": {
                "evidence": result.get("evidence", []),
                "query": result.get("query", ""),
                "agent": self.name,
                "context_used": result.get("context_used", 0)
            }
        }
