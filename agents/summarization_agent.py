"""
Summarization Agent for Multi-Agent PDF Analysis System

Handles:
- Single document summarization
- Multi-document summarization
- Map-reduce summarization strategy
"""
from typing import List, Dict, Any
from llm_utils import get_llm
from vector_store_manager import VectorStoreManager
from config import Config


class SummarizationAgent:
    """Agent for document summarization using map-reduce strategy"""
    
    def __init__(
        self,
        name: str = "Summarization_Agent",
        vector_store: VectorStoreManager = None
    ):
        """
        Initialize Summarization Agent
        
        Args:
            name: Agent name
            vector_store: Vector store manager instance
        """
        self.name = name
        self.vector_store = vector_store
        
        # Initialize LLM
        self.llm = get_llm()
    
    def summarize_chunk(self, text: str) -> str:
        """
        Summarize a single chunk of text
        
        Args:
            text: Text chunk to summarize
            
        Returns:
            Summary string
        """
        from langchain_core.messages import SystemMessage, HumanMessage
        
        messages = [
            SystemMessage(content="You are an expert summarizer. Summarize the provided text concisely, preserving key information."),
            HumanMessage(content=f"TEXT:\n{text}")
        ]
        
        try:
            response = self.llm.invoke(messages)
            content = response.content
            if isinstance(content, list):
                return "".join(item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text")
            return content
        except Exception as e:
            return f"Error summarizing chunk: {str(e)}"
    
    def merge_summaries(self, summaries: List[str]) -> str:
        """
        Merge multiple summaries into a coherent final summary
        
        Args:
            summaries: List of summary strings
            
        Returns:
            Final merged summary
        """
        combined = "\n\n".join([f"Section {i+1}:\n{s}" for i, s in enumerate(summaries)])
        
        from langchain_core.messages import SystemMessage, HumanMessage
        
        messages = [
            SystemMessage(content="You are an expert summarizer. Merge the provided section summaries into a coherent, comprehensive summary. Remove redundancy and maintain logical flow."),
            HumanMessage(content=f"SECTION SUMMARIES:\n{combined}")
        ]
        
        try:
            response = self.llm.invoke(messages)
            content = response.content
            if isinstance(content, list):
                return "".join(item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text")
            return content
        except Exception as e:
            return f"Error merging summaries: {str(e)}"
    
    def summarize_single_doc(self, doc_name: str) -> Dict[str, Any]:
        """
        Summarize a single document using map-reduce
        
        Args:
            doc_name: Name of document to summarize
            
        Returns:
            Dict with summary and metadata
        """
        if not self.vector_store:
            return {
                "summary": "No vector store available",
                "doc_name": doc_name,
                "agent": self.name
            }
        
        # Get all chunks for this document (using a generic query to get all)
        all_chunks = self.vector_store.retrieve_with_metadata(
            query=f"content from {doc_name}",
            top_k=50,  # Get many chunks
            doc_filter=doc_name
        )
        
        if not all_chunks:
            return {
                "summary": f"No content found for document: {doc_name}",
                "doc_name": doc_name,
                "agent": self.name
            }
        
        # MAP: Summarize each chunk
        chunk_summaries = []
        for chunk in all_chunks[:20]:  # Limit to avoid token limits
            summary = self.summarize_chunk(chunk['text'])
            chunk_summaries.append(summary)
        
        # REDUCE: Merge summaries
        final_summary = self.merge_summaries(chunk_summaries)
        
        return {
            "summary": final_summary,
            "doc_name": doc_name,
            "chunks_processed": len(chunk_summaries),
            "agent": self.name
        }
    
    def summarize_multiple_docs(self, doc_names: List[str] = None) -> Dict[str, Any]:
        """
        Summarize multiple documents
        
        Args:
            doc_names: List of document names (if None, summarize all)
            
        Returns:
            Dict with summary and metadata
        """
        if not self.vector_store:
            return {
                "summary": "No vector store available",
                "agent": self.name
            }
        
        # Get all indexed documents if no doc_names provided
        if doc_names is None:
            doc_names = self.vector_store.get_indexed_documents()
        
        if not doc_names:
            return {
                "summary": "No documents available to summarize",
                "agent": self.name
            }
        
        # Summarize each document
        doc_summaries = []
        for doc_name in doc_names:
            result = self.summarize_single_doc(doc_name)
            doc_summaries.append(f"{doc_name}:\n{result['summary']}")
        
        # Merge all document summaries
        if len(doc_summaries) == 1:
            final_summary = doc_summaries[0]
        else:
            final_summary = self.merge_summaries(doc_summaries)
        
        return {
            "summary": final_summary,
            "documents_summarized": doc_names,
            "agent": self.name
        }
    
    def reply(self, x: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Reply method for compatibility with orchestrator
        
        Args:
            x: Input dict
            
        Returns:
            Output dict with summary
        """
        if x is None:
            return {"name": self.name, "content": "No input provided", "metadata": {}}
        
        # Extract parameters from input metadata
        doc_filter = None
        doc_names = None
        
        if isinstance(x, dict) and "metadata" in x:
            doc_filter = x["metadata"].get('doc_filter')
            doc_names = x["metadata"].get('doc_names')
        
        # Determine if single or multi-document summarization
        if doc_filter:
            result = self.summarize_single_doc(doc_filter)
        elif doc_names:
            result = self.summarize_multiple_docs(doc_names)
        else:
            result = self.summarize_multiple_docs()
        
        # Return dict with result
        return {
            "name": self.name,
            "content": result["summary"],
            "metadata": result
        }
