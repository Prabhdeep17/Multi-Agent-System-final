"""
Comparator Agent for Multi-Agent PDF Analysis System

Handles:
- Comparing and contrasting information across documents
- Identifying similarities and differences
- Preserving citations from RAG agent output
"""
from typing import List, Dict, Any
from llm_utils import get_llm
from config import Config


class ComparatorAgent:
    """Specialized agent for comparing and contrasting across documents"""
    
    def __init__(self, name: str = "Comparator_Agent"):
        """Initialize Comparator Agent"""
        self.name = name
        
        # Initialize LLM
        self.llm = get_llm()
        
        self.system_prompt = """You are a comparison specialist that analyzes evidence from multiple documents.

INSTRUCTIONS:
1. Compare and contrast the provided evidence
2. Identify similarities and differences
3. Maintain citations from the original evidence
4. Organize comparisons clearly (Similarities / Differences)
5. Be objective and factual

Format your response as:
**SIMILARITIES:**
- Point 1 [citation]
- Point 2 [citation]

**DIFFERENCES:**
- Difference 1 [citation]
- Difference 2 [citation]

**ANALYSIS:**
Brief synthesis of the comparison"""
    
    def compare(self, evidence: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """
        Compare evidence across documents
        
        Args:
            evidence: List of evidence from RAG agent
            query: Original user query
            
        Returns:
            Dict with comparison result
        """
        if not evidence:
            return {
                "comparison": "No evidence provided for comparison.",
                "query": query,
                "agent": self.name
            }
        
        # Format evidence with citations
        formatted_evidence = []
        for idx, ev in enumerate(evidence, 1):
            formatted_evidence.append(
                f"[{idx}] {ev['doc_name']} (Page {ev['page_number']}): {ev['text']}"
            )
        
        evidence_text = "\n\n".join(formatted_evidence)
        
        # Create comparison prompt
        prompt = f"""{self.system_prompt}

QUERY: {query}

EVIDENCE:
{evidence_text}

COMPARISON:"""
        
        # Generate comparison
        try:
            response = self.llm.invoke(prompt)
            comparison = response.content
        except Exception as e:
            comparison = f"Error generating comparison: {str(e)}"
        
        return {
            "comparison": comparison,
            "evidence": evidence,
            "query": query,
            "agent": self.name,
            "evidence_count": len(evidence)
        }
    
    def reply(self, x: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Reply method for compatibility with orchestrator
        
        Args:
            x: Input dict (should contain evidence from RAG agent)
            
        Returns:
            Output dict with comparison
        """
        if x is None:
            return {"name": self.name, "content": "No input provided", "metadata": {}}
        
        # Extract evidence and query from input
        evidence = []
        query = x.get("content", "") if isinstance(x, dict) else str(x)
        
        if isinstance(x, dict) and "metadata" in x:
            evidence = x["metadata"].get('evidence', [])
            query = x["metadata"].get('query', query)
        
        # Perform comparison
        result = self.compare(evidence, query)
        
        # Return dict with result
        return {
            "name": self.name,
            "content": result["comparison"],
            "metadata": result
        }
