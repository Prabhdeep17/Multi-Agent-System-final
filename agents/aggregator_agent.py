"""
Aggregator Agent for Multi-Agent PDF Analysis System

Handles:
- Merging overlapping evidence
- Removing redundancy
- Consolidating citations
"""
from typing import List, Dict, Any
from llm_utils import get_llm
from config import Config


class AggregatorAgent:
    """Specialized agent for evidence aggregation and deduplication"""
    
    def __init__(self, name: str = "Aggregator_Agent"):
        """Initialize Aggregator Agent"""
        self.name = name
        
        # Initialize LLM
        self.llm = get_llm()
        
        self.system_prompt = """You are an evidence aggregation specialist.

INSTRUCTIONS:
1. Identify overlapping or redundant information in the evidence
2. Merge similar evidence into consolidated statements
3. Preserve ALL citations for each consolidated point
4. Remove pure redundancy while preserving unique details
5. Create a coherent, non-redundant summary

Format your response as:
**AGGREGATED EVIDENCE:**

1. Consolidated point [citation1, citation2, ...]
2. Consolidated point [citation1, citation2, ...]

**REMOVED REDUNDANCY:**
- Brief note on what redundant information was merged"""
    
    def aggregate(self, evidence: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """
        Aggregate and deduplicate evidence
        
        Args:
            evidence: List of evidence from RAG agent
            query: Original user query
            
        Returns:
            Dict with aggregated result
        """
        if not evidence:
            return {
                "aggregated": "No evidence provided for aggregation.",
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
        
        # Create aggregation prompt
        prompt = f"""{self.system_prompt}

QUERY: {query}

EVIDENCE:
{evidence_text}

AGGREGATED EVIDENCE:"""
        
        # Generate aggregated result
        try:
            response = self.llm.invoke(prompt)
            aggregated = response.content
        except Exception as e:
            aggregated = f"Error aggregating evidence: {str(e)}"
        
        return {
            "aggregated": aggregated,
            "evidence": evidence,
            "query": query,
            "agent": self.name,
            "original_evidence_count": len(evidence)
        }
    
    def reply(self, x: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Reply method for compatibility with orchestrator
        
        Args:
            x: Input dict (should contain evidence from RAG agent)
            
        Returns:
            Output dict with aggregated evidence
        """
        if x is None:
            return {"name": self.name, "content": "No input provided", "metadata": {}}
        
        # Extract evidence and query from input
        evidence = []
        query = x.get("content", "") if isinstance(x, dict) else str(x)
        
        if isinstance(x, dict) and "metadata" in x:
            evidence = x["metadata"].get('evidence', [])
            query = x["metadata"].get('query', query)
        
        # Aggregate evidence
        result = self.aggregate(evidence, query)
        
        # Return dict with result
        return {
            "name": self.name,
            "content": result["aggregated"],
            "metadata": result
        }
