"""
Timeline Agent for Multi-Agent PDF Analysis System

Handles:
- Extracting temporal information from evidence
- Arranging events chronologically
- Maintaining document citations
"""
from typing import List, Dict, Any
from llm_utils import get_llm
from config import Config


class TimelineAgent:
    """Specialized agent for chronological arrangement of events"""
    
    def __init__(self, name: str = "Timeline_Agent"):
        """Initialize Timeline Agent"""
        self.name = name
        
        # Initialize LLM
        self.llm = get_llm()
        
        self.system_prompt = """You are a timeline specialist that extracts and organizes temporal information.

INSTRUCTIONS:
1. Extract all events with temporal markers from the evidence
2. Arrange events chronologically
3. Maintain citations for each event
4. Infer relative ordering when exact dates are not provided
5. Group events by time periods if appropriate

Format your response as:
**TIMELINE:**

[Date/Period] Event description [citation]
[Date/Period] Event description [citation]

**NOTES:**
- Any assumptions made about ordering
- Missing temporal information"""
    
    def build_timeline(self, evidence: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """
        Build a timeline from evidence
        
        Args:
            evidence: List of evidence from RAG agent
            query: Original user query
            
        Returns:
            Dict with timeline result
        """
        if not evidence:
            return {
                "timeline": "No evidence provided for timeline construction.",
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
        
        # Create timeline prompt
        prompt = f"""{self.system_prompt}

QUERY: {query}

EVIDENCE:
{evidence_text}

TIMELINE:"""
        
        # Generate timeline
        try:
            response = self.llm.invoke(prompt)
            timeline = response.content
        except Exception as e:
            timeline = f"Error generating timeline: {str(e)}"
        
        return {
            "timeline": timeline,
            "evidence": evidence,
            "query": query,
            "agent": self.name,
            "events_processed": len(evidence)
        }
    
    def reply(self, x: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Reply method for compatibility with orchestrator
        
        Args:
            x: Input dict (should contain evidence from RAG agent)
            
        Returns:
            Output dict with timeline
        """
        if x is None:
            return {"name": self.name, "content": "No input provided", "metadata": {}}
        
        # Extract evidence and query from input
        evidence = []
        query = x.get("content", "") if isinstance(x, dict) else str(x)
        
        if isinstance(x, dict) and "metadata" in x:
            evidence = x["metadata"].get('evidence', [])
            query = x["metadata"].get('query', query)
        
        # Build timeline
        result = self.build_timeline(evidence, query)
        
        # Return dict with result
        return {
            "name": self.name,
            "content": result["timeline"],
            "metadata": result
        }
