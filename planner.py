"""
Dynamic Planner for Multi-Agent PDF Analysis System

Handles:
- Intent detection from user queries
- Agent routing and orchestration
- Agent chain execution
- Reasoning trace generation
"""
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from llm_utils import get_llm
from agents import (
    RAGAgent,
    SummarizationAgent,
    ComparatorAgent,
    TimelineAgent,
    AggregatorAgent
)
from vector_store_manager import VectorStoreManager
from config import Config


class IntentType(Enum):
    """Types of user intents"""
    QUERY = "query"
    SUMMARIZATION = "summarization"
    COMPARISON = "comparison"
    TIMELINE = "timeline"
    AGGREGATION = "aggregation"
    UNKNOWN = "unknown"


class IntentDetector:
    """Detect user intent from query"""
    
    def __init__(self):
        """Initialize intent detector"""
        self.llm = get_llm(temperature=0.3, max_tokens=100)
        
        self.classification_prompt = """Classify the user's intent into ONE of these categories:

1. QUERY - User is asking a specific question about document content
2. SUMMARIZATION - User wants a summary of document(s)
3. COMPARISON - User wants to compare/contrast information across documents
4. TIMELINE - User wants chronological arrangement of events
5. AGGREGATION - User wants consolidated evidence across multiple sources

Examples:
- "What are the main findings?" → QUERY
- "Summarize this document" → SUMMARIZATION
- "Compare the methodologies in doc1 and doc2" → COMPARISON
- "Create a timeline of events" → TIMELINE
- "What evidence supports this claim across all documents?" → AGGREGATION

USER QUERY: {query}

CLASSIFICATION (respond with only one word - QUERY, SUMMARIZATION, COMPARISON, TIMELINE, or AGGREGATION):"""
    
    def detect(self, query: str) -> IntentType:
        """
        Detect intent from user query
        
        Args:
            query: User query string
            
        Returns:
            IntentType enum value
        """
        prompt = self.classification_prompt.format(query=query)
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content
            if isinstance(content, list):
                content = "".join(item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text")
            
            intent_str = content.strip().upper()
            
            # Map to IntentType
            intent_mapping = {
                "QUERY": IntentType.QUERY,
                "SUMMARIZATION": IntentType.SUMMARIZATION,
                "COMPARISON": IntentType.COMPARISON,
                "TIMELINE": IntentType.TIMELINE,
                "AGGREGATION": IntentType.AGGREGATION
            }
            
            return intent_mapping.get(intent_str, IntentType.UNKNOWN)
        
        except Exception as e:
            print(f"Error detecting intent: {str(e)}")
            return IntentType.UNKNOWN


class AgentOrchestrator:
    """Orchestrate agent execution based on user intent"""
    
    def __init__(self, vector_store: VectorStoreManager):
        """
        Initialize agent orchestrator
        
        Args:
            vector_store: Vector store manager instance
        """
        self.vector_store = vector_store
        self.intent_detector = IntentDetector()
        
        # Initialize all agents
        self.rag_agent = RAGAgent(
            name="RAG_Agent",
            vector_store=vector_store
        )
        self.summarization_agent = SummarizationAgent(
            name="Summarization_Agent",
            vector_store=vector_store
        )
        self.comparator_agent = ComparatorAgent(name="Comparator_Agent")
        self.timeline_agent = TimelineAgent(name="Timeline_Agent")
        self.aggregator_agent = AggregatorAgent(name="Aggregator_Agent")
        
        self.reasoning_trace = []
    
    def reset_trace(self):
        """Reset reasoning trace"""
        self.reasoning_trace = []
    
    def add_to_trace(self, step: str):
        """Add step to reasoning trace"""
        self.reasoning_trace.append(step)
    
    def execute_query_workflow(self, query: str) -> Dict[str, Any]:
        """
        Execute workflow for QUERY intent (RAG only)
        
        Args:
            query: User query
            
        Returns:
            Result dict
        """
        self.add_to_trace("Detected intent: QUERY")
        self.add_to_trace("Routing to: RAG Agent")
        
        # Execute RAG agent
        msg = {"content": query}
        result_msg = self.rag_agent.reply(msg)
        
        return {
            "answer": result_msg["content"],
            "evidence": result_msg.get("metadata", {}).get("evidence", []),
            "reasoning_trace": self.reasoning_trace.copy(),
            "agent_chain": ["RAG_Agent"]
        }
    
    def execute_summarization_workflow(
        self,
        query: str,
        doc_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute workflow for SUMMARIZATION intent
        
        Args:
            query: User query
            doc_filter: Optional document filter
            
        Returns:
            Result dict
        """
        self.add_to_trace("Detected intent: SUMMARIZATION")
        self.add_to_trace("Routing to: Summarization Agent")
        
        # Execute summarization agent
        msg = {
            "content": query,
            "metadata": {"doc_filter": doc_filter}
        }
        result_msg = self.summarization_agent.reply(msg)
        
        return {
            "answer": result_msg["content"],
            "evidence": [],
            "reasoning_trace": self.reasoning_trace.copy(),
            "agent_chain": ["Summarization_Agent"]
        }
    
    def execute_comparison_workflow(self, query: str) -> Dict[str, Any]:
        """
        Execute workflow for COMPARISON intent (RAG → Comparator)
        
        Args:
            query: User query
            
        Returns:
            Result dict
        """
        self.add_to_trace("Detected intent: COMPARISON")
        self.add_to_trace("Routing to: RAG Agent → Comparator Agent")
        
        # Step 1: RAG Agent retrieves evidence
        rag_msg = {"content": query}
        rag_result = self.rag_agent.reply(rag_msg)
        
        self.add_to_trace("RAG Agent: Retrieved evidence")
        
        # Step 2: Comparator Agent processes evidence
        comparator_msg = {
            "content": query,
            "metadata": {
                "evidence": rag_result.get("metadata", {}).get("evidence", []),
                "query": query
            }
        }
        comparator_result = self.comparator_agent.reply(comparator_msg)
        
        self.add_to_trace("Comparator Agent: Generated comparison")
        
        return {
            "answer": comparator_result["content"],
            "evidence": rag_result.get("metadata", {}).get("evidence", []),
            "reasoning_trace": self.reasoning_trace.copy(),
            "agent_chain": ["RAG_Agent", "Comparator_Agent"]
        }
    
    def execute_timeline_workflow(self, query: str) -> Dict[str, Any]:
        """
        Execute workflow for TIMELINE intent (RAG → Timeline)
        
        Args:
            query: User query
            
        Returns:
            Result dict
        """
        self.add_to_trace("Detected intent: TIMELINE")
        self.add_to_trace("Routing to: RAG Agent → Timeline Agent")
        
        # Step 1: RAG Agent retrieves evidence
        rag_msg = {"content": query}
        rag_result = self.rag_agent.reply(rag_msg)
        
        self.add_to_trace("RAG Agent: Retrieved evidence")
        
        # Step 2: Timeline Agent processes evidence
        timeline_msg = {
            "content": query,
            "metadata": {
                "evidence": rag_result.get("metadata", {}).get("evidence", []),
                "query": query
            }
        }
        timeline_result = self.timeline_agent.reply(timeline_msg)
        
        self.add_to_trace("Timeline Agent: Built timeline")
        
        return {
            "answer": timeline_result["content"],
            "evidence": rag_result.get("metadata", {}).get("evidence", []),
            "reasoning_trace": self.reasoning_trace.copy(),
            "agent_chain": ["RAG_Agent", "Timeline_Agent"]
        }
    
    def execute_aggregation_workflow(self, query: str) -> Dict[str, Any]:
        """
        Execute workflow for AGGREGATION intent (RAG → Aggregator)
        
        Args:
            query: User query
            
        Returns:
            Result dict
        """
        self.add_to_trace("Detected intent: AGGREGATION")
        self.add_to_trace("Routing to: RAG Agent → Aggregator Agent")
        
        # Step 1: RAG Agent retrieves evidence
        rag_msg = {"content": query}
        rag_result = self.rag_agent.reply(rag_msg)
        
        self.add_to_trace("RAG Agent: Retrieved evidence")
        
        # Step 2: Aggregator Agent processes evidence
        aggregator_msg = {
            "content": query,
            "metadata": {
                "evidence": rag_result.get("metadata", {}).get("evidence", []),
                "query": query
            }
        }
        aggregator_result = self.aggregator_agent.reply(aggregator_msg)
        
        self.add_to_trace("Aggregator Agent: Aggregated evidence")
        
        return {
            "answer": aggregator_result["content"],
            "evidence": rag_result.get("metadata", {}).get("evidence", []),
            "reasoning_trace": self.reasoning_trace.copy(),
            "agent_chain": ["RAG_Agent", "Aggregator_Agent"]
        }
    
    def execute(self, query: str, doc_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute agent workflow based on detected intent
        
        Args:
            query: User query
            doc_filter: Optional document filter
            
        Returns:
            Result dict with answer, evidence, and reasoning trace
        """
        # Reset trace
        self.reset_trace()
        
        # Detect intent
        intent = self.intent_detector.detect(query)
        
        # Route to appropriate workflow
        if intent == IntentType.QUERY:
            return self.execute_query_workflow(query)
        elif intent == IntentType.SUMMARIZATION:
            return self.execute_summarization_workflow(query, doc_filter)
        elif intent == IntentType.COMPARISON:
            return self.execute_comparison_workflow(query)
        elif intent == IntentType.TIMELINE:
            return self.execute_timeline_workflow(query)
        elif intent == IntentType.AGGREGATION:
            return self.execute_aggregation_workflow(query)
        else:
            # Default to RAG for unknown intent
            self.add_to_trace("Intent unclear, defaulting to RAG Agent")
            return self.execute_query_workflow(query)
