"""
LLM utility for getting the configured language model
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from config import Config


def get_llm(temperature: float = None, max_tokens: int = None):
    """
    Get the configured LLM instance
    
    Args:
        temperature: Override temperature (defaults to Config.AGENT_TEMPERATURE)
        max_tokens: Override max tokens (defaults to Config.MAX_TOKENS)
        
    Returns:
        LLM instance (ChatGroq or ChatGoogleGenerativeAI)
    """
    temp = temperature if temperature is not None else Config.AGENT_TEMPERATURE
    tokens = max_tokens if max_tokens is not None else Config.MAX_TOKENS
    
    if Config.LLM_PROVIDER == "groq":
        return ChatGroq(
            model=Config.LLM_MODEL,
            api_key=Config.GROQ_API_KEY,
            temperature=temp,
            max_tokens=tokens
        )
    elif Config.LLM_PROVIDER == "google":
        return ChatGoogleGenerativeAI(
            model=Config.LLM_MODEL,
            google_api_key=Config.GOOGLE_API_KEY,
            temperature=temp,
            max_output_tokens=tokens
        )
    else:
        raise ValueError(f"Unknown LLM provider: {Config.LLM_PROVIDER}")
