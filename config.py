"""
Configuration management for Multi-Agent PDF Analysis System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables explicitly from the same directory as config.py
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)


class Config:
    """Centralized configuration for the application"""
    
    # API Configuration
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    # Directory Configuration
    BASE_DIR = Path(__file__).parent
    CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db_multi_agent")
    PDF_STORAGE_DIR = os.getenv("PDF_STORAGE_DIR", "./uploaded_pdfs")
    
    # Model Configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mistral-embed")
    LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # groq or google
    
    # Text Processing Configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Retrieval Configuration
    TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    
    # Agent Configuration
    AGENT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.7"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
    
    # UI Configuration
    MAX_FILE_SIZE_MB = 50
    ALLOWED_EXTENSIONS = [".pdf"]
    
    # Vector Store Configuration
    COLLECTION_NAME = "pdf_documents"
    DISTANCE_METRIC = "cosine"
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist"""
        Path(cls.CHROMA_DB_DIR).mkdir(parents=True, exist_ok=True)
        Path(cls.PDF_STORAGE_DIR).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        # Check API keys based on provider
        if cls.LLM_PROVIDER == "groq" and not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        elif cls.LLM_PROVIDER == "google" and not cls.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        cls.ensure_directories()
        return True


# Validate configuration on import
Config.validate()
