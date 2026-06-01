"""
Mistral Embeddings wrapper for LangChain compatibility
"""
import os
import time
import random
from typing import List, Any
from mistralai.client.sdk import Mistral
from langchain_core.embeddings import Embeddings


class MistralEmbeddings(Embeddings):
    """Mistral AI embeddings wrapper for LangChain"""
    
    def __init__(self, api_key: str = None, model: str = "mistral-embed", max_retries: int = 5):
        """
        Initialize Mistral embeddings
        
        Args:
            api_key: Mistral API key (defaults to MISTRAL_API_KEY env var)
            model: Embedding model name
            max_retries: Maximum number of retries for API calls
        """
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment variables")
        
        self.model = model
        self.max_retries = max_retries
        self.client = Mistral(api_key=self.api_key)
    
    def _execute_with_retry(self, func, *args, **kwargs) -> Any:
        """Execute a function with exponential backoff retry logic"""
        retries = 0
        while True:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = str(e).lower()
                if "429" in error_msg or "capacity exceeded" in error_msg:
                    retries += 1
                    if retries > self.max_retries:
                        print(f"Max retries ({self.max_retries}) exceeded for Mistral API.")
                        raise
                    
                    # Exponential backoff with jitter
                    sleep_time = (2 ** retries) + random.uniform(0, 1)
                    print(f"Mistral API rate limit hit. Retrying in {sleep_time:.2f}s...")
                    time.sleep(sleep_time)
                else:
                    raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            # Mistral has a batch size limit, process in smaller batches if needed
            # For now, we rely on the caller to handle batching or Mistral's limit
            
            response = self._execute_with_retry(
                self.client.embeddings.create,
                model=self.model,
                inputs=texts
            )
            
            # Extract embeddings from response
            embeddings = [item.embedding for item in response.data]
            return embeddings
            
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text
        
        Args:
            text: Query string to embed
            
        Returns:
            Embedding vector
        """
        try:
            response = self._execute_with_retry(
                self.client.embeddings.create,
                model=self.model,
                inputs=[text]
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            print(f"Error generating query embedding: {str(e)}")
            raise
