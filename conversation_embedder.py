"""
conversation_embedder.py - Vector-based conversation memory
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from langchain_ollama import OllamaEmbeddings
import os

logger = logging.getLogger(__name__)

class ConversationEmbedder:
    """
    Embeds conversation interactions for vector-based retrieval.
    This enables semantic search over conversation history.
    """

    def __init__(self, model: str = None, base_url: str = None):
        """
        Initialize the conversation embedder.

        Args:
            model: The embedding model to use
            base_url: The base URL for the Ollama API
        """
        self.model = model or os.getenv('EMBEDDING_MODEL', 'mistral')
        self.base_url = base_url or os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')

        self.embeddings = OllamaEmbeddings(
            model=self.model,
            base_url=self.base_url
        )
        logger.info(f"Initialized ConversationEmbedder with model: {self.model}")

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a query string.

        Args:
            query: The query to embed

        Returns:
            The embedding vector
        """
        try:
            return self.embeddings.embed_query(query)
        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            # Return a zero vector as fallback
            return [0.0] * 384  # Default embedding dimension

    def embed_interaction(self, query: str, response: str) -> List[float]:
        """
        Embed a conversation interaction (query + response).

        Args:
            query: The user's query
            response: The system's response

        Returns:
            The embedding vector
        """
        try:
            # Combine query and response for a more comprehensive embedding
            combined_text = f"Q: {query} A: {response}"
            return self.embeddings.embed_query(combined_text)
        except Exception as e:
            logger.error(f"Error embedding interaction: {str(e)}")
            # Return a zero vector as fallback
            return [0.0] * 384  # Default embedding dimension

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity (between -1 and 1)
        """
        if not vec1 or not vec2:
            return 0.0

        try:
            # Convert to numpy arrays
            a = np.array(vec1)
            b = np.array(vec2)

            # Calculate cosine similarity
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {str(e)}")
            return 0.0

    def find_most_similar(self, query_embedding: List[float],
                         embeddings: List[List[float]],
                         top_k: int = 2) -> List[Tuple[int, float]]:
        """
        Find the most similar embeddings to a query embedding.

        Args:
            query_embedding: The query embedding
            embeddings: List of embeddings to compare against
            top_k: Number of results to return

        Returns:
            List of (index, similarity) tuples
        """
        if not embeddings:
            return []

        # Calculate similarities
        similarities = []
        for i, emb in enumerate(embeddings):
            similarity = self.cosine_similarity(query_embedding, emb)
            similarities.append((i, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top_k results
        return similarities[:top_k]