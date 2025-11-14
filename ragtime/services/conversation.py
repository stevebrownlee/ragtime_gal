"""
Unified conversation management module for Ragtime.

This module provides comprehensive conversation management with:
- Session-based conversation history tracking
- Vector-based semantic retrieval of relevant interactions
- Conversation summarization for context management
- Follow-up question detection
- Integration with ConPort for feedback

Classes:
    Interaction: Represents a single conversation turn
    ConversationEmbedder: Provides vector embeddings for semantic search
    ConversationSummarizer: Generates conversation summaries
    Conversation: Base conversation management
    EnhancedConversation: Extended conversation with vector retrieval
    ConversationManager: Main orchestrator for conversation operations

Examples:
    >>> from ragtime.services.conversation import ConversationManager
    >>> manager = ConversationManager()
    >>> conversation = manager.create_conversation()
    >>> conversation.add_interaction("Hello", "Hi there!")
    >>> relevant = conversation.get_relevant_interactions("greeting")
"""

import time
import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from langchain_ollama import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama

from ragtime.config.settings import settings
from ragtime.monitoring.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Interaction:
    """
    Represents a single interaction in a conversation.

    Attributes:
        query: User's input query
        response: System's response
        timestamp: Unix timestamp of the interaction
        document_ids: List of referenced document IDs
        metadata: Additional metadata (tags, content type, etc.)

    Examples:
        >>> interaction = Interaction(
        ...     query="What is RAG?",
        ...     response="RAG stands for Retrieval-Augmented Generation...",
        ...     document_ids=["doc1", "doc2"]
        ... )
    """
    query: str
    response: str
    timestamp: float = field(default_factory=time.time)
    document_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationEmbedder:
    """
    Embeds conversation interactions for vector-based retrieval.

    This class enables semantic search over conversation history by creating
    vector embeddings of queries and responses. It uses Ollama embeddings
    with configuration from settings.

    Attributes:
        model: Embedding model name
        base_url: Ollama API base URL
        embeddings: OllamaEmbeddings instance

    Examples:
        >>> embedder = ConversationEmbedder()
        >>> embedding = embedder.embed_query("Hello world")
        >>> len(embedding)
        384  # Depends on model
    """

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize the conversation embedder.

        Args:
            model: Embedding model name (defaults to settings)
            base_url: Ollama API URL (defaults to settings)
        """
        self.model = model or settings.EMBEDDING_MODEL
        self.base_url = base_url or settings.OLLAMA_BASE_URL

        try:
            self.embeddings = OllamaEmbeddings(
                model=self.model,
                base_url=self.base_url
            )
            logger.info(
                "Initialized conversation embedder",
                extra={
                    "model": self.model,
                    "base_url": self.base_url
                }
            )
        except Exception as e:
            logger.error(
                "Failed to initialize conversation embedder",
                extra={"error": str(e)},
                exc_info=True
            )
            raise

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a query string.

        Args:
            query: The query to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            Exception: If embedding generation fails
        """
        try:
            embedding = self.embeddings.embed_query(query)
            logger.debug(
                "Generated query embedding",
                extra={"query_length": len(query), "embedding_dim": len(embedding)}
            )
            return embedding
        except Exception as e:
            logger.error(
                "Error embedding query",
                extra={"query": query[:100], "error": str(e)},
                exc_info=True
            )
            # Return zero vector as fallback
            return [0.0] * 384

    def embed_interaction(self, query: str, response: str) -> List[float]:
        """
        Embed a conversation interaction (query + response).

        Combines query and response for a more comprehensive embedding that
        captures the full context of the interaction.

        Args:
            query: User's query
            response: System's response

        Returns:
            Embedding vector for the combined interaction
        """
        try:
            combined_text = f"Q: {query} A: {response}"
            embedding = self.embeddings.embed_query(combined_text)
            logger.debug(
                "Generated interaction embedding",
                extra={
                    "query_length": len(query),
                    "response_length": len(response),
                    "embedding_dim": len(embedding)
                }
            )
            return embedding
        except Exception as e:
            logger.error(
                "Error embedding interaction",
                extra={"error": str(e)},
                exc_info=True
            )
            return [0.0] * 384

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score (between -1 and 1)
        """
        if not vec1 or not vec2:
            return 0.0

        try:
            a = np.array(vec1)
            b = np.array(vec2)
            similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            return float(similarity)
        except Exception as e:
            logger.error(
                "Error calculating cosine similarity",
                extra={"error": str(e)},
                exc_info=True
            )
            return 0.0

    def find_most_similar(
        self,
        query_embedding: List[float],
        embeddings: List[List[float]],
        top_k: int = 2
    ) -> List[Tuple[int, float]]:
        """
        Find the most similar embeddings to a query embedding.

        Args:
            query_embedding: Query vector
            embeddings: List of vectors to compare against
            top_k: Number of results to return

        Returns:
            List of (index, similarity_score) tuples sorted by similarity
        """
        if not embeddings:
            return []

        similarities = []
        for i, emb in enumerate(embeddings):
            similarity = self.cosine_similarity(query_embedding, emb)
            similarities.append((i, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


class ConversationSummarizer:
    """
    Summarizes conversation history to provide compact context.

    This class uses an LLM to generate concise summaries of conversation
    history, helping manage token usage by compressing older interactions.

    Attributes:
        model: LLM model name for summarization
        base_url: Ollama API base URL
        llm: ChatOllama instance

    Examples:
        >>> summarizer = ConversationSummarizer()
        >>> conversation = Conversation()
        >>> # ... add interactions ...
        >>> summary = summarizer.summarize(conversation, max_tokens=500)
    """

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize the conversation summarizer.

        Args:
            model: LLM model name (defaults to settings)
            base_url: Ollama API URL (defaults to settings)
        """
        self.model = model or settings.LLM_MODEL
        self.base_url = base_url or settings.OLLAMA_BASE_URL

        try:
            self.llm = ChatOllama(
                model=self.model,
                base_url=self.base_url,
                temperature=0.3,  # Lower temperature for factual summaries
                num_predict=1024  # Shorter generations for summaries
            )
            logger.info(
                "Initialized conversation summarizer",
                extra={
                    "model": self.model,
                    "base_url": self.base_url
                }
            )
        except Exception as e:
            logger.error(
                "Failed to initialize conversation summarizer",
                extra={"error": str(e)},
                exc_info=True
            )
            raise

    def summarize(self, conversation: 'Conversation', max_tokens: int = 500) -> str:
        """
        Generate a summary of the conversation history.

        Args:
            conversation: Conversation to summarize
            max_tokens: Approximate maximum tokens for summary

        Returns:
            Conversation summary as string
        """
        if not conversation or not hasattr(conversation, 'get_history'):
            return ""

        history = conversation.get_history()

        # For very short conversations, format without summarizing
        if len(history) <= 2:
            return self._format_short_conversation(history)

        try:
            history_text = self._format_full_history(history)

            summary_prompt = f"""
            Summarize the following conversation in a concise way that preserves the key points,
            questions asked, and information provided. Focus on the main topics and any conclusions reached.
            Keep your summary under {max_tokens} tokens.

            {history_text}

            Summary:
            """

            response = self.llm.invoke(summary_prompt)
            summary = response.content.strip()

            logger.info(
                "Generated conversation summary",
                extra={
                    "history_length": len(history),
                    "summary_length": len(summary)
                }
            )
            return summary
        except Exception as e:
            logger.error(
                "Error summarizing conversation",
                extra={"error": str(e)},
                exc_info=True
            )
            return self._format_recent_history(history)

    def _format_short_conversation(self, history: List[Interaction]) -> str:
        """Format a short conversation without summarization."""
        if not history:
            return ""

        formatted = "Conversation history:\n\n"
        for i, interaction in enumerate(history):
            formatted += f"User: {interaction.query}\n"
            formatted += f"Assistant: {interaction.response}\n"
            if i < len(history) - 1:
                formatted += "\n"

        return formatted

    def _format_full_history(self, history: List[Interaction]) -> str:
        """Format the full conversation history as a string."""
        if not history:
            return ""

        formatted = "Conversation history:\n\n"
        for i, interaction in enumerate(history):
            formatted += f"User: {interaction.query}\n"
            formatted += f"Assistant: {interaction.response}\n"
            if i < len(history) - 1:
                formatted += "\n"

        return formatted

    def _format_recent_history(
        self,
        history: List[Interaction],
        max_recent: int = 3
    ) -> str:
        """Format recent history with a brief summary of older interactions."""
        if not history:
            return ""

        if len(history) <= max_recent:
            return self._format_full_history(history)

        recent = history[-max_recent:]
        older = history[:-max_recent]

        summary = "Previous conversation summary:\n"

        if older:
            summary += f"- {len(older)} earlier exchanges occurred.\n"
            summary += f"- The conversation began with: \"{older[0].query}\"\n"

            # Extract topics
            topics = set()
            for interaction in older:
                words = interaction.query.lower().split()
                for word in words:
                    if len(word) > 4 and word not in [
                        "about", "would", "could", "should",
                        "there", "their", "these", "those"
                    ]:
                        topics.add(word)

            if topics:
                topic_list = list(topics)[:5]
                summary += f"- Topics discussed: {', '.join(topic_list)}\n"

        summary += "\nRecent conversation:\n"
        for i, interaction in enumerate(recent):
            summary += f"User: {interaction.query}\n"
            summary += f"Assistant: {interaction.response}\n"
            if i < len(recent) - 1:
                summary += "\n"

        return summary


class Conversation:
    """
    Base conversation management with history tracking.

    Manages a conversation session with history, search, and context retrieval.
    This is the base class that can be used directly or extended.

    Attributes:
        history: List of Interaction objects

    Examples:
        >>> conv = Conversation()
        >>> conv.add_interaction("Hi", "Hello!")
        >>> conv.get_last_interaction().query
        'Hi'
    """

    def __init__(self):
        """Initialize an empty conversation."""
        self.history: List[Interaction] = []
        logger.debug("Created new Conversation instance")

    def add_interaction(
        self,
        query: str,
        response: str,
        document_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a new interaction to the conversation history.

        Args:
            query: User's query
            response: System's response
            document_ids: Optional list of referenced document IDs
            metadata: Optional metadata (e.g., content_type, tags)
        """
        interaction = Interaction(
            query=query,
            response=response,
            timestamp=time.time(),
            document_ids=document_ids or [],
            metadata=metadata or {}
        )
        self.history.append(interaction)
        logger.info(
            "Added interaction to conversation",
            extra={
                "total_interactions": len(self.history),
                "has_documents": bool(document_ids)
            }
        )

    def get_history(self) -> List[Interaction]:
        """Get the full conversation history."""
        return self.history

    def clear(self) -> None:
        """Clear the conversation history."""
        interaction_count = len(self.history)
        self.history = []
        logger.info(
            "Cleared conversation history",
            extra={"cleared_interactions": interaction_count}
        )

    def get_last_interaction(self) -> Optional[Interaction]:
        """Get the most recent interaction."""
        if not self.history:
            return None
        return self.history[-1]

    def get_last_n_interactions(self, n: int = 1) -> List[Interaction]:
        """
        Get the last n interactions.

        Args:
            n: Number of interactions to retrieve

        Returns:
            List of the most recent n interactions
        """
        return self.history[-n:] if len(self.history) >= n else self.history[:]

    def find_interactions_by_query_keywords(
        self,
        keywords: List[str]
    ) -> List[Interaction]:
        """
        Find interactions where query contains any of the given keywords.

        Args:
            keywords: List of keywords to search for

        Returns:
            List of matching interactions
        """
        if not keywords:
            return []

        result = []
        for interaction in self.history:
            if any(kw.lower() in interaction.query.lower() for kw in keywords):
                result.append(interaction)

        logger.debug(
            "Searched interactions by keywords",
            extra={
                "keywords": keywords,
                "matches": len(result)
            }
        )
        return result

    def find_interactions_by_metadata(self, **metadata) -> List[Interaction]:
        """
        Find interactions matching the given metadata criteria.

        Args:
            **metadata: Key-value pairs to match

        Returns:
            List of matching interactions
        """
        result = []
        for interaction in self.history:
            if all(
                interaction.metadata.get(key) == value
                for key, value in metadata.items()
            ):
                result.append(interaction)

        logger.debug(
            "Searched interactions by metadata",
            extra={
                "criteria": metadata,
                "matches": len(result)
            }
        )
        return result

    def get_generated_content(self) -> str:
        """
        Get all generated content from conversation history.

        Returns:
            String containing all responses
        """
        if not self.history:
            return ""

        content = "Previously generated content:\n\n"
        for i, interaction in enumerate(self.history):
            content += f"Content {i+1}:\n"
            content += f"Query: {interaction.query}\n"
            content += f"Response: {interaction.response}\n\n"

        return content

    def get_most_relevant_content(
        self,
        query: str,
        max_interactions: int = 2
    ) -> str:
        """
        Get most relevant previous content based on current query.

        Uses keyword matching to identify potential references to previous
        content in the query.

        Args:
            query: Current query
            max_interactions: Maximum relevant interactions to include

        Returns:
            String containing relevant previous content
        """
        if not self.history:
            return ""

        reference_indicators = [
            "previous", "last", "earlier", "before", "above",
            "chapter", "section", "paragraph", "summary",
            "you said", "you mentioned", "you wrote", "you generated",
            "that", "it", "this"
        ]

        is_reference_query = any(
            indicator.lower() in query.lower()
            for indicator in reference_indicators
        )

        if is_reference_query:
            relevant_interactions = self.get_last_n_interactions(max_interactions)

            content = "Relevant previous content:\n\n"
            for i, interaction in enumerate(relevant_interactions):
                content += f"Content {i+1} (most recent):\n"
                content += f"Query: {interaction.query}\n"
                content += f"Response: {interaction.response}\n\n"

            return content

        return ""

    def is_follow_up_question(self, query: str) -> bool:
        """
        Determine if query is a follow-up to previous interactions.

        Uses regex patterns to detect pronouns and references suggesting
        a follow-up question.

        Args:
            query: Current query

        Returns:
            True if query appears to be a follow-up
        """
        if not self.history:
            return False

        follow_up_indicators = [
            r"\bit\b", r"\bthis\b", r"\bthat\b", r"\bthese\b", r"\bthose\b",
            r"\bthe\b", r"\byour\b", r"\byou\b", r"\bprevious\b", r"\blast\b",
            r"\babove\b", r"\bearlier\b", r"\bbefore\b"
        ]

        for pattern in follow_up_indicators:
            if re.search(pattern, query, re.IGNORECASE):
                logger.debug(
                    "Detected follow-up question",
                    extra={"pattern": pattern, "query": query[:100]}
                )
                return True

        return False

    def get_summary(self, max_interactions: int = 3) -> str:
        """
        Generate a summary of conversation history.

        Args:
            max_interactions: Maximum recent interactions to include in full

        Returns:
            Formatted conversation summary
        """
        if not self.history:
            return ""

        if len(self.history) <= max_interactions:
            return self._format_full_history()

        recent = self.history[-max_interactions:]
        older = self.history[:-max_interactions]

        summary = "Previous conversation summary:\n"

        if older:
            summary += f"- {len(older)} earlier exchanges occurred.\n"
            summary += f"- The conversation began with: \"{older[0].query}\"\n"

        summary += "\nRecent conversation:\n"
        for i, interaction in enumerate(recent):
            summary += f"User: {interaction.query}\n"
            summary += f"Assistant: {interaction.response}\n"
            if i < len(recent) - 1:
                summary += "\n"

        return summary

    def _format_full_history(self) -> str:
        """Format the full conversation history as a string."""
        if not self.history:
            return ""

        formatted = "Conversation history:\n"
        for i, interaction in enumerate(self.history):
            formatted += f"User: {interaction.query}\n"
            formatted += f"Assistant: {interaction.response}\n"
            if i < len(self.history) - 1:
                formatted += "\n"

        return formatted

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert conversation to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "history": [
                {
                    "query": interaction.query,
                    "response": interaction.response,
                    "timestamp": interaction.timestamp,
                    "document_ids": interaction.document_ids,
                    "metadata": interaction.metadata
                }
                for interaction in self.history
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Conversation':
        """
        Create Conversation instance from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            Conversation instance
        """
        conversation = cls()

        for item in data.get("history", []):
            interaction = Interaction(
                query=item["query"],
                response=item["response"],
                timestamp=item.get("timestamp", time.time()),
                document_ids=item.get("document_ids", []),
                metadata=item.get("metadata", {})
            )
            conversation.history.append(interaction)

        return conversation


class EnhancedConversation(Conversation):
    """
    Enhanced conversation with vector-based retrieval and summarization.

    Extends the base Conversation class with:
    - Vector embeddings for semantic search
    - Semantic follow-up detection
    - Conversation summarization

    Attributes:
        history: List of Interaction objects
        embedder: ConversationEmbedder instance
        interaction_embeddings: List of embedding vectors
        summary: Current conversation summary
        summary_updated_at: Timestamp of last summary update

    Examples:
        >>> conv = EnhancedConversation()
        >>> conv.add_interaction("What is AI?", "AI stands for...")
        >>> relevant = conv.get_relevant_interactions("machine learning")
    """

    def __init__(self):
        """Initialize enhanced conversation with embedder."""
        super().__init__()
        self.embedder = ConversationEmbedder()
        self.interaction_embeddings: List[List[float]] = []
        self.summary = ""
        self.summary_updated_at = 0.0
        logger.debug("Created new EnhancedConversation instance")

    def add_interaction(
        self,
        query: str,
        response: str,
        document_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add interaction with embedding generation.

        Args:
            query: User's query
            response: System's response
            document_ids: Optional list of referenced document IDs
            metadata: Optional metadata
        """
        super().add_interaction(query, response, document_ids, metadata)

        try:
            embedding = self.embedder.embed_interaction(query, response)
            self.interaction_embeddings.append(embedding)
            logger.debug(
                "Added embedding for interaction",
                extra={"total_embeddings": len(self.interaction_embeddings)}
            )
        except Exception as e:
            logger.error(
                "Error creating embedding for interaction",
                extra={"error": str(e)},
                exc_info=True
            )
            self.interaction_embeddings.append([0.0] * 384)

        # Mark summary as outdated
        self.summary_updated_at = 0.0

    def clear(self) -> None:
        """Clear conversation history and embeddings."""
        super().clear()
        self.interaction_embeddings = []
        self.summary = ""
        self.summary_updated_at = 0.0
        logger.info("Cleared enhanced conversation history and embeddings")

    def get_relevant_interactions(
        self,
        query: str,
        top_k: int = 2
    ) -> List[Interaction]:
        """
        Get most relevant previous interactions using vector similarity.

        Args:
            query: Current query
            top_k: Maximum number of relevant interactions

        Returns:
            List of relevant interactions sorted by relevance
        """
        if not self.history or not self.interaction_embeddings:
            return []

        try:
            query_embedding = self.embedder.embed_query(query)

            similar_indices = self.embedder.find_most_similar(
                query_embedding,
                self.interaction_embeddings,
                top_k
            )

            relevant = [self.history[idx] for idx, _ in similar_indices]
            logger.debug(
                "Found relevant interactions",
                extra={
                    "query": query[:100],
                    "num_relevant": len(relevant)
                }
            )
            return relevant
        except Exception as e:
            logger.error(
                "Error finding relevant interactions",
                extra={"error": str(e)},
                exc_info=True
            )
            return self.get_last_n_interactions(top_k)

    def get_most_relevant_content(
        self,
        query: str,
        max_interactions: int = 2
    ) -> str:
        """
        Get most relevant content using vector similarity.

        Overrides base method to use semantic search instead of keywords.

        Args:
            query: Current query
            max_interactions: Maximum interactions to include

        Returns:
            String containing relevant previous content
        """
        if not self.history:
            return ""

        relevant_interactions = self.get_relevant_interactions(
            query, max_interactions
        )

        if not relevant_interactions:
            return ""

        content = "Relevant previous content:\n\n"
        for i, interaction in enumerate(relevant_interactions):
            similarity = "high" if i == 0 else "medium"
            content += f"Content {i+1} (relevance: {similarity}):\n"
            content += f"Query: {interaction.query}\n"
            content += f"Response: {interaction.response}\n\n"

        return content

    def is_follow_up_question(self, query: str) -> bool:
        """
        Determine if query is a follow-up using both regex and semantics.

        Enhances base method with semantic similarity detection.

        Args:
            query: Current query

        Returns:
            True if query appears to be a follow-up
        """
        # First check with regex
        is_regex_follow_up = super().is_follow_up_question(query)

        if is_regex_follow_up:
            return True

        # Check semantic similarity with last interaction
        if self.history and self.interaction_embeddings:
            try:
                last_interaction = self.history[-1]

                query_embedding = self.embedder.embed_query(query)
                last_query_embedding = self.embedder.embed_query(
                    last_interaction.query
                )

                similarity = self.embedder.cosine_similarity(
                    query_embedding, last_query_embedding
                )

                if similarity > 0.8:
                    logger.info(
                        "Detected semantic follow-up question",
                        extra={"similarity": similarity}
                    )
                    return True
            except Exception as e:
                logger.error(
                    "Error in semantic follow-up detection",
                    extra={"error": str(e)},
                    exc_info=True
                )

        return False

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary including embeddings.

        Returns:
            Dictionary representation with embeddings
        """
        base_dict = super().to_dict()

        if self.interaction_embeddings:
            base_dict["embeddings"] = self.interaction_embeddings

        return base_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnhancedConversation':
        """
        Create EnhancedConversation from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            EnhancedConversation instance
        """
        conversation = cls()

        for item in data.get("history", []):
            interaction = Interaction(
                query=item["query"],
                response=item["response"],
                timestamp=item.get("timestamp", time.time()),
                document_ids=item.get("document_ids", []),
                metadata=item.get("metadata", {})
            )
            conversation.history.append(interaction)

        # Load or generate embeddings
        if "embeddings" in data and len(data["embeddings"]) == len(
            conversation.history
        ):
            conversation.interaction_embeddings = data["embeddings"]
        else:
            # Generate embeddings if not available
            conversation.interaction_embeddings = []
            for interaction in conversation.history:
                try:
                    embedding = conversation.embedder.embed_interaction(
                        interaction.query, interaction.response
                    )
                    conversation.interaction_embeddings.append(embedding)
                except Exception as e:
                    logger.error(
                        "Error generating embedding during load",
                        extra={"error": str(e)},
                        exc_info=True
                    )
                    conversation.interaction_embeddings.append([0.0] * 384)

        return conversation


class ConversationManager:
    """
    Main orchestrator for conversation operations.

    Provides a high-level interface for creating and managing conversations,
    including integration with session storage and summarization.

    Attributes:
        summarizer: ConversationSummarizer instance

    Examples:
        >>> manager = ConversationManager()
        >>> conversation = manager.create_conversation(enhanced=True)
        >>> manager.add_interaction(conversation, "Hi", "Hello!")
    """

    def __init__(self):
        """Initialize the conversation manager."""
        self.summarizer = ConversationSummarizer()
        logger.info("Initialized ConversationManager")

    def create_conversation(self, enhanced: bool = True) -> Conversation:
        """
        Create a new conversation instance.

        Args:
            enhanced: If True, creates EnhancedConversation with vector search

        Returns:
            New conversation instance
        """
        if enhanced:
            return EnhancedConversation()
        return Conversation()

    def add_interaction(
        self,
        conversation: Conversation,
        query: str,
        response: str,
        document_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add an interaction to a conversation.

        Args:
            conversation: Target conversation
            query: User's query
            response: System's response
            document_ids: Optional document IDs
            metadata: Optional metadata
        """
        conversation.add_interaction(query, response, document_ids, metadata)

    def get_summary(
        self,
        conversation: Conversation,
        max_tokens: int = 500
    ) -> str:
        """
        Generate a summary of the conversation.

        Args:
            conversation: Conversation to summarize
            max_tokens: Maximum tokens for summary

        Returns:
            Conversation summary
        """
        return self.summarizer.summarize(conversation, max_tokens)

    def get_relevant_context(
        self,
        conversation: Conversation,
        query: str,
        max_interactions: int = 2
    ) -> str:
        """
        Get relevant context for a query.

        Args:
            conversation: Source conversation
            query: Current query
            max_interactions: Maximum relevant interactions

        Returns:
            Relevant context string
        """
        return conversation.get_most_relevant_content(query, max_interactions)


# Session helper functions for Flask integration
def get_conversation_from_session(
    session: Dict[str, Any],
    enhanced: bool = True
) -> Conversation:
    """
    Get or create conversation from Flask session.

    Args:
        session: Flask session object
        enhanced: If True, uses EnhancedConversation

    Returns:
        Conversation instance from session

    Examples:
        >>> from flask import session
        >>> conv = get_conversation_from_session(session)
    """
    conversation_class = EnhancedConversation if enhanced else Conversation

    if "conversation" not in session:
        session["conversation"] = conversation_class().to_dict()
        logger.info(
            "Created new conversation in session",
            extra={"enhanced": enhanced}
        )

    return conversation_class.from_dict(session["conversation"])


def update_conversation_in_session(
    session: Dict[str, Any],
    conversation: Conversation
) -> None:
    """
    Update conversation in Flask session.

    Args:
        session: Flask session object
        conversation: Conversation to store

    Examples:
        >>> from flask import session
        >>> update_conversation_in_session(session, conversation)
    """
    session["conversation"] = conversation.to_dict()
    logger.debug("Updated conversation in session")


def clear_conversation_in_session(session: Dict[str, Any]) -> None:
    """
    Clear conversation from Flask session.

    Args:
        session: Flask session object

    Examples:
        >>> from flask import session
        >>> clear_conversation_in_session(session)
    """
    if "conversation" in session:
        del session["conversation"]
    logger.info("Cleared conversation from session")


# Maintain compatibility with legacy imports
def get_enhanced_conversation_from_session(
    session: Dict[str, Any]
) -> EnhancedConversation:
    """Legacy compatibility: get enhanced conversation from session."""
    return get_conversation_from_session(session, enhanced=True)


def update_enhanced_conversation_in_session(
    session: Dict[str, Any],
    conversation: EnhancedConversation
) -> None:
    """Legacy compatibility: update enhanced conversation in session."""
    update_conversation_in_session(session, conversation)