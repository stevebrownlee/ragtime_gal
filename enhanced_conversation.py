"""
enhanced_conversation.py - Enhanced conversation management with vector-based retrieval
"""

import logging
import time
from typing import List, Dict, Any, Optional

from conversation import Conversation, Interaction
from conversation_embedder import ConversationEmbedder

logger = logging.getLogger(__name__)

class EnhancedConversation(Conversation):
    """
    Enhanced conversation class with vector-based retrieval.
    This extends the base Conversation class with embedding-based retrieval
    for more accurate identification of relevant previous interactions.
    """

    def __init__(self):
        """Initialize the enhanced conversation."""
        super().__init__()
        self.embedder = ConversationEmbedder()
        self.interaction_embeddings = []
        self.summary = ""
        self.summary_updated_at = 0

    def add_interaction(self, query: str, response: str, document_ids: Optional[List[str]] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a new interaction to the conversation history with embedding.

        Args:
            query: The user's query
            response: The system's response
            document_ids: Optional list of document IDs that were referenced
            metadata: Optional metadata about the interaction
        """
        # Add to history using parent method
        super().add_interaction(query, response, document_ids, metadata)

        # Create and store embedding for the interaction
        try:
            embedding = self.embedder.embed_interaction(query, response)
            self.interaction_embeddings.append(embedding)
            logger.info(f"Added embedding for interaction (total: {len(self.interaction_embeddings)})")
        except Exception as e:
            logger.error(f"Error creating embedding for interaction: {str(e)}")
            # Add a zero vector as fallback
            self.interaction_embeddings.append([0.0] * 384)

        # Mark summary as outdated
        self.summary_updated_at = 0

    def clear(self) -> None:
        """Clear the conversation history and embeddings."""
        super().clear()
        self.interaction_embeddings = []
        self.summary = ""
        self.summary_updated_at = 0
        logger.info("Cleared conversation history and embeddings")

    def get_relevant_interactions(self, query: str, top_k: int = 2) -> List[Interaction]:
        """
        Get the most relevant previous interactions using vector similarity.

        Args:
            query: The current query
            top_k: Maximum number of relevant interactions to return

        Returns:
            List of relevant interactions
        """
        if not self.history or not self.interaction_embeddings:
            return []

        try:
            # Embed the query
            query_embedding = self.embedder.embed_query(query)

            # Find most similar interactions
            similar_indices = self.embedder.find_most_similar(
                query_embedding,
                self.interaction_embeddings,
                top_k
            )

            # Return the interactions
            return [self.history[idx] for idx, _ in similar_indices]
        except Exception as e:
            logger.error(f"Error finding relevant interactions: {str(e)}")
            # Fall back to recency-based retrieval
            return self.get_last_n_interactions(top_k)

    def get_most_relevant_content(self, query: str, max_interactions: int = 2) -> str:
        """
        Get the most relevant previous content based on the current query.
        This overrides the base method to use vector similarity instead of keywords.

        Args:
            query: The current query
            max_interactions: Maximum number of relevant interactions to include

        Returns:
            A string containing the most relevant previous content
        """
        if not self.history:
            return ""

        # Get relevant interactions using vector similarity
        relevant_interactions = self.get_relevant_interactions(query, max_interactions)

        if not relevant_interactions:
            return ""

        # Format the relevant content
        content = "Relevant previous content:\n\n"
        for i, interaction in enumerate(relevant_interactions):
            similarity = "high" if i == 0 else "medium"
            content += f"Content {i+1} (relevance: {similarity}):\n"
            content += f"Query: {interaction.query}\n"
            content += f"Response: {interaction.response}\n\n"

        return content

    def is_follow_up_question(self, query: str) -> bool:
        """
        Determine if the current query is a follow-up question.
        This enhances the base method with semantic similarity.

        Args:
            query: The current query

        Returns:
            True if the query appears to be a follow-up question, False otherwise
        """
        # First use the regex-based detection from the parent class
        is_regex_follow_up = super().is_follow_up_question(query)

        # If regex already detected it as a follow-up, return True
        if is_regex_follow_up:
            return True

        # If we have history, check semantic similarity
        if self.history and self.interaction_embeddings:
            try:
                # Get the most recent interaction
                last_interaction = self.history[-1]

                # Embed the current query and the last query
                query_embedding = self.embedder.embed_query(query)
                last_query_embedding = self.embedder.embed_query(last_interaction.query)

                # Calculate similarity
                similarity = self.embedder.cosine_similarity(query_embedding, last_query_embedding)

                # If similarity is high, consider it a follow-up
                if similarity > 0.8:  # Threshold can be adjusted
                    logger.info(f"Detected follow-up question based on semantic similarity: {similarity:.2f}")
                    return True
            except Exception as e:
                logger.error(f"Error in semantic follow-up detection: {str(e)}")

        return False

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the conversation to a dictionary for serialization.
        This extends the base method to include embeddings.

        Returns:
            Dictionary representation of the conversation
        """
        base_dict = super().to_dict()

        # Add embeddings if available
        if self.interaction_embeddings:
            base_dict["embeddings"] = self.interaction_embeddings

        return base_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnhancedConversation':
        """
        Create an EnhancedConversation instance from a dictionary.

        Args:
            data: Dictionary representation of the conversation

        Returns:
            EnhancedConversation instance
        """
        conversation = cls()

        # Load history
        for item in data.get("history", []):
            interaction = Interaction(
                query=item["query"],
                response=item["response"],
                timestamp=item.get("timestamp", time.time()),
                document_ids=item.get("document_ids", []),
                metadata=item.get("metadata", {})
            )
            conversation.history.append(interaction)

        # Load embeddings if available
        if "embeddings" in data and len(data["embeddings"]) == len(conversation.history):
            conversation.interaction_embeddings = data["embeddings"]
        else:
            # Generate embeddings if not available or mismatched
            conversation.interaction_embeddings = []
            for interaction in conversation.history:
                try:
                    embedding = conversation.embedder.embed_interaction(
                        interaction.query, interaction.response
                    )
                    conversation.interaction_embeddings.append(embedding)
                except Exception as e:
                    logger.error(f"Error generating embedding during load: {str(e)}")
                    conversation.interaction_embeddings.append([0.0] * 384)

        return conversation


def get_enhanced_conversation_from_session(session: Dict[str, Any]) -> EnhancedConversation:
    """
    Get the enhanced conversation object from a Flask session.

    Args:
        session: The Flask session object

    Returns:
        An EnhancedConversation instance
    """
    if "conversation" not in session:
        session["conversation"] = EnhancedConversation().to_dict()
        logger.info("Created new enhanced conversation in session")

    return EnhancedConversation.from_dict(session["conversation"])


def update_enhanced_conversation_in_session(session: Dict[str, Any], conversation: EnhancedConversation) -> None:
    """
    Update the enhanced conversation object in a Flask session.

    Args:
        session: The Flask session object
        conversation: The EnhancedConversation instance to store
    """
    session["conversation"] = conversation.to_dict()
    logger.info("Updated enhanced conversation in session")