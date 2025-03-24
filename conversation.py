"""
conversation.py - Module for managing conversation history and context
"""

import logging
import time
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Interaction:
    """Represents a single interaction in a conversation"""
    query: str
    response: str
    timestamp: float = field(default_factory=time.time)
    document_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class Conversation:
    """Manages a conversation session with history and context"""

    def __init__(self):
        self.history: List[Interaction] = []

    def add_interaction(self, query: str, response: str, document_ids: Optional[List[str]] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a new interaction to the conversation history

        Args:
            query: The user's query
            response: The system's response
            document_ids: Optional list of document IDs that were referenced
            metadata: Optional metadata about the interaction (e.g., content_type, tags)
        """
        interaction = Interaction(
            query=query,
            response=response,
            timestamp=time.time(),
            document_ids=document_ids or [],
            metadata=metadata or {}
        )
        self.history.append(interaction)
        logger.info(f"Added interaction to conversation history (total: {len(self.history)})")

    def get_history(self) -> List[Interaction]:
        """Get the full conversation history"""
        return self.history

    def clear(self) -> None:
        """Clear the conversation history"""
        self.history = []
        logger.info("Cleared conversation history")

    def get_last_interaction(self) -> Optional[Interaction]:
        """Get the most recent interaction"""
        if not self.history:
            return None
        return self.history[-1]

    def get_last_n_interactions(self, n: int = 1) -> List[Interaction]:
        """Get the last n interactions"""
        return self.history[-n:] if len(self.history) >= n else self.history[:]

    def find_interactions_by_query_keywords(self, keywords: List[str]) -> List[Interaction]:
        """Find interactions where the query contains any of the given keywords"""
        if not keywords:
            return []

        result = []
        for interaction in self.history:
            if any(keyword.lower() in interaction.query.lower() for keyword in keywords):
                result.append(interaction)

        return result

    def find_interactions_by_metadata(self, **metadata) -> List[Interaction]:
        """Find interactions that match the given metadata criteria"""
        result = []
        for interaction in self.history:
            if all(interaction.metadata.get(key) == value for key, value in metadata.items()):
                result.append(interaction)

        return result

    def get_generated_content(self) -> str:
        """
        Get all generated content from the conversation history

        Returns:
            A string containing all responses from the system
        """
        if not self.history:
            return ""

        content = "Previously generated content:\n\n"
        for i, interaction in enumerate(self.history):
            content += f"Content {i+1}:\nQuery: {interaction.query}\nResponse: {interaction.response}\n\n"

        return content

    def get_most_relevant_content(self, query: str, max_interactions: int = 2) -> str:
        """
        Get the most relevant previous content based on the current query

        Args:
            query: The current query
            max_interactions: Maximum number of relevant interactions to include

        Returns:
            A string containing the most relevant previous content
        """
        if not self.history:
            return ""

        # Extract potential keywords from the query
        # Look for words like "previous", "last", "chapter", "summary", etc.
        reference_indicators = [
            "previous", "last", "earlier", "before", "above",
            "chapter", "section", "paragraph", "summary",
            "you said", "you mentioned", "you wrote", "you generated",
            "that", "it", "this"
        ]

        # Check if the query is likely referring to previous content
        is_reference_query = any(indicator.lower() in query.lower() for indicator in reference_indicators)

        if is_reference_query:
            # For reference queries, prioritize the most recent interactions
            relevant_interactions = self.get_last_n_interactions(max_interactions)

            content = "Relevant previous content:\n\n"
            for i, interaction in enumerate(relevant_interactions):
                content += f"Content {i+1} (most recent):\nQuery: {interaction.query}\nResponse: {interaction.response}\n\n"

            return content
        else:
            # For non-reference queries, return an empty string
            # This will fall back to the standard RAG process
            return ""

    def is_follow_up_question(self, query: str) -> bool:
        """
        Determine if the current query is a follow-up question to previous interactions

        Args:
            query: The current query

        Returns:
            True if the query appears to be a follow-up question, False otherwise
        """
        if not self.history:
            return False

        # Check for pronouns and references that suggest a follow-up
        follow_up_indicators = [
            r"\bit\b", r"\bthis\b", r"\bthat\b", r"\bthese\b", r"\bthose\b",
            r"\bthe\b", r"\byour\b", r"\byou\b", r"\bprevious\b", r"\blast\b",
            r"\babove\b", r"\bearlier\b", r"\bbefore\b"
        ]

        for pattern in follow_up_indicators:
            if re.search(pattern, query, re.IGNORECASE):
                return True

        return False

    def get_summary(self, max_interactions: int = 3) -> str:
        """
        Generate a summary of the conversation history

        Args:
            max_interactions: Maximum number of recent interactions to include in full

        Returns:
            A formatted string containing the conversation summary
        """
        if not self.history:
            return ""

        # For short conversations, include everything
        if len(self.history) <= max_interactions:
            return self._format_full_history()

        # For longer conversations, include recent interactions and summarize older ones
        recent = self.history[-max_interactions:]
        older = self.history[:-max_interactions]

        summary = "Previous conversation summary:\n"

        # Add a brief summary of older interactions
        if older:
            summary += f"- {len(older)} earlier exchanges occurred.\n"
            # Include the very first interaction for context
            summary += f"- The conversation began with: \"{older[0].query}\"\n"

        # Add recent interactions in full
        summary += "\nRecent conversation:\n"
        for i, interaction in enumerate(recent):
            summary += f"User: {interaction.query}\n"
            summary += f"Assistant: {interaction.response}\n"
            if i < len(recent) - 1:
                summary += "\n"

        return summary

    def _format_full_history(self) -> str:
        """Format the full conversation history as a string"""
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
        """Convert the conversation to a dictionary for serialization"""
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
        """Create a Conversation instance from a dictionary"""
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


def get_conversation_from_session(session: Dict[str, Any]) -> Conversation:
    """
    Get the conversation object from a Flask session

    Args:
        session: The Flask session object

    Returns:
        A Conversation instance
    """
    if "conversation" not in session:
        session["conversation"] = Conversation().to_dict()
        logger.info("Created new conversation in session")

    return Conversation.from_dict(session["conversation"])


def update_conversation_in_session(session: Dict[str, Any], conversation: Conversation) -> None:
    """
    Update the conversation object in a Flask session

    Args:
        session: The Flask session object
        conversation: The Conversation instance to store
    """
    session["conversation"] = conversation.to_dict()
    logger.info("Updated conversation in session")


def clear_conversation_in_session(session: Dict[str, Any]) -> None:
    """
    Clear the conversation history in a Flask session

    Args:
        session: The Flask session object
    """
    if "conversation" in session:
        del session["conversation"]
    logger.info("Cleared conversation from session")