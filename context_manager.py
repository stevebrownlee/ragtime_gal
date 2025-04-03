"""
context_manager.py - Manages context for conversational RAG
"""

import logging
from typing import Dict, Any, Optional, List

from template_manager import TemplateManager
from query_classifier import QueryClassifier
from conversation_summarizer import ConversationSummarizer

logger = logging.getLogger(__name__)

class ContextManager:
    """
    Manages context for conversational RAG.
    This class handles the selection and formatting of context for different query types.
    """

    def __init__(self, template_manager: Optional[TemplateManager] = None):
        """
        Initialize the context manager.

        Args:
            template_manager: Optional TemplateManager instance. If None, creates a new one.
        """
        self.template_manager = template_manager or TemplateManager()
        self.query_classifier = QueryClassifier()
        self.summarizer = ConversationSummarizer()

        logger.info("Initialized ContextManager with QueryClassifier and ConversationSummarizer")

    def get_context_params(self, query: str, conversation: Any, retrieved_docs: List[Any],
                          classification: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Get context parameters for prompt formatting.

        Args:
            query: The user's query
            conversation: The conversation object
            retrieved_docs: List of retrieved documents
            classification: Optional pre-computed query classification

        Returns:
            Dictionary of context parameters
        """
        # Format retrieved context
        context = "\n".join(doc.page_content for doc in retrieved_docs) if retrieved_docs else "No relevant documents found."

        # Get previous content if available
        previous_content = ""
        if conversation is not None:
            # Use enhanced retrieval if available
            if hasattr(conversation, 'get_relevant_interactions'):
                relevant = conversation.get_relevant_interactions(query, top_k=2)
                if relevant:
                    previous_content = "Relevant previous exchanges:\n\n"
                    for i, interaction in enumerate(relevant):
                        previous_content += f"Exchange {i+1}:\n"
                        previous_content += f"Q: {interaction.query}\n"
                        previous_content += f"A: {interaction.response}\n\n"
            # Fall back to standard retrieval
            elif hasattr(conversation, 'get_most_relevant_content'):
                previous_content = conversation.get_most_relevant_content(query)

        # Get conversation summary if available
        conversation_summary = ""
        if conversation is not None:
            if hasattr(conversation, 'get_summary'):
                # Use built-in summary method if available
                conversation_summary = conversation.get_summary()
            elif hasattr(self.summarizer, 'summarize'):
                # Generate summary using the summarizer
                conversation_summary = self.summarizer.summarize(conversation)

        return {
            "context": context,
            "question": query,
            "previous_content": previous_content,
            "conversation_history": conversation_summary,
            "conversation_summary": conversation_summary
        }

    def get_prompt(self, query: str, conversation: Any, retrieved_docs: List[Any], style: str = "standard") -> Dict[str, Any]:
        """
        Get a formatted prompt and metadata for the given query.

        Args:
            query: The user's query
            conversation: The conversation object
            retrieved_docs: List of retrieved documents
            style: The prompt style (standard, creative, sixthwood)

        Returns:
            Dictionary containing prompt and metadata
        """
        # Classify query using the query classifier
        classification = self.query_classifier.classify(query, conversation)
        query_type = classification["query_type"]

        # Get context parameters
        context_params = self.get_context_params(
            query,
            conversation,
            retrieved_docs,
            classification
        )

        # Get prompt from template manager
        prompt = self.template_manager.get_prompt(style, query_type, context_params)

        # Get system instruction
        system_instruction = self.template_manager.get_system_instruction(style)

        # Add few-shot examples if available (placeholder for future enhancement)
        few_shot_examples = ""

        # Combine system instruction with few-shot examples if needed
        if few_shot_examples and system_instruction:
            system_instruction = f"{system_instruction}\n\n{few_shot_examples}"

        # Return prompt and metadata
        return {
            "prompt": prompt,
            "system_instruction": system_instruction,
            "query_type": query_type,
            "is_follow_up": classification["is_follow_up"],
            "has_previous_content": bool(context_params.get("previous_content")),
            "style": style,
            "classification": classification
        }

    def get_conversation_memory(self, query: str, conversation: Any) -> Dict[str, Any]:
        """
        Get structured memory from conversation history.

        Args:
            query: The user's query
            conversation: The conversation object

        Returns:
            Dictionary with different types of conversation memory
        """
        if not conversation or not hasattr(conversation, 'get_history'):
            return {
                "recent": "",
                "summary": "",
                "relevant": ""
            }

        # Get recent interactions
        recent = ""
        if hasattr(conversation, 'get_last_n_interactions'):
            last_interactions = conversation.get_last_n_interactions(2)
            if last_interactions:
                recent = "Recent exchanges:\n\n"
                for i, interaction in enumerate(last_interactions):
                    recent += f"User: {interaction.query}\n"
                    recent += f"Assistant: {interaction.response}\n"
                    if i < len(last_interactions) - 1:
                        recent += "\n"

        # Get conversation summary
        summary = ""
        if hasattr(self.summarizer, 'summarize'):
            summary = self.summarizer.summarize(conversation)
        elif hasattr(conversation, 'get_summary'):
            summary = conversation.get_summary()

        # Get relevant interactions
        relevant = ""
        if hasattr(conversation, 'get_relevant_interactions'):
            relevant_interactions = conversation.get_relevant_interactions(query, top_k=2)
            if relevant_interactions:
                relevant = "Semantically relevant exchanges:\n\n"
                for i, interaction in enumerate(relevant_interactions):
                    relevant += f"User: {interaction.query}\n"
                    relevant += f"Assistant: {interaction.response}\n"
                    if i < len(relevant_interactions) - 1:
                        relevant += "\n"

        return {
            "recent": recent,
            "summary": summary,
            "relevant": relevant
        }