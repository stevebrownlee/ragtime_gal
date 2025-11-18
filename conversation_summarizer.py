"""
conversation_summarizer.py - Summarizes conversation history
"""

import logging
from typing import List
from langchain_community.chat_models import ChatOllama
import os

from conversation import Conversation, Interaction

logger = logging.getLogger(__name__)

class ConversationSummarizer:
    """
    Summarizes conversation history to provide compact context.
    This helps manage token usage by compressing older conversation turns.
    """

    def __init__(self, model: str = None, base_url: str = None):
        """
        Initialize the conversation summarizer.

        Args:
            model: The LLM model to use for summarization
            base_url: The base URL for the Ollama API
        """
        self.model = model or os.getenv('LLM_MODEL', 'ilyr')
        self.base_url = base_url or os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')

        # Initialize LLM for summarization
        self.llm = ChatOllama(
            model=self.model,
            base_url=self.base_url,
            temperature=0.3,  # Lower temperature for more factual summaries
            num_predict=1024  # Shorter generations for summaries
        )
        logger.info(f"Initialized ConversationSummarizer with model: {self.model}")

    def summarize(self, conversation: Conversation, max_tokens: int = 500) -> str:
        """
        Generate a summary of the conversation history.

        Args:
            conversation: The conversation to summarize
            max_tokens: Approximate maximum tokens for the summary

        Returns:
            A summary of the conversation
        """
        if not conversation or not hasattr(conversation, 'get_history'):
            return ""

        history = conversation.get_history()

        # For very short conversations, just format without summarizing
        if len(history) <= 2:
            return self._format_short_conversation(history)

        try:
            # Format the conversation history
            history_text = self._format_full_history(history)

            # Create a summarization prompt
            summary_prompt = f"""
            Summarize the following conversation in a concise way that preserves the key points,
            questions asked, and information provided. Focus on the main topics and any conclusions reached.
            Keep your summary under {max_tokens} tokens.

            {history_text}

            Summary:
            """

            # Generate summary
            response = self.llm.invoke(summary_prompt)
            summary = response.content.strip()

            logger.info(f"Generated conversation summary ({len(summary)} chars)")
            return summary
        except Exception as e:
            logger.error(f"Error summarizing conversation: {str(e)}")
            # Fall back to formatting without LLM summarization
            return self._format_recent_history(history)

    def _format_short_conversation(self, history: List[Interaction]) -> str:
        """
        Format a short conversation without summarization.

        Args:
            history: List of conversation interactions

        Returns:
            Formatted conversation string
        """
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
        """
        Format the full conversation history as a string.

        Args:
            history: List of conversation interactions

        Returns:
            Formatted conversation string
        """
        if not history:
            return ""

        formatted = "Conversation history:\n\n"
        for i, interaction in enumerate(history):
            formatted += f"User: {interaction.query}\n"
            formatted += f"Assistant: {interaction.response}\n"
            if i < len(history) - 1:
                formatted += "\n"

        return formatted

    def _format_recent_history(self, history: List[Interaction], max_recent: int = 3) -> str:
        """
        Format recent history with a brief summary of older interactions.

        Args:
            history: List of conversation interactions
            max_recent: Maximum number of recent interactions to include in full

        Returns:
            Formatted conversation string with recent history
        """
        if not history:
            return ""

        # For short conversations, include everything
        if len(history) <= max_recent:
            return self._format_full_history(history)

        # For longer conversations, include recent interactions and summarize older ones
        recent = history[-max_recent:]
        older = history[:-max_recent]

        summary = "Previous conversation summary:\n"

        # Add a brief summary of older interactions
        if older:
            summary += f"- {len(older)} earlier exchanges occurred.\n"
            # Include the very first interaction for context
            summary += f"- The conversation began with: \"{older[0].query}\"\n"
            # Add a few keywords from older interactions
            topics = set()
            for interaction in older:
                # This is a simple keyword extraction - could be enhanced
                words = interaction.query.lower().split()
                for word in words:
                    if len(word) > 4 and word not in ["about", "would", "could", "should", "there", "their", "these", "those"]:
                        topics.add(word)
            if topics:
                topic_list = list(topics)[:5]  # Limit to 5 topics
                summary += f"- Topics discussed: {', '.join(topic_list)}\n"

        # Add recent interactions in full
        summary += "\nRecent conversation:\n"
        for i, interaction in enumerate(recent):
            summary += f"User: {interaction.query}\n"
            summary += f"Assistant: {interaction.response}\n"
            if i < len(recent) - 1:
                summary += "\n"

        return summary