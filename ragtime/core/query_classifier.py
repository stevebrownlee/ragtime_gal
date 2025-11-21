"""
Query Classifier for RAG System

Classifies queries to determine the appropriate context handling strategy.
This enables more sophisticated query understanding beyond simple regex patterns.

Migrated from root query_classifier.py to ragtime.core package.
"""

import logging
import re
from typing import Dict, Any, Optional

from ragtime.services.conversation import ConversationEmbedder

logger = logging.getLogger(__name__)

class QueryClassifier:
    """
    Classifies queries to determine the appropriate context handling strategy.
    This enables more sophisticated query understanding beyond simple regex patterns.
    """

    def __init__(self, embedder: Optional[ConversationEmbedder] = None):
        """
        Initialize the query classifier.

        Args:
            embedder: Optional ConversationEmbedder instance. If None, creates a new one.
        """
        self.embedder = embedder or ConversationEmbedder()

        # Regex patterns for follow-up detection
        self.follow_up_indicators = [
            r"\bit\b", r"\bthis\b", r"\bthat\b", r"\bthese\b", r"\bthose\b",
            r"\bthe\b", r"\byour\b", r"\byou\b", r"\bprevious\b", r"\blast\b",
            r"\babove\b", r"\bearlier\b", r"\bbefore\b", r"\bmentioned\b"
        ]

        # Keywords that suggest a new topic
        self.new_topic_indicators = [
            r"\bnew\b", r"\bdifferent\b", r"\bchange\b", r"\bswitch\b",
            r"\bunrelated\b", r"\banother\b", r"\bseparate\b"
        ]

        # Question types
        self.question_types = {
            "clarification": [r"\bclarify\b", r"\bexplain\b", r"\bwhat do you mean\b", r"\bwhat does that mean\b"],
            "elaboration": [r"\bmore\b", r"\belaborate\b", r"\bexpand\b", r"\bdetail\b", r"\btell me more\b"],
            "confirmation": [r"\bconfirm\b", r"\bis that\b", r"\bare you\b", r"\bdoes that\b", r"\bdo you\b"],
            "comparison": [r"\bcompare\b", r"\bdifference\b", r"\bsimilar\b", r"\bversus\b", r"\bvs\b"],
            "example": [r"\bexample\b", r"\binstance\b", r"\bsample\b", r"\billustrate\b", r"\bshow me\b"]
        }

        logger.info("Initialized QueryClassifier")

    def classify(self, query: str, conversation: Any) -> Dict[str, Any]:
        """
        Classify a query based on its relationship to conversation history.

        Args:
            query: The user's query
            conversation: The conversation object

        Returns:
            Dictionary with classification results
        """
        # Default classification
        classification = {
            "query_type": "initial",
            "is_follow_up": False,
            "confidence": 0.0,
            "question_type": None,
            "requires_previous_content": False
        }

        # Check if there's conversation history
        has_conversation = (conversation is not None and
                           hasattr(conversation, 'get_history') and
                           len(conversation.get_history()) > 0)

        if not has_conversation:
            return classification

        # Get the last interaction
        last_interaction = conversation.get_history()[-1]

        # Check for regex patterns indicating a follow-up
        regex_follow_up = False
        for pattern in self.follow_up_indicators:
            if re.search(pattern, query, re.IGNORECASE):
                regex_follow_up = True
                break

        # Check for new topic indicators
        new_topic = False
        for pattern in self.new_topic_indicators:
            if re.search(pattern, query, re.IGNORECASE):
                new_topic = True
                break

        # If explicit new topic, override follow-up detection
        if new_topic:
            regex_follow_up = False

        # Determine question type
        question_type = None
        for qtype, patterns in self.question_types.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    question_type = qtype
                    break
            if question_type:
                break

        # Calculate semantic similarity if we have embeddings
        semantic_similarity = 0.0
        if hasattr(conversation, 'interaction_embeddings') and conversation.interaction_embeddings:
            try:
                # Embed the current query
                query_embedding = self.embedder.embed_query(query)

                # Get the last query embedding
                if hasattr(conversation, 'get_relevant_interactions'):
                    # For enhanced conversation with vector retrieval
                    relevant = conversation.get_relevant_interactions(query, top_k=1)
                    if relevant:
                        last_query = relevant[0].query
                        last_query_embedding = self.embedder.embed_query(last_query)
                        semantic_similarity = self.embedder.cosine_similarity(query_embedding, last_query_embedding)
                else:
                    # For regular conversation
                    last_query_embedding = self.embedder.embed_query(last_interaction.query)
                    semantic_similarity = self.embedder.cosine_similarity(query_embedding, last_query_embedding)
            except Exception as e:
                logger.error(f"Error calculating semantic similarity: {str(e)}")

        # Determine if this is a follow-up based on combined evidence
        is_follow_up = regex_follow_up or semantic_similarity > 0.8

        # Determine confidence
        if regex_follow_up and semantic_similarity > 0.8:
            confidence = 0.95  # Very high confidence
        elif regex_follow_up:
            confidence = 0.8   # High confidence
        elif semantic_similarity > 0.8:
            confidence = 0.7   # Medium-high confidence
        elif semantic_similarity > 0.6:
            confidence = 0.5   # Medium confidence
        else:
            confidence = 0.3   # Low confidence

        # Determine query type
        if is_follow_up:
            query_type = "follow_up"
        elif has_conversation:
            query_type = "with_previous_content"
        else:
            query_type = "initial"

        # Update classification
        classification.update({
            "query_type": query_type,
            "is_follow_up": is_follow_up,
            "confidence": confidence,
            "question_type": question_type,
            "semantic_similarity": semantic_similarity,
            "requires_previous_content": is_follow_up or semantic_similarity > 0.6
        })

        logger.info(f"Classified query as {query_type} (confidence: {confidence:.2f}, similarity: {semantic_similarity:.2f})")
        return classification