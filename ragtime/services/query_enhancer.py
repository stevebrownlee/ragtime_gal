"""
Query Enhancement Service

This service implements intelligent query enhancement and document re-ranking
based on historical user feedback patterns.

Key Features:
- Query expansion with successful terms
- Query reformatting based on patterns
- Adaptive similarity threshold calculation
- Document re-ranking by historical performance
- Enhancement suggestions for users

Migrated from root-level query_enhancer.py to follow project maturity standards:
- Uses structured logging with structlog
- Integrates with centralized Settings
- Enhanced error handling and monitoring
- Supports dependency injection
"""

import structlog
from typing import Dict, List, Any, Optional, Tuple
import re
from collections import defaultdict
import math

# Import from new structure
from ragtime.config.settings import get_settings

# Initialize structured logger
logger = structlog.get_logger(__name__)


class QueryEnhancer:
    """
    Enhances queries and optimizes retrieval based on feedback-driven insights.

    This service learns from user feedback to:
    - Expand queries with successful terms
    - Reformat queries to match successful patterns
    - Calculate adaptive similarity thresholds
    - Re-rank documents based on historical performance
    """

    def __init__(self, feedback_analyzer=None, settings=None):
        """
        Initialize the query enhancer.

        Args:
            feedback_analyzer: FeedbackAnalyzer instance for pattern insights
            settings: Application settings (uses default if not provided)
        """
        self.feedback_analyzer = feedback_analyzer
        self.settings = settings or get_settings()
        self.enhancement_cache = {}
        self.similarity_threshold_cache = {}

        self.logger = logger.bind(
            component="query_enhancer",
            has_feedback_analyzer=bool(feedback_analyzer)
        )

        self.logger.info("query_enhancer_initialized")

    def enhance_query(
        self,
        original_query: str,
        enhancement_mode: str = "auto"
    ) -> Dict[str, Any]:
        """
        Enhance a query based on successful patterns from feedback analysis.

        Args:
            original_query: The original user query
            enhancement_mode: Enhancement strategy ("auto", "expand", "rephrase", "none")

        Returns:
            Dictionary containing enhanced query and metadata
        """
        try:
            if enhancement_mode == "none":
                return {
                    "enhanced_query": original_query,
                    "original_query": original_query,
                    "enhancements_applied": [],
                    "confidence": 1.0
                }

            self.logger.info(
                "enhancing_query",
                mode=enhancement_mode,
                query_length=len(original_query)
            )

            # Get successful patterns from feedback analyzer
            patterns = self._get_successful_patterns()

            # Apply enhancements based on patterns
            enhanced_query = original_query
            enhancements_applied = []

            # Query expansion based on successful terms
            if enhancement_mode in ["auto", "expand"]:
                expanded_query, expansion_terms = self._expand_query(
                    enhanced_query,
                    patterns
                )
                if expansion_terms:
                    enhanced_query = expanded_query
                    enhancements_applied.append(
                        f"expanded_with_terms: {expansion_terms}"
                    )

            # Query reformatting based on successful patterns
            if enhancement_mode in ["auto", "rephrase"]:
                reformatted_query, reformatting_applied = self._reformat_query(
                    enhanced_query,
                    patterns
                )
                if reformatting_applied:
                    enhanced_query = reformatted_query
                    enhancements_applied.extend(reformatting_applied)

            # Calculate confidence based on how many enhancements were applied
            confidence = min(1.0, 0.7 + (len(enhancements_applied) * 0.1))

            result = {
                "enhanced_query": enhanced_query,
                "original_query": original_query,
                "enhancements_applied": enhancements_applied,
                "confidence": confidence
            }

            self.logger.info(
                "query_enhanced",
                enhancements_count=len(enhancements_applied),
                confidence=confidence,
                changed=enhanced_query != original_query
            )

            return result

        except Exception as e:
            self.logger.error(
                "query_enhancement_failed",
                error=str(e),
                exc_info=True
            )
            return {
                "enhanced_query": original_query,
                "original_query": original_query,
                "enhancements_applied": [],
                "confidence": 0.5,
                "error": str(e)
            }

    def _get_successful_patterns(self) -> Dict[str, Any]:
        """
        Get successful patterns from feedback analyzer with caching.

        Returns:
            Dictionary containing successful patterns
        """
        try:
            if not self.feedback_analyzer:
                self.logger.debug("no_feedback_analyzer_using_defaults")
                return {
                    "query_expansion_terms": [],
                    "similarity_threshold": 0.7
                }

            # Use cached patterns if available
            cache_key = "successful_patterns"
            if cache_key in self.enhancement_cache:
                self.logger.debug("using_cached_patterns")
                return self.enhancement_cache[cache_key]

            # Get fresh patterns from feedback analyzer
            analysis_result = self.feedback_analyzer.identify_successful_patterns(
                days_back=30
            )
            patterns = analysis_result.get("recommendations", {})

            # Cache the results
            self.enhancement_cache[cache_key] = patterns

            self.logger.debug(
                "patterns_retrieved",
                expansion_terms_count=len(patterns.get("query_expansion_terms", [])),
                similarity_threshold=patterns.get("similarity_threshold", 0.7)
            )

            return patterns

        except Exception as e:
            self.logger.error(
                "pattern_retrieval_failed",
                error=str(e)
            )
            return {
                "query_expansion_terms": [],
                "similarity_threshold": 0.7
            }

    def _expand_query(
        self,
        query: str,
        patterns: Dict[str, Any]
    ) -> Tuple[str, List[str]]:
        """
        Expand query with terms from successful queries.

        Args:
            query: Original query string
            patterns: Successful patterns from feedback analysis

        Returns:
            Tuple of (expanded_query, expansion_terms_used)
        """
        try:
            expansion_terms = patterns.get("query_expansion_terms", [])
            if not expansion_terms:
                return query, []

            # Analyze query to determine if expansion would be beneficial
            query_words = set(query.lower().split())

            # Find relevant expansion terms not already in query
            relevant_terms = []
            for term in expansion_terms[:5]:  # Limit to top 5 terms
                if term.lower() not in query_words and len(term) > 2:
                    if self._is_term_relevant(query, term):
                        relevant_terms.append(term)

            if not relevant_terms:
                return query, []

            # Add expansion terms strategically
            if query.strip().endswith('?'):
                # For questions, add terms before the question mark
                expanded_query = query.rstrip('?').strip() + \
                    f" {' '.join(relevant_terms)}?"
            else:
                # For statements, append terms
                expanded_query = f"{query} {' '.join(relevant_terms)}"

            self.logger.debug(
                "query_expanded",
                terms_added=len(relevant_terms),
                terms=relevant_terms
            )

            return expanded_query, relevant_terms

        except Exception as e:
            self.logger.error(
                "query_expansion_failed",
                error=str(e)
            )
            return query, []

    def _is_term_relevant(self, query: str, term: str) -> bool:
        """
        Simple relevance check for expansion terms.

        Args:
            query: Original query
            term: Potential expansion term

        Returns:
            Boolean indicating if term is relevant
        """
        try:
            query_lower = query.lower()
            term_lower = term.lower()

            # Skip very common words
            common_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
                'to', 'for', 'of', 'with', 'by'
            }
            if term_lower in common_words:
                return False

            # Skip if term is too similar to existing words
            for word in query.split():
                if abs(len(word) - len(term)) <= 2 and \
                   word.lower().startswith(term_lower[:3]):
                    return False

            # Accept term if it's not too long and seems substantive
            return len(term) <= 15 and len(term) >= 3

        except Exception:
            return False

    def _reformat_query(
        self,
        query: str,
        patterns: Dict[str, Any]
    ) -> Tuple[str, List[str]]:
        """
        Reformat query based on successful patterns.

        Args:
            query: Query to reformat
            patterns: Successful patterns from feedback analysis

        Returns:
            Tuple of (reformatted_query, reformatting_applied)
        """
        try:
            reformatted_query = query
            reformatting_applied = []

            # Get successful query characteristics
            successful_patterns = patterns.get("successful_patterns", {})
            if not successful_patterns:
                return query, []

            question_percentage = successful_patterns.get("question_percentage", 50)
            avg_length = successful_patterns.get("average_length", 5)

            # Convert to question format if successful queries are mostly questions
            if question_percentage > 70 and not query.strip().endswith('?'):
                if not any(q_word in query.lower() for q_word in
                          ['what', 'how', 'why', 'when', 'where', 'who', 'which']):
                    # Add question word based on query content
                    if any(word in query.lower() for word in
                          ['explain', 'describe', 'tell']):
                        reformatted_query = \
                            f"What can you tell me about {query.lower()}?"
                    elif any(word in query.lower() for word in
                            ['process', 'method', 'way']):
                        reformatted_query = f"How does {query.lower()}?"
                    else:
                        reformatted_query = f"What is {query.lower()}?"

                    reformatting_applied.append("converted_to_question")

            # Encourage longer queries if successful ones are longer
            current_length = len(query.split())
            if avg_length > current_length + 2 and current_length < 8:
                context_phrases = [
                    "Please provide detailed information about",
                    "I need comprehensive details on",
                    "Can you explain in detail"
                ]

                if not any(phrase.lower() in query.lower()
                          for phrase in context_phrases):
                    reformatted_query = \
                        f"Please provide detailed information about {query.lower()}"
                    reformatting_applied.append("encouraged_detail")

            if reformatting_applied:
                self.logger.debug(
                    "query_reformatted",
                    reformattings=reformatting_applied
                )

            return reformatted_query, reformatting_applied

        except Exception as e:
            self.logger.error(
                "query_reformatting_failed",
                error=str(e)
            )
            return query, []

    def get_adaptive_similarity_threshold(
        self,
        query: str,
        default_threshold: float = 0.7
    ) -> float:
        """
        Get adaptive similarity threshold based on feedback patterns.

        Args:
            query: The query being processed
            default_threshold: Default threshold if no patterns available

        Returns:
            Optimized similarity threshold
        """
        try:
            patterns = self._get_successful_patterns()
            recommended_threshold = patterns.get(
                "similarity_threshold",
                default_threshold
            )

            # Adjust threshold based on query characteristics
            query_length = len(query.split())

            # Longer queries can use lower thresholds (more specific)
            if query_length > 10:
                recommended_threshold = max(0.5, recommended_threshold - 0.1)
            elif query_length < 3:
                # Short queries need higher thresholds (more precision)
                recommended_threshold = min(0.9, recommended_threshold + 0.1)

            # Ensure threshold is within reasonable bounds
            final_threshold = max(0.3, min(0.95, recommended_threshold))

            self.logger.debug(
                "adaptive_threshold_calculated",
                query_length=query_length,
                threshold=final_threshold
            )

            return final_threshold

        except Exception as e:
            self.logger.error(
                "adaptive_threshold_calculation_failed",
                error=str(e)
            )
            return default_threshold

    def rerank_documents(
        self,
        documents: List[Any],
        query: str,
        original_scores: Optional[List[float]] = None
    ) -> List[Any]:
        """
        Re-rank documents based on historical performance patterns.

        Args:
            documents: List of retrieved documents
            query: Original query
            original_scores: Original similarity scores (if available)

        Returns:
            Re-ranked list of documents
        """
        try:
            if not documents:
                return documents

            patterns = self._get_successful_patterns()
            doc_preferences = patterns.get("document_preferences", {})

            if not doc_preferences:
                self.logger.debug("no_document_preferences_available")
                return documents

            # Create scoring for documents based on historical performance
            scored_docs = []
            for i, doc in enumerate(documents):
                base_score = (
                    original_scores[i]
                    if original_scores and i < len(original_scores)
                    else 0.5
                )

                # Get document ID for preference lookup
                doc_id = None
                if hasattr(doc, 'metadata') and 'id' in doc.metadata:
                    doc_id = doc.metadata['id']
                elif hasattr(doc, 'id'):
                    doc_id = doc.id

                # Apply preference boost
                preference_boost = 0
                if doc_id and doc_id in doc_preferences:
                    # Up to 20% boost based on historical performance
                    preference_boost = doc_preferences[doc_id] * 0.2

                final_score = base_score + preference_boost
                scored_docs.append((final_score, doc))

            # Sort by final score (descending)
            scored_docs.sort(key=lambda x: x[0], reverse=True)

            # Return re-ranked documents
            reranked_docs = [doc for score, doc in scored_docs]

            self.logger.info(
                "documents_reranked",
                count=len(documents),
                preferences_applied=len([d for d in documents
                                        if hasattr(d, 'metadata') and
                                        d.metadata.get('id') in doc_preferences])
            )

            return reranked_docs

        except Exception as e:
            self.logger.error(
                "document_reranking_failed",
                error=str(e),
                exc_info=True
            )
            return documents

    def get_enhancement_suggestions(self, query: str) -> Dict[str, Any]:
        """
        Get suggestions for query enhancement without applying them.

        Args:
            query: Query to analyze

        Returns:
            Dictionary containing enhancement suggestions
        """
        try:
            patterns = self._get_successful_patterns()
            suggestions = {
                "original_query": query,
                "suggestions": [],
                "confidence": 0.5
            }

            # Analyze current query
            query_length = len(query.split())
            is_question = query.strip().endswith('?')

            # Generate suggestions based on patterns
            if patterns.get("query_expansion_terms"):
                expansion_terms = patterns["query_expansion_terms"][:3]
                suggestions["suggestions"].append({
                    "type": "expansion",
                    "description": f"Consider adding terms: {', '.join(expansion_terms)}",
                    "example": f"{query} {' '.join(expansion_terms)}"
                })

            successful_patterns = patterns.get("successful_patterns", {})
            if successful_patterns:
                question_pct = successful_patterns.get("question_percentage", 50)
                avg_length = successful_patterns.get("average_length", 5)

                if question_pct > 70 and not is_question:
                    suggestions["suggestions"].append({
                        "type": "format",
                        "description": "Questions tend to get better results",
                        "example": f"What can you tell me about {query}?"
                    })

                if avg_length > query_length + 2:
                    suggestions["suggestions"].append({
                        "type": "detail",
                        "description": "More detailed queries tend to perform better",
                        "example": f"Please provide comprehensive information about {query}"
                    })

            suggestions["confidence"] = min(
                1.0,
                0.6 + (len(suggestions["suggestions"]) * 0.15)
            )

            self.logger.debug(
                "enhancement_suggestions_generated",
                suggestion_count=len(suggestions["suggestions"]),
                confidence=suggestions["confidence"]
            )

            return suggestions

        except Exception as e:
            self.logger.error(
                "enhancement_suggestions_failed",
                error=str(e)
            )
            return {
                "original_query": query,
                "suggestions": [],
                "confidence": 0.5,
                "error": str(e)
            }

    def clear_cache(self):
        """Clear enhancement caches to force fresh analysis."""
        self.enhancement_cache.clear()
        self.similarity_threshold_cache.clear()
        self.logger.info("query_enhancer_cache_cleared")


# Factory function for backward compatibility and external use
def create_query_enhancer(feedback_analyzer=None) -> QueryEnhancer:
    """
    Factory function to create a QueryEnhancer instance.

    Args:
        feedback_analyzer: FeedbackAnalyzer instance

    Returns:
        QueryEnhancer instance
    """
    enhancer = QueryEnhancer(feedback_analyzer=feedback_analyzer)

    logger.info(
        "query_enhancer_created",
        has_feedback_analyzer=bool(feedback_analyzer)
    )

    return enhancer