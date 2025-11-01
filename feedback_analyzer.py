"""
Feedback Analyzer Module for RAG System

This module processes historical user feedback data stored in ConPort to identify patterns
and generate insights for improving retrieval quality.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import statistics
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeedbackAnalyzer:
    """
    Analyzes user feedback patterns to identify successful query characteristics
    and provide recommendations for query enhancement.
    """

    def __init__(self, conport_client=None, workspace_id: str = None):
        """
        Initialize the feedback analyzer.

        Args:
            conport_client: ConPort MCP client for data access
            workspace_id: Workspace identifier for ConPort operations
        """
        self.conport_client = conport_client
        self.workspace_id = workspace_id
        self.feedback_cache = {}
        self.analysis_cache = {}

    def get_feedback_data(self, days_back: int = 30, min_rating: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve feedback data from ConPort storage.

        Args:
            days_back: Number of days to look back for feedback data
            min_rating: Minimum rating to filter by (1-5)

        Returns:
            List of feedback entries
        """
        try:
            if not self.conport_client or not self.workspace_id:
                logger.warning("ConPort client or workspace_id not configured")
                return []

            # Search for feedback data in ConPort
            # Since we don't have a direct way to get all custom data by category,
            # we'll use search to find feedback entries
            search_result = self.conport_client.search_custom_data_value_fts({
                "workspace_id": self.workspace_id,
                "query_term": "rating",
                "category_filter": "UserFeedback",
                "limit": 100
            })

            feedback_entries = []
            if search_result and isinstance(search_result, list):
                for entry in search_result:
                    try:
                        # Parse the feedback data
                        feedback_data = entry.get('value', {})
                        if isinstance(feedback_data, str):
                            feedback_data = json.loads(feedback_data)

                        # Filter by date if specified
                        if days_back > 0:
                            entry_date = datetime.fromisoformat(feedback_data.get('timestamp', ''))
                            cutoff_date = datetime.now() - timedelta(days=days_back)
                            if entry_date < cutoff_date:
                                continue

                        # Filter by minimum rating if specified
                        if min_rating is not None:
                            rating = feedback_data.get('rating', 0)
                            if rating < min_rating:
                                continue

                        feedback_entries.append(feedback_data)

                    except (json.JSONDecodeError, ValueError, KeyError) as e:
                        logger.warning(f"Error parsing feedback entry: {e}")
                        continue

            logger.info(f"Retrieved {len(feedback_entries)} feedback entries")
            return feedback_entries

        except Exception as e:
            logger.error(f"Error retrieving feedback data: {e}")
            return []

    def analyze_rating_patterns(self, feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze patterns in user ratings to identify successful characteristics.

        Args:
            feedback_data: List of feedback entries

        Returns:
            Dictionary containing rating analysis results
        """
        if not feedback_data:
            return {"error": "No feedback data available"}

        try:
            ratings = [entry.get('rating', 0) for entry in feedback_data]
            high_rated = [entry for entry in feedback_data if entry.get('rating', 0) >= 4]
            low_rated = [entry for entry in feedback_data if entry.get('rating', 0) <= 2]

            analysis = {
                "total_feedback": len(feedback_data),
                "average_rating": statistics.mean(ratings) if ratings else 0,
                "rating_distribution": dict(Counter(ratings)),
                "high_rated_count": len(high_rated),
                "low_rated_count": len(low_rated),
                "high_rated_percentage": (len(high_rated) / len(feedback_data)) * 100 if feedback_data else 0
            }

            # Analyze characteristics of high-rated queries
            if high_rated:
                high_rated_queries = [entry.get('query', '') for entry in high_rated]
                analysis["successful_query_characteristics"] = self._analyze_query_characteristics(high_rated_queries)

                # Analyze document types that perform well
                high_rated_docs = []
                for entry in high_rated:
                    docs = entry.get('document_ids', [])
                    high_rated_docs.extend(docs)
                analysis["successful_document_patterns"] = dict(Counter(high_rated_docs))

            # Analyze characteristics of low-rated queries
            if low_rated:
                low_rated_queries = [entry.get('query', '') for entry in low_rated]
                analysis["problematic_query_characteristics"] = self._analyze_query_characteristics(low_rated_queries)

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing rating patterns: {e}")
            return {"error": str(e)}

    def _analyze_query_characteristics(self, queries: List[str]) -> Dict[str, Any]:
        """
        Analyze characteristics of a set of queries.

        Args:
            queries: List of query strings

        Returns:
            Dictionary containing query characteristics
        """
        if not queries:
            return {}

        try:
            # Basic statistics
            query_lengths = [len(query.split()) for query in queries]

            # Common words and phrases
            all_words = []
            for query in queries:
                words = query.lower().split()
                all_words.extend(words)

            common_words = dict(Counter(all_words).most_common(10))

            # Query types (questions vs statements)
            question_indicators = ['what', 'how', 'why', 'when', 'where', 'who', 'which', '?']
            question_count = sum(1 for query in queries
                               if any(indicator in query.lower() for indicator in question_indicators))

            return {
                "total_queries": len(queries),
                "average_length": statistics.mean(query_lengths) if query_lengths else 0,
                "median_length": statistics.median(query_lengths) if query_lengths else 0,
                "common_words": common_words,
                "question_percentage": (question_count / len(queries)) * 100 if queries else 0,
                "sample_queries": queries[:5]  # First 5 queries as examples
            }

        except Exception as e:
            logger.error(f"Error analyzing query characteristics: {e}")
            return {}

    def identify_successful_patterns(self, days_back: int = 30) -> Dict[str, Any]:
        """
        Identify patterns from successful (high-rated) queries and responses.

        Args:
            days_back: Number of days to analyze

        Returns:
            Dictionary containing successful patterns and recommendations
        """
        try:
            # Get feedback data
            feedback_data = self.get_feedback_data(days_back=days_back)

            if not feedback_data:
                return {
                    "status": "no_data",
                    "message": "No feedback data available for analysis",
                    "recommendations": {
                        "query_enhancement": [],
                        "similarity_threshold": 0.7,  # Default threshold
                        "document_preferences": {}
                    }
                }

            # Analyze rating patterns
            rating_analysis = self.analyze_rating_patterns(feedback_data)

            # Generate recommendations based on analysis
            recommendations = self._generate_recommendations(rating_analysis, feedback_data)

            return {
                "status": "success",
                "analysis_period": f"{days_back} days",
                "data_summary": {
                    "total_feedback": rating_analysis.get("total_feedback", 0),
                    "average_rating": rating_analysis.get("average_rating", 0),
                    "high_rated_percentage": rating_analysis.get("high_rated_percentage", 0)
                },
                "successful_patterns": rating_analysis.get("successful_query_characteristics", {}),
                "problematic_patterns": rating_analysis.get("problematic_query_characteristics", {}),
                "recommendations": recommendations
            }

        except Exception as e:
            logger.error(f"Error identifying successful patterns: {e}")
            return {
                "status": "error",
                "message": str(e),
                "recommendations": {
                    "query_enhancement": [],
                    "similarity_threshold": 0.7,
                    "document_preferences": {}
                }
            }

    def _generate_recommendations(self, rating_analysis: Dict[str, Any],
                                feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate recommendations based on feedback analysis.

        Args:
            rating_analysis: Results from rating pattern analysis
            feedback_data: Raw feedback data

        Returns:
            Dictionary containing recommendations
        """
        try:
            recommendations = {
                "query_enhancement": [],
                "similarity_threshold": 0.7,  # Default
                "document_preferences": {},
                "query_expansion_terms": []
            }

            # Extract successful query terms for expansion
            successful_chars = rating_analysis.get("successful_query_characteristics", {})
            if successful_chars and "common_words" in successful_chars:
                # Get top words from successful queries (excluding common stop words)
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
                expansion_terms = [word for word, count in successful_chars["common_words"].items()
                                 if word not in stop_words and len(word) > 2][:10]
                recommendations["query_expansion_terms"] = expansion_terms

            # Analyze similarity thresholds from successful queries
            high_rated = [entry for entry in feedback_data if entry.get('rating', 0) >= 4]
            if high_rated:
                # If we have metadata about similarity scores, we could analyze optimal thresholds
                # For now, we'll use a conservative approach
                if len(high_rated) > len(feedback_data) * 0.6:  # If >60% are high-rated
                    recommendations["similarity_threshold"] = 0.6  # Lower threshold for broader results
                else:
                    recommendations["similarity_threshold"] = 0.75  # Higher threshold for precision

            # Document preference analysis
            successful_docs = rating_analysis.get("successful_document_patterns", {})
            if successful_docs:
                # Identify document types or sources that perform well
                total_successful = sum(successful_docs.values())
                doc_preferences = {doc_id: count/total_successful
                                 for doc_id, count in successful_docs.items()}
                recommendations["document_preferences"] = doc_preferences

            # Query enhancement suggestions
            if successful_chars:
                avg_length = successful_chars.get("average_length", 0)
                question_pct = successful_chars.get("question_percentage", 0)

                if avg_length > 5:
                    recommendations["query_enhancement"].append("Encourage longer, more detailed queries")
                if question_pct > 70:
                    recommendations["query_enhancement"].append("Question-format queries perform better")
                if question_pct < 30:
                    recommendations["query_enhancement"].append("Statement-format queries perform better")

            return recommendations

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return {
                "query_enhancement": [],
                "similarity_threshold": 0.7,
                "document_preferences": {},
                "query_expansion_terms": []
            }

    def get_feedback_summary(self, days_back: int = 7) -> Dict[str, Any]:
        """
        Get a quick summary of recent feedback for monitoring purposes.

        Args:
            days_back: Number of days to summarize

        Returns:
            Dictionary containing feedback summary
        """
        try:
            feedback_data = self.get_feedback_data(days_back=days_back)

            if not feedback_data:
                return {
                    "period": f"Last {days_back} days",
                    "total_feedback": 0,
                    "average_rating": 0,
                    "trend": "no_data"
                }

            ratings = [entry.get('rating', 0) for entry in feedback_data]

            return {
                "period": f"Last {days_back} days",
                "total_feedback": len(feedback_data),
                "average_rating": round(statistics.mean(ratings), 2) if ratings else 0,
                "rating_distribution": dict(Counter(ratings)),
                "trend": "positive" if statistics.mean(ratings) >= 3.5 else "needs_improvement" if statistics.mean(ratings) < 2.5 else "neutral"
            }

        except Exception as e:
            logger.error(f"Error generating feedback summary: {e}")
            return {
                "period": f"Last {days_back} days",
                "total_feedback": 0,
                "average_rating": 0,
                "trend": "error",
                "error": str(e)
            }

# Utility function for external use
def create_feedback_analyzer(conport_client=None, workspace_id: str = None) -> FeedbackAnalyzer:
    """
    Factory function to create a FeedbackAnalyzer instance.

    Args:
        conport_client: ConPort MCP client
        workspace_id: Workspace identifier

    Returns:
        FeedbackAnalyzer instance
    """
    return FeedbackAnalyzer(conport_client=conport_client, workspace_id=workspace_id)