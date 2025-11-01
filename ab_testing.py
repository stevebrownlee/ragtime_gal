
"""
A/B Testing Framework for Embedding Model Comparison

This module implements a comprehensive A/B testing system to compare
original vs fine-tuned embedding models with statistical significance testing.
"""

import logging
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict
import random
import hashlib
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ABTestConfig:
    """Configuration for A/B testing"""
    test_name: str
    model_a_name: str  # Original model
    model_b_name: str  # Fine-tuned model
    traffic_split: float = 0.5  # Percentage of traffic to model B
    min_samples_per_variant: int = 100
    significance_level: float = 0.05
    test_duration_days: int = 7
    metrics_to_track: List[str] = None
    gradual_rollout: bool = True
    rollout_schedule: List[Tuple[str, float]] = None  # (date, traffic_percentage)

@dataclass
class QueryResult:
    """Result of a single query execution"""
    query_id: str
    query_text: str
    model_variant: str  # 'A' or 'B'
    timestamp: datetime
    response_time_ms: float
    similarity_scores: List[float]
    retrieved_documents: List[str]
    user_rating: Optional[int] = None
    user_feedback: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ABTestMetrics:
    """Metrics for A/B test evaluation"""
    variant: str
    total_queries: int
    avg_response_time: float
    avg_similarity_score: float
    avg_user_rating: float
    user_satisfaction_rate: float  # % of ratings >= 4
    precision_at_k: Dict[int, float]  # Precision at different k values
    recall_at_k: Dict[int, float]
    ndcg_at_k: Dict[int, float]  # Normalized Discounted Cumulative Gain
    conversion_rate: float  # % of queries leading to positive feedback
    error_rate: float

class ABTestingFramework:
    """
    Comprehensive A/B testing framework for embedding model comparison.
    """

    def __init__(self, config: ABTestConfig, storage_path: str = "./ab_test_data"):
        """
        Initialize the A/B testing framework.

        Args:
            config: A/B test configuration
            storage_path: Path to store test data and results
        """
        self.config = config
        self.storage_path = storage_path
        self.test_data_path = os.path.join(storage_path, f"test_{config.test_name}")

        # Initialize metrics tracking
        if config.metrics_to_track is None:
            self.config.metrics_to_track = [
                'response_time', 'similarity_score', 'user_rating',
                'precision', 'recall', 'ndcg'
            ]

        # Create storage directories
        os.makedirs(self.test_data_path, exist_ok=True)
        os.makedirs(os.path.join(self.test_data_path, "results"), exist_ok=True)
        os.makedirs(os.path.join(self.test_data_path, "reports"), exist_ok=True)

        # Initialize data storage
        self.results_storage = []
        self.current_traffic_split = 0.0  # Start with 0% traffic to model B

        # Load existing test data if available
        self._load_existing_data()

    def _load_existing_data(self):
        """Load existing test data from storage."""
        try:
            results_file = os.path.join(self.test_data_path, "query_results.json")
            if os.path.exists(results_file):
                with open(results_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Convert back to QueryResult objects
                for item in data:
                    item['timestamp'] = datetime.fromisoformat(item['timestamp'])
                    result = QueryResult(**item)
                    self.results_storage.append(result)

                logger.info(f"Loaded {len(self.results_storage)} existing test results")

        except Exception as e:
            logger.warning(f"Could not load existing test data: {e}")

    def _save_results(self):
        """Save current results to storage."""
        try:
            results_file = os.path.join(self.test_data_path, "query_results.json")

            # Convert QueryResult objects to dictionaries for JSON serialization
            serializable_data = []
            for result in self.results_storage:
                data = asdict(result)
                data['timestamp'] = result.timestamp.isoformat()
                serializable_data.append(data)

            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Error saving results: {e}")

    def assign_variant(self, query_id: str, user_id: str = None) -> str:
        """
        Assign a variant (A or B) to a query based on current traffic split.

        Args:
            query_id: Unique identifier for the query
            user_id: Optional user identifier for consistent assignment

        Returns:
            Variant assignment ('A' or 'B')
        """
        try:
            # Use consistent hashing for user-based assignment
            if user_id:
                hash_input = f"{user_id}_{self.config.test_name}"
            else:
                hash_input = f"{query_id}_{self.config.test_name}"

            # Create hash and convert to probability
            hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
            probability = (hash_value % 10000) / 10000.0

            # Assign based on current traffic split
            if probability < self.current_traffic_split:
                return 'B'
            else:
                return 'A'

        except Exception as e:
            logger.error(f"Error assigning variant: {e}")
            return 'A'  # Default to variant A

    def update_traffic_split(self, new_split: float) -> bool:
        """
        Update the traffic split for gradual rollout.

        Args:
            new_split: New traffic percentage for variant B (0.0 to 1.0)

        Returns:
            True if successful, False otherwise
        """
        try:
            if not 0.0 <= new_split <= 1.0:
                raise ValueError("Traffic split must be between 0.0 and 1.0")

            old_split = self.current_traffic_split
            self.current_traffic_split = new_split

            logger.info(f"Updated traffic split from {old_split:.1%} to {new_split:.1%}")

            # Save configuration update
            config_file = os.path.join(self.test_data_path, "current_config.json")
            config_data = asdict(self.config)
            config_data['current_traffic_split'] = new_split
            config_data['last_updated'] = datetime.now().isoformat()

            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2)

            return True

        except Exception as e:
            logger.error(f"Error updating traffic split: {e}")
            return False

    def record_query_result(self, query_result: QueryResult):
        """
        Record the result of a query execution.

        Args:
            query_result: Query result to record
        """
        try:
            self.results_storage.append(query_result)

            # Periodically save results (every 10 queries)
            if len(self.results_storage) % 10 == 0:
                self._save_results()

        except Exception as e:
            logger.error(f"Error recording query result: {e}")

    def calculate_metrics(self, variant: str, start_date: datetime = None,
                         end_date: datetime = None) -> ABTestMetrics:
        """
        Calculate comprehensive metrics for a specific variant.

        Args:
            variant: Variant to calculate metrics for ('A' or 'B')
            start_date: Start date for metric calculation
            end_date: End date for metric calculation

        Returns:
            ABTestMetrics object with calculated metrics
        """
        try:
            # Filter results by variant and date range
            filtered_results = [
                r for r in self.results_storage
                if r.model_variant == variant
            ]

            if start_date:
                filtered_results = [r for r in filtered_results if r.timestamp >= start_date]
            if end_date:
                filtered_results = [r for r in filtered_results if r.timestamp <= end_date]

            if not filtered_results:
                return ABTestMetrics(
                    variant=variant,
                    total_queries=0,
                    avg_response_time=0.0,
                    avg_similarity_score=0.0,
                    avg_user_rating=0.0,
                    user_satisfaction_rate=0.0,
                    precision_at_k={},
                    recall_at_k={},
                    ndcg_at_k={},
                    conversion_rate=0.0,
                    error_rate=0.0
                )

            # Basic metrics
            total_queries = len(filtered_results)
            response_times = [r.response_time_ms for r in filtered_results if r.response_time_ms is not None]
            similarity_scores = []
            for r in filtered_results:
                if r.similarity_scores:
                    similarity_scores.extend(r.similarity_scores)

            user_ratings = [r.user_rating for r in filtered_results if r.user_rating is not None]

            # Calculate averages
            avg_response_time = np.mean(response_times) if response_times else 0.0
            avg_similarity_score = np.mean(similarity_scores) if similarity_scores else 0.0
            avg_user_rating = np.mean(user_ratings) if user_ratings else 0.0

            # User satisfaction (ratings >= 4)
            satisfied_users = sum(1 for rating in user_ratings if rating >= 4)
            user_satisfaction_rate = satisfied_users / len(user_ratings) if user_ratings else 0.0

            # Conversion rate (positive feedback)
            positive_feedback = sum(1 for r in filtered_results
                                  if r.user_rating and r.user_rating >= 4)
            conversion_rate = positive_feedback / total_queries if total_queries > 0 else 0.0

            # Error rate (queries with no results or errors)
            errors = sum(1 for r in filtered_results
                        if not r.retrieved_documents or len(r.retrieved_documents) == 0)
            error_rate = errors / total_queries if total_queries > 0 else 0.0

            # Precision, Recall, and NDCG at different k values
            k_values = [1, 3, 5, 10]
            precision_at_k = {}
            recall_at_k = {}
            ndcg_at_k = {}

            for k in k_values:
                precision_scores = []
                recall_scores = []
                ndcg_scores = []

                for result in filtered_results:
                    if result.user_rating and result.retrieved_documents:
                        # Simple relevance: rating >= 4 means relevant
                        relevant = result.user_rating >= 4
                        retrieved_k = result.retrieved_documents[:k]

                        if retrieved_k:
                            # Precision@k: assume first document is most relevant
                            precision = 1.0 if relevant else 0.0
                            precision_scores.append(precision)

                            # Recall@k: simplified calculation
                            recall = 1.0 if relevant else 0.0
                            recall_scores.append(recall)

                            # NDCG@k: simplified calculation
                            if relevant:
                                # DCG = relevance / log2(position + 1)
                                dcg = 1.0 / np.log2(2)  # Position 1
                                idcg = 1.0 / np.log2(2)  # Ideal DCG
                                ndcg = dcg / idcg
                            else:
                                ndcg = 0.0
                            ndcg_scores.append(ndcg)

                precision_at_k[k] = np.mean(precision_scores) if precision_scores else 0.0
                recall_at_k[k] = np.mean(recall_scores) if recall_scores else 0.0
                ndcg_at_k[k] = np.mean(ndcg_scores) if ndcg_scores else 0.0

            return ABTestMetrics(
                variant=variant,
                total_queries=total_queries,
                avg_response_time=avg_response_time,
                avg_similarity_score=avg_similarity_score,
                avg_user_rating=avg_user_rating,
                user_satisfaction_rate=user_satisfaction_rate,
                precision_at_k=precision_at_k,
                recall_at_k=recall_at_k,
                ndcg_at_k=ndcg_at_k,
                conversion_rate=conversion_rate,
                error_rate=error_rate
            )

        except Exception as e:
            logger.error(f"Error calculating metrics for variant {variant}: {e}")
            return ABTestMetrics(
                variant=variant,
                total_queries=0,
                avg_response_time=0.0,
                avg_similarity_score=0.0,
                avg_user_rating=0.0,
                user_satisfaction_rate=0.0,
                precision_at_k={},
                recall_at_k={},
                ndcg_at_k={},
                conversion_rate=0.0,
                error_rate=0.0
            )

    def perform_statistical_test(self, metric_name: str,
                               confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Perform statistical significance testing between variants.

        Args:
            metric_name: Name of the metric to test
            confidence_level: Confidence level for the test

        Returns:
            Dictionary containing test results
        """
        try:
            # Get data for both variants
            variant_a_data = [r for r in self.results_storage if r.model_variant == 'A']
            variant_b_data = [r for r in self.results_storage if r.model_variant == 'B']

            if len(variant_a_data) < self.config.min_samples_per_variant or \
               len(variant_b_data) < self.config.min_samples_per_variant:
                return {
                    "test_name": metric_name,
                    "sufficient_data": False,
                    "variant_a_samples": len(variant_a_data),
                    "variant_b_samples": len(variant_b_data),
                    "min_required": self.config.min_samples_per_variant
                }

            # Extract metric values
            def extract_metric_values(data, metric):
                values = []
                for result in data:
                    if metric == 'response_time':
                        if result.response_time_ms is not None:
                            values.append(result.response_time_ms)
                    elif metric == 'similarity_score':
                        if result.similarity_scores:
                            values.extend(result.similarity_scores)
                    elif metric == 'user_rating':
                        if result.user_rating is not None:
                            values.append(result.user_rating)
                    elif metric == 'conversion':
                        # Binary: 1 if rating >= 4, 0 otherwise
                        if result.user_rating is not None:
                            values.append(1 if result.user_rating >= 4 else 0)
                return values

            values_a = extract_metric_values(variant_a_data, metric_name)
            values_b = extract_metric_values(variant_b_data, metric_name)

            if not values_a or not values_b:
                return {
                    "test_name": metric_name,
                    "sufficient_data": False,
                    "error": "No metric values found for one or both variants"
                }

            # Perform appropriate statistical test
            if metric_name == 'conversion':
                # Use Chi-square test for binary outcomes
                successes_a = sum(values_a)
                successes_b = sum(values_b)
                total_a = len(values_a)
                total_b = len(values_b)

                # Create contingency table
                observed = np.array([[successes_a, total_a - successes_a],
                                   [successes_b, total_b - successes_b]])

                chi2, p_value = stats.chi2_contingency(observed)[:2]
                test_statistic = chi2
                test_type = "Chi-square"

                effect_size = (successes_b / total_b) - (successes_a / total_a)

            else:
                # Use t-test for continuous metrics
                t_stat, p_value = stats.ttest_ind(values_b, values_a)
                test_statistic = t_stat
                test_type = "Independent t-test"

                # Calculate effect size (Cohen's d)
                pooled_std = np.sqrt(((len(values_a) - 1) * np.var(values_a, ddof=1) +
                                    (len(values_b) - 1) * np.var(values_b, ddof=1)) /
                                   (len(values_a) + len(values_b) - 2))
                effect_size = (np.mean(values_b) - np.mean(values_a)) / pooled_std if pooled_std > 0 else 0

            # Determine significance
            alpha = 1 - confidence_level
            is_significant = p_value < alpha

            # Calculate confidence interval for the difference
            if metric_name != 'conversion':
                se_diff = np.sqrt(np.var(values_a, ddof=1) / len(values_a) +
                                np.var(values_b, ddof=1) / len(values_b))
                t_critical = stats.t.ppf(1 - alpha/2, len(values_a) + len(values_b) - 2)
                margin_error = t_critical * se_diff
                mean_diff = np.mean(values_b) - np.mean(values_a)
                ci_lower = mean_diff - margin_error
                ci_upper = mean_diff + margin_error
            else:
                # For proportions
                p_a = sum(values_a) / len(values_a)
                p_b = sum(values_b) / len(values_b)
                se_diff = np.sqrt(p_a * (1 - p_a) / len(values_a) +
                                p_b * (1 - p_b) / len(values_b))
                z_critical = stats.norm.ppf(1 - alpha/2)
                margin_error = z_critical * se_diff
                mean_diff = p_b - p_a
                ci_lower = mean_diff - margin_error
                ci_upper = mean_diff + margin_error

            return {
                "test_name": metric_name,
                "test_type": test_type,
                "sufficient_data": True,
                "variant_a_samples": len(values_a),
                "variant_b_samples": len(values_b),
                "variant_a_mean": np.mean(values_a),
                "variant_b_mean": np.mean(values_b),
                "test_statistic": test_statistic,
                "p_value": p_value,
                "is_significant": is_significant,
                "confidence_level": confidence_level,
                "effect_size": effect_size,
                "mean_difference": np.mean(values_b) - np.mean(values_a),
                "confidence_interval": [ci_lower, ci_upper],
                "interpretation": self._interpret_test_result(is_significant, effect_size, metric_name)
            }

        except Exception as e:
            logger.error(f"Error performing statistical test for {metric_name}: {e}")
            return {
                "test_name": metric_name,
                "error": str(e),
                "sufficient_data": False
            }

    def _interpret_test_result(self, is_significant: bool, effect_size: float,
                             metric_name: str) -> str:
        """Interpret the statistical test result."""
        if not is_significant:
            return "No statistically significant difference detected between variants."

        direction = "higher" if effect_size > 0 else "lower"
        magnitude = "large" if abs(effect_size) > 0.8 else "medium" if abs(effect_size) > 0.5 else "small"

        return f"Variant B shows a statistically significant {direction} {metric_name} with a {magnitude} effect size."

    def generate_comparison_report(self, output_path: str = None) -> Dict[str, Any]:
        """
        Generate a comprehensive comparison report between variants.

        Args:
            output_path: Path to save the report

        Returns:
            Dictionary containing the complete report
        """
        try:
            if output_path is None:
                output_path = os.path.join(self.test_data_path, "reports",
                                         f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

            # Calculate metrics for both variants
            metrics_a = self.calculate_metrics('A')
            metrics_b = self.calculate_metrics('B')

            # Perform statistical tests
            statistical_tests = {}
            test_metrics = ['response_time', 'similarity_score', 'user_rating', 'conversion']

            for metric in test_metrics:
                statistical_tests[metric] = self.perform_statistical_test(metric)

            # Compile report
            report = {
                "test_configuration": asdict(self.config),
                "report_generated": datetime.now().isoformat(),
                "test_duration": {
                    "start_date": min(r.timestamp for r in self.results_storage).isoformat() if self.results_storage else None,
                    "end_date": max(r.timestamp for r in self.results_storage).isoformat() if self.results_storage else None,
                    "total_days": (max(r.timestamp for r in self.results_storage) -
                                 min(r.timestamp for r in self.results_storage)).days if self.results_storage else 0
                },
                "variant_metrics": {
                    "A": asdict(metrics_a),
                    "B": asdict(metrics_b)
                },
                "statistical_tests": statistical_tests,
                "summary": self._generate_summary(metrics_a, metrics_b, statistical_tests),
                "recommendations": self._generate_recommendations(metrics_a, metrics_b, statistical_tests)
            }

            # Save report
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            logger.info(f"Comparison report generated: {output_path}")
            return report

        except Exception as e:
            logger.error(f"Error generating comparison report: {e}")
            return {"error": str(e)}

    def _generate_summary(self, metrics_a: ABTestMetrics, metrics_b: ABTestMetrics,
                         statistical_tests: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of the A/B test results."""
        try:
            significant_improvements = []
            significant_degradations = []

            for metric, test_result in statistical_tests.items():
                if test_result.get('is_significant', False):
                    if test_result.get('effect_size', 0) > 0:
                        significant_improvements.append(metric)
                    else:
                        significant_degradations.append(metric)

            # Overall recommendation
            if len(significant_improvements) > len(significant_degradations):
                overall_recommendation = "DEPLOY"
                confidence = "HIGH" if len(significant_improvements) >= 2 else "MEDIUM"
            elif len(significant_degradations) > len(significant_improvements):
                overall_recommendation = "DO_NOT_DEPLOY"
                confidence = "HIGH" if len(significant_degradations) >= 2 else "MEDIUM"
            else:
                overall_recommendation = "INCONCLUSIVE"
                confidence = "LOW"

            return {
                "total_queries_tested": metrics_a.total_queries + metrics_b.total_queries,
                "test_balance": {
                    "variant_a_percentage": metrics_a.total_queries / (metrics_a.total_queries + metrics_b.total_queries) * 100 if (metrics_a.total_queries + metrics_b.total_queries) > 0 else 0,
                    "variant_b_percentage": metrics_b.total_queries / (metrics_a.total_queries + metrics_b.total_queries) * 100 if (metrics_a.total_queries + metrics_b.total_queries) > 0 else 0
                },
                "significant_improvements": significant_improvements,
                "significant_degradations": significant_degradations,
                "overall_recommendation": overall_recommendation,
                "confidence_level": confidence,
                "key_findings": [
                    f"Variant B processed {metrics_b.total_queries} queries vs {metrics_a.total_queries} for Variant A",
                    f"Average user rating: A={metrics_a.avg_user_rating:.2f}, B={metrics_b.avg_user_rating:.2f}",
                    f"User satisfaction rate: A={metrics_a.user_satisfaction_rate:.1%}, B={metrics_b.user_satisfaction_rate:.1%}",
                    f"Average response time: A={metrics_a.avg_response_time:.1f}ms, B={metrics_b.avg_response_time:.1f}ms"
                ]
            }

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {"error": str(e)}

    def _generate_recommendations(self, metrics_a: ABTestMetrics, metrics_b: ABTestMetrics,
                                statistical_tests: Dict[str, Any]) -> Dict[str, Any]:
        """Generate actionable recommendations based on test results."""
        try:
            recommendations = {
                "deployment_decision": "HOLD",
                "reasoning": [],
                "next_steps": [],
                "risk_assessment": "MEDIUM",
                "rollout_strategy": None
            }

            # Analyze each metric
            user_rating_test = statistical_tests.get('user_rating', {})
            response_time_test = statistical_tests.get('response_time', {})
            conversion_test = statistical_tests.get('conversion', {})

            # Decision logic
            positive_signals = 0
            negative_signals = 0

            if user_rating_test.get('is_significant') and user_rating_test.get('effect_size', 0) > 0:
                positive_signals += 2  # User rating is very important
                recommendations["reasoning"].append("Significant improvement in user ratings")
            elif user_rating_test.get('is_significant') and user_rating_test.get('effect_size', 0) < 0:
                negative_signals += 2
                recommendations["reasoning"].append("Significant degradation in user ratings")

            if conversion_test.get('is_significant') and conversion_test.get('effect_size', 0) > 0:
                positive_signals += 2  # Conversion is very important
                recommendations["reasoning"].append("Significant improvement in conversion rate")
            elif conversion_test.get('is_significant') and conversion_test.get('effect_size', 0) < 0:
                negative_signals += 2
                recommendations["reasoning"].append("Significant degradation in conversion rate")

            if response_time_test.get('is_significant'):
                if response_time_test.get('effect_size', 0) < 0:  # Lower response time is better
                    positive_signals += 1
                    recommendations["reasoning"].append("Significant improvement in response time")
                else:
                    negative_signals += 1
                    recommendations["reasoning"].append("Significant increase in response time")

            # Make deployment decision
            if positive_signals >= 2 and negative_signals == 0:
                recommendations["deployment_decision"] = "DEPLOY"
                recommendations["risk_assessment"] = "LOW"
                recommendations["rollout_strategy"] = {
                    "type": "gradual",
                    "schedule": [
                        {"traffic_percentage": 0.25, "duration_days": 2},
                        {"traffic_percentage": 0.50, "duration_days": 3},
                        {"traffic_percentage": 0.75, "duration_days": 2},
                        {"traffic_percentage": 1.00, "duration_days": 0}
                    ]
                }
            elif positive_signals > negative_signals:
                recommendations["deployment_decision"] = "DEPLOY_WITH_CAUTION"
                recommendations["risk_assessment"] = "MEDIUM"
                recommendations["rollout_strategy"] = {
                    "type": "conservative",
                    "schedule": [
                        {"traffic_percentage": 0.10, "duration_days": 3},
                        {"traffic_percentage": 0.25, "duration_days": 4},
                        {"traffic_percentage": 0.50, "duration_days": 7}
                    ]
                }
            elif negative_signals > positive_signals:
                recommendations["deployment_decision"] = "DO_NOT_DEPLOY"
                recommendations["risk_assessment"] = "HIGH"
                recommendations["next_steps"].append("Investigate performance degradations")
                recommendations["next_steps"].append("Consider additional model fine-tuning")
            else:
                recommendations["deployment_decision"] = "EXTEND_TEST"
                recommendations["next_steps"].append("Collect more data to reach statistical significance")
                recommendations["next_steps"].append("Consider testing with different user segments")

            # Add general next steps
            if recommendations["deployment_decision"] in ["DEPLOY", "DEPLOY_WITH_CAUTION"]:
                recommendations["next_steps"].extend([
                    "Monitor key metrics closely during rollout",
                    "Set up automated rollback triggers",
                    "Prepare communication for stakeholders"
                ])

            return recommendations

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return {"error": str(e)}

    def create_visualizations(self, output_dir: str = None) -> List[str]:
        """
        Create visualizations for the A/B test results.

        Args:
            output_dir: Directory to save visualizations

        Returns:
            List of paths to generated visualization files
        """
        try:
            if output_dir is None:
                output_dir = os.path.join(self.test_data_path, "reports", "visualizations")

            os.makedirs(output_dir, exist_ok=True)

            generated_files = []

            # Get data for both variants
            variant_a_data = [r for r in self.results_storage if r.model_variant == 'A']
            variant_b_data = [r for r in self.results_storage if r.model_variant == 'B']

            if not variant_a_data or not variant_b_data:
                logger.warning("Insufficient data for visualizations")
                return []

            # 1. Response Time Comparison
            response_times_a = [r.response_time_ms for r in variant_a_data if r.response_time_ms is not None]
            response_times_b = [r.response_time_ms for r in variant_b_data if r.response_time_ms is not None]

            if response_times_a and response_times_b:
                plt.figure(figsize=(10, 6))
                plt.hist(response_times_a, alpha=0.7, label='Variant A', bins=20)
                plt.hist(response_times_b, alpha=0.7, label='Variant B', bins=20)
                plt.xlabel('Response Time (ms)')
                plt.ylabel('Frequency')
                plt.title('Response Time Distribution Comparison')
                plt.legend()
                plt.grid(True, alpha=0.3)

                response_time_plot = os.path.join(output_dir, 'response_time_comparison.png')
                plt.savefig(response_time_plot, dpi=300, bbox_inches='tight')
                plt.close()
                generated_files.append(response_time_plot)

            # 2. User Rating Comparison
            ratings_a = [r.user_rating for r in variant_a_data if r.user_rating is not None]
            ratings_b = [r.user_rating for r in variant_b_data if r.user_rating is not None]

            if ratings_a and ratings_b:
                plt.figure(figsize=(10, 6))

                # Create side-by-side box plots
                data_to_plot = [ratings_a, ratings_b]
                plt.boxplot(data_to_plot, labels=['Variant A', 'Variant B'])
                plt.ylabel('User Rating')
                plt.title('User Rating Distribution Comparison')
                plt.grid(True, alpha=0.3)

                rating_plot = os.path.join(output_dir, 'user_rating_comparison.png')
                plt.savefig(rating_plot, dpi=300, bbox_inches='tight')
                plt.close()
                generated_files.append(rating_plot)

            # 3. Metrics Over Time
            if len(self.results_storage) > 10:  # Only if we have enough data points
                # Group by day
                daily_metrics_a = defaultdict(list)
                daily_metrics_b = defaultdict(list)

                for result in variant_a_data:
                    day = result.timestamp.date()
                    if result.user_rating is not None:
                        daily_metrics_a[day].append(result.user_rating)

                for result in variant_b_data:
                    day = result.timestamp.date()
                    if result.user_rating is not None:
                        daily_metrics_b[day].append(result.user_rating)

                # Calculate daily averages
                dates_a = sorted(daily_metrics_a.keys())
                dates_b = sorted(daily_metrics_b.keys())
                avg_ratings_a = [np.mean(daily_metrics_a[date]) for date in dates_a]
                avg_ratings_b = [np.mean(daily_metrics_b[date]) for date in dates_b]

                if dates_a and dates_b:
                    plt.figure(figsize=(12, 6))
                    plt.plot(dates_a, avg_ratings_a, marker='o', label='Variant A', linewidth=2)
                    plt.plot(dates_b, avg_ratings_b, marker='s', label='Variant B', linewidth=2)
                    plt.xlabel('Date')
                    plt.ylabel('Average User Rating')
                    plt.title('User Rating Trends Over Time')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)

                    trends_plot = os.path.join(output_dir, 'rating_trends.png')
                    plt.savefig(trends_plot, dpi=300, bbox_inches='tight')
                    plt.close()
                    generated_files.append(trends_plot)

            # 4. Conversion Rate Comparison
            conversion_a = sum(1 for r in variant_a_data if r.user_rating and r.user_rating >= 4)
            conversion_b = sum(1 for r in variant_b_data if r.user_rating and r.user_rating >= 4)
            total_a = len([r for r in variant_a_data if r.user_rating is not None])
            total_b = len([r for r in variant_b_data if r.user_rating is not None])

            if total_a > 0 and total_b > 0:
                conversion_rate_a = conversion_a / total_a
                conversion_rate_b = conversion_b / total_b

                plt.figure(figsize=(8, 6))
                variants = ['Variant A', 'Variant B']
                conversion_rates = [conversion_rate_a, conversion_rate_b]
                colors = ['skyblue', 'lightcoral']

                bars = plt.bar(variants, conversion_rates, color=colors)
                plt.ylabel('Conversion Rate')
                plt.title('Conversion Rate Comparison (Rating ≥ 4)')
                plt.ylim(0, 1)

                # Add value labels on bars
                for bar, rate in zip(bars, conversion_rates):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{rate:.1%}', ha='center', va='bottom')

                plt.grid(True, alpha=0.3, axis='y')

                conversion_plot = os.path.join(output_dir, 'conversion_rate_comparison.png')
                plt.savefig(conversion_plot, dpi=300, bbox_inches='tight')
                plt.close()
                generated_files.append(conversion_plot)

            logger.info(f"Generated {len(generated_files)} visualization files in {output_dir}")
            return generated_files

        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            return []

    def rollback_to_variant_a(self) -> bool:
        """
        Emergency rollback to variant A (original model).

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.warning("Initiating emergency rollback to Variant A")

            # Set traffic split to 0% for variant B
            success = self.update_traffic_split(0.0)

            if success:
                # Log rollback event
                rollback_event = {
                    "event_type": "rollback",
                    "timestamp": datetime.now().isoformat(),
                    "reason": "Emergency rollback initiated",
                    "previous_traffic_split": self.current_traffic_split
                }

                rollback_file = os.path.join(self.test_data_path, "rollback_events.json")

                # Load existing rollback events
                rollback_events = []
                if os.path.exists(rollback_file):
                    try:
                        with open(rollback_file, 'r') as f:
                            rollback_events = json.load(f)
                    except:
                        pass

                rollback_events.append(rollback_event)

                with open(rollback_file, 'w') as f:
                    json.dump(rollback_events, f, indent=2)

                logger.info("Rollback completed successfully")
                return True
            else:
                logger.error("Failed to update traffic split during rollback")
                return False

        except Exception as e:
            logger.error(f"Error during rollback: {e}")
            return False

    def get_test_status(self) -> Dict[str, Any]:
        """
        Get current status of the A/B test.

        Returns:
            Dictionary containing test status information
        """
        try:
            # Calculate basic statistics
            total_queries = len(self.results_storage)
            variant_a_queries = len([r for r in self.results_storage if r.model_variant == 'A'])
            variant_b_queries = len([r for r in self.results_storage if r.model_variant == 'B'])

            # Calculate time-based statistics
            if self.results_storage:
                start_time = min(r.timestamp for r in self.results_storage)
                end_time = max(r.timestamp for r in self.results_storage)
                test_duration = (end_time - start_time).total_seconds() / 86400  # days
            else:
                start_time = None
                end_time = None
                test_duration = 0

            # Check if we have sufficient data for statistical significance
            sufficient_data = (variant_a_queries >= self.config.min_samples_per_variant and
                             variant_b_queries >= self.config.min_samples_per_variant)

            # Calculate recent performance (last 24 hours)
            recent_cutoff = datetime.now() - timedelta(hours=24)
            recent_results = [r for r in self.results_storage if r.timestamp >= recent_cutoff]
            recent_ratings = [r.user_rating for r in recent_results if r.user_rating is not None]

            status = {
                "test_name": self.config.test_name,
                "test_status": "ACTIVE" if self.current_traffic_split > 0 else "PAUSED",
                "current_traffic_split": self.current_traffic_split,
                "test_duration_days": round(test_duration, 1),
                "total_queries": total_queries,
                "variant_distribution": {
                    "A": variant_a_queries,
                    "B": variant_b_queries,
                    "A_percentage": (variant_a_queries / total_queries * 100) if total_queries > 0 else 0,
                    "B_percentage": (variant_b_queries / total_queries * 100) if total_queries > 0 else 0
                },
                "data_sufficiency": {
                    "sufficient_for_analysis": sufficient_data,
                    "min_required_per_variant": self.config.min_samples_per_variant,
                    "variant_a_samples": variant_a_queries,
                    "variant_b_samples": variant_b_queries
                },
                "recent_performance": {
                    "last_24h_queries": len(recent_results),
                    "last_24h_avg_rating": np.mean(recent_ratings) if recent_ratings else None
                },
                "test_timeline": {
                    "start_time": start_time.isoformat() if start_time else None,
                    "last_update": end_time.isoformat() if end_time else None,
                    "planned_duration_days": self.config.test_duration_days
                }
            }

            return status

        except Exception as e:
            logger.error(f"Error getting test status: {e}")
            return {"error": str(e)}


# Utility functions for external use
def create_ab_test(config: ABTestConfig, storage_path: str = "./ab_test_data") -> ABTestingFramework:
    """
    Factory function to create an ABTestingFramework instance.

    Args:
        config: A/B test configuration
        storage_path: Path to store test data

    Returns:
        ABTestingFramework instance
    """
    return ABTestingFramework(config=config, storage_path=storage_path)

def quick_ab_test_setup(test_name: str, model_a_name: str, model_b_name: str,
                       traffic_split: float = 0.1) -> ABTestingFramework:
    """
    Quick setup for A/B testing with default configuration.

    Args:
        test_name: Name of the test
        model_a_name: Name of the original model
        model_b_name: Name of the fine-tuned model
        traffic_split: Initial traffic split for model B

    Returns:
        ABTestingFramework instance
    """
    config = ABTestConfig(
        test_name=test_name,
        model_a_name=model_a_name,
        model_b_name=model_b_name,
        traffic_split=traffic_split
    )

    framework = ABTestingFramework(config)
    framework.update_traffic_split(traffic_split)

    return framework