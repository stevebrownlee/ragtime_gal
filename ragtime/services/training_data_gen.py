"""
Training Data Generation Service for Ragtime.

This module generates training data for embedding model fine-tuning from user
feedback stored in ConPort. It creates positive/negative pairs and mines hard
negatives for improved model training.

Classes:
    TrainingDataGenerator: Main class for generating training data from feedback

Examples:
    >>> from ragtime.services.training_data_gen import TrainingDataGenerator
    >>> from ragtime.storage.conport_client import ConPortClient
    >>>
    >>> conport = ConPortClient()
    >>> generator = TrainingDataGenerator(
    ...     conport_client=conport,
    ...     workspace_id="/path/to/workspace"
    ... )
    >>> result = generator.generate_training_data(days_back=90)
"""

import json
import os
import csv
import hashlib
import numpy as np
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass

from ragtime.config.settings import settings
from ragtime.monitoring.logging import get_logger
from ragtime.models.training import (
    TrainingDataPair,
    TrainingDataset,
    TrainingDataConfig
)

logger = get_logger(__name__)


@dataclass
class TrainingPair:
    """
    Represents a training pair for embedding fine-tuning.

    This is a lightweight internal representation that gets converted
    to Pydantic models for API responses.

    Attributes:
        query: User's query text
        document: Document content
        label: 1.0 for positive, 0.0 for negative
        rating: Optional user rating (1-5)
        pair_type: Type of pair (positive, negative, hard_negative)
        metadata: Additional metadata
    """
    query: str
    document: str
    label: float
    rating: Optional[int] = None
    pair_type: str = "standard"
    metadata: Optional[Dict[str, Any]] = None


class TrainingDataGenerator:
    """
    Generates training data for embedding model fine-tuning from user feedback.

    This class retrieves feedback from ConPort, extracts document content from
    the vector database, and creates training pairs suitable for fine-tuning
    sentence-transformer models.

    Attributes:
        conport_client: ConPort client for feedback access
        workspace_id: Workspace identifier
        chroma_db: ChromaDB instance for document retrieval
        embedding_model: Current embedding model
        config: Training data configuration

    Examples:
        >>> generator = TrainingDataGenerator(
        ...     conport_client=conport,
        ...     workspace_id="/workspace"
        ... )
        >>> feedback = generator.get_feedback_data(days_back=30)
        >>> positive_pairs = generator.generate_positive_pairs(feedback)
    """

    def __init__(
        self,
        conport_client=None,
        workspace_id: Optional[str] = None,
        chroma_db=None,
        embedding_model=None,
        config: Optional[TrainingDataConfig] = None
    ):
        """
        Initialize the training data generator.

        Args:
            conport_client: ConPort MCP client for feedback data access
            workspace_id: Workspace identifier for ConPort operations
            chroma_db: ChromaDB instance for document retrieval
            embedding_model: Current embedding model for similarity calculations
            config: Training data configuration (uses defaults if not provided)
        """
        self.conport_client = conport_client
        self.workspace_id = workspace_id or os.getcwd()
        self.chroma_db = chroma_db
        self.embedding_model = embedding_model
        self.config = config or TrainingDataConfig()

        # Configuration parameters from config or defaults
        self.positive_rating_threshold = self.config.positive_threshold
        self.negative_rating_threshold = self.config.negative_threshold
        self.hard_negative_similarity_threshold = self.config.hard_negative_similarity
        self.min_query_length = 3
        self.max_pairs_per_query = self.config.max_pairs_per_query

        # Caches for efficiency
        self.feedback_cache: Dict[str, Any] = {}
        self.document_cache: Dict[str, str] = {}
        self.similarity_cache: Dict[str, float] = {}

        logger.info(
            "Initialized TrainingDataGenerator",
            extra={
                "workspace_id": self.workspace_id,
                "has_conport": bool(conport_client),
                "has_chroma": bool(chroma_db),
                "positive_threshold": self.positive_rating_threshold,
                "negative_threshold": self.negative_rating_threshold
            }
        )

    def get_feedback_data(
        self,
        days_back: int = 90,
        min_samples: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve feedback data from ConPort for training data generation.

        Args:
            days_back: Number of days to look back for feedback data
            min_samples: Minimum number of samples required

        Returns:
            List of feedback entries suitable for training
        """
        try:
            if not self.conport_client or not self.workspace_id:
                logger.warning(
                    "ConPort client or workspace_id not configured",
                    extra={
                        "has_client": bool(self.conport_client),
                        "has_workspace": bool(self.workspace_id)
                    }
                )
                return []

            # Search for feedback data in ConPort
            search_result = self.conport_client.search_custom_data_value_fts({
                "workspace_id": self.workspace_id,
                "query_term": "rating",
                "category_filter": "UserFeedback",
                "limit": 500
            })

            feedback_entries = []
            if search_result and isinstance(search_result, list):
                for entry in search_result:
                    try:
                        feedback_data = entry.get('value', {})
                        if isinstance(feedback_data, str):
                            feedback_data = json.loads(feedback_data)

                        # Filter by date
                        if days_back > 0:
                            timestamp = feedback_data.get('timestamp')
                            if isinstance(timestamp, (int, float)):
                                entry_date = datetime.fromtimestamp(timestamp)
                            elif isinstance(timestamp, str):
                                entry_date = datetime.fromisoformat(timestamp)
                            else:
                                logger.warning(
                                    "Invalid timestamp type",
                                    extra={"type": type(timestamp).__name__}
                                )
                                continue

                            cutoff_date = datetime.now() - timedelta(days=days_back)
                            if entry_date < cutoff_date:
                                continue

                        # Validate required fields
                        if not all(
                            key in feedback_data
                            for key in ['query', 'rating', 'document_ids']
                        ):
                            continue

                        # Filter by query quality
                        query = feedback_data.get('query', '').strip()
                        if len(query.split()) < self.min_query_length:
                            continue

                        feedback_entries.append(feedback_data)

                    except (json.JSONDecodeError, ValueError, KeyError) as e:
                        logger.warning(
                            "Error parsing feedback entry",
                            extra={"error": str(e)},
                            exc_info=True
                        )
                        continue

            logger.info(
                "Retrieved feedback entries",
                extra={
                    "count": len(feedback_entries),
                    "days_back": days_back,
                    "sufficient": len(feedback_entries) >= min_samples
                }
            )

            if len(feedback_entries) < min_samples:
                logger.warning(
                    "Insufficient feedback data",
                    extra={
                        "retrieved": len(feedback_entries),
                        "required": min_samples
                    }
                )

            return feedback_entries

        except Exception as e:
            logger.error(
                "Error retrieving feedback data",
                extra={"error": str(e)},
                exc_info=True
            )
            return []

    def get_document_content(
        self,
        document_ids: List[str]
    ) -> Dict[str, str]:
        """
        Retrieve document content from ChromaDB.

        Args:
            document_ids: List of document IDs to retrieve

        Returns:
            Dictionary mapping document IDs to their content
        """
        try:
            if not self.chroma_db or not document_ids:
                return {}

            # Check cache first
            cached_docs = {}
            missing_ids = []

            for doc_id in document_ids:
                if doc_id in self.document_cache:
                    cached_docs[doc_id] = self.document_cache[doc_id]
                else:
                    missing_ids.append(doc_id)

            # Retrieve missing documents
            if missing_ids:
                try:
                    results = self.chroma_db.get(ids=missing_ids)

                    if results and 'documents' in results and 'ids' in results:
                        for doc_id, content in zip(
                            results['ids'],
                            results['documents']
                        ):
                            if content:
                                self.document_cache[doc_id] = content
                                cached_docs[doc_id] = content

                    logger.debug(
                        "Retrieved documents from ChromaDB",
                        extra={
                            "requested": len(missing_ids),
                            "retrieved": len(cached_docs) - len(cached_docs) + len(missing_ids)
                        }
                    )

                except Exception as e:
                    logger.warning(
                        "Error retrieving documents from ChromaDB",
                        extra={"error": str(e)},
                        exc_info=True
                    )

            return cached_docs

        except Exception as e:
            logger.error(
                "Error getting document content",
                extra={"error": str(e)},
                exc_info=True
            )
            return {}

    def generate_positive_pairs(
        self,
        feedback_data: List[Dict[str, Any]]
    ) -> List[TrainingPair]:
        """
        Generate positive training pairs from high-rated feedback.

        Args:
            feedback_data: List of feedback entries

        Returns:
            List of positive training pairs
        """
        positive_pairs = []

        try:
            high_rated = [
                entry for entry in feedback_data
                if entry.get('rating', 0) >= self.positive_rating_threshold
            ]

            logger.info(
                "Processing high-rated feedback",
                extra={"count": len(high_rated)}
            )

            for entry in high_rated:
                query = entry.get('query', '').strip()
                document_ids = entry.get('document_ids', [])
                rating = entry.get('rating', 0)

                if not query or not document_ids:
                    continue

                documents = self.get_document_content(document_ids)

                pair_count = 0
                for doc_id, content in documents.items():
                    if content and pair_count < self.max_pairs_per_query:
                        content = self._truncate_document(content, max_length=1000)

                        pair = TrainingPair(
                            query=query,
                            document=content,
                            label=1.0,
                            rating=rating,
                            pair_type="positive",
                            metadata={
                                "document_id": doc_id,
                                "feedback_timestamp": entry.get('timestamp'),
                                "original_rating": rating
                            }
                        )
                        positive_pairs.append(pair)
                        pair_count += 1

            logger.info(
                "Generated positive training pairs",
                extra={"count": len(positive_pairs)}
            )
            return positive_pairs

        except Exception as e:
            logger.error(
                "Error generating positive pairs",
                extra={"error": str(e)},
                exc_info=True
            )
            return []

    def generate_negative_pairs(
        self,
        feedback_data: List[Dict[str, Any]]
    ) -> List[TrainingPair]:
        """
        Generate negative training pairs from low-rated feedback.

        Args:
            feedback_data: List of feedback entries

        Returns:
            List of negative training pairs
        """
        negative_pairs = []

        try:
            low_rated = [
                entry for entry in feedback_data
                if entry.get('rating', 0) <= self.negative_rating_threshold
            ]

            logger.info(
                "Processing low-rated feedback",
                extra={"count": len(low_rated)}
            )

            for entry in low_rated:
                query = entry.get('query', '').strip()
                document_ids = entry.get('document_ids', [])
                rating = entry.get('rating', 0)

                if not query or not document_ids:
                    continue

                documents = self.get_document_content(document_ids)

                pair_count = 0
                for doc_id, content in documents.items():
                    if content and pair_count < self.max_pairs_per_query:
                        content = self._truncate_document(content, max_length=1000)

                        pair = TrainingPair(
                            query=query,
                            document=content,
                            label=0.0,
                            rating=rating,
                            pair_type="negative",
                            metadata={
                                "document_id": doc_id,
                                "feedback_timestamp": entry.get('timestamp'),
                                "original_rating": rating
                            }
                        )
                        negative_pairs.append(pair)
                        pair_count += 1

            logger.info(
                "Generated negative training pairs",
                extra={"count": len(negative_pairs)}
            )
            return negative_pairs

        except Exception as e:
            logger.error(
                "Error generating negative pairs",
                extra={"error": str(e)},
                exc_info=True
            )
            return []

    def generate_hard_negative_pairs(
        self,
        positive_pairs: List[TrainingPair],
        feedback_data: List[Dict[str, Any]],
        num_hard_negatives: Optional[int] = None
    ) -> List[TrainingPair]:
        """
        Generate hard negative pairs by finding similar queries with different outcomes.

        Args:
            positive_pairs: List of positive training pairs
            feedback_data: All feedback data for mining
            num_hard_negatives: Number of hard negatives to generate

        Returns:
            List of hard negative training pairs
        """
        hard_negatives = []

        try:
            if not positive_pairs or not self.embedding_model:
                logger.warning(
                    "Cannot generate hard negatives",
                    extra={
                        "has_positive_pairs": bool(positive_pairs),
                        "has_embedding_model": bool(self.embedding_model)
                    }
                )
                return []

            if num_hard_negatives is None:
                num_hard_negatives = min(len(positive_pairs), 100)

            low_rated = [
                entry for entry in feedback_data
                if entry.get('rating', 0) <= self.negative_rating_threshold
            ]

            if not low_rated:
                logger.warning("No low-rated feedback for hard negative mining")
                return []

            logger.info(
                "Mining hard negatives",
                extra={
                    "low_rated_entries": len(low_rated),
                    "target_count": num_hard_negatives
                }
            )

            for pos_pair in positive_pairs[:num_hard_negatives]:
                try:
                    similar_entries = self._find_similar_queries(
                        pos_pair.query,
                        low_rated,
                        threshold=self.hard_negative_similarity_threshold
                    )

                    if similar_entries:
                        similar_entry = similar_entries[0]
                        document_ids = similar_entry.get('document_ids', [])

                        if document_ids:
                            documents = self.get_document_content(document_ids[:1])

                            for doc_id, content in documents.items():
                                if content:
                                    content = self._truncate_document(
                                        content,
                                        max_length=1000
                                    )

                                    hard_neg = TrainingPair(
                                        query=pos_pair.query,
                                        document=content,
                                        label=0.0,
                                        rating=similar_entry.get('rating', 0),
                                        pair_type="hard_negative",
                                        metadata={
                                            "document_id": doc_id,
                                            "similar_query": similar_entry.get('query'),
                                            "similarity_score": similar_entries[0].get('similarity', 0),
                                            "original_rating": similar_entry.get('rating', 0)
                                        }
                                    )
                                    hard_negatives.append(hard_neg)
                                    break

                except Exception as e:
                    logger.warning(
                        "Error processing hard negative",
                        extra={
                            "query": pos_pair.query[:50],
                            "error": str(e)
                        }
                    )
                    continue

            logger.info(
                "Generated hard negative pairs",
                extra={"count": len(hard_negatives)}
            )
            return hard_negatives

        except Exception as e:
            logger.error(
                "Error generating hard negative pairs",
                extra={"error": str(e)},
                exc_info=True
            )
            return []

    def _find_similar_queries(
        self,
        target_query: str,
        feedback_entries: List[Dict[str, Any]],
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Find queries similar to target query using Jaccard similarity.

        Args:
            target_query: Query to find similarities for
            feedback_entries: List of feedback entries to search
            threshold: Similarity threshold

        Returns:
            List of similar feedback entries with similarity scores
        """
        try:
            similar_entries = []
            target_words = set(target_query.lower().split())

            for entry in feedback_entries:
                query = entry.get('query', '').strip()
                if not query or query == target_query:
                    continue

                query_words = set(query.lower().split())
                intersection = len(target_words.intersection(query_words))
                union = len(target_words.union(query_words))

                if union > 0:
                    similarity = intersection / union
                    if similarity >= threshold:
                        entry_copy = entry.copy()
                        entry_copy['similarity'] = similarity
                        similar_entries.append(entry_copy)

            similar_entries.sort(key=lambda x: x.get('similarity', 0), reverse=True)
            return similar_entries

        except Exception as e:
            logger.error(
                "Error finding similar queries",
                extra={"error": str(e)},
                exc_info=True
            )
            return []

    def _truncate_document(self, content: str, max_length: int = 1000) -> str:
        """
        Truncate document content to reasonable length for training.

        Args:
            content: Document content
            max_length: Maximum number of characters

        Returns:
            Truncated content
        """
        if len(content) <= max_length:
            return content

        truncated = content[:max_length]
        last_period = truncated.rfind('.')
        last_newline = truncated.rfind('\n')

        boundary = max(last_period, last_newline)
        if boundary > max_length * 0.7:
            return content[:boundary + 1]
        else:
            return content[:max_length] + "..."

    def validate_training_data(
        self,
        training_pairs: List[TrainingPair]
    ) -> Dict[str, Any]:
        """
        Validate the quality of generated training data.

        Args:
            training_pairs: List of training pairs to validate

        Returns:
            Dictionary containing validation results
        """
        try:
            if not training_pairs:
                return {
                    "valid": False,
                    "error": "No training pairs provided",
                    "statistics": {}
                }

            # Basic statistics
            total_pairs = len(training_pairs)
            positive_pairs = sum(1 for pair in training_pairs if pair.label == 1.0)
            negative_pairs = sum(1 for pair in training_pairs if pair.label == 0.0)
            hard_negatives = sum(
                1 for pair in training_pairs
                if pair.pair_type == "hard_negative"
            )

            # Query and document length statistics
            query_lengths = [len(pair.query.split()) for pair in training_pairs]
            doc_lengths = [len(pair.document.split()) for pair in training_pairs]

            # Check for duplicates
            pair_hashes: Set[str] = set()
            duplicates = 0
            for pair in training_pairs:
                pair_hash = hashlib.md5(
                    f"{pair.query}|{pair.document}".encode()
                ).hexdigest()
                if pair_hash in pair_hashes:
                    duplicates += 1
                else:
                    pair_hashes.add(pair_hash)

            # Validation criteria
            min_pairs = 20
            min_positive_ratio = 0.2
            max_positive_ratio = 0.8
            max_duplicate_ratio = 0.1

            positive_ratio = positive_pairs / total_pairs if total_pairs > 0 else 0
            duplicate_ratio = duplicates / total_pairs if total_pairs > 0 else 0

            validation_results = {
                "valid": (
                    total_pairs >= min_pairs and
                    min_positive_ratio <= positive_ratio <= max_positive_ratio and
                    duplicate_ratio <= max_duplicate_ratio
                ),
                "statistics": {
                    "total_pairs": total_pairs,
                    "positive_pairs": positive_pairs,
                    "negative_pairs": negative_pairs,
                    "hard_negatives": hard_negatives,
                    "positive_ratio": round(positive_ratio, 3),
                    "duplicate_ratio": round(duplicate_ratio, 3),
                    "avg_query_length": round(np.mean(query_lengths), 1) if query_lengths else 0,
                    "avg_document_length": round(np.mean(doc_lengths), 1) if doc_lengths else 0,
                    "query_length_range": [min(query_lengths), max(query_lengths)] if query_lengths else [0, 0],
                    "document_length_range": [min(doc_lengths), max(doc_lengths)] if doc_lengths else [0, 0]
                },
                "warnings": []
            }

            # Add warnings
            if total_pairs < min_pairs:
                validation_results["warnings"].append(
                    f"Insufficient training data: {total_pairs} < {min_pairs}"
                )

            if positive_ratio < min_positive_ratio:
                validation_results["warnings"].append(
                    f"Too few positive examples: {positive_ratio:.1%}"
                )
            elif positive_ratio > max_positive_ratio:
                validation_results["warnings"].append(
                    f"Too many positive examples: {positive_ratio:.1%}"
                )

            if duplicate_ratio > max_duplicate_ratio:
                validation_results["warnings"].append(
                    f"High duplicate ratio: {duplicate_ratio:.1%}"
                )

            logger.info(
                "Validated training data",
                extra={
                    "valid": validation_results["valid"],
                    "total_pairs": total_pairs,
                    "warnings": len(validation_results["warnings"])
                }
            )

            return validation_results

        except Exception as e:
            logger.error(
                "Error validating training data",
                extra={"error": str(e)},
                exc_info=True
            )
            return {
                "valid": False,
                "error": str(e),
                "statistics": {}
            }

    def export_training_data(
        self,
        training_pairs: List[TrainingPair],
        output_format: str = "sentence_transformers",
        output_path: str = "./training_data"
    ) -> Dict[str, Any]:
        """
        Export training data in the specified format.

        Args:
            training_pairs: List of training pairs to export
            output_format: Export format (sentence_transformers, triplets, json)
            output_path: Output directory path

        Returns:
            Dictionary containing export results
        """
        try:
            if not training_pairs:
                return {"success": False, "error": "No training pairs to export"}

            os.makedirs(output_path, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if output_format == "sentence_transformers":
                return self._export_sentence_transformers_format(
                    training_pairs, output_path, timestamp
                )
            elif output_format == "triplets":
                return self._export_triplets_format(
                    training_pairs, output_path, timestamp
                )
            elif output_format == "json":
                return self._export_json_format(
                    training_pairs, output_path, timestamp
                )
            else:
                return {
                    "success": False,
                    "error": f"Unsupported format: {output_format}"
                }

        except Exception as e:
            logger.error(
                "Error exporting training data",
                extra={"error": str(e)},
                exc_info=True
            )
            return {"success": False, "error": str(e)}

    def _export_sentence_transformers_format(
        self,
        training_pairs: List[TrainingPair],
        output_path: str,
        timestamp: str
    ) -> Dict[str, Any]:
        """Export in sentence-transformers format (CSV)."""
        try:
            filename = f"training_data_{timestamp}.csv"
            filepath = os.path.join(output_path, filename)

            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['query', 'document', 'label'])

                for pair in training_pairs:
                    writer.writerow([pair.query, pair.document, int(pair.label)])

            # Export metadata
            metadata_filename = f"training_metadata_{timestamp}.json"
            metadata_filepath = os.path.join(output_path, metadata_filename)

            metadata = {
                "export_timestamp": datetime.now().isoformat(),
                "total_pairs": len(training_pairs),
                "format": "sentence_transformers",
                "pair_statistics": {
                    "positive": sum(1 for p in training_pairs if p.label == 1.0),
                    "negative": sum(1 for p in training_pairs if p.label == 0.0),
                    "hard_negatives": sum(
                        1 for p in training_pairs
                        if p.pair_type == "hard_negative"
                    )
                }
            }

            with open(metadata_filepath, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)

            logger.info(
                "Exported training data",
                extra={
                    "format": "sentence_transformers",
                    "filepath": filepath,
                    "total_pairs": len(training_pairs)
                }
            )

            return {
                "success": True,
                "files": [filepath, metadata_filepath],
                "format": "sentence_transformers",
                "total_pairs": len(training_pairs)
            }

        except Exception as e:
            logger.error(
                "Error exporting sentence-transformers format",
                extra={"error": str(e)},
                exc_info=True
            )
            return {"success": False, "error": str(e)}

    def _export_triplets_format(
        self,
        training_pairs: List[TrainingPair],
        output_path: str,
        timestamp: str
    ) -> Dict[str, Any]:
        """Export in triplets format (anchor, positive, negative)."""
        try:
            query_groups = defaultdict(lambda: {"positive": [], "negative": []})

            for pair in training_pairs:
                if pair.label == 1.0:
                    query_groups[pair.query]["positive"].append(pair.document)
                else:
                    query_groups[pair.query]["negative"].append(pair.document)

            filename = f"training_triplets_{timestamp}.csv"
            filepath = os.path.join(output_path, filename)

            triplet_count = 0
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['anchor', 'positive', 'negative'])

                for query, docs in query_groups.items():
                    positives = docs["positive"]
                    negatives = docs["negative"]

                    for pos_doc in positives:
                        for neg_doc in negatives:
                            writer.writerow([query, pos_doc, neg_doc])
                            triplet_count += 1

            logger.info(
                "Exported training triplets",
                extra={"filepath": filepath, "total_triplets": triplet_count}
            )

            return {
                "success": True,
                "files": [filepath],
                "format": "triplets",
                "total_triplets": triplet_count
            }

        except Exception as e:
            logger.error(
                "Error exporting triplets format",
                extra={"error": str(e)},
                exc_info=True
            )
            return {"success": False, "error": str(e)}

    def _export_json_format(
        self,
        training_pairs: List[TrainingPair],
        output_path: str,
        timestamp: str
    ) -> Dict[str, Any]:
        """Export in JSON format with full metadata."""
        try:
            filename = f"training_data_{timestamp}.json"
            filepath = os.path.join(output_path, filename)

            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "total_pairs": len(training_pairs),
                "training_pairs": []
            }

            for pair in training_pairs:
                pair_data = {
                    "query": pair.query,
                    "document": pair.document,
                    "label": pair.label,
                    "pair_type": pair.pair_type,
                    "rating": pair.rating,
                    "metadata": pair.metadata
                }
                export_data["training_pairs"].append(pair_data)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            logger.info(
                "Exported training data as JSON",
                extra={"filepath": filepath, "total_pairs": len(training_pairs)}
            )

            return {
                "success": True,
                "files": [filepath],
                "format": "json",
                "total_pairs": len(training_pairs)
            }

        except Exception as e:
            logger.error(
                "Error exporting JSON format",
                extra={"error": str(e)},
                exc_info=True
            )
            return {"success": False, "error": str(e)}

    def generate_training_data(
        self,
        days_back: int = 90,
        include_hard_negatives: bool = True,
        output_format: str = "sentence_transformers",
        output_path: str = "./training_data"
    ) -> Dict[str, Any]:
        """
        Complete pipeline to generate training data from feedback.

        Args:
            days_back: Number of days to look back for feedback
            include_hard_negatives: Whether to include hard negative mining
            output_format: Export format for training data
            output_path: Output directory path

        Returns:
            Dictionary containing generation results
        """
        try:
            logger.info(
                "Starting training data generation pipeline",
                extra={
                    "days_back": days_back,
                    "include_hard_negatives": include_hard_negatives,
                    "output_format": output_format
                }
            )

            # Step 1: Get feedback data
            feedback_data = self.get_feedback_data(days_back=days_back)
            if not feedback_data:
                return {
                    "success": False,
                    "error": "No feedback data available",
                    "stage": "data_retrieval"
                }

            # Step 2: Generate positive pairs
            positive_pairs = self.generate_positive_pairs(feedback_data)

            # Step 3: Generate negative pairs
            negative_pairs = self.generate_negative_pairs(feedback_data)

            # Step 4: Generate hard negatives (optional)
            hard_negatives = []
            if include_hard_negatives and positive_pairs:
                hard_negatives = self.generate_hard_negative_pairs(
                    positive_pairs, feedback_data
                )

            # Step 5: Combine all pairs
            all_pairs = positive_pairs + negative_pairs + hard_negatives

            if not all_pairs:
                return {
                    "success": False,
                    "error": "No training pairs generated",
                    "stage": "pair_generation"
                }

            # Step 6: Validate training data
            validation_results = self.validate_training_data(all_pairs)

            # Step 7: Export training data
            export_results = self.export_training_data(
                all_pairs, output_format, output_path
            )

            # Compile final results
            results = {
                "success": export_results.get("success", False),
                "generation_timestamp": datetime.now().isoformat(),
                "feedback_period_days": days_back,
                "data_statistics": {
                    "total_feedback_entries": len(feedback_data),
                    "positive_pairs": len(positive_pairs),
                    "negative_pairs": len(negative_pairs),
                    "hard_negatives": len(hard_negatives),
                    "total_training_pairs": len(all_pairs)
                },
                "validation": validation_results,
                "export": export_results
            }

            if validation_results.get("warnings"):
                results["warnings"] = validation_results["warnings"]

            logger.info(
                "Training data generation completed",
                extra={
                    "success": results["success"],
                    "total_pairs": len(all_pairs)
                }
            )

            return results

        except Exception as e:
            logger.error(
                "Error in training data generation pipeline",
                extra={"error": str(e)},
                exc_info=True
            )
            return {
                "success": False,
                "error": str(e),
                "stage": "pipeline_execution"
            }


def create_training_data_generator(
    conport_client=None,
    workspace_id: Optional[str] = None,
    chroma_db=None,
    embedding_model=None,
    config: Optional[TrainingDataConfig] = None
) -> TrainingDataGenerator:
    """
    Factory function to create a TrainingDataGenerator instance.

    Args:
        conport_client: ConPort MCP client
        workspace_id: Workspace identifier
        chroma_db: ChromaDB instance
        embedding_model: Current embedding model
        config: Training data configuration

    Returns:
        TrainingDataGenerator instance

    Examples:
        >>> generator = create_training_data_generator(
        ...     conport_client=conport,
        ...     workspace_id="/workspace"
        ... )
    """
    return TrainingDataGenerator(
        conport_client=conport_client,
        workspace_id=workspace_id,
        chroma_db=chroma_db,
        embedding_model=embedding_model,
        config=config
    )