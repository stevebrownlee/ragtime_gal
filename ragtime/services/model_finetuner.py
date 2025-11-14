"""
Embedding Model Fine-tuning Service for Ragtime.

This module implements fine-tuning of sentence-transformer embedding models
using feedback-generated training data for improved semantic understanding.

Classes:
    ModelFineTuner: Main class for fine-tuning embedding models

Examples:
    >>> from ragtime.services.model_finetuner import ModelFineTuner
    >>> from ragtime.models.training import FineTuningConfig
    >>>
    >>> config = FineTuningConfig(base_model="all-MiniLM-L6-v2")
    >>> finetuner = ModelFineTuner(config)
    >>> finetuner.load_base_model()
    >>> result = finetuner.fine_tune_model(train_examples, val_examples)
"""

import os
import json
import shutil
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

# Sentence Transformers imports
try:
    from sentence_transformers import (
        SentenceTransformer,
        InputExample,
        losses,
        evaluation
    )
    from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
    from torch.utils.data import DataLoader
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from ragtime.config.settings import settings
from ragtime.monitoring.logging import get_logger
from ragtime.models.training import FineTuningConfig

logger = get_logger(__name__)


@dataclass
class TrainingMetrics:
    """
    Training metrics tracking during fine-tuning.

    Attributes:
        epoch: Current training epoch
        train_loss: Training loss value
        eval_loss: Evaluation loss (optional)
        eval_cosine_accuracy: Cosine similarity accuracy (optional)
        eval_manhattan_accuracy: Manhattan distance accuracy (optional)
        eval_euclidean_accuracy: Euclidean distance accuracy (optional)
        timestamp: ISO timestamp of metric recording
    """
    epoch: int
    train_loss: float
    eval_loss: Optional[float] = None
    eval_cosine_accuracy: Optional[float] = None
    eval_manhattan_accuracy: Optional[float] = None
    eval_euclidean_accuracy: Optional[float] = None
    timestamp: Optional[str] = None


class ModelFineTuner:
    """
    Fine-tunes sentence-transformer embedding models using training data.

    This class handles the complete fine-tuning pipeline from loading base
    models through training to evaluation and deployment preparation.

    Attributes:
        config: Fine-tuning configuration
        model: Current SentenceTransformer model
        training_history: List of training metrics
        best_model_path: Path to best performing model
        best_eval_score: Best evaluation score achieved

    Examples:
        >>> config = FineTuningConfig(num_epochs=5, batch_size=32)
        >>> finetuner = ModelFineTuner(config)
        >>> success = finetuner.load_base_model("all-MiniLM-L6-v2")
        >>> train_data, val_data = finetuner.load_training_data("data.csv")
        >>> result = finetuner.fine_tune_model(train_data, val_data)
    """

    def __init__(self, config: Optional[FineTuningConfig] = None):
        """
        Initialize the model fine-tuner.

        Args:
            config: Fine-tuning configuration (uses defaults if not provided)

        Raises:
            ImportError: If sentence-transformers is not installed
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for fine-tuning. "
                "Install with: pip install sentence-transformers"
            )

        self.config = config or FineTuningConfig()
        self.model: Optional[SentenceTransformer] = None
        self.training_history: List[TrainingMetrics] = []
        self.best_model_path: Optional[str] = None
        self.best_eval_score = float('-inf')

        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        # Create output directory
        output_path = self.config.output_path or "./fine_tuned_models"
        os.makedirs(output_path, exist_ok=True)

        logger.info(
            "Initialized ModelFineTuner",
            extra={
                "base_model": self.config.base_model,
                "output_path": output_path,
                "num_epochs": self.config.num_epochs,
                "batch_size": self.config.batch_size
            }
        )

    def load_base_model(self, model_name: Optional[str] = None) -> bool:
        """
        Load the base sentence-transformer model.

        Args:
            model_name: Name of the base model to load (defaults to config)

        Returns:
            True if successful, False otherwise
        """
        try:
            model_name = model_name or self.config.base_model
            logger.info(
                "Loading base model",
                extra={"model_name": model_name}
            )

            self.model = SentenceTransformer(model_name)
            self.model.max_seq_length = 512

            logger.info(
                "Successfully loaded base model",
                extra={
                    "model_name": model_name,
                    "max_seq_length": self.model.max_seq_length
                }
            )
            return True

        except Exception as e:
            logger.error(
                "Error loading base model",
                extra={"model_name": model_name, "error": str(e)},
                exc_info=True
            )
            return False

    def load_training_data(
        self,
        data_path: str,
        format_type: str = "csv"
    ) -> Tuple[List[InputExample], List[InputExample]]:
        """
        Load training data and split into train/validation sets.

        Args:
            data_path: Path to training data file
            format_type: Format of the data (csv or json)

        Returns:
            Tuple of (train_examples, val_examples)
        """
        try:
            logger.info(
                "Loading training data",
                extra={"path": data_path, "format": format_type}
            )

            if format_type == "csv":
                return self._load_csv_data(data_path)
            elif format_type == "json":
                return self._load_json_data(data_path)
            else:
                raise ValueError(f"Unsupported format: {format_type}")

        except Exception as e:
            logger.error(
                "Error loading training data",
                extra={"path": data_path, "error": str(e)},
                exc_info=True
            )
            return [], []

    def _load_csv_data(
        self,
        csv_path: str
    ) -> Tuple[List[InputExample], List[InputExample]]:
        """Load training data from CSV format."""
        try:
            df = pd.read_csv(csv_path)

            # Validate required columns
            required_cols = ['query', 'document', 'label']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"CSV must contain columns: {required_cols}")

            # Create InputExamples
            examples = []
            for _, row in df.iterrows():
                example = InputExample(
                    texts=[str(row['query']), str(row['document'])],
                    label=float(row['label'])
                )
                examples.append(example)

            # Split into train/validation
            val_size = int(len(examples) * 0.2)  # 20% validation
            train_examples = examples[val_size:]
            val_examples = examples[:val_size]

            logger.info(
                "Loaded CSV data",
                extra={
                    "total": len(examples),
                    "train": len(train_examples),
                    "validation": len(val_examples)
                }
            )
            return train_examples, val_examples

        except Exception as e:
            logger.error(
                "Error loading CSV data",
                extra={"error": str(e)},
                exc_info=True
            )
            return [], []

    def _load_json_data(
        self,
        json_path: str
    ) -> Tuple[List[InputExample], List[InputExample]]:
        """Load training data from JSON format."""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            training_pairs = data.get('training_pairs', [])

            # Create InputExamples
            examples = []
            for pair in training_pairs:
                example = InputExample(
                    texts=[pair['query'], pair['document']],
                    label=float(pair['label'])
                )
                examples.append(example)

            # Split into train/validation
            val_size = int(len(examples) * 0.2)
            train_examples = examples[val_size:]
            val_examples = examples[:val_size]

            logger.info(
                "Loaded JSON data",
                extra={
                    "total": len(examples),
                    "train": len(train_examples),
                    "validation": len(val_examples)
                }
            )
            return train_examples, val_examples

        except Exception as e:
            logger.error(
                "Error loading JSON data",
                extra={"error": str(e)},
                exc_info=True
            )
            return [], []

    def create_loss_function(self, loss_name: Optional[str] = None):
        """
        Create the appropriate loss function for training.

        Args:
            loss_name: Name of the loss function (defaults to CosineSimilarityLoss)

        Returns:
            Loss function instance
        """
        try:
            loss_name = loss_name or "CosineSimilarityLoss"

            if loss_name == "CosineSimilarityLoss":
                return losses.CosineSimilarityLoss(self.model)
            elif loss_name == "TripletLoss":
                return losses.TripletLoss(self.model)
            elif loss_name == "MultipleNegativesRankingLoss":
                return losses.MultipleNegativesRankingLoss(self.model)
            elif loss_name == "ContrastiveLoss":
                return losses.ContrastiveLoss(self.model)
            else:
                logger.warning(
                    "Unknown loss function, using CosineSimilarityLoss",
                    extra={"requested": loss_name}
                )
                return losses.CosineSimilarityLoss(self.model)

        except Exception as e:
            logger.error(
                "Error creating loss function",
                extra={"error": str(e)},
                exc_info=True
            )
            return losses.CosineSimilarityLoss(self.model)

    def create_evaluator(
        self,
        val_examples: List[InputExample]
    ) -> Optional[evaluation.SentenceEvaluator]:
        """
        Create evaluator for validation during training.

        Args:
            val_examples: Validation examples

        Returns:
            Evaluator instance or None
        """
        try:
            if not val_examples:
                logger.warning("No validation examples provided")
                return None

            sentences1 = [example.texts[0] for example in val_examples]
            sentences2 = [example.texts[1] for example in val_examples]
            scores = [example.label for example in val_examples]

            evaluator = EmbeddingSimilarityEvaluator(
                sentences1=sentences1,
                sentences2=sentences2,
                scores=scores,
                name="validation"
            )

            logger.debug(
                "Created evaluator",
                extra={"num_examples": len(val_examples)}
            )
            return evaluator

        except Exception as e:
            logger.error(
                "Error creating evaluator",
                extra={"error": str(e)},
                exc_info=True
            )
            return None

    def fine_tune_model(
        self,
        train_examples: List[InputExample],
        val_examples: Optional[List[InputExample]] = None,
        model_name_suffix: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fine-tune the embedding model with training data.

        Args:
            train_examples: Training examples
            val_examples: Validation examples (optional)
            model_name_suffix: Suffix for the output model name

        Returns:
            Dictionary containing training results
        """
        try:
            if not self.model:
                logger.error("No model loaded")
                return {"success": False, "error": "No model loaded"}

            if not train_examples:
                logger.error("No training examples provided")
                return {"success": False, "error": "No training examples"}

            logger.info(
                "Starting fine-tuning",
                extra={"train_examples": len(train_examples)}
            )

            # Create data loader
            train_dataloader = DataLoader(
                train_examples,
                shuffle=True,
                batch_size=self.config.batch_size
            )

            # Create loss function
            train_loss = self.create_loss_function()

            # Create evaluator
            evaluator = self.create_evaluator(val_examples) if val_examples else None

            # Generate model name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_suffix = model_name_suffix or "finetuned"
            model_name = f"{model_suffix}_{timestamp}"
            output_path = os.path.join(
                self.config.output_path or "./fine_tuned_models",
                model_name
            )

            # Training arguments
            training_args = {
                "train_objectives": [(train_dataloader, train_loss)],
                "epochs": self.config.num_epochs,
                "warmup_steps": self.config.warmup_steps,
                "output_path": output_path,
                "save_best_model": True,
                "show_progress_bar": True,
                "use_amp": self.config.use_amp
            }

            if evaluator:
                training_args.update({
                    "evaluator": evaluator,
                    "evaluation_steps": 500,
                    "callback": self._training_callback
                })

            # Start training
            logger.info("Beginning model fine-tuning")
            start_time = datetime.now()

            self.model.fit(**training_args)

            end_time = datetime.now()
            training_duration = (end_time - start_time).total_seconds()

            # Save training metadata
            metadata = {
                "model_name": model_name,
                "base_model": self.config.base_model,
                "training_start": start_time.isoformat(),
                "training_end": end_time.isoformat(),
                "training_duration_seconds": training_duration,
                "config": {
                    "batch_size": self.config.batch_size,
                    "num_epochs": self.config.num_epochs,
                    "learning_rate": self.config.learning_rate,
                    "warmup_steps": self.config.warmup_steps,
                    "use_amp": self.config.use_amp
                },
                "data_statistics": {
                    "train_examples": len(train_examples),
                    "val_examples": len(val_examples) if val_examples else 0
                },
                "training_history": [
                    {
                        "epoch": m.epoch,
                        "train_loss": m.train_loss,
                        "eval_cosine_accuracy": m.eval_cosine_accuracy,
                        "timestamp": m.timestamp
                    }
                    for m in self.training_history
                ]
            }

            metadata_path = os.path.join(output_path, "training_metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)

            logger.info(
                "Fine-tuning completed",
                extra={
                    "duration_seconds": training_duration,
                    "output_path": output_path
                }
            )

            return {
                "success": True,
                "model_path": output_path,
                "model_name": model_name,
                "training_duration": training_duration,
                "training_history": self.training_history,
                "metadata_path": metadata_path
            }

        except Exception as e:
            logger.error(
                "Error during fine-tuning",
                extra={"error": str(e)},
                exc_info=True
            )
            return {"success": False, "error": str(e)}

    def _training_callback(self, score: float, epoch: int, steps: int):
        """Callback function called during training for monitoring."""
        try:
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=0.0,
                eval_cosine_accuracy=score,
                timestamp=datetime.now().isoformat()
            )

            self.training_history.append(metrics)

            if score > self.best_eval_score:
                self.best_eval_score = score
                logger.info(
                    "New best validation score",
                    extra={"score": score, "epoch": epoch}
                )

        except Exception as e:
            logger.warning(
                "Error in training callback",
                extra={"error": str(e)}
            )

    def evaluate_model(
        self,
        model_path: str,
        test_data_path: str,
        format_type: str = "csv"
    ) -> Dict[str, Any]:
        """
        Evaluate a fine-tuned model on test data.

        Args:
            model_path: Path to the fine-tuned model
            test_data_path: Path to test data
            format_type: Format of test data (csv or json)

        Returns:
            Dictionary containing evaluation results
        """
        try:
            logger.info(
                "Evaluating model",
                extra={"model_path": model_path}
            )

            model = SentenceTransformer(model_path)

            if format_type == "csv":
                test_df = pd.read_csv(test_data_path)
                queries = test_df['query'].tolist()
                documents = test_df['document'].tolist()
                labels = test_df['label'].tolist()
            else:
                raise ValueError(f"Unsupported test data format: {format_type}")

            evaluator = EmbeddingSimilarityEvaluator(
                sentences1=queries,
                sentences2=documents,
                scores=labels,
                name="test_evaluation"
            )

            eval_score = evaluator(model)

            # Additional metrics
            embeddings1 = model.encode(queries)
            embeddings2 = model.encode(documents)

            cosine_similarities = []
            for emb1, emb2 in zip(embeddings1, embeddings2):
                similarity = np.dot(emb1, emb2) / (
                    np.linalg.norm(emb1) * np.linalg.norm(emb2)
                )
                cosine_similarities.append(similarity)

            correlation = np.corrcoef(cosine_similarities, labels)[0, 1]

            results = {
                "model_path": model_path,
                "test_data_path": test_data_path,
                "evaluation_score": eval_score,
                "correlation_with_labels": correlation,
                "num_test_samples": len(queries),
                "evaluation_timestamp": datetime.now().isoformat()
            }

            logger.info(
                "Evaluation completed",
                extra={
                    "score": eval_score,
                    "correlation": correlation
                }
            )
            return results

        except Exception as e:
            logger.error(
                "Error evaluating model",
                extra={"error": str(e)},
                exc_info=True
            )
            return {"error": str(e)}

    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List all available fine-tuned models.

        Returns:
            List of model information dictionaries
        """
        try:
            models = []
            models_dir = Path(self.config.output_path or "./fine_tuned_models")

            if not models_dir.exists():
                return models

            for model_dir in models_dir.iterdir():
                if model_dir.is_dir() and not model_dir.name.startswith('.'):
                    metadata_path = model_dir / "training_metadata.json"

                    model_info = {
                        "name": model_dir.name,
                        "path": str(model_dir),
                        "created": datetime.fromtimestamp(
                            model_dir.stat().st_ctime
                        ).isoformat()
                    }

                    if metadata_path.exists():
                        try:
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                            model_info.update({
                                "base_model": metadata.get("base_model"),
                                "training_duration": metadata.get(
                                    "training_duration_seconds"
                                ),
                                "train_examples": metadata.get(
                                    "data_statistics", {}
                                ).get("train_examples"),
                                "config": metadata.get("config")
                            })
                        except Exception as e:
                            logger.warning(
                                "Error loading metadata",
                                extra={
                                    "model": model_dir.name,
                                    "error": str(e)
                                }
                            )

                    models.append(model_info)

            models.sort(key=lambda x: x["created"], reverse=True)

            logger.info(
                "Listed available models",
                extra={"count": len(models)}
            )
            return models

        except Exception as e:
            logger.error(
                "Error listing models",
                extra={"error": str(e)},
                exc_info=True
            )
            return []

    def delete_model(self, model_name: str) -> bool:
        """
        Delete a fine-tuned model.

        Args:
            model_name: Name of the model to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            model_path = os.path.join(
                self.config.output_path or "./fine_tuned_models",
                model_name
            )

            if os.path.exists(model_path):
                shutil.rmtree(model_path)
                logger.info(
                    "Deleted model",
                    extra={"model_name": model_name}
                )
                return True
            else:
                logger.warning(
                    "Model not found",
                    extra={"model_name": model_name}
                )
                return False

        except Exception as e:
            logger.error(
                "Error deleting model",
                extra={"model_name": model_name, "error": str(e)},
                exc_info=True
            )
            return False


def create_model_finetuner(
    config: Optional[FineTuningConfig] = None
) -> ModelFineTuner:
    """
    Factory function to create a ModelFineTuner instance.

    Args:
        config: Fine-tuning configuration

    Returns:
        ModelFineTuner instance

    Examples:
        >>> config = FineTuningConfig(num_epochs=5)
        >>> finetuner = create_model_finetuner(config)
    """
    return ModelFineTuner(config=config)


def quick_finetune(
    training_data_path: str,
    base_model: str = "all-MiniLM-L6-v2",
    output_path: str = "./fine_tuned_models"
) -> Dict[str, Any]:
    """
    Quick fine-tuning function with default settings.

    Args:
        training_data_path: Path to training data CSV
        base_model: Base model name
        output_path: Output directory

    Returns:
        Fine-tuning results

    Examples:
        >>> result = quick_finetune("training_data.csv")
        >>> print(result["model_path"])
    """
    try:
        config = FineTuningConfig(
            base_model=base_model,
            output_path=output_path
        )

        finetuner = ModelFineTuner(config)

        if not finetuner.load_base_model():
            return {"success": False, "error": "Failed to load base model"}

        train_examples, val_examples = finetuner.load_training_data(
            training_data_path
        )

        if not train_examples:
            return {"success": False, "error": "No training data loaded"}

        results = finetuner.fine_tune_model(train_examples, val_examples)
        return results

    except Exception as e:
        logger.error(
            "Error in quick fine-tuning",
            extra={"error": str(e)},
            exc_info=True
        )
        return {"success": False, "error": str(e)}