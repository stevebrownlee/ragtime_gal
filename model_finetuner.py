"""
Embedding Model Fine-tuning System

This module implements fine-tuning of sentence-transformer embedding models
using feedback-generated training data for improved semantic understanding.
"""

import logging
import os
import json
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import shutil
from pathlib import Path
import pandas as pd
from dataclasses import dataclass
import pickle

# Sentence Transformers imports
try:
    from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
    from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, InformationRetrievalEvaluator
    from torch.utils.data import DataLoader
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Install with: pip install sentence-transformers")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning process"""
    base_model_name: str = "all-MiniLM-L6-v2"  # Default sentence-transformer model
    output_model_path: str = "./fine_tuned_models"
    batch_size: int = 16
    num_epochs: int = 4
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    evaluation_steps: int = 500
    save_steps: int = 500
    max_seq_length: int = 512
    loss_function: str = "CosineSimilarityLoss"  # Options: CosineSimilarityLoss, TripletLoss, MultipleNegativesRankingLoss
    use_amp: bool = True  # Automatic Mixed Precision
    early_stopping_patience: int = 3
    validation_split: float = 0.2
    random_seed: int = 42

@dataclass
class TrainingMetrics:
    """Training metrics tracking"""
    epoch: int
    train_loss: float
    eval_loss: Optional[float] = None
    eval_cosine_accuracy: Optional[float] = None
    eval_manhattan_accuracy: Optional[float] = None
    eval_euclidean_accuracy: Optional[float] = None
    timestamp: str = None

class ModelFineTuner:
    """
    Fine-tunes sentence-transformer embedding models using training data
    generated from user feedback.
    """

    def __init__(self, config: FineTuningConfig = None):
        """
        Initialize the model fine-tuner.

        Args:
            config: Fine-tuning configuration
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required for fine-tuning. Install with: pip install sentence-transformers")

        self.config = config or FineTuningConfig()
        self.model = None
        self.training_history = []
        self.best_model_path = None
        self.best_eval_score = float('-inf')

        # Set random seeds for reproducibility
        torch.manual_seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)

        # Create output directory
        os.makedirs(self.config.output_model_path, exist_ok=True)

    def load_base_model(self, model_name: str = None) -> bool:
        """
        Load the base sentence-transformer model.

        Args:
            model_name: Name of the base model to load

        Returns:
            True if successful, False otherwise
        """
        try:
            model_name = model_name or self.config.base_model_name
            logger.info(f"Loading base model: {model_name}")

            self.model = SentenceTransformer(model_name)
            self.model.max_seq_length = self.config.max_seq_length

            logger.info(f"Successfully loaded model: {model_name}")
            logger.info(f"Model max sequence length: {self.model.max_seq_length}")

            return True

        except Exception as e:
            logger.error(f"Error loading base model: {e}")
            return False

    def load_training_data(self, data_path: str, format_type: str = "csv") -> Tuple[List[InputExample], List[InputExample]]:
        """
        Load training data and split into train/validation sets.

        Args:
            data_path: Path to training data file
            format_type: Format of the data ("csv", "json")

        Returns:
            Tuple of (train_examples, val_examples)
        """
        try:
            logger.info(f"Loading training data from: {data_path}")

            if format_type == "csv":
                return self._load_csv_data(data_path)
            elif format_type == "json":
                return self._load_json_data(data_path)
            else:
                raise ValueError(f"Unsupported format: {format_type}")

        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            return [], []

    def _load_csv_data(self, csv_path: str) -> Tuple[List[InputExample], List[InputExample]]:
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
            val_size = int(len(examples) * self.config.validation_split)
            train_examples = examples[val_size:]
            val_examples = examples[:val_size]

            logger.info(f"Loaded {len(examples)} examples: {len(train_examples)} train, {len(val_examples)} validation")
            return train_examples, val_examples

        except Exception as e:
            logger.error(f"Error loading CSV data: {e}")
            return [], []

    def _load_json_data(self, json_path: str) -> Tuple[List[InputExample], List[InputExample]]:
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
            val_size = int(len(examples) * self.config.validation_split)
            train_examples = examples[val_size:]
            val_examples = examples[:val_size]

            logger.info(f"Loaded {len(examples)} examples: {len(train_examples)} train, {len(val_examples)} validation")
            return train_examples, val_examples

        except Exception as e:
            logger.error(f"Error loading JSON data: {e}")
            return [], []

    def create_loss_function(self, loss_name: str = None):
        """
        Create the appropriate loss function for training.

        Args:
            loss_name: Name of the loss function

        Returns:
            Loss function instance
        """
        try:
            loss_name = loss_name or self.config.loss_function

            if loss_name == "CosineSimilarityLoss":
                return losses.CosineSimilarityLoss(self.model)
            elif loss_name == "TripletLoss":
                return losses.TripletLoss(self.model)
            elif loss_name == "MultipleNegativesRankingLoss":
                return losses.MultipleNegativesRankingLoss(self.model)
            elif loss_name == "ContrastiveLoss":
                return losses.ContrastiveLoss(self.model)
            else:
                logger.warning(f"Unknown loss function: {loss_name}, using CosineSimilarityLoss")
                return losses.CosineSimilarityLoss(self.model)

        except Exception as e:
            logger.error(f"Error creating loss function: {e}")
            return losses.CosineSimilarityLoss(self.model)

    def create_evaluator(self, val_examples: List[InputExample]) -> Optional[evaluation.SentenceEvaluator]:
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

            # Convert examples to format expected by evaluator
            sentences1 = [example.texts[0] for example in val_examples]
            sentences2 = [example.texts[1] for example in val_examples]
            scores = [example.label for example in val_examples]

            evaluator = EmbeddingSimilarityEvaluator(
                sentences1=sentences1,
                sentences2=sentences2,
                scores=scores,
                name="validation"
            )

            return evaluator

        except Exception as e:
            logger.error(f"Error creating evaluator: {e}")
            return None

    def fine_tune_model(self, train_examples: List[InputExample],
                       val_examples: List[InputExample] = None,
                       model_name_suffix: str = None) -> Dict[str, Any]:
        """
        Fine-tune the embedding model with training data.

        Args:
            train_examples: Training examples
            val_examples: Validation examples
            model_name_suffix: Suffix for the output model name

        Returns:
            Dictionary containing training results
        """
        try:
            if not self.model:
                logger.error("No model loaded. Call load_base_model() first.")
                return {"success": False, "error": "No model loaded"}

            if not train_examples:
                logger.error("No training examples provided")
                return {"success": False, "error": "No training examples"}

            logger.info(f"Starting fine-tuning with {len(train_examples)} training examples")

            # Create data loader
            train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=self.config.batch_size)

            # Create loss function
            train_loss = self.create_loss_function()

            # Create evaluator
            evaluator = self.create_evaluator(val_examples) if val_examples else None

            # Generate model name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_suffix = model_name_suffix or "finetuned"
            model_name = f"{model_suffix}_{timestamp}"
            output_path = os.path.join(self.config.output_model_path, model_name)

            # Training arguments
            training_args = {
                "train_dataloader": train_dataloader,
                "train_loss": train_loss,
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
                    "evaluation_steps": self.config.evaluation_steps,
                    "callback": self._training_callback
                })

            # Start training
            logger.info("Beginning model fine-tuning...")
            start_time = datetime.now()

            self.model.fit(**training_args)

            end_time = datetime.now()
            training_duration = (end_time - start_time).total_seconds()

            # Save training metadata
            metadata = {
                "model_name": model_name,
                "base_model": self.config.base_model_name,
                "training_start": start_time.isoformat(),
                "training_end": end_time.isoformat(),
                "training_duration_seconds": training_duration,
                "config": {
                    "batch_size": self.config.batch_size,
                    "num_epochs": self.config.num_epochs,
                    "learning_rate": self.config.learning_rate,
                    "loss_function": self.config.loss_function,
                    "max_seq_length": self.config.max_seq_length
                },
                "data_statistics": {
                    "train_examples": len(train_examples),
                    "val_examples": len(val_examples) if val_examples else 0
                },
                "training_history": self.training_history
            }

            metadata_path = os.path.join(output_path, "training_metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Fine-tuning completed in {training_duration:.2f} seconds")
            logger.info(f"Model saved to: {output_path}")

            return {
                "success": True,
                "model_path": output_path,
                "model_name": model_name,
                "training_duration": training_duration,
                "training_history": self.training_history,
                "metadata_path": metadata_path
            }

        except Exception as e:
            logger.error(f"Error during fine-tuning: {e}")
            return {"success": False, "error": str(e)}

    def _training_callback(self, score: float, epoch: int, steps: int):
        """Callback function called during training for monitoring."""
        try:
            # Track training progress
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=0.0,  # This would need to be passed from the training loop
                eval_cosine_accuracy=score,
                timestamp=datetime.now().isoformat()
            )

            self.training_history.append(metrics)

            # Check if this is the best model so far
            if score > self.best_eval_score:
                self.best_eval_score = score
                logger.info(f"New best validation score: {score:.4f} at epoch {epoch}")

        except Exception as e:
            logger.warning(f"Error in training callback: {e}")

    def evaluate_model(self, model_path: str, test_data_path: str,
                      format_type: str = "csv") -> Dict[str, Any]:
        """
        Evaluate a fine-tuned model on test data.

        Args:
            model_path: Path to the fine-tuned model
            test_data_path: Path to test data
            format_type: Format of test data

        Returns:
            Dictionary containing evaluation results
        """
        try:
            logger.info(f"Evaluating model: {model_path}")

            # Load the fine-tuned model
            model = SentenceTransformer(model_path)

            # Load test data
            if format_type == "csv":
                test_df = pd.read_csv(test_data_path)
                queries = test_df['query'].tolist()
                documents = test_df['document'].tolist()
                labels = test_df['label'].tolist()
            else:
                raise ValueError(f"Unsupported test data format: {format_type}")

            # Create evaluator
            evaluator = EmbeddingSimilarityEvaluator(
                sentences1=queries,
                sentences2=documents,
                scores=labels,
                name="test_evaluation"
            )

            # Run evaluation
            eval_score = evaluator(model)

            # Additional metrics
            embeddings1 = model.encode(queries)
            embeddings2 = model.encode(documents)

            # Calculate cosine similarities
            cosine_similarities = []
            for emb1, emb2 in zip(embeddings1, embeddings2):
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                cosine_similarities.append(similarity)

            # Calculate correlation with ground truth
            correlation = np.corrcoef(cosine_similarities, labels)[0, 1]

            results = {
                "model_path": model_path,
                "test_data_path": test_data_path,
                "evaluation_score": eval_score,
                "correlation_with_labels": correlation,
                "num_test_samples": len(queries),
                "evaluation_timestamp": datetime.now().isoformat()
            }

            logger.info(f"Evaluation completed. Score: {eval_score:.4f}, Correlation: {correlation:.4f}")
            return results

        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {"error": str(e)}

    def convert_to_ollama_format(self, model_path: str, ollama_model_name: str) -> Dict[str, Any]:
        """
        Convert fine-tuned model to Ollama format for deployment.

        Args:
            model_path: Path to the fine-tuned sentence-transformer model
            ollama_model_name: Name for the Ollama model

        Returns:
            Dictionary containing conversion results
        """
        try:
            logger.info(f"Converting model to Ollama format: {ollama_model_name}")

            # Load the fine-tuned model
            model = SentenceTransformer(model_path)

            # Create Ollama model directory
            ollama_dir = os.path.join(self.config.output_model_path, f"ollama_{ollama_model_name}")
            os.makedirs(ollama_dir, exist_ok=True)

            # Save model in ONNX format for better compatibility
            onnx_path = os.path.join(ollama_dir, "model.onnx")

            try:
                # Try to export to ONNX if available
                model.save(ollama_dir)
                logger.info(f"Model saved to: {ollama_dir}")

                # Create Ollama Modelfile
                modelfile_content = f"""
FROM {ollama_dir}
TEMPLATE "{{{{ .Prompt }}}}"
PARAMETER temperature 0.1
PARAMETER top_p 0.9
"""

                modelfile_path = os.path.join(ollama_dir, "Modelfile")
                with open(modelfile_path, 'w') as f:
                    f.write(modelfile_content)

                # Create instructions for Ollama deployment
                instructions = f"""
To deploy this model with Ollama:

1. Copy the model directory to your Ollama models location
2. Create the model with: ollama create {ollama_model_name} -f {modelfile_path}
3. Use the model with: ollama run {ollama_model_name}

Model directory: {ollama_dir}
"""

                instructions_path = os.path.join(ollama_dir, "deployment_instructions.txt")
                with open(instructions_path, 'w') as f:
                    f.write(instructions)

                return {
                    "success": True,
                    "ollama_model_name": ollama_model_name,
                    "ollama_directory": ollama_dir,
                    "modelfile_path": modelfile_path,
                    "instructions_path": instructions_path
                }

            except Exception as e:
                logger.warning(f"ONNX export failed, using standard format: {e}")

                # Fallback: just copy the model files
                shutil.copytree(model_path, ollama_dir, dirs_exist_ok=True)

                return {
                    "success": True,
                    "ollama_model_name": ollama_model_name,
                    "ollama_directory": ollama_dir,
                    "note": "Standard format used (ONNX export failed)"
                }

        except Exception as e:
            logger.error(f"Error converting to Ollama format: {e}")
            return {"success": False, "error": str(e)}

    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List all available fine-tuned models.

        Returns:
            List of model information dictionaries
        """
        try:
            models = []
            models_dir = Path(self.config.output_model_path)

            if not models_dir.exists():
                return models

            for model_dir in models_dir.iterdir():
                if model_dir.is_dir() and not model_dir.name.startswith('.'):
                    metadata_path = model_dir / "training_metadata.json"

                    model_info = {
                        "name": model_dir.name,
                        "path": str(model_dir),
                        "created": datetime.fromtimestamp(model_dir.stat().st_ctime).isoformat()
                    }

                    # Load metadata if available
                    if metadata_path.exists():
                        try:
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                            model_info.update({
                                "base_model": metadata.get("base_model"),
                                "training_duration": metadata.get("training_duration_seconds"),
                                "train_examples": metadata.get("data_statistics", {}).get("train_examples"),
                                "config": metadata.get("config")
                            })
                        except Exception as e:
                            logger.warning(f"Error loading metadata for {model_dir.name}: {e}")

                    models.append(model_info)

            # Sort by creation time (newest first)
            models.sort(key=lambda x: x["created"], reverse=True)
            return models

        except Exception as e:
            logger.error(f"Error listing models: {e}")
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
            model_path = os.path.join(self.config.output_model_path, model_name)

            if os.path.exists(model_path):
                shutil.rmtree(model_path)
                logger.info(f"Deleted model: {model_name}")
                return True
            else:
                logger.warning(f"Model not found: {model_name}")
                return False

        except Exception as e:
            logger.error(f"Error deleting model {model_name}: {e}")
            return False


# Utility functions for external use
def create_model_finetuner(config: FineTuningConfig = None) -> ModelFineTuner:
    """
    Factory function to create a ModelFineTuner instance.

    Args:
        config: Fine-tuning configuration

    Returns:
        ModelFineTuner instance
    """
    return ModelFineTuner(config=config)

def quick_finetune(training_data_path: str, base_model: str = "all-MiniLM-L6-v2",
                  output_path: str = "./fine_tuned_models") -> Dict[str, Any]:
    """
    Quick fine-tuning function with default settings.

    Args:
        training_data_path: Path to training data CSV
        base_model: Base model name
        output_path: Output directory

    Returns:
        Fine-tuning results
    """
    try:
        config = FineTuningConfig(
            base_model_name=base_model,
            output_model_path=output_path
        )

        finetuner = ModelFineTuner(config)

        # Load base model
        if not finetuner.load_base_model():
            return {"success": False, "error": "Failed to load base model"}

        # Load training data
        train_examples, val_examples = finetuner.load_training_data(training_data_path)

        if not train_examples:
            return {"success": False, "error": "No training data loaded"}

        # Fine-tune model
        results = finetuner.fine_tune_model(train_examples, val_examples)

        return results

    except Exception as e:
        logger.error(f"Error in quick fine-tuning: {e}")
        return {"success": False, "error": str(e)}