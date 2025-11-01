"""
Phase 3 Integration Module

This module provides integration points for the complete Phase 3 implementation
with the existing RAG system, including Flask app integration and workflow management.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from flask import Flask, request, jsonify

# Import Phase 3 modules
from training_data_generator import create_training_data_generator
from model_finetuner import create_model_finetuner, FineTuningConfig
from ab_testing import create_ab_test, ABTestConfig, QueryResult
from embed_enhanced import model_manager, register_fine_tuned_model, switch_embedding_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Phase3Manager:
    """
    Central manager for Phase 3 functionality integration.
    """

    def __init__(self, conport_client=None, workspace_id: str = None, chroma_db=None):
        """
        Initialize Phase 3 manager.

        Args:
            conport_client: ConPort MCP client
            workspace_id: Workspace identifier
            chroma_db: ChromaDB instance
        """
        self.conport_client = conport_client
        self.workspace_id = workspace_id
        self.chroma_db = chroma_db

        # Initialize components
        self.training_generator = None
        self.model_finetuner = None
        self.ab_test_framework = None

        # Configuration
        self.config = {
            "training_data_path": "./training_data",
            "fine_tuned_models_path": "./fine_tuned_models",
            "ab_test_data_path": "./ab_test_data",
            "auto_training_enabled": False,
            "auto_training_threshold": 100,  # Minimum feedback entries
            "current_ab_test": None
        }

        self._initialize_components()

    def _initialize_components(self):
        """Initialize Phase 3 components."""
        try:
            # Initialize training data generator
            if self.conport_client and self.workspace_id:
                self.training_generator = create_training_data_generator(
                    conport_client=self.conport_client,
                    workspace_id=self.workspace_id,
                    chroma_db=self.chroma_db
                )
                logger.info("Training data generator initialized")

            # Initialize model fine-tuner
            finetuning_config = FineTuningConfig(
                output_model_path=self.config["fine_tuned_models_path"]
            )
            self.model_finetuner = create_model_finetuner(finetuning_config)
            logger.info("Model fine-tuner initialized")

        except Exception as e:
            logger.error(f"Error initializing Phase 3 components: {e}")

    def generate_training_data(self, days_back: int = 90,
                             include_hard_negatives: bool = True) -> Dict[str, Any]:
        """
        Generate training data from feedback.

        Args:
            days_back: Number of days to look back for feedback
            include_hard_negatives: Whether to include hard negative mining

        Returns:
            Dictionary containing generation results
        """
        try:
            if not self.training_generator:
                return {"success": False, "error": "Training generator not initialized"}

            logger.info("Starting training data generation...")

            results = self.training_generator.generate_training_data(
                days_back=days_back,
                include_hard_negatives=include_hard_negatives,
                output_format="sentence_transformers",
                output_path=self.config["training_data_path"]
            )

            if results["success"]:
                logger.info(f"Training data generated successfully: {results['data_statistics']['total_training_pairs']} pairs")
            else:
                logger.error(f"Training data generation failed: {results.get('error')}")

            return results

        except Exception as e:
            logger.error(f"Error generating training data: {e}")
            return {"success": False, "error": str(e)}

    def fine_tune_model(self, training_data_path: str = None,
                       model_name_suffix: str = None) -> Dict[str, Any]:
        """
        Fine-tune an embedding model.

        Args:
            training_data_path: Path to training data (optional)
            model_name_suffix: Suffix for the model name

        Returns:
            Dictionary containing fine-tuning results
        """
        try:
            if not self.model_finetuner:
                return {"success": False, "error": "Model fine-tuner not initialized"}

            # Use default training data path if not provided
            if not training_data_path:
                training_data_path = os.path.join(self.config["training_data_path"], "training_data_*.csv")
                # Find the most recent training data file
                import glob
                files = glob.glob(training_data_path)
                if not files:
                    return {"success": False, "error": "No training data found"}
                training_data_path = max(files, key=os.path.getctime)

            logger.info(f"Starting model fine-tuning with data: {training_data_path}")

            # Load base model
            if not self.model_finetuner.load_base_model():
                return {"success": False, "error": "Failed to load base model"}

            # Load training data
            train_examples, val_examples = self.model_finetuner.load_training_data(training_data_path)

            if not train_examples:
                return {"success": False, "error": "No training examples loaded"}

            # Fine-tune model
            results = self.model_finetuner.fine_tune_model(
                train_examples,
                val_examples,
                model_name_suffix
            )

            if results["success"]:
                logger.info(f"Model fine-tuning completed: {results['model_name']}")

                # Register the fine-tuned model
                model_path = results["model_path"]
                model_name = f"finetuned_{results['model_name']}"

                if register_fine_tuned_model(model_name, model_path):
                    logger.info(f"Fine-tuned model registered: {model_name}")
                    results["registered_name"] = model_name
                else:
                    logger.warning("Failed to register fine-tuned model")

            return results

        except Exception as e:
            logger.error(f"Error fine-tuning model: {e}")
            return {"success": False, "error": str(e)}

    def start_ab_test(self, original_model: str, fine_tuned_model: str,
                     test_name: str = None, traffic_split: float = 0.1) -> Dict[str, Any]:
        """
        Start an A/B test between original and fine-tuned models.

        Args:
            original_model: Name of the original model
            fine_tuned_model: Name of the fine-tuned model
            test_name: Name of the test (optional)
            traffic_split: Initial traffic split for fine-tuned model

        Returns:
            Dictionary containing test setup results
        """
        try:
            if not test_name:
                test_name = f"ab_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Create A/B test configuration
            config = ABTestConfig(
                test_name=test_name,
                model_a_name=original_model,
                model_b_name=fine_tuned_model,
                traffic_split=traffic_split
            )

            # Create A/B test framework
            self.ab_test_framework = create_ab_test(config, self.config["ab_test_data_path"])

            # Start with gradual rollout
            success = self.ab_test_framework.update_traffic_split(traffic_split)

            if success:
                self.config["current_ab_test"] = test_name
                logger.info(f"A/B test started: {test_name} with {traffic_split:.1%} traffic to {fine_tuned_model}")

                return {
                    "success": True,
                    "test_name": test_name,
                    "original_model": original_model,
                    "fine_tuned_model": fine_tuned_model,
                    "traffic_split": traffic_split
                }
            else:
                return {"success": False, "error": "Failed to set traffic split"}

        except Exception as e:
            logger.error(f"Error starting A/B test: {e}")
            return {"success": False, "error": str(e)}

    def record_query_feedback(self, query_id: str, query_text: str,
                            response_time_ms: float, similarity_scores: List[float],
                            retrieved_documents: List[str], user_rating: int = None,
                            user_feedback: str = None) -> bool:
        """
        Record query feedback for A/B testing.

        Args:
            query_id: Unique query identifier
            query_text: The query text
            response_time_ms: Response time in milliseconds
            similarity_scores: List of similarity scores
            retrieved_documents: List of retrieved document IDs
            user_rating: User rating (1-5)
            user_feedback: User feedback text

        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.ab_test_framework:
                return False

            # Determine which model variant was used
            variant = self.ab_test_framework.assign_variant(query_id)

            # Switch to appropriate model for this query
            if variant == 'A':
                model_name = self.ab_test_framework.config.model_a_name
            else:
                model_name = self.ab_test_framework.config.model_b_name

            # Record the result
            result = QueryResult(
                query_id=query_id,
                query_text=query_text,
                model_variant=variant,
                timestamp=datetime.now(),
                response_time_ms=response_time_ms,
                similarity_scores=similarity_scores,
                retrieved_documents=retrieved_documents,
                user_rating=user_rating,
                user_feedback=user_feedback,
                metadata={"model_name": model_name}
            )

            self.ab_test_framework.record_query_result(result)
            return True

        except Exception as e:
            logger.error(f"Error recording query feedback: {e}")
            return False

    def get_ab_test_status(self) -> Dict[str, Any]:
        """Get current A/B test status."""
        try:
            if not self.ab_test_framework:
                return {"status": "no_active_test"}

            return self.ab_test_framework.get_test_status()

        except Exception as e:
            logger.error(f"Error getting A/B test status: {e}")
            return {"error": str(e)}

    def generate_ab_test_report(self) -> Dict[str, Any]:
        """Generate A/B test comparison report."""
        try:
            if not self.ab_test_framework:
                return {"error": "No active A/B test"}

            return self.ab_test_framework.generate_comparison_report()

        except Exception as e:
            logger.error(f"Error generating A/B test report: {e}")
            return {"error": str(e)}

    def auto_training_check(self) -> Dict[str, Any]:
        """
        Check if automatic training should be triggered based on feedback volume.

        Returns:
            Dictionary containing check results and actions taken
        """
        try:
            if not self.config["auto_training_enabled"] or not self.training_generator:
                return {"auto_training": False, "reason": "Auto training disabled"}

            # Get recent feedback count
            feedback_data = self.training_generator.get_feedback_data(days_back=30)
            feedback_count = len(feedback_data)

            if feedback_count >= self.config["auto_training_threshold"]:
                logger.info(f"Auto training triggered: {feedback_count} feedback entries")

                # Generate training data
                training_results = self.generate_training_data(days_back=30)

                if training_results["success"]:
                    # Trigger fine-tuning
                    finetuning_results = self.fine_tune_model()

                    return {
                        "auto_training": True,
                        "feedback_count": feedback_count,
                        "training_data_generated": True,
                        "model_finetuned": finetuning_results["success"],
                        "results": {
                            "training": training_results,
                            "finetuning": finetuning_results
                        }
                    }
                else:
                    return {
                        "auto_training": True,
                        "feedback_count": feedback_count,
                        "training_data_generated": False,
                        "error": training_results.get("error")
                    }
            else:
                return {
                    "auto_training": False,
                    "feedback_count": feedback_count,
                    "threshold": self.config["auto_training_threshold"],
                    "reason": "Insufficient feedback data"
                }

        except Exception as e:
            logger.error(f"Error in auto training check: {e}")
            return {"auto_training": False, "error": str(e)}

# Global Phase 3 manager instance
phase3_manager = None

def initialize_phase3(conport_client=None, workspace_id: str = None, chroma_db=None):
    """Initialize Phase 3 manager."""
    global phase3_manager
    phase3_manager = Phase3Manager(conport_client, workspace_id, chroma_db)
    return phase3_manager

def add_phase3_routes(app: Flask):
    """Add Phase 3 routes to Flask app."""

    @app.route('/api/phase3/generate-training-data', methods=['POST'])
    def generate_training_data():
        """Generate training data from feedback."""
        try:
            if not phase3_manager:
                return jsonify({"error": "Phase 3 not initialized"}), 500

            data = request.get_json() or {}
            days_back = data.get('days_back', 90)
            include_hard_negatives = data.get('include_hard_negatives', True)

            results = phase3_manager.generate_training_data(days_back, include_hard_negatives)

            if results["success"]:
                return jsonify(results), 200
            else:
                return jsonify(results), 400

        except Exception as e:
            logger.error(f"Error in generate-training-data endpoint: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/phase3/fine-tune-model', methods=['POST'])
    def fine_tune_model():
        """Fine-tune embedding model."""
        try:
            if not phase3_manager:
                return jsonify({"error": "Phase 3 not initialized"}), 500

            data = request.get_json() or {}
            training_data_path = data.get('training_data_path')
            model_name_suffix = data.get('model_name_suffix')

            results = phase3_manager.fine_tune_model(training_data_path, model_name_suffix)

            if results["success"]:
                return jsonify(results), 200
            else:
                return jsonify(results), 400

        except Exception as e:
            logger.error(f"Error in fine-tune-model endpoint: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/phase3/start-ab-test', methods=['POST'])
    def start_ab_test():
        """Start A/B test."""
        try:
            if not phase3_manager:
                return jsonify({"error": "Phase 3 not initialized"}), 500

            data = request.get_json() or {}
            original_model = data.get('original_model', 'default')
            fine_tuned_model = data.get('fine_tuned_model')
            test_name = data.get('test_name')
            traffic_split = data.get('traffic_split', 0.1)

            if not fine_tuned_model:
                return jsonify({"error": "fine_tuned_model is required"}), 400

            results = phase3_manager.start_ab_test(
                original_model, fine_tuned_model, test_name, traffic_split
            )

            if results["success"]:
                return jsonify(results), 200
            else:
                return jsonify(results), 400

        except Exception as e:
            logger.error(f"Error in start-ab-test endpoint: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/phase3/ab-test-status', methods=['GET'])
    def ab_test_status():
        """Get A/B test status."""
        try:
            if not phase3_manager:
                return jsonify({"error": "Phase 3 not initialized"}), 500

            status = phase3_manager.get_ab_test_status()
            return jsonify(status), 200

        except Exception as e:
            logger.error(f"Error in ab-test-status endpoint: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/phase3/ab-test-report', methods=['GET'])
    def ab_test_report():
        """Generate A/B test report."""
        try:
            if not phase3_manager:
                return jsonify({"error": "Phase 3 not initialized"}), 500

            report = phase3_manager.generate_ab_test_report()
            return jsonify(report), 200

        except Exception as e:
            logger.error(f"Error in ab-test-report endpoint: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/phase3/models', methods=['GET'])
    def list_models():
        """List available embedding models."""
        try:
            models = model_manager.get_model_info()
            return jsonify(models), 200

        except Exception as e:
            logger.error(f"Error in list-models endpoint: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/phase3/switch-model', methods=['POST'])
    def switch_model():
        """Switch active embedding model."""
        try:
            data = request.get_json() or {}
            model_name = data.get('model_name')

            if not model_name:
                return jsonify({"error": "model_name is required"}), 400

            success = switch_embedding_model(model_name)

            if success:
                return jsonify({"success": True, "active_model": model_name}), 200
            else:
                return jsonify({"error": "Failed to switch model"}), 400

        except Exception as e:
            logger.error(f"Error in switch-model endpoint: {e}")
            return jsonify({"error": str(e)}), 500

    logger.info("Phase 3 API routes added to Flask app")