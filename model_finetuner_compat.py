"""
Backward compatibility wrapper for model_finetuner.py

This module maintains compatibility with existing code that imports from
model_finetuner.py. All imports are redirected to the new
ragtime.services.model_finetuner module.

Examples:
    >>> from model_finetuner import ModelFineTuner  # Old style
    >>> from ragtime.services.model_finetuner import ModelFineTuner  # New style
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "Importing from 'model_finetuner' is deprecated. "
    "Use 'from ragtime.services.model_finetuner import ModelFineTuner' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from new location
from ragtime.services.model_finetuner import (
    TrainingMetrics,
    ModelFineTuner,
    create_model_finetuner,
    quick_finetune,
)

# Import config from models
from ragtime.models.training import FineTuningConfig

__all__ = [
    'FineTuningConfig',
    'TrainingMetrics',
    'ModelFineTuner',
    'create_model_finetuner',
    'quick_finetune',
]