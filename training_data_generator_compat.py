"""
Backward compatibility wrapper for training_data_generator.py

This module maintains compatibility with existing code that imports from
training_data_generator.py. All imports are redirected to the new
ragtime.services.training_data_gen module.

Examples:
    >>> from training_data_generator import TrainingDataGenerator  # Old style
    >>> from ragtime.services.training_data_gen import TrainingDataGenerator  # New style
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "Importing from 'training_data_generator' is deprecated. "
    "Use 'from ragtime.services.training_data_gen import TrainingDataGenerator' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from new location
from ragtime.services.training_data_gen import (
    TrainingPair,
    TrainingDataGenerator,
    create_training_data_generator,
)

__all__ = [
    'TrainingPair',
    'TrainingDataGenerator',
    'create_training_data_generator',
]