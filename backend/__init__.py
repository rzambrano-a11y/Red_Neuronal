"""Backend reutilizable para la clasificación didáctica de conexiones."""

from .config import ACADEMIC_NOTICE, BackendConfig, DataConfig, TrainingConfig
from .data import ConnectionDataset, generate_dataset
from .exceptions import (
    BackendError,
    DatasetValidationError,
    InputValidationError,
    ModelArtifactError,
    ModelNotTrainedError,
)
from .results import BaselineMetrics, EvaluationMetrics, PredictionResult, TrainingResult
from .service import ConnectionRiskService

__all__ = [
    "ACADEMIC_NOTICE",
    "BackendConfig",
    "BackendError",
    "BaselineMetrics",
    "ConnectionDataset",
    "ConnectionRiskService",
    "DataConfig",
    "DatasetValidationError",
    "EvaluationMetrics",
    "InputValidationError",
    "ModelArtifactError",
    "ModelNotTrainedError",
    "PredictionResult",
    "TrainingConfig",
    "TrainingResult",
    "generate_dataset",
]
