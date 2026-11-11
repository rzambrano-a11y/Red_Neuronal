"""Excepciones públicas del backend."""


class BackendError(Exception):
    """Error base controlado por el backend."""


class InputValidationError(BackendError, ValueError):
    """Los datos enviados para analizar no son válidos."""


class DatasetValidationError(BackendError, ValueError):
    """El conjunto de datos no tiene la forma o etiquetas esperadas."""


class ModelNotTrainedError(BackendError, RuntimeError):
    """Se solicitó inferencia antes de entrenar o cargar un modelo."""


class ModelArtifactError(BackendError, RuntimeError):
    """El archivo del modelo está incompleto o no se puede leer."""
