"""Tipos de respuesta fáciles de adaptar a JSON o a una interfaz."""

from dataclasses import dataclass
from typing import Any

from .config import ACADEMIC_NOTICE


@dataclass(frozen=True)
class BaselineMetrics:
    """Métricas para comparar la red con una regla simple de clase mayoritaria."""

    accuracy: float  # Aciertos totales del clasificador de referencia.
    precision: float  # Qué parte de sus alertas positivas era correcta.
    recall: float  # Qué parte de las conexiones positivas detectó.
    f1: float  # Balance entre precisión y recall.
    confusion_matrix: tuple[tuple[int, int], tuple[int, int]]  # Real x predicha.

    def to_dict(self) -> dict[str, Any]:
        """Convierte las métricas y la matriz a tipos que acepta JSON."""
        return {
            "strategy": "most_frequent",
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "confusion_matrix": [list(row) for row in self.confusion_matrix],
        }


@dataclass(frozen=True)
class EvaluationMetrics:
    """Indicadores del modelo sobre un conjunto que no usó para entrenar."""

    loss: float  # Entropía cruzada binaria; menor suele indicar mejor ajuste.
    accuracy: float  # Fracción de predicciones correctas.
    precision: float  # Proporción de alertas sospechosas que eran positivas.
    recall: float  # Proporción de positivos reales que el modelo detectó.
    f1: float  # Media armónica que combina precision y recall.
    specificity: float  # Proporción de conexiones normales bien reconocidas.
    balanced_accuracy: float | None  # Promedia recall por clase; None si falta una.
    roc_auc: float | None  # Discriminación entre clases; None si falta una clase.
    brier_score: float  # Error cuadrático de probabilidad; menor es mejor.
    threshold: float  # Corte aplicado para convertir probabilidad en clase.
    sample_count: int  # Número de ejemplos usados para estas métricas.
    normal_count: int  # Cantidad de etiquetas reales 0.
    suspicious_count: int  # Cantidad de etiquetas reales 1.
    confusion_matrix: tuple[tuple[int, int], tuple[int, int]]  # Real x predicha.

    def to_dict(self) -> dict[str, Any]:
        """Devuelve nombres de métricas estables para la interfaz y los informes."""
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "specificity": self.specificity,
            "balanced_accuracy": self.balanced_accuracy,
            "roc_auc": self.roc_auc,
            "brier_score": self.brier_score,
            "threshold": self.threshold,
            "sample_count": self.sample_count,
            "normal_count": self.normal_count,
            "suspicious_count": self.suspicious_count,
            "confusion_matrix": [list(row) for row in self.confusion_matrix],
        }


@dataclass(frozen=True)
class TrainingResult:
    """Resumen del experimento, incluido el historial por época de Keras."""

    metrics: EvaluationMetrics  # Evaluación de la red sobre la prueba final.
    baseline: BaselineMetrics  # Referencia simple, ajustada con y_train.
    dataset: dict[str, int]  # Recuento de filas y clases generadas.
    experiment: dict[str, Any]  # Parámetros para repetir la corrida.
    train_count: int  # Filas entregadas a fit antes de reservar validación.
    test_count: int  # Filas reservadas para la evaluación final.
    epochs_completed: int  # Épocas ejecutadas antes de detener el ajuste.
    best_epoch: int  # Época con menor val_loss.
    history: dict[str, tuple[float, ...]]  # Curvas loss/accuracy por época.
    architecture: str = "2-8-4-1"

    def to_dict(self) -> dict[str, Any]:
        """Serializa el resumen sin exponer objetos internos de TensorFlow."""
        return {
            "architecture": self.architecture,
            "academic_notice": ACADEMIC_NOTICE,
            "dataset": dict(self.dataset),
            "experiment": dict(self.experiment),
            "train_count": self.train_count,
            "test_count": self.test_count,
            "epochs_completed": self.epochs_completed,
            "best_epoch": self.best_epoch,
            "metrics": self.metrics.to_dict(),
            "baseline": self.baseline.to_dict(),
            "history": {name: list(values) for name, values in self.history.items()},
        }


@dataclass(frozen=True)
class PredictionResult:
    """Respuesta completa de una inferencia individual."""

    failed_attempts: int  # Entrada validada en el rango de entrenamiento.
    traffic_mb: float  # Segunda entrada, expresada en megabytes.
    probability: float  # Salida de la sigmoide, comprendida entre 0 y 1.
    threshold: float  # Umbral que produjo label.
    label: int  # 0 normal, 1 sospechosa.
    classification: str  # Nombre legible de la clase asignada.
    explanation: str  # Probabilidad y comparación con el umbral.
    academic_notice: str = ACADEMIC_NOTICE  # Alcance didáctico de la predicción.
    trace: dict[str, Any] | None = None  # Activaciones reales de esta inferencia.

    def to_dict(self) -> dict[str, Any]:
        """Produce el diccionario que la UI puede mostrar o enviar como JSON."""
        payload = {
            "failed_attempts": self.failed_attempts,
            "traffic_mb": self.traffic_mb,
            "probability": self.probability,
            "threshold": self.threshold,
            "label": self.label,
            "classification": self.classification,
            "explanation": self.explanation,
            "academic_notice": self.academic_notice,
        }
        if self.trace is not None:
            payload["trace"] = self.trace
        return payload
