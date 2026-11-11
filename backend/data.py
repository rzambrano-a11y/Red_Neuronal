"""Generación y validación de los datos didácticos."""

from dataclasses import dataclass

import numpy as np

from .config import DataConfig
from .exceptions import DatasetValidationError


@dataclass(frozen=True)
class ConnectionDataset:
    """Datos en forma de matriz ``(n, 2)`` y etiquetas en forma ``(n,)``.

    Se copian y protegen los arreglos para que otro componente no altere el
    conjunto después de haberlo validado.
    """

    features: np.ndarray
    labels: np.ndarray

    def __post_init__(self) -> None:
        # Normalizar tipos de entrada antes de verificar dimensiones y valores.
        features = np.asarray(self.features, dtype=np.float32)
        raw_labels = np.asarray(self.labels)

        # Cada fila debe representar exactamente una conexión con dos variables.
        if features.ndim != 2 or features.shape[1] != 2:
            raise DatasetValidationError(
                "features debe tener dos columnas: intentos y tráfico."
            )
        if raw_labels.ndim != 1 or raw_labels.shape[0] != features.shape[0]:
            raise DatasetValidationError(
                "labels debe tener una etiqueta por fila de features."
            )
        if features.shape[0] < 2:
            raise DatasetValidationError("El conjunto debe incluir al menos dos filas.")
        if not np.isfinite(features).all():
            raise DatasetValidationError("features contiene valores no finitos.")
        if not np.isin(raw_labels, (0, 1)).all():
            raise DatasetValidationError("Las etiquetas permitidas son 0 y 1.")

        # La copia evita retener referencias mutables que entregue el llamador.
        feature_copy = np.array(features, copy=True)
        label_copy = np.array(raw_labels, dtype=np.int32, copy=True)
        # Los arreglos de una instancia validada quedan de solo lectura.
        feature_copy.setflags(write=False)
        label_copy.setflags(write=False)
        object.__setattr__(self, "features", feature_copy)
        object.__setattr__(self, "labels", label_copy)

    @property
    def size(self) -> int:
        """Cantidad total de conexiones del conjunto."""
        return int(self.labels.shape[0])

    @property
    def normal_count(self) -> int:
        """Número de filas cuya etiqueta es 0 (normal)."""
        return int(np.count_nonzero(self.labels == 0))

    @property
    def suspicious_count(self) -> int:
        """Número de filas cuya etiqueta es 1 (sospechosa)."""
        return int(np.count_nonzero(self.labels == 1))

    def summary(self) -> dict[str, int]:
        """Resumen simple que se puede incluir en un informe JSON."""
        return {
            "total": self.size,
            "normal": self.normal_count,
            "suspicious": self.suspicious_count,
        }


def generate_dataset(config: DataConfig | None = None) -> ConnectionDataset:
    """Genera conexiones simuladas con una semilla reproducible.

    La regla de etiquetado conserva el ejercicio del taller. Los datos no
    representan tráfico de una red real.
    """

    # Si no se especifican opciones, usar los parámetros reproducibles del taller.
    settings = config or DataConfig()
    # Un generador local evita depender del estado aleatorio global de NumPy.
    rng = np.random.default_rng(settings.seed)

    # Crear cada columna por separado para que sus rangos sean explícitos.
    failed_attempts = rng.integers(
        0, settings.max_failed_attempts + 1, size=settings.sample_count
    )
    traffic_mb = rng.uniform(0, settings.max_traffic_mb, size=settings.sample_count)
    noise = rng.normal(0, settings.noise_std, size=settings.sample_count)

    # Puntaje didáctico: intentos pesan 55 %, tráfico 45 % y se agrega ruido.
    risk_score = (
        0.55 * (failed_attempts / settings.max_failed_attempts)
        + 0.45 * (traffic_mb / settings.max_traffic_mb)
        + noise
    )
    # La red recibe solo las dos entradas; el puntaje se usa para crear respuestas.
    labels = (risk_score >= settings.label_threshold).astype(np.int32)
    # Empacar las dos características en el orden público definido por el backend.
    features = np.column_stack((failed_attempts, traffic_mb))
    return ConnectionDataset(features=features, labels=labels)
