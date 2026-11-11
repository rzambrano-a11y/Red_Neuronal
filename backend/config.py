"""Configuración del experimento y de la red neuronal."""

from dataclasses import dataclass, field
from math import isfinite
from numbers import Integral


# Orden fijo de las columnas que recibe la red y que debe usar la interfaz.
FEATURE_NAMES = ("intentos_fallidos", "trafico_mb")
# Aviso incluido en las respuestas para dejar claro el alcance académico.
ACADEMIC_NOTICE = (
    "Demostración académica con datos simulados. Este resultado no confirma "
    "ni descarta un ataque informático real."
)


@dataclass(frozen=True)
class DataConfig:
    """Parámetros reproducibles para generar y dividir los datos.

    Los valores por defecto mantienen el ejercicio del taller. Se pueden
    cambiar al crear un BackendConfig propio.
    """

    sample_count: int = 2000  # Conexiones simuladas que se van a generar.
    seed: int = 42  # Semilla para repetir datos y división.
    test_size: float = 0.20  # Fracción reservada para la evaluación final.
    validation_split: float = 0.20  # Fracción del entrenamiento para validar épocas.
    max_failed_attempts: int = 20  # Límite superior del primer dato de entrada.
    max_traffic_mb: float = 2000.0  # Límite superior de tráfico usado al simular.
    label_threshold: float = 0.55  # Corte del puntaje simulado para asignar la clase.
    noise_std: float = 0.08  # Desviación del ruido que evita etiquetas perfectas.

    def __post_init__(self) -> None:
        # Rechazar tipos incorrectos antes de que NumPy o Keras fallen más tarde.
        for name, value in (
            ("sample_count", self.sample_count),
            ("seed", self.seed),
            ("max_failed_attempts", self.max_failed_attempts),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} debe ser un entero.")
        if self.sample_count < 10:
            raise ValueError("sample_count debe ser al menos 10.")
        if self.seed < 0:
            raise ValueError("seed debe ser un entero no negativo.")
        if not isfinite(self.test_size) or not 0 < self.test_size < 1:
            raise ValueError("test_size debe estar entre 0 y 1.")
        if not isfinite(self.validation_split) or not 0 < self.validation_split < 1:
            raise ValueError("validation_split debe estar entre 0 y 1.")
        if self.max_failed_attempts <= 0:
            raise ValueError("max_failed_attempts debe ser mayor que cero.")
        if not isfinite(self.max_traffic_mb) or self.max_traffic_mb <= 0:
            raise ValueError("max_traffic_mb debe ser mayor que cero.")
        if not isfinite(self.label_threshold) or not 0 < self.label_threshold < 1:
            raise ValueError("label_threshold debe estar entre 0 y 1.")
        if not isfinite(self.noise_std) or self.noise_std < 0:
            raise ValueError("noise_std no puede ser negativo.")


@dataclass(frozen=True)
class TrainingConfig:
    """Parámetros del optimizador, las épocas y la salida de entrenamiento."""

    seed: int = 42  # Semilla que inicializa los pesos de la red.
    epochs: int = 100  # Máximo de recorridos completos por los datos.
    batch_size: int = 32  # Ejemplos procesados antes de actualizar los pesos.
    threshold: float = 0.5  # Convierte la probabilidad en clase 0 o 1.
    learning_rate: float = 0.001  # Tamaño de cada ajuste del optimizador Adam.
    early_stopping: bool = True  # Detiene el ajuste si val_loss deja de mejorar.
    early_stopping_patience: int = 12  # Épocas sin mejora toleradas.
    early_stopping_min_delta: float = 0.0001  # Mejora mínima para contar como avance.
    verbose: int = 0  # Nivel de mensajes de Keras: 0 silencioso, 1 barra, 2 por época.

    def __post_init__(self) -> None:
        # Validar valores al crear la configuración, no durante el entrenamiento.
        for name, value in (
            ("seed", self.seed),
            ("epochs", self.epochs),
            ("batch_size", self.batch_size),
            ("early_stopping_patience", self.early_stopping_patience),
            ("verbose", self.verbose),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} debe ser un entero.")
        if not isinstance(self.early_stopping, bool):
            raise ValueError("early_stopping debe ser verdadero o falso.")
        if self.seed < 0:
            raise ValueError("seed debe ser un entero no negativo.")
        if self.epochs <= 0:
            raise ValueError("epochs debe ser mayor que cero.")
        if self.batch_size <= 0:
            raise ValueError("batch_size debe ser mayor que cero.")
        if not isfinite(self.threshold) or not 0 < self.threshold < 1:
            raise ValueError("threshold debe estar entre 0 y 1.")
        if not isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate debe ser mayor que cero.")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience debe ser mayor que cero.")
        if (
            not isfinite(self.early_stopping_min_delta)
            or self.early_stopping_min_delta < 0
        ):
            raise ValueError("early_stopping_min_delta no puede ser negativo.")
        if self.verbose not in (0, 1, 2):
            raise ValueError("verbose debe ser 0, 1 o 2.")


@dataclass(frozen=True)
class BackendConfig:
    """Agrupa opciones de datos y entrenamiento para compartirlas entre módulos."""

    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
