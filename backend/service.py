"""Servicio de entrenamiento, evaluación, inferencia y persistencia.

La interfaz debe usar esta capa para no duplicar preprocesamiento, reglas de
clasificación ni acceso directo a TensorFlow.
"""

import json
import math
import os
import tempfile
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence
from zipfile import ZIP_STORED, ZipFile

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from .config import (
    ACADEMIC_NOTICE,
    FEATURE_NAMES,
    BackendConfig,
    DataConfig,
    TrainingConfig,
)
from .data import ConnectionDataset, generate_dataset
from .exceptions import (
    DatasetValidationError,
    InputValidationError,
    ModelArtifactError,
    ModelNotTrainedError,
)
from .network import build_model
from .results import (
    BaselineMetrics,
    EvaluationMetrics,
    PredictionResult,
    TrainingResult,
)


_ARTIFACT_VERSION = 2
_ARTIFACT_FILES = {"model.keras", "metadata.json"}


class ConnectionRiskService:
    """API independiente de Streamlit, Flask, FastAPI o cualquier interfaz.

    Entrenar ocurre únicamente cuando se llama a :meth:`train`. La interfaz
    puede cargar un artefacto una vez al iniciar y reutilizar esta instancia.
    """

    def __init__(self, config: BackendConfig | None = None) -> None:
        # La configuración queda inyectable para cambiar datos o entrenamiento.
        self.config = config or BackendConfig()
        # Estos objetos solo existen después de entrenar o cargar un artefacto.
        self._model: Any | None = None
        self._scaler: StandardScaler | None = None
        # Se conserva la prueba para permitir evaluar nuevamente la misma corrida.
        self._test_dataset: ConnectionDataset | None = None
        self._threshold = self.config.training.threshold
        # El informe se comparte con model_info() y se incluye al guardar.
        self._last_training_report: dict[str, Any] | None = None

    @property
    def is_ready(self) -> bool:
        """Indica si hay un modelo y normalizador listos para inferencia."""

        return self._model is not None and self._scaler is not None

    def train(
        self,
        dataset: ConnectionDataset | None = None,
        callbacks: Sequence[Any] | None = None,
        verbose: int | None = None,
    ) -> TrainingResult:
        """Genera (si hace falta), divide, normaliza, entrena y evalúa.

        La transformación de escala se ajusta solo con los datos de
        entrenamiento, para evitar filtrar información desde el conjunto de
        prueba. El estado anterior del servicio se conserva si el nuevo
        entrenamiento falla antes de completarse.
        """

        # Permitir tanto datos simulados como un conjunto preparado por el usuario.
        samples = dataset if dataset is not None else generate_dataset(self.config.data)
        self._validate_training_dataset(samples)

        # Apartar la prueba desde el inicio y mantener proporciones de clases.
        x_train, x_test, y_train, y_test = train_test_split(
            samples.features,
            samples.labels,
            test_size=self.config.data.test_size,
            random_state=self.config.data.seed,
            stratify=samples.labels,
        )

        # Ajustar el escalador solo con entrenamiento previene fuga de información.
        candidate_scaler = StandardScaler()
        x_train_scaled = candidate_scaler.fit_transform(x_train)
        x_test_scaled = candidate_scaler.transform(x_test)

        # Crear una red nueva garantiza que esta corrida no herede pesos anteriores.
        candidate_model = build_model(
            seed=self.config.training.seed,
            threshold=self.config.training.threshold,
            learning_rate=self.config.training.learning_rate,
        )

        selected_verbose = (
            self.config.training.verbose if verbose is None else verbose
        )
        if selected_verbose not in (0, 1, 2):
            raise ValueError("verbose debe ser 0, 1 o 2.")

        # Dejar que la interfaz agregue callbacks y activar parada temprana por defecto.
        training_callbacks = list(callbacks or ())
        if self.config.training.early_stopping:
            training_callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.config.training.early_stopping_patience,
                    min_delta=self.config.training.early_stopping_min_delta,
                    restore_best_weights=True,
                )
            )

        # Keras ajusta los pesos con train y vigila val_loss con validation_split.
        history = candidate_model.fit(
            x_train_scaled,
            y_train,
            epochs=self.config.training.epochs,
            batch_size=self.config.training.batch_size,
            validation_split=self.config.data.validation_split,
            callbacks=training_callbacks,
            verbose=selected_verbose,
        )
        # La prueba se usa después del ajuste, únicamente para medir generalización.
        evaluation = self._evaluate_with(
            candidate_model,
            candidate_scaler,
            ConnectionDataset(x_test, y_test),
            threshold=self.config.training.threshold,
        )

        # Convertir el historial de Keras a números y tuplas fáciles de serializar.
        history_values = {
            name: tuple(float(value) for value in values)
            for name, values in history.history.items()
        }
        validation_losses = history_values.get("val_loss", ())
        best_epoch = (
            int(np.argmin(validation_losses)) + 1
            if validation_losses
            else len(history_values.get("loss", ()))
        )
        # Comparar con una regla simple ayuda a saber si la red aporta aprendizaje.
        baseline = self._evaluate_baseline(y_train, y_test)
        report = TrainingResult(
            metrics=evaluation,
            baseline=baseline,
            dataset=samples.summary(),
            experiment=asdict(self.config),
            train_count=int(len(y_train)),
            test_count=int(len(y_test)),
            epochs_completed=len(history_values.get("loss", ())),
            best_epoch=best_epoch,
            history=history_values,
        )

        # Publicar el nuevo estado solo después de completar entrenamiento e informe.
        self._model = candidate_model
        self._scaler = candidate_scaler
        self._test_dataset = ConnectionDataset(x_test, y_test)
        self._threshold = self.config.training.threshold
        self._last_training_report = report.to_dict()
        return report

    def evaluate(self, dataset: ConnectionDataset | None = None) -> EvaluationMetrics:
        """Evalúa con una muestra dada o con la prueba del último entrenamiento.

        Un artefacto cargado no incluye las filas originales de prueba; en ese caso
        se debe entregar un ConnectionDataset explícito.
        """

        self._require_ready()
        selected_dataset = dataset if dataset is not None else self._test_dataset
        if selected_dataset is None:
            raise DatasetValidationError(
                "No hay datos de prueba guardados; proporcione un ConnectionDataset."
            )
        if not isinstance(selected_dataset, ConnectionDataset):
            raise DatasetValidationError(
                "dataset debe ser una instancia de ConnectionDataset."
            )
        assert self._model is not None and self._scaler is not None
        return self._evaluate_with(
            self._model, self._scaler, selected_dataset, threshold=self._threshold
        )

    def predict(self, failed_attempts: Any, traffic_mb: Any) -> PredictionResult:
        """Devuelve probabilidad, clase, explicación breve y aviso académico."""

        self._require_ready()
        # Validar antes de llamar al modelo para mostrar errores claros en la UI.
        attempts = self._validate_attempts(failed_attempts)
        traffic = self._validate_traffic(traffic_mb)

        assert self._model is not None and self._scaler is not None
        # Aplicar el mismo escalador ajustado durante entrenamiento.
        features = np.asarray([[attempts, traffic]], dtype=np.float32)
        scaled_features = self._scaler.transform(features)
        # Sigmoide devuelve una probabilidad; el umbral produce la etiqueta binaria.
        probability = float(
            np.asarray(self._model.predict(scaled_features, verbose=0)).reshape(-1)[0]
        )
        trace = self._forward_trace(scaled_features, probability)
        label = int(probability >= self._threshold)
        classification = "sospechosa" if label else "normal"
        relation = "igual o superior" if label else "inferior"
        explanation = (
            f"La probabilidad ({probability:.1%}) es {relation} al umbral "
            f"({self._threshold:.1%}); el modelo la clasifica como {classification}."
        )
        return PredictionResult(
            failed_attempts=attempts,
            traffic_mb=traffic,
            probability=probability,
            threshold=self._threshold,
            label=label,
            classification=classification,
            explanation=explanation,
            academic_notice=ACADEMIC_NOTICE,
            trace=trace,
        )

    def _forward_trace(self, scaled_features: np.ndarray, probability: float) -> dict[str, Any]:
        """Activaciones reales de cada capa para la misma entrada ya escalada."""

        assert self._model is not None
        current = np.asarray(scaled_features, dtype=np.float32)
        layers: list[dict[str, Any]] = []
        for layer in self._model.layers:
            if not isinstance(layer, keras.layers.Dense):
                continue
            kernel, bias = layer.get_weights()
            pre = current @ kernel + bias
            name = layer.activation.__name__
            if name == "relu":
                activated = np.maximum(pre, 0)
            elif name == "sigmoid":
                activated = 1.0 / (1.0 + np.exp(-np.clip(pre, -60, 60)))
            else:
                activated = np.asarray(layer(current), dtype=np.float32)
            layers.append(
                {
                    "name": layer.name,
                    "activation": name,
                    "values": [round(float(value), 4) for value in np.asarray(activated).reshape(-1)],
                }
            )
            current = np.asarray(activated, dtype=np.float32)
        if layers and layers[-1]["activation"] == "sigmoid":
            layers[-1]["values"] = [round(float(probability), 4)]
        return {
            "scaled": [round(float(value), 4) for value in np.asarray(scaled_features).reshape(-1)],
            "layers": layers,
        }

    def connection_weights(self) -> list[list[list[float]]]:
        """Pesos reales de cada conexión, en el mismo orden que las capas 2-8-4-1."""

        self._require_ready()
        assert self._model is not None
        matrices: list[list[list[float]]] = []
        for layer in self._model.layers:
            if not isinstance(layer, keras.layers.Dense):
                continue
            kernel = np.asarray(layer.get_weights()[0], dtype=np.float64)
            matrices.append([[round(float(value), 4) for value in row] for row in kernel])
        return matrices

    def predict_many(
        self, connections: Sequence[tuple[Any, Any]]
    ) -> list[PredictionResult]:
        """Analiza pares (intentos fallidos, tráfico en MB) con las mismas reglas."""

        return [self.predict(attempts, traffic) for attempts, traffic in connections]

    def model_summary(self) -> str:
        """Retorna capas y parámetros de Keras como texto para CLI o una pantalla."""

        self._require_ready()
        assert self._model is not None
        lines: list[str] = []
        self._model.summary(print_fn=lines.append)
        return "\n".join(lines)

    def model_info(self) -> dict[str, Any]:
        """Describe entradas, umbral, estado y métricas sin exponer TensorFlow."""

        return {
            "architecture": "2-8-4-1",
            "features": list(FEATURE_NAMES),
            "threshold": self._threshold,
            "is_ready": self.is_ready,
            "academic_notice": ACADEMIC_NOTICE,
            "training_metrics": (
                deepcopy(self._last_training_report.get("metrics"))
                if self._last_training_report
                else None
            ),
        }

    def last_training_report(self) -> dict[str, Any] | None:
        """Copia del informe del último entrenamiento o del artefacto cargado.

        La interfaz puede mostrar métricas, curvas y la comparación de referencia
        sin leer atributos internos ni volver a evaluar.
        """

        if self._last_training_report is None:
            return None
        return deepcopy(self._last_training_report)

    def save(self, path: str | Path) -> Path:
        """Guarda red, normalizador y resumen en un único archivo ZIP.

        El bundle se escribe primero en un archivo temporal y luego se reemplaza
        el destino para evitar dejar un artefacto parcialmente escrito.
        """

        self._require_ready()
        assert self._model is not None and self._scaler is not None

        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary_bundle: str | None = None
        try:
            with tempfile.TemporaryDirectory(dir=target.parent) as temp_dir:
                model_path = Path(temp_dir) / "model.keras"
                # El formato .keras conserva pesos y configuración de la red.
                self._model.save(model_path)
                metadata = self._artifact_metadata()

                descriptor, temporary_bundle = tempfile.mkstemp(
                    prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
                )
                os.close(descriptor)
                # Los nombres fijos facilitan validar el archivo al cargarlo.
                with ZipFile(temporary_bundle, "w", compression=ZIP_STORED) as bundle:
                    bundle.write(model_path, "model.keras")
                    bundle.writestr(
                        "metadata.json",
                        json.dumps(metadata, ensure_ascii=False, indent=2),
                    )
            os.replace(temporary_bundle, target)
            temporary_bundle = None
            return target
        except Exception as exc:
            if temporary_bundle and os.path.exists(temporary_bundle):
                os.unlink(temporary_bundle)
            raise ModelArtifactError(f"No se pudo guardar el modelo: {exc}") from exc

    @classmethod
    def load(cls, path: str | Path) -> "ConnectionRiskService":
        """Carga un bundle y verifica versión, entradas, normalizador y arquitectura."""

        source = Path(path).expanduser().resolve()
        try:
            # Revisar el bundle antes de entregar sus datos al cargador de Keras.
            with ZipFile(source, "r") as bundle:
                names = bundle.namelist()
                if len(names) != 2 or set(names) != _ARTIFACT_FILES:
                    raise ModelArtifactError(
                        "El artefacto debe contener model.keras y metadata.json."
                    )
                corrupt_member = bundle.testzip()
                if corrupt_member is not None:
                    raise ModelArtifactError(
                        f"El archivo del modelo está dañado: {corrupt_member}."
                    )
                metadata = json.loads(bundle.read("metadata.json"))
                model_bytes = bundle.read("model.keras")

            if metadata.get("artifact_version") != _ARTIFACT_VERSION:
                raise ModelArtifactError("La versión del artefacto no es compatible.")
            if tuple(metadata.get("feature_names", ())) != FEATURE_NAMES:
                raise ModelArtifactError(
                    "El artefacto no usa las características esperadas."
                )

            data_config = DataConfig(**metadata["data_config"])
            training_config = TrainingConfig(**metadata["training_config"])
            training_report = metadata.get("training_report")
            if training_report is not None and not isinstance(training_report, dict):
                raise ModelArtifactError("El informe guardado no tiene un formato válido.")
            service = cls(BackendConfig(data=data_config, training=training_config))
            mean = np.asarray(metadata["scaler"]["mean"], dtype=np.float64)
            scale = np.asarray(metadata["scaler"]["scale"], dtype=np.float64)
            variance = np.asarray(metadata["scaler"]["variance"], dtype=np.float64)
            samples_seen = int(metadata["scaler"]["samples_seen"])
            if (
                mean.shape != (2,)
                or scale.shape != (2,)
                or variance.shape != (2,)
                or not np.isfinite(mean).all()
                or not np.isfinite(scale).all()
                or not np.isfinite(variance).all()
                or np.any(scale <= 0)
                or np.any(variance < 0)
                or samples_seen <= 0
            ):
                raise ModelArtifactError("El normalizador guardado no es válido.")

            # Reconstruir StandardScaler desde valores JSON, sin usar pickle.
            scaler = StandardScaler()
            scaler.mean_ = mean
            scaler.scale_ = scale
            scaler.var_ = variance
            scaler.n_features_in_ = 2
            scaler.n_samples_seen_ = samples_seen

            # Keras lee el archivo temporal; los pesos quedan cargados en memoria.
            with tempfile.TemporaryDirectory() as temp_dir:
                model_path = Path(temp_dir) / "model.keras"
                model_path.write_bytes(model_bytes)
                model = keras.models.load_model(model_path)

            input_width = model.input_shape[-1]
            output_width = model.output_shape[-1]
            # Evitar cargar un modelo con forma distinta a la que espera la interfaz.
            dense_layers = [
                layer for layer in model.layers if isinstance(layer, keras.layers.Dense)
            ]
            dense_units = [layer.units for layer in dense_layers]
            dense_activations = [
                getattr(layer.activation, "__name__", "") for layer in dense_layers
            ]
            if (
                input_width != 2
                or output_width != 1
                or dense_units != [8, 4, 1]
                or dense_activations != ["relu", "relu", "sigmoid"]
                or metadata.get("architecture") != "2-8-4-1"
            ):
                raise ModelArtifactError("La arquitectura guardada no es 2-8-4-1.")
            saved_threshold = float(metadata["threshold"])
            if (
                not 0 < saved_threshold < 1
                or not math.isclose(saved_threshold, training_config.threshold)
            ):
                raise ModelArtifactError("El umbral guardado no es válido.")

            service._model = model
            service._scaler = scaler
            service._threshold = saved_threshold
            service._last_training_report = training_report
            return service
        except ModelArtifactError:
            raise
        except Exception as exc:
            raise ModelArtifactError(f"No se pudo cargar el modelo: {exc}") from exc

    def _artifact_metadata(self) -> dict[str, Any]:
        """Recoge configuración e información del preprocesamiento del modelo."""
        assert self._scaler is not None
        return {
            "artifact_version": _ARTIFACT_VERSION,
            "architecture": "2-8-4-1",
            "feature_names": list(FEATURE_NAMES),
            "threshold": self._threshold,
            "data_config": self.config.data.__dict__,
            "training_config": self.config.training.__dict__,
            "training_report": self._last_training_report,
            "scaler": {
                "mean": self._scaler.mean_.tolist(),
                "scale": self._scaler.scale_.tolist(),
                "variance": self._scaler.var_.tolist(),
                "samples_seen": int(self._scaler.n_samples_seen_),
            },
        }

    def _evaluate_with(
        self,
        model: Any,
        scaler: StandardScaler,
        dataset: ConnectionDataset,
        threshold: float,
    ) -> EvaluationMetrics:
        """Calcula métricas de clasificación y de calidad de probabilidades."""

        # Escalar la muestra con estadísticas aprendidas exclusivamente en train.
        x_scaled = scaler.transform(dataset.features)
        loss_results = model.evaluate(
            x_scaled, dataset.labels, verbose=0, return_dict=True
        )
        probabilities = np.asarray(model.predict(x_scaled, verbose=0)).reshape(-1)
        predicted = (probabilities >= threshold).astype(np.int32)
        # Orden de la matriz: clase real en filas y predicción en columnas.
        matrix = confusion_matrix(dataset.labels, predicted, labels=[0, 1])
        has_both_classes = np.unique(dataset.labels).size == 2
        true_negatives = int(matrix[0, 0])
        false_positives = int(matrix[0, 1])
        specificity = (
            true_negatives / (true_negatives + false_positives)
            if true_negatives + false_positives
            else 0.0
        )
        return EvaluationMetrics(
            loss=float(loss_results["loss"]),
            accuracy=float(accuracy_score(dataset.labels, predicted)),
            precision=float(precision_score(dataset.labels, predicted, zero_division=0)),
            recall=float(recall_score(dataset.labels, predicted, zero_division=0)),
            f1=float(f1_score(dataset.labels, predicted, zero_division=0)),
            specificity=float(specificity),
            balanced_accuracy=(
                float(balanced_accuracy_score(dataset.labels, predicted))
                if has_both_classes
                else None
            ),
            roc_auc=(
                float(roc_auc_score(dataset.labels, probabilities))
                if has_both_classes
                else None
            ),
            brier_score=float(brier_score_loss(dataset.labels, probabilities)),
            threshold=threshold,
            sample_count=dataset.size,
            normal_count=dataset.normal_count,
            suspicious_count=dataset.suspicious_count,
            confusion_matrix=(
                (int(matrix[0, 0]), int(matrix[0, 1])),
                (int(matrix[1, 0]), int(matrix[1, 1])),
            ),
        )

    def _evaluate_baseline(
        self, y_train: np.ndarray, y_test: np.ndarray
    ) -> BaselineMetrics:
        """Mide cuánto se logra prediciendo siempre la clase mayoritaria de train."""

        # Las columnas de entrada son irrelevantes para esta estrategia base.
        baseline = DummyClassifier(strategy="most_frequent")
        baseline.fit(np.zeros((len(y_train), 1)), y_train)
        predictions = baseline.predict(np.zeros((len(y_test), 1)))
        matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
        return BaselineMetrics(
            accuracy=float(accuracy_score(y_test, predictions)),
            precision=float(precision_score(y_test, predictions, zero_division=0)),
            recall=float(recall_score(y_test, predictions, zero_division=0)),
            f1=float(f1_score(y_test, predictions, zero_division=0)),
            confusion_matrix=(
                (int(matrix[0, 0]), int(matrix[0, 1])),
                (int(matrix[1, 0]), int(matrix[1, 1])),
            ),
        )

    def _validate_training_dataset(self, dataset: ConnectionDataset) -> None:
        """Evita divisiones estratificadas imposibles o métricas sin clases."""
        if not isinstance(dataset, ConnectionDataset):
            raise DatasetValidationError(
                "dataset debe ser una instancia de ConnectionDataset."
            )
        counts = np.bincount(dataset.labels, minlength=2)
        if np.any(counts < 2):
            raise DatasetValidationError(
                "El entrenamiento requiere al menos dos ejemplos de cada clase."
            )
        test_count = math.ceil(dataset.size * self.config.data.test_size)
        train_count = dataset.size - test_count
        if test_count < 2 or train_count < 2:
            raise DatasetValidationError(
                "La división necesita al menos dos filas tanto para prueba "
                "como para entrenamiento."
            )

    def _validate_attempts(self, value: Any) -> int:
        """Convierte intentos de un formulario y exige entero dentro del rango."""
        if isinstance(value, bool):
            raise InputValidationError("Los intentos fallidos deben ser un entero.")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise InputValidationError(
                "Los intentos fallidos deben ser un número entero."
            ) from exc
        if not math.isfinite(number) or not number.is_integer():
            raise InputValidationError("Los intentos fallidos deben ser un entero.")
        attempts = int(number)
        if not 0 <= attempts <= self.config.data.max_failed_attempts:
            raise InputValidationError(
                "Los intentos fallidos deben estar entre 0 y "
                f"{self.config.data.max_failed_attempts}."
            )
        return attempts

    def _validate_traffic(self, value: Any) -> float:
        """Convierte tráfico de un formulario y exige un valor finito en rango."""
        if isinstance(value, bool):
            raise InputValidationError("El tráfico debe ser un número finito.")
        try:
            traffic = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise InputValidationError(
                "El tráfico debe ser un número finito en MB."
            ) from exc
        if not math.isfinite(traffic):
            raise InputValidationError("El tráfico debe ser un número finito.")
        if not 0 <= traffic <= self.config.data.max_traffic_mb:
            raise InputValidationError(
                f"El tráfico debe estar entre 0 y {self.config.data.max_traffic_mb:g} MB."
            )
        return traffic

    def _require_ready(self) -> None:
        """Falla pronto si todavía no hay modelo ni normalizador disponibles."""
        if not self.is_ready:
            raise ModelNotTrainedError(
                "Entrene o cargue el modelo antes de evaluar o predecir."
            )
