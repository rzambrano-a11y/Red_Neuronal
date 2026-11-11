"""Definición de la arquitectura 2-8-4-1 del taller."""

from tensorflow import keras
from tensorflow.keras import layers


def build_model(
    seed: int = 42,
    threshold: float = 0.5,
    learning_rate: float = 0.001,
) -> keras.Sequential:
    """Construye y compila la red binaria usada para clasificar conexiones.

    La arquitectura 2-8-4-1 conserva las capas y activaciones del taller.
    """

    # Fijar la aleatoriedad antes de crear las capas y sus pesos iniciales.
    keras.utils.set_random_seed(seed)
    model = keras.Sequential(
        [
            # Cada ejemplo trae dos números: intentos fallidos y tráfico en MB.
            keras.Input(shape=(2,), name="caracteristicas_conexion"),
            # Capas ocultas: ReLU permite aprender relaciones no lineales.
            layers.Dense(8, activation="relu", name="oculta_8_relu"),
            layers.Dense(4, activation="relu", name="oculta_4_relu"),
            # Una salida sigmoide produce la probabilidad de clase sospechosa.
            layers.Dense(1, activation="sigmoid", name="probabilidad_sospechosa"),
        ],
        name="red_estudiantes_2_8_4_1",
    )
    # BCE es la pérdida para dos clases; Adam ajusta pesos y sesgos.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            # Las métricas de Keras usan el mismo corte que predict().
            keras.metrics.BinaryAccuracy(name="accuracy", threshold=threshold),
            keras.metrics.Precision(name="precision", thresholds=threshold),
            keras.metrics.Recall(name="recall", thresholds=threshold),
        ],
    )
    return model
