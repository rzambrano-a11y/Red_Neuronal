"""Interfaz gráfica del taller de clasificación de conexiones simuladas.

Tres secciones:
- Analizar: tarea principal, clasificar una conexión.
- Resultados: métricas, matriz, curvas y un nuevo entrenamiento.
- Cómo funciona: el experimento, en el lenguaje del taller.

El modelo se carga una vez por sesión. Entrenar reemplaza los pesos y
descarta clasificaciones hechas con el modelo anterior.
"""

from __future__ import annotations

import html
import os
import time
from pathlib import Path
from types import SimpleNamespace

import streamlit as st

ROOT = Path(__file__).resolve().parent
ARTIFACT_PATH = ROOT / "artifacts" / "red_estudiantes.zip"
SECTIONS = ("Analizar conexión", "Resultados", "Cómo funciona")
EXAMPLES = (
    ("2 intentos · 100 MB", 2, 100),
    ("8 intentos · 600 MB", 8, 600),
    ("18 intentos · 1.800 MB", 18, 1800),
)
HEADLINE_METRICS = (
    ("accuracy", "Exactitud", "De cada 100 conexiones de prueba, cuántas clasificó bien."),
    ("precision", "Precisión", "De las que marcó como sospechosas, cuántas lo eran."),
    ("recall", "Recall", "De las sospechosas reales, cuántas detectó."),
    ("f1", "F1", "Equilibrio entre precisión y recall."),
)
EXTRA_METRICS = (
    ("specificity", "Especificidad", "De las normales reales, cuántas reconoció.", "rate"),
    (
        "balanced_accuracy",
        "Exactitud balanceada",
        "Promedio del acierto en cada clase.",
        "rate",
    ),
    (
        "roc_auc",
        "ROC-AUC",
        "Capacidad de ordenar sospechosas por encima de normales. 0,50 equivale al azar.",
        "score",
    ),
    ("brier_score", "Brier", "Error de las probabilidades. Más bajo es mejor.", "score"),
)

_API: SimpleNamespace | None = None


def api() -> SimpleNamespace:
    """Importa el backend una sola vez, después de silenciar avisos de TensorFlow."""

    global _API
    if _API is None:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        from backend import (
            ACADEMIC_NOTICE,
            BackendConfig,
            BackendError,
            ConnectionRiskService,
            InputValidationError,
            ModelArtifactError,
            TrainingConfig,
        )

        _API = SimpleNamespace(
            ACADEMIC_NOTICE=ACADEMIC_NOTICE,
            BackendConfig=BackendConfig,
            BackendError=BackendError,
            ConnectionRiskService=ConnectionRiskService,
            InputValidationError=InputValidationError,
            ModelArtifactError=ModelArtifactError,
            TrainingConfig=TrainingConfig,
        )
    return _API


def inject_css() -> None:
    st.markdown(
        """
        <style>
          .block-container { max-width: 1120px; padding-top: 1.6rem; }
          .hero h1 {
            font-size: 2.05rem;
            line-height: 1.15;
            letter-spacing: -0.03em;
            margin: 0 0 0.35rem;
          }
          .eyebrow {
            margin: 0 0 0.25rem;
            color: #0f6e6b;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
          }
          .notice, .result-card, .balance {
            color: #1c1915;
          }
          .notice {
            background: #fffdf8;
            border: 1px solid #e4ddd2;
            border-left: 4px solid #0f6e6b;
            border-radius: 12px;
            padding: 0.75rem 0.95rem;
            margin: 0.4rem 0 1rem;
          }
          .side-brand {
            margin: 0 0 0.8rem;
            font-weight: 700;
            letter-spacing: -0.02em;
          }
          .status {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-weight: 650;
            margin-top: 0.4rem;
          }
          .status span {
            width: 0.65rem;
            height: 0.65rem;
            border-radius: 99px;
            display: inline-block;
          }
          .status.ready span { background: #1f7a4d; }
          .status.wait span { background: #b86e00; }
          .result-card {
            background: #fffdf8;
            border: 1px solid #e4ddd2;
            border-radius: 16px;
            padding: 1rem 1rem 0.85rem;
            min-height: 100%;
          }
          .result-card h3 {
            margin: 0;
            font-size: 1.7rem;
            letter-spacing: -0.03em;
          }
          .result-card.normal { background: #f3faf6; border-color: #b7dfc8; }
          .result-card.normal h3 { color: #146c43; }
          .result-card.suspicious { background: #fff8ee; border-color: #f0d7a8; }
          .result-card.suspicious h3 { color: #8a4b08; }
          .result-card.idle h3 { color: #6b645c; font-size: 1.35rem; }
          .result-card.stale { opacity: 0.78; }
          .prob {
            margin: 0.15rem 0 0;
            font-size: 2rem;
            font-weight: 700;
            letter-spacing: -0.04em;
          }
          .meter {
            position: relative;
            height: 10px;
            background: #e7e1d6;
            border-radius: 99px;
            margin: 0.85rem 0 0.35rem;
          }
          .meter-fill {
            display: block;
            height: 100%;
            border-radius: 99px;
          }
          .result-card.normal .meter-fill { background: #1f7a4d; }
          .result-card.suspicious .meter-fill { background: #c47d12; }
          .meter-mark {
            position: absolute;
            top: -4px;
            width: 2px;
            height: 18px;
            background: #1c1915;
            transform: translateX(-1px);
          }
          .explain, .meta, .card-note { margin: 0.45rem 0 0; }
          .meta { color: #4e4944; font-weight: 650; }
          .card-note { color: #4e4944; font-size: 0.92rem; }
          .stale-note {
            margin: 0 0 0.55rem;
            color: #8a4b08;
            font-weight: 650;
          }
          .arch {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.55rem;
            margin: 0.4rem 0 0.8rem;
          }
          .node {
            background: #fffdf8;
            border: 1px solid #e4ddd2;
            border-radius: 14px;
            min-width: 7.2rem;
            padding: 0.7rem 0.8rem;
            text-align: center;
          }
          .node strong { display: block; font-size: 1.35rem; letter-spacing: -0.03em; }
          .node span { color: #4e4944; }
          .arrow { color: #0f6e6b; font-size: 1.3rem; }
          .cm {
            display: grid;
            grid-template-columns: 9.5rem 1fr 1fr;
            gap: 0.45rem;
            margin-top: 0.4rem;
          }
          .cm-head {
            color: #4e4944;
            font-size: 0.86rem;
            font-weight: 650;
          }
          .cm-label {
            display: flex;
            align-items: end;
            font-weight: 650;
            font-size: 0.92rem;
          }
          .cm-cell {
            border-radius: 12px;
            padding: 0.7rem 0.75rem;
          }
          .cm-cell strong { display: block; font-size: 1.35rem; }
          .cm-cell span { color: #4e4944; font-size: 0.84rem; }
          .cm-cell.hit { background: #e7f6ee; }
          .cm-cell.miss { background: #fff1d6; }
          .balance-bar {
            display: flex;
            height: 14px;
            overflow: hidden;
            border-radius: 99px;
            background: #e7e1d6;
          }
          .balance-bar .ok { background: #1f7a4d; }
          .balance-bar .is-suspicious { background: #c47d12; }
          .balance p { margin: 0.4rem 0 0; color: #4e4944; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def boot() -> None:
    """Carga el artefacto guardado, si existe, una sola vez por sesión."""

    if st.session_state.get("booted"):
        return
    with st.spinner("Cargando la red neuronal…"):
        module = api()
        st.session_state.booted = True
        st.session_state.setdefault("nav", SECTIONS[0])
        st.session_state.setdefault("epochs", 100)
        st.session_state.setdefault("save_model", True)
        st.session_state.setdefault("attempts", 8)
        st.session_state.setdefault("traffic", 600)
        st.session_state.setdefault("history", [])
        st.session_state.service = None
        st.session_state.report = None
        st.session_state.model_source = None
        st.session_state.last_result = None
        st.session_state.showcase = None
        st.session_state.load_error = None
        st.session_state.predict_error = None
        if ARTIFACT_PATH.exists():
            try:
                service = module.ConnectionRiskService.load(ARTIFACT_PATH)
            except module.ModelArtifactError as exc:
                st.session_state.load_error = str(exc)
            else:
                st.session_state.service = service
                st.session_state.report = service.last_training_report()
                st.session_state.model_source = "archivo"


def ready_service():
    service = st.session_state.get("service")
    if service is not None and service.is_ready:
        return service
    return None


def request_train() -> None:
    st.session_state.do_train = True


def queue_example(attempts: int, traffic: int) -> None:
    st.session_state.attempts = attempts
    st.session_state.traffic = traffic
    st.session_state.classify_now = True


def run_training() -> tuple[bool, str]:
    """Ajusta una red nueva y solo entonces reemplaza la de la sesión."""

    module = api()
    epochs = int(st.session_state.get("epochs", 100))
    save_model = bool(st.session_state.get("save_model", True))
    started = time.perf_counter()
    try:
        config = module.BackendConfig(training=module.TrainingConfig(epochs=epochs, verbose=0))
        service = module.ConnectionRiskService(config)
        from tensorflow import keras

        with st.status("Entrenando la red…", expanded=True) as status:
            bar = st.progress(0, text="Generando 2.000 conexiones y ajustando los pesos…")

            class Progress(keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    logs = logs or {}
                    done = epoch + 1
                    detail = f"Época {done} de {epochs}"
                    if "loss" in logs:
                        detail += f" · pérdida {float(logs['loss']):.4f}"
                    if "val_loss" in logs:
                        detail += f" · validación {float(logs['val_loss']):.4f}"
                    bar.progress(min(done / epochs, 1), text=detail)

            report = service.train(callbacks=[Progress()], verbose=0)
            bar.progress(1.0, text="Prueba final evaluada.")
            status.update(label="Entrenamiento terminado", state="complete", expanded=False)
    except module.BackendError as exc:
        return False, str(exc)
    except Exception as exc:
        return False, f"No se pudo completar el entrenamiento: {exc}"

    if save_model:
        try:
            service.save(ARTIFACT_PATH)
        except module.ModelArtifactError as exc:
            save_note = (
                f" No se pudo guardar el archivo ({exc}). "
                "Puede seguir usándolo durante esta sesión."
            )
        else:
            save_note = " El modelo quedó guardado para la próxima apertura."
    else:
        save_note = " No se guardó el archivo: al cerrar habrá que entrenar de nuevo."

    st.session_state.service = service
    st.session_state.report = report.to_dict()
    st.session_state.model_source = "sesión"
    st.session_state.history = []
    st.session_state.last_result = None
    st.session_state.showcase = None
    st.session_state.predict_error = None
    st.session_state.load_error = None
    elapsed = time.perf_counter() - started
    return True, (
        f"Entrenamiento listo en {elapsed:.0f} s. "
        f"Se completaron {report.epochs_completed} épocas y se conservaron "
        f"los pesos de la época {report.best_epoch}."
        + save_note
    )


def classify(attempts: int, traffic: int) -> None:
    service = ready_service()
    if service is None:
        return
    try:
        result = service.predict(attempts, traffic)
    except api().InputValidationError as exc:
        st.session_state.predict_error = str(exc)
        return
    st.session_state.predict_error = None
    st.session_state.last_result = result
    history = st.session_state.history
    history.insert(0, result.to_dict())
    del history[10:]


def compare_examples() -> None:
    service = ready_service()
    if service is None:
        return
    try:
        st.session_state.showcase = service.predict_many(
            [(attempts, traffic) for _, attempts, traffic in EXAMPLES]
        )
    except api().InputValidationError as exc:
        st.session_state.predict_error = str(exc)


def render_sidebar() -> str:
    st.sidebar.markdown('<p class="side-brand">Red neuronal</p>', unsafe_allow_html=True)
    section = st.sidebar.radio("Sección", SECTIONS, key="nav", label_visibility="collapsed")
    service = ready_service()
    if service is None:
        st.sidebar.markdown(
            '<div class="status wait"><span></span>Sin modelo</div>',
            unsafe_allow_html=True,
        )
        st.sidebar.caption("Entrénela para clasificar conexiones.")
        if st.session_state.get("load_error"):
            st.sidebar.caption("El archivo guardado no se pudo abrir. Entrenar crea uno nuevo.")
        return section

    source = (
        "Abierto desde el archivo guardado."
        if st.session_state.get("model_source") == "archivo"
        else "Entrenado en esta sesión."
    )
    st.sidebar.markdown(
        '<div class="status ready"><span></span>Modelo listo</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.caption(source)
    st.sidebar.caption(f"Umbral de decisión: {service.model_info()['threshold']:.0%}")
    return section


def render_header() -> None:
    st.markdown(
        """
        <header class="hero">
          <p class="eyebrow">Taller · red neuronal de tres capas</p>
          <h1>Conexiones simuladas</h1>
        </header>
        """,
        unsafe_allow_html=True,
    )
    notice = html.escape(api().ACADEMIC_NOTICE)
    st.markdown(f'<div class="notice">{notice}</div>', unsafe_allow_html=True)


def render_train_panel(prominent: bool) -> None:
    if prominent:
        with st.container(border=True):
            st.markdown("##### Entrenar la red")
            st.caption(
                "Misma semilla que la consola (42). La primera vez puede tardar "
                "cerca de un minuto."
            )
            _train_controls(show_steps=True)
        return
    with st.expander("Entrenar de nuevo"):
        _train_controls(show_steps=False)


def _train_controls(show_steps: bool) -> None:
    if show_steps:
        controls, story = st.columns([1, 1])
    else:
        controls = st.container()
        story = None
    with controls:
        st.number_input(
            "Épocas máximas",
            min_value=1,
            max_value=300,
            step=1,
            key="epochs",
            help="Tope de recorridos sobre los datos. La parada temprana puede terminar antes.",
        )
        st.checkbox(
            "Guardar el modelo para la próxima apertura",
            key="save_model",
            help="Escribe artifacts/red_estudiantes.zip y evita entrenar otra vez al abrir.",
        )
        st.button(
            "Entrenar red",
            type="primary",
            on_click=request_train,
            key="train_button",
            width="stretch",
        )
    if story is not None:
        with story:
            st.markdown(
                """
                **Qué va a pasar**

                1. Se simulan 2.000 conexiones.
                2. Se aparta el 20 % para la prueba final.
                3. La red se ajusta y se detiene si la validación deja de mejorar.
                4. Después podrá clasificar y ver las métricas.
                """
            )


def render_analyze() -> None:
    if st.session_state.pop("classify_now", False):
        classify(int(st.session_state.attempts), int(st.session_state.traffic))

    if ready_service() is None:
        st.subheader("Primero hay que entrenar")
        st.markdown(
            "La red todavía no tiene pesos. El entrenamiento simula 2.000 conexiones, "
            "ajusta la arquitectura 2-8-4-1 y aparta un 20 % para medirla. "
            "Cuando termine, esta pantalla sirve para clasificar."
        )
        if st.session_state.get("load_error"):
            st.error(st.session_state.load_error)
        render_train_panel(prominent=True)
        return

    st.subheader("Clasificar una conexión")
    st.caption("Elija los dos datos de una conexión simulada. El resultado no cambia hasta pulsar el botón.")
    controls, outcome = st.columns([1.05, 0.95], gap="large")
    with controls:
        attempts = st.slider(
            "Intentos fallidos",
            min_value=0,
            max_value=20,
            step=1,
            key="attempts",
            help="Inicios de sesión fallidos. En el taller van de 0 a 20.",
        )
        traffic = st.slider(
            "Tráfico enviado",
            min_value=0,
            max_value=2000,
            step=10,
            format="%d MB",
            key="traffic",
            help="Megabytes enviados. En el taller van de 0 a 2.000, de 10 en 10.",
        )
        result = st.session_state.get("last_result")
        stale = result is not None and not _same_inputs(result, attempts, traffic)
        label = "Actualizar clasificación" if stale else "Clasificar conexión"
        if st.button(label, type="primary", width="stretch"):
            classify(int(attempts), int(traffic))
            result = st.session_state.get("last_result")
            stale = result is not None and not _same_inputs(result, attempts, traffic)
        if st.session_state.get("predict_error"):
            st.error(st.session_state.predict_error)
    with outcome:
        _show_result(result, stale)

    st.caption("Probar un caso de la consola. El control se mueve y la clasificación se actualiza.")
    shortcuts = st.columns(3)
    for column, (caption, example_attempts, example_traffic) in zip(shortcuts, EXAMPLES):
        with column:
            st.button(
                caption,
                key=f"example-{example_attempts}-{example_traffic}",
                on_click=queue_example,
                args=(example_attempts, example_traffic),
                width="stretch",
            )

    st.divider()
    st.markdown("**Comparar los tres ejemplos**")
    st.caption("Los mismos tres casos que imprime el programa de consola, uno al lado del otro.")
    if st.button("Comparar ejemplos"):
        compare_examples()
    showcase = st.session_state.get("showcase")
    if showcase:
        cards = st.columns(3, gap="medium")
        for column, prediction in zip(cards, showcase):
            with column:
                st.markdown(_result_html(prediction, compact=True), unsafe_allow_html=True)

    history = st.session_state.get("history") or []
    if not history:
        return
    st.divider()
    st.markdown("**Clasificaciones de esta sesión**")
    st.caption("La más reciente queda arriba. Corresponden al modelo actual.")
    st.dataframe(
        [
            {
                "Intentos fallidos": item["failed_attempts"],
                "Tráfico (MB)": f"{item['traffic_mb']:.0f}",
                "Probabilidad": f"{item['probability']:.1%}",
                "Clasificación": item["classification"].capitalize(),
            }
            for item in history
        ],
        hide_index=True,
        width="stretch",
    )
    if st.button("Borrar historial"):
        st.session_state.history = []
        st.rerun()


def render_results() -> None:
    ready = ready_service() is not None
    render_train_panel(prominent=not ready)
    report = st.session_state.get("report")
    if not report:
        if st.session_state.get("load_error"):
            st.error(st.session_state.load_error)
        st.caption(
            "Después de entrenar, esta pantalla muestra el reparto de los datos, "
            "las métricas de la prueba, la matriz de confusión y las curvas."
        )
        return

    source = (
        "Informe del archivo guardado."
        if st.session_state.get("model_source") == "archivo"
        else "Informe del entrenamiento de esta sesión."
    )
    st.caption(source)
    _render_dataset(report)
    _render_metrics(report)
    _render_matrices(report)
    _render_curves(report)
    _render_architecture()
    _render_download(report)


def render_guide() -> None:
    st.subheader("Cómo funciona el experimento")
    st.markdown(
        """
        Aquí se hace una sola tarea: decir si una conexión **simulada** se parece
        más a las que el taller etiquetó como normales o como sospechosas.

        1. Si todavía no hay modelo, entrénelo en **Analizar conexión**. Para repetirlo, use **Resultados**.
        2. Clasifique una conexión con intentos fallidos y tráfico.
        3. Compare las métricas con la referencia, que siempre elige la clase más frecuente.
        """
    )
    st.markdown("#### Las dos entradas")
    st.markdown(
        """
        Cada conexión tiene solo dos números, en este orden:

        - **Intentos fallidos**, de 0 a 20.
        - **Tráfico enviado**, de 0 a 2.000 MB.

        La red no ve direcciones, puertos ni usuarios. Si un valor cae fuera de
        esos rangos, la interfaz lo rechaza: son los mismos límites del conjunto simulado.
        """
    )
    st.markdown("#### De dónde salen las respuestas")
    st.markdown(
        """
        Las 2.000 conexiones se generan con la semilla 42. La etiqueta no viene de
        un ataque real. Sale de esta regla del taller, más un poco de ruido aleatorio
        para que el problema no tenga una frontera perfecta:
        """
    )
    st.code(
        "puntaje  = 0,55 × (intentos / 20) + 0,45 × (tráfico MB / 2000) + ruido\n"
        "etiqueta = sospechosa si puntaje ≥ 0,55; si no, normal",
        language=None,
    )
    st.markdown("#### La red")
    st.markdown(_architecture_html(), unsafe_allow_html=True)
    st.markdown(
        """
        Son 65 parámetros entrenables. El optimizador es Adam y la pérdida es la
        entropía cruzada binaria. La salida sigmoide es una probabilidad: si llega
        al **50 %**, la clase es sospechosa; por debajo, normal.

        La parada temprana vigila la pérdida de validación. Si no mejora durante
        12 épocas, el ajuste se detiene y **se restauran los pesos de la mejor época**,
        no los de la última.
        """
    )
    st.markdown("#### Cómo se reparte el conjunto")
    st.markdown(
        """
        La división es estratificada, así que cada parte conserva la proporción de clases:

        - **64 %** ajusta los pesos.
        - **16 %** valida cada época y decide la parada.
        - **20 %** queda intacto para las métricas finales.

        El normalizador se ajusta solo con el entrenamiento. La prueba no influye
        en los pesos ni en la elección de la mejor época.
        """
    )
    with st.expander("Cómo leer las métricas"):
        st.markdown(
            """
            - **Exactitud:** aciertos sobre el total de la prueba.
            - **Precisión:** de las alertas de sospecha, cuántas lo eran.
            - **Recall:** de las sospechosas reales, cuántas detectó.
            - **F1:** equilibrio entre precisión y recall.
            - **Especificidad:** de las normales reales, cuántas reconoció.
            - **ROC-AUC:** qué tan bien ordena sospechosas por encima de normales. 0,50 se parece al azar.
            - **Brier:** error de las probabilidades. Más bajo es mejor.
            - **Matriz de confusión:** filas = clase real, columnas = clase predicha.

            La referencia predice siempre la clase mayoritaria del entrenamiento.
            Sirve para ver si la red aprendió algo más que repetir esa clase.
            """
        )


def _render_dataset(report: dict) -> None:
    dataset = report.get("dataset") or {}
    total = int(dataset.get("total", 0))
    normal = int(dataset.get("normal", 0))
    suspicious = int(dataset.get("suspicious", 0))
    st.subheader("Conjunto simulado")
    first, second, third = st.columns(3)
    first.metric("Simuladas", _grouped(total))
    second.metric(
        "Usadas para ajustar",
        _grouped(int(report.get("train_count", 0))),
        help="Incluye la fracción que vigila la parada temprana. No es el conjunto de las métricas.",
    )
    third.metric(
        "Apartadas para la prueba",
        _grouped(int(report.get("test_count", 0))),
        help="Estas conexiones no ajustan pesos ni eligen la mejor época.",
    )
    st.markdown(_balance_html(normal, suspicious), unsafe_allow_html=True)
    seed = ((report.get("experiment") or {}).get("data") or {}).get("seed")
    seed_text = f" Semilla {seed}." if seed is not None else ""
    st.caption(
        f"De las {_grouped(total)} simuladas, "
        f"{_grouped(int(report.get('test_count', 0)))} se apartan. "
        f"Las métricas de abajo salen solo de ese grupo.{seed_text}"
    )


def _render_metrics(report: dict) -> None:
    metrics = report.get("metrics") or {}
    baseline = report.get("baseline") or {}
    st.subheader("Métricas en el conjunto de prueba")
    st.caption(
        "La variación de la primera fila se mide contra la referencia, "
        "que siempre predice la clase más frecuente."
    )
    cards = st.columns(4)
    for column, (key, label, help_text) in zip(cards, HEADLINE_METRICS):
        delta = None
        if key in baseline and metrics.get(key) is not None:
            delta = f"{metrics[key] - baseline[key]:+.2%}"
        column.metric(label, _format_metric(key, metrics.get(key)), delta=delta, help=help_text)

    extra = st.columns(4)
    for column, (key, label, help_text, _kind) in zip(extra, EXTRA_METRICS):
        column.metric(label, _format_metric(key, metrics.get(key)), help=help_text)

    loss = metrics.get("loss")
    threshold = metrics.get("threshold")
    loss_text = f"{loss:.4f}" if isinstance(loss, (int, float)) else "—"
    threshold_text = f"{threshold:.0%}" if isinstance(threshold, (int, float)) else "—"
    st.caption(
        f"Pérdida {loss_text} · umbral {threshold_text} · "
        f"{_grouped(int(metrics.get('sample_count', 0)))} conexiones de prueba "
        f"({_grouped(int(metrics.get('normal_count', 0)))} normales y "
        f"{_grouped(int(metrics.get('suspicious_count', 0)))} sospechosas). "
        "En la pérdida y en Brier, más bajo es mejor."
    )
    st.markdown(_baseline_sentence(baseline), unsafe_allow_html=False)


def _render_matrices(report: dict) -> None:
    metrics = report.get("metrics") or {}
    baseline = report.get("baseline") or {}
    st.subheader("Matriz de confusión")
    st.caption("Verde: acierto. Ámbar: error. Las filas son la clase real.")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Red neuronal**")
        st.markdown(
            _confusion_html(metrics.get("confusion_matrix")),
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("**Referencia mayoritaria**")
        st.markdown(
            _confusion_html(baseline.get("confusion_matrix")),
            unsafe_allow_html=True,
        )


def _render_curves(report: dict) -> None:
    history = report.get("history") or {}
    losses = list(history.get("loss") or [])
    if not losses:
        return
    import pandas as pd

    epochs = pd.Index(range(1, len(losses) + 1), name="Época")
    st.subheader("Curvas de entrenamiento")
    st.caption(
        f"La época {int(report.get('best_epoch', 0))} tuvo la menor pérdida de validación. "
        "Esos son los pesos que quedaron. Si la validación se separa del entrenamiento, "
        "no se conservan los pesos de la última época."
    )
    loss_frame = {"Entrenamiento": losses}
    if history.get("val_loss"):
        loss_frame["Validación"] = list(history["val_loss"])
    accuracy_frame = {}
    if history.get("accuracy"):
        accuracy_frame["Entrenamiento"] = list(history["accuracy"])
    if history.get("val_accuracy"):
        accuracy_frame["Validación"] = list(history["val_accuracy"])

    loss_chart, accuracy_chart = st.columns(2, gap="large")
    with loss_chart:
        st.markdown("**Pérdida**")
        st.line_chart(pd.DataFrame(loss_frame, index=epochs), height=240)
    if accuracy_frame:
        with accuracy_chart:
            st.markdown("**Exactitud**")
            st.line_chart(pd.DataFrame(accuracy_frame, index=epochs), height=240)
    st.caption("El informe JSON también guarda precisión y recall de cada época.")


def _render_architecture() -> None:
    service = ready_service()
    st.subheader("Arquitectura")
    st.markdown(_architecture_html(), unsafe_allow_html=True)
    if service is None:
        return
    with st.expander("Resumen técnico de Keras"):
        st.code(service.model_summary(), language=None)


def _render_download(report: dict) -> None:
    import json

    payload = dict(report)
    history = st.session_state.get("history") or []
    if history:
        payload["session_predictions"] = history
    st.download_button(
        "Descargar informe JSON",
        data=json.dumps(payload, ensure_ascii=False, indent=2),
        file_name="entrenamiento.json",
        mime="application/json",
        help="Incluye configuración, métricas, curvas y, si ya clasificó, las pruebas de esta sesión.",
    )


def _show_result(result, stale: bool) -> None:
    if result is None:
        st.markdown(
            """
            <article class="result-card idle">
              <p class="eyebrow">Resultado</p>
              <h3>Sin clasificar</h3>
              <p>Ajuste los dos controles y pulse <strong>Clasificar conexión</strong>.
              Si abre un ejemplo, verá el resultado y los valores que lo produjeron.</p>
            </article>
            """,
            unsafe_allow_html=True,
        )
        return
    st.markdown(_result_html(result, stale=stale), unsafe_allow_html=True)


def _result_html(result, stale: bool = False, compact: bool = False) -> str:
    kind = "suspicious" if result.label else "normal"
    title = "Sospechosa" if result.label else "Normal"
    classes = f"result-card {kind}"
    if stale:
        classes += " stale"
    stale_line = (
        '<p class="stale-note">Los controles ya no coinciden con este resultado.</p>'
        if stale
        else ""
    )
    probability = max(0.0, min(float(result.probability), 1.0))
    threshold = max(0.0, min(float(result.threshold), 1.0))
    notice = "" if compact else f'<p class="card-note">{html.escape(result.academic_notice)}</p>'
    return f"""
    <article class="{classes}">
      {stale_line}
      <p class="eyebrow">Clasificación</p>
      <h3>{title}</h3>
      <p class="prob">{probability:.1%}</p>
      <div class="meter" role="img" aria-label="Probabilidad {probability:.1%}">
        <span class="meter-fill" style="width:{probability * 100:.1f}%"></span>
        <span class="meter-mark" style="left:{threshold * 100:.1f}%"></span>
      </div>
      <p class="card-note">La línea marca el umbral de decisión ({threshold:.0%}).</p>
      <p class="explain">{html.escape(result.explanation)}</p>
      <p class="meta">{result.failed_attempts} intentos · {result.traffic_mb:.0f} MB</p>
      {notice}
    </article>
    """


def _architecture_html() -> str:
    nodes = (
        ("2", "entradas"),
        ("8", "ReLU"),
        ("4", "ReLU"),
        ("1", "sigmoide"),
    )
    parts = ['<div class="arch">']
    for index, (units, caption) in enumerate(nodes):
        if index:
            parts.append('<span class="arrow" aria-hidden="true">→</span>')
        parts.append(f'<div class="node"><strong>{units}</strong><span>{caption}</span></div>')
    parts.append("</div>")
    return "".join(parts)


def _confusion_html(matrix) -> str:
    if not matrix or len(matrix) != 2:
        return '<p class="card-note">No hay matriz para esta corrida.</p>'
    true_negative, false_positive = (int(matrix[0][0]), int(matrix[0][1]))
    false_negative, true_positive = (int(matrix[1][0]), int(matrix[1][1]))
    cells = (
        ("Real normal", true_negative, "hit", "normal correcta"),
        ("", false_positive, "miss", "falsa sospecha"),
        ("Real sospechosa", false_negative, "miss", "sospecha no vista"),
        ("", true_positive, "hit", "sospecha detectada"),
    )
    parts = [
        '<div class="cm">',
        "<div></div>",
        '<div class="cm-head">Predicha normal</div>',
        '<div class="cm-head">Predicha sospechosa</div>',
    ]
    for index, (label, count, tone, caption) in enumerate(cells):
        if index % 2 == 0:
            parts.append(f'<div class="cm-label">{label}</div>')
        parts.append(
            f'<div class="cm-cell {tone}"><strong>{_grouped(count)}</strong><span>{caption}</span></div>'
        )
    parts.append("</div>")
    return "".join(parts)


def _balance_html(normal: int, suspicious: int) -> str:
    total = max(normal + suspicious, 1)
    normal_width = 100 * normal / total
    suspicious_width = 100 * suspicious / total
    return f"""
    <div class="balance">
      <div class="balance-bar" role="img" aria-label="{normal} normales y {suspicious} sospechosas">
        <span class="ok" style="width:{normal_width:.2f}%"></span>
        <span class="is-suspicious" style="width:{suspicious_width:.2f}%"></span>
      </div>
      <p>{_grouped(normal)} normales · {_grouped(suspicious)} sospechosas</p>
    </div>
    """


def _baseline_sentence(baseline: dict) -> str:
    matrix = baseline.get("confusion_matrix")
    if not matrix or len(matrix) != 2:
        return "La referencia predice siempre la clase más frecuente del entrenamiento."
    _true_negative, false_positive = int(matrix[0][0]), int(matrix[0][1])
    _false_negative, true_positive = int(matrix[1][0]), int(matrix[1][1])
    if true_positive + false_positive == 0:
        guessed = "normales"
    elif _true_negative + _false_negative == 0:
        guessed = "sospechosas"
    else:
        guessed = "la clase mayoritaria"
    return (
        "La referencia clasificó todas las pruebas como "
        f"{guessed}, porque esa fue la clase más frecuente al entrenar. "
        "Una variación positiva en la primera fila indica que la red supera esa regla."
    )


def _same_inputs(result, attempts: int, traffic: int) -> bool:
    return result.failed_attempts == int(attempts) and float(result.traffic_mb) == float(traffic)


def _format_metric(key: str, value) -> str:
    if value is None:
        return "No aplica"
    if key in {"roc_auc", "brier_score", "loss"}:
        return f"{float(value):.4f}"
    return f"{float(value):.2%}"


def _grouped(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def _running_with_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx(suppress_warning=True) is not None
    except Exception:
        return False


def main() -> None:
    if not _running_with_streamlit():
        print("Abra la interfaz con:  python -m streamlit run interfaz.py")
        raise SystemExit(0)

    st.set_page_config(
        page_title="Conexiones simuladas",
        page_icon="◎",
        layout="wide",
        initial_sidebar_state="auto",
    )
    inject_css()
    boot()
    if st.session_state.pop("go_results", False):
        st.session_state.nav = SECTIONS[1]

    section = render_sidebar()
    render_header()
    if st.session_state.pop("do_train", False):
        ok, message = run_training()
        if ok:
            st.session_state.flash = message
            st.session_state.go_results = True
        else:
            st.session_state.flash_error = message
        st.rerun()

    if message := st.session_state.pop("flash", None):
        st.success(message)
    if error := st.session_state.pop("flash_error", None):
        st.error(error)

    if section == SECTIONS[0]:
        render_analyze()
    elif section == SECTIONS[1]:
        render_results()
    else:
        render_guide()


main()
