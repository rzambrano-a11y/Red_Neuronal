"""Interfaz gráfica con la red en movimiento.

Sirve la página de web/ y expone el modelo por una API local. El
entrenamiento corre en segundo plano; la página consulta el avance.
"""

from __future__ import annotations

import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
ARTIFACT_PATH = ROOT / "artifacts" / "red_estudiantes.zip"
HOST = "127.0.0.1"
PORT = 8501

_API: SimpleNamespace | None = None
_LOCK = threading.Lock()
_STATE: dict = {
    "loading": True,
    "ready": False,
    "training": False,
    "source": None,
    "progress": {
        "epoch": 0,
        "total": 0,
        "loss": None,
        "val_loss": None,
        "fraction": 0,
    },
    "report": None,
    "error": None,
    "message": None,
    "revision": 0,
    "academic_notice": (
        "Demostración académica con datos simulados. Este resultado no confirma "
        "ni descarta un ataque informático real."
    ),
    "threshold": 0.5,
    "weights": None,
}
_SERVICE = None


def api() -> SimpleNamespace:
    global _API
    if _API is None:
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


def _public_state() -> dict:
    with _LOCK:
        return {
            "loading": _STATE["loading"],
            "ready": _STATE["ready"],
            "training": _STATE["training"],
            "source": _STATE["source"],
            "progress": dict(_STATE["progress"]),
            "report": _STATE["report"],
            "error": _STATE["error"],
            "message": _STATE["message"],
            "revision": _STATE["revision"],
            "academic_notice": _STATE["academic_notice"],
            "threshold": _STATE["threshold"],
            "weights": _STATE["weights"],
        }


def boot() -> None:
    """Importa el backend y abre el modelo guardado, si existe."""

    module = api()
    with _LOCK:
        _STATE["academic_notice"] = module.ACADEMIC_NOTICE
    service = None
    source = None
    error = None
    if ARTIFACT_PATH.exists():
        try:
            service = module.ConnectionRiskService.load(ARTIFACT_PATH)
            source = "archivo"
        except module.ModelArtifactError as exc:
            error = str(exc)
    with _LOCK:
        global _SERVICE
        _SERVICE = service
        _STATE["loading"] = False
        _STATE["ready"] = service is not None and service.is_ready
        _STATE["source"] = source
        _STATE["error"] = error
        _STATE["threshold"] = service.model_info()["threshold"] if service else 0.5
        _STATE["report"] = service.last_training_report() if service else None
        _STATE["weights"] = service.connection_weights() if service and service.is_ready else None
        if service:
            _STATE["revision"] += 1


def _train(epochs: int, save: bool) -> None:
    module = api()
    try:
        from tensorflow import keras

        config = module.BackendConfig(
            training=module.TrainingConfig(epochs=epochs, verbose=0)
        )
        service = module.ConnectionRiskService(config)

        class Progress(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                done = epoch + 1
                with _LOCK:
                    _STATE["progress"] = {
                        "epoch": done,
                        "total": epochs,
                        "loss": float(logs["loss"]) if "loss" in logs else None,
                        "val_loss": float(logs["val_loss"]) if "val_loss" in logs else None,
                        "fraction": min(done / epochs, 1),
                    }

        report = service.train(callbacks=[Progress()], verbose=0)
        save_note = " No se guardó el archivo."
        if save:
            try:
                service.save(ARTIFACT_PATH)
            except module.ModelArtifactError as exc:
                save_note = f" No se pudo guardar el archivo: {exc}"
            else:
                save_note = " El modelo quedó guardado para la próxima apertura."
        message = (
            f"Entrenamiento listo. Se completaron {report.epochs_completed} épocas "
            f"y se conservaron los pesos de la época {report.best_epoch}."
            + save_note
        )
        with _LOCK:
            global _SERVICE
            _SERVICE = service
            _STATE["ready"] = True
            _STATE["source"] = "sesión"
            _STATE["report"] = report.to_dict()
            _STATE["threshold"] = report.metrics.threshold
            _STATE["weights"] = service.connection_weights()
            _STATE["message"] = message
            _STATE["error"] = None
            _STATE["revision"] += 1
            _STATE["progress"]["fraction"] = 1
    except Exception as exc:
        with _LOCK:
            _STATE["error"] = str(exc)
            _STATE["message"] = None
            _STATE["revision"] += 1
    finally:
        with _LOCK:
            _STATE["training"] = False


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args) -> None:
        if str(args[0]).startswith("GET /api/state"):
            return
        super().log_message(format, *args)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/state":
            self._send(200, _public_state())
            return
        self._send_file(path)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length > 100_000:
            self._send(413, {"error": "La solicitud es demasiado grande."})
            return
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._send(400, {"error": "El cuerpo no es JSON válido."})
            return
        if path == "/api/train":
            self._start_training(body)
            return
        if path == "/api/predict":
            self._predict(body)
            return
        self._send(404, {"error": "Ruta no encontrada."})

    def _start_training(self, body: dict) -> None:
        with _LOCK:
            if _STATE["loading"]:
                self._send(409, {"error": "La red todavía se está preparando."})
                return
            if _STATE["training"]:
                self._send(409, {"error": "Ya hay un entrenamiento en curso."})
                return
            try:
                epochs = int(body.get("epochs", 100))
            except (TypeError, ValueError):
                self._send(400, {"error": "Las épocas deben ser un entero."})
                return
            if not 1 <= epochs <= 300:
                self._send(400, {"error": "Las épocas deben estar entre 1 y 300."})
                return
            save = bool(body.get("save", True))
            _STATE["training"] = True
            _STATE["error"] = None
            _STATE["message"] = None
            _STATE["progress"] = {
                "epoch": 0,
                "total": epochs,
                "loss": None,
                "val_loss": None,
                "fraction": 0,
            }
        threading.Thread(target=_train, args=(epochs, save), daemon=True).start()
        self._send(202, {"ok": True})

    def _predict(self, body: dict) -> None:
        with _LOCK:
            service = _SERVICE
            ready = _STATE["ready"]
        if not ready or service is None:
            self._send(409, {"error": "Entrene la red antes de analizar una conexión."})
            return
        try:
            result = service.predict(body.get("failed_attempts"), body.get("traffic_mb"))
        except api().InputValidationError as exc:
            self._send(400, {"error": str(exc)})
            return
        self._send(200, result.to_dict())

    def _send(self, status: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: str) -> None:
        routes = {"/": "index.html", "": "index.html", "/simulador": "simulador.html", "/simulador/": "simulador.html"}
        relative = routes.get(path, path.lstrip("/"))
        target = (WEB / relative).resolve()
        if not str(target).startswith(str(WEB.resolve())) or not target.is_file():
            self._send(404, {"error": "No se encontró el archivo."})
            return
        kind = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "text/javascript; charset=utf-8",
            ".svg": "image/svg+xml",
        }.get(target.suffix, "application/octet-stream")
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    if not WEB.joinpath("index.html").is_file():
        raise SystemExit("Falta web/index.html.")
    threading.Thread(target=boot, daemon=True).start()
    ThreadingHTTPServer.allow_reuse_address = True
    try:
        server = ThreadingHTTPServer((HOST, PORT), Handler)
    except OSError as exc:
        raise SystemExit(
            f"No se pudo abrir el puerto {PORT}. Cierre la ventana que lo esté usando.\n{exc}"
        ) from exc
    url = f"http://{HOST}:{PORT}"
    print(f"Interfaz en {url}")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nInterfaz cerrada.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
