"""Punto de entrada de consola para el backend de la red neuronal."""

import argparse
import json
from pathlib import Path

from backend import (
    BackendConfig,
    ConnectionRiskService,
    TrainingConfig,
)


def main() -> None:
    # Definir opciones para entrenar y guardar resultados desde la terminal.
    parser = argparse.ArgumentParser(
        description="Entrena y presenta el modelo académico de clasificación."
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        help="Ruta opcional donde guardar el modelo y su normalizador (.zip).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Cantidad de épocas de entrenamiento (por defecto: 100).",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        help="Ruta opcional donde guardar métricas, configuración e historial.",
    )
    args = parser.parse_args()

    # CLI muestra por época; las interfaces pueden dejar verbose=0.
    config = BackendConfig(training=TrainingConfig(epochs=args.epochs, verbose=2))
    service = ConnectionRiskService(config)
    # train() genera datos, ajusta el modelo, evalúa y retorna un informe.
    report = service.train()

    # Mostrar distribución y resumen de la arquitectura aprendida.
    print("\nDISTRIBUCIÓN DEL CONJUNTO SIMULADO")
    print(f"Total: {report.dataset['total']}")
    print(f"Normales: {report.dataset['normal']}")
    print(f"Sospechosas: {report.dataset['suspicious']}")
    print(
        f"Épocas completadas: {report.epochs_completed}; "
        f"mejor época de validación: {report.best_epoch}"
    )

    print("\nARQUITECTURA DE LA RED")
    print(service.model_summary())

    # Reportar solo métricas calculadas con el bloque reservado como test.
    print("\nMÉTRICAS EN EL CONJUNTO DE PRUEBA")
    print(f"Pérdida: {report.metrics.loss:.4f}")
    print(f"Exactitud: {report.metrics.accuracy:.2%}")
    print(f"Precisión: {report.metrics.precision:.2%}")
    print(f"Recall: {report.metrics.recall:.2%}")
    print(f"F1: {report.metrics.f1:.2%}")
    print(f"Especificidad: {report.metrics.specificity:.2%}")
    if report.metrics.balanced_accuracy is not None:
        print(f"Exactitud balanceada: {report.metrics.balanced_accuracy:.2%}")
    if report.metrics.roc_auc is not None:
        print(f"ROC-AUC: {report.metrics.roc_auc:.4f}")
    print(f"Brier score: {report.metrics.brier_score:.4f}")
    print(f"Umbral de clasificación: {report.metrics.threshold:.2f}")
    print(f"Matriz de confusión: {report.metrics.confusion_matrix}")

    print("\nREFERENCIA: CLASIFICADOR MAYORITARIO")
    print(f"Exactitud: {report.baseline.accuracy:.2%}")
    print(f"Precisión: {report.baseline.precision:.2%}")
    print(f"Recall: {report.baseline.recall:.2%}")
    print(f"F1: {report.baseline.f1:.2%}")

    # Ejemplos variados para mostrar cómo cambia la probabilidad según entradas.
    print("\nPREDICCIONES DE EJEMPLO")
    example_predictions = []
    for attempts, traffic in ((2, 100), (8, 600), (18, 1800)):
        result = service.predict(attempts, traffic)
        example_predictions.append(result.to_dict())
        print(result.explanation)
        print(f"  Intentos: {result.failed_attempts}; tráfico: {result.traffic_mb:.0f} MB")
        print(f"  {result.academic_notice}")

    if args.artifact:
        # La UI puede cargar este bundle sin repetir el entrenamiento.
        saved_path = service.save(args.artifact)
        print(f"\nModelo guardado en: {saved_path}")

    if args.report_json:
        # El JSON facilita entregar evidencia de configuración, métricas y ejemplos.
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        report_content = report.to_dict()
        report_content["example_predictions"] = example_predictions
        args.report_json.write_text(
            json.dumps(report_content, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Informe guardado en: {args.report_json.resolve()}")


if __name__ == "__main__":
    main()
