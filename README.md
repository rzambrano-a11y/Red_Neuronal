# Taller: red neuronal de tres capas

Proyecto académico que entrena y evalúa una red neuronal para clasificar
conexiones informáticas **simuladas** como `normal` o `sospechosa`. Incluye un
programa de consola y un paquete `backend` reutilizable desde cualquier
interfaz (Streamlit, Flask, FastAPI, etc.).

> **Aviso:** los datos se generan con una regla didáctica y ruido aleatorio.
> El programa **no** es un sistema real de detección de ataques informáticos.

## Requisitos

- Python 3.10 o superior
- Librerías: `numpy`, `tensorflow`, `scikit-learn`

```powershell
python -m pip install -r requirements.txt
```

## Uso rápido

Entrenar, ver la arquitectura, las métricas y tres predicciones de ejemplo:

```powershell
python red_neuronal_estudiantes.py
```

Entrenar y además guardar el modelo y un informe JSON:

```powershell
python red_neuronal_estudiantes.py --artifact artifacts/red_estudiantes.zip --report-json informes/entrenamiento.json
```

| Opción          | Descripción                                                        |
|-----------------|--------------------------------------------------------------------|
| `--epochs N`    | Máximo de épocas de entrenamiento (por defecto 100).               |
| `--artifact`    | Ruta `.zip` donde guardar el modelo y su normalizador.             |
| `--report-json` | Ruta donde guardar configuración, métricas, historial y ejemplos.  |

## Interfaz gráfica

La interfaz carga el modelo una sola vez. Desde ahí se clasifica una conexión,
se leen las métricas contra la referencia y se consulta cómo está armado el experimento.

```powershell
python servidor_interfaz.py
```

En Windows también puede abrirse con `iniciar_interfaz.bat`.

Si existe `artifacts/red_estudiantes.zip`, la interfaz lo usa y no vuelve a entrenar.
Si no existe, el botón **Entrenar red** genera los datos, ajusta el modelo y puede
guardarlo en esa ruta para la próxima apertura.

## Cómo funciona

**Datos.** Se generan 2.000 conexiones con semilla 42. Cada una tiene dos
entradas: intentos fallidos (0 a 20) y tráfico enviado (0 a 2.000 MB). La
etiqueta sigue la regla del taller:

```text
puntaje  = 0,55 * (intentos / 20) + 0,45 * (tráfico_MB / 2000) + ruido
etiqueta = 1 (sospechosa) si puntaje >= 0,55; en otro caso 0 (normal)
```

**División.** Estratificada: 64 % entrenamiento, 16 % validación y 20 % prueba
final. El `StandardScaler` se ajusta solo con el entrenamiento.

**Red.** 65 parámetros entrenables, optimizador Adam y entropía cruzada binaria:

```text
2 entradas -> 8 neuronas ReLU -> 4 neuronas ReLU -> 1 salida sigmoide
```

**Entrenamiento.** Parada temprana sobre `val_loss` (paciencia 12) con
restauración de los mejores pesos. Umbral de clasificación: 0,5.

**Evaluación.** Sobre el conjunto de prueba se reportan pérdida, exactitud,
precisión, recall, F1, especificidad, exactitud balanceada, ROC-AUC, Brier score
y matriz de confusión. Los resultados se comparan con un clasificador de
referencia que siempre predice la clase mayoritaria.

## Uso desde una interfaz

Cargue el modelo **una sola vez** al iniciar la aplicación y reutilice la
instancia:

```python
from backend import ConnectionRiskService, InputValidationError

servicio = ConnectionRiskService.load("artifacts/red_estudiantes.zip")

try:
    resultado = servicio.predict(failed_attempts=8, traffic_mb=600)
    print(resultado.to_dict())
except InputValidationError as error:
    print(f"Entrada no válida: {error}")
```

La respuesta incluye `probability`, `threshold`, `label`, `classification`,
`explanation` y `academic_notice`. Si aún no existe el artefacto, puede
entrenarse con `ConnectionRiskService().train()` y guardarse con `save(...)`.

La API completa, el formato de las respuestas y las excepciones están
documentados en [BACKEND_GUIDE.md](BACKEND_GUIDE.md).

## Estructura del proyecto

```text
01_estudiantes_consola/
├── red_neuronal_estudiantes.py   # Punto de entrada de consola
├── servidor_interfaz.py          # Servidor de la interfaz animada
├── web/                          # Página, estilos y animación de la red
├── interfaz.py                   # Interfaz alternativa con Streamlit
├── iniciar_interfaz.bat          # Atajo de Windows para la interfaz
├── requirements.txt              # Dependencias
├── BACKEND_GUIDE.md              # Guía técnica del backend
├── LEAME_ESTUDIANTES.txt         # Instrucciones breves para estudiantes
├── .streamlit/config.toml        # Tema de la interfaz alternativa
├── doct/                         # Enunciado del taller (PDF)
└── backend/
    ├── __init__.py               # API pública del paquete
    ├── config.py                 # Configuración y validación de parámetros
    ├── data.py                   # Generación y validación del conjunto simulado
    ├── network.py                # Arquitectura Keras 2-8-4-1
    ├── service.py                # Entrenamiento, evaluación, predicción y persistencia
    ├── results.py                # Tipos de respuesta serializables a JSON
    └── exceptions.py             # Excepciones públicas
```

## Documentación adicional

- [LEAME_ESTUDIANTES.txt](LEAME_ESTUDIANTES.txt): instrucciones breves para los estudiantes.
- [BACKEND_GUIDE.md](BACKEND_GUIDE.md): API, métricas y contrato con la interfaz.
- `doct/TALLER PRACTICO REDES NEURONALES 28 OCT.pdf`: enunciado del taller.
