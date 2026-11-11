# Guía del backend de la red neuronal

El backend conserva el ejercicio del taller y ofrece una API de Python que la
interfaz puede llamar sin depender de Streamlit, Flask, FastAPI u otra
tecnología visual.

## Qué hace cada archivo

- `backend/config.py`: define las opciones del experimento y valida sus rangos.
- `backend/data.py`: comprueba la forma de los datos y genera las conexiones
  simuladas con sus etiquetas.
- `backend/network.py`: crea y compila la arquitectura de Keras.
- `backend/service.py`: coordina entrenamiento, evaluación, predicción y
  persistencia del modelo.
- `backend/results.py`: define las respuestas del backend y las convierte a
  diccionarios listos para JSON.
- `backend/exceptions.py`: reúne errores que la interfaz puede capturar.
- `backend/__init__.py`: expone los componentes públicos para importarlos desde
  `backend`.
- `red_neuronal_estudiantes.py`: ejecuta el flujo completo desde la terminal.

## Qué incluye

- Generación reproducible de 2.000 conexiones simuladas.
- Separación estratificada de entrenamiento y prueba.
- Ajuste de `StandardScaler` solo con los datos de entrenamiento.
- Red neuronal 2-8-4-1 con ReLU, sigmoide, Adam y entropía cruzada binaria.
- Parada temprana según la pérdida de validación, con restauración de los mejores
  pesos observados.
- Evaluación con pérdida, exactitud, precisión, recall, F1, especificidad,
  exactitud balanceada, ROC-AUC, Brier score y matriz de confusión.
- Comparación contra un clasificador de referencia que siempre predice la clase
  mayoritaria.
- Predicción con validación de rangos, umbral configurable y explicación breve.
- Guardado y carga del modelo junto con el normalizador y el informe de
  entrenamiento.

## Protocolo experimental

Las entradas están en este orden: intentos fallidos (0 a 20) y tráfico enviado
(0 a 2.000 MB). La etiqueta simulada sigue la regla del taller:

```text
puntaje = 0,55 * (intentos / 20) + 0,45 * (tráfico_MB / 2000) + ruido
etiqueta = 1 si puntaje >= 0,55; en otro caso, 0
```

La semilla 42 hace repetible la generación, la división y el entrenamiento. El
conjunto se divide de forma estratificada: 64 % para ajustar los pesos, 16 % para
validación temprana y 20 % para la prueba final. El escalador se ajusta solo con
el 64 % de entrenamiento.

```text
2 entradas -> 8 neuronas ReLU -> 4 neuronas ReLU -> 1 salida sigmoide
```

La red tiene 65 parámetros entrenables. La parada temprana observa `val_loss` y
restaura los pesos de la mejor época. La prueba final se reserva para reportar
las métricas y no participa en el entrenamiento.

## Entrenar y guardar una vez

Desde la carpeta del proyecto:

```powershell
python red_neuronal_estudiantes.py --artifact artifacts/red_estudiantes.zip --report-json informes/entrenamiento.json
```

El programa entrena la red, imprime las métricas y analiza tres ejemplos. Si se
indica `--artifact`, guarda el modelo para que la interfaz lo reutilice sin
entrenar en cada solicitud. `--report-json` guarda configuración, métricas,
comparación de referencia, curvas de entrenamiento/validación y predicciones de
ejemplo en JSON.

## Contrato para la interfaz

La interfaz incluida está en `interfaz.py` y se abre con
`python -m streamlit run interfaz.py`. Carga una sola instancia al iniciar:

```python
from backend import ConnectionRiskService, InputValidationError

servicio = ConnectionRiskService.load("artifacts/red_estudiantes.zip")

try:
    resultado = servicio.predict(failed_attempts=8, traffic_mb=600)
    datos_para_mostrar = resultado.to_dict()
except InputValidationError as error:
    mensaje = str(error)
```

`datos_para_mostrar` contiene `probability` (de 0 a 1), `threshold`, `label`
(0 o 1), `classification` (`normal` o `sospechosa`), `explanation` y
`academic_notice`. La interfaz puede presentar la probabilidad como porcentaje
multiplicándola por 100. Las entradas aceptadas son intentos enteros de 0 a 20 y
tráfico de 0 a 2.000 MB, los mismos rangos del conjunto simulado.

La carga debe ocurrir una vez al arrancar la aplicación, no dentro del callback
de cada botón o ruta. Si todavía no existe un artefacto, se puede entrenar una
vez con `servicio = ConnectionRiskService(); resumen = servicio.train()` y
guardarlo con `servicio.save("artifacts/red_estudiantes.zip")`.

## API disponible

- `generate_dataset()` crea un `ConnectionDataset` simulado.
- `ConnectionRiskService.train()` entrena y retorna `TrainingResult`.
- `ConnectionRiskService.evaluate()` evalúa el último conjunto de prueba o uno
  suministrado.
- `ConnectionRiskService.predict(intentos, trafico)` analiza una conexión.
- `ConnectionRiskService.predict_many([(intentos, trafico), ...])` analiza varias.
- `ConnectionRiskService.model_info()` retorna arquitectura, entradas, umbral,
  estado del servicio y las métricas guardadas en el artefacto.
- `ConnectionRiskService.last_training_report()` devuelve una copia del informe
  del último entrenamiento o del artefacto cargado.
- `TrainingResult`, `EvaluationMetrics` y `PredictionResult` tienen `to_dict()`
  para facilitar su conversión a JSON.

Los errores de datos se entregan como `InputValidationError` o
`DatasetValidationError`; el estado del modelo se comunica mediante
`ModelNotTrainedError`. Todas son excepciones públicas de `backend`.

## Cómo leer las métricas

- **Precisión**: de las conexiones marcadas como sospechosas, cuántas lo eran.
- **Recall**: de las sospechosas reales, cuántas detectó el modelo.
- **F1**: equilibrio entre precisión y recall.
- **Especificidad**: proporción de conexiones normales reconocidas.
- **ROC-AUC**: capacidad de ordenar positivos por encima de negativos en varios
  umbrales; 0,5 equivale aproximadamente a una elección al azar.
- **Brier score**: error cuadrático medio de las probabilidades; menor es mejor.
- **Matriz de confusión**: filas = clase real (0 normal, 1 sospechosa), columnas
  = clase predicha en el mismo orden.

Las métricas del modelo se comparan con un clasificador mayoritario. Esa
referencia ayuda a identificar si la red aprendió algo más que la clase más
frecuente. El umbral sigue fijo en 0,5 para coincidir con el taller.

## Alcance

Las etiquetas se generan con una regla didáctica basada en las dos entradas y
ruido aleatorio. Las probabilidades describen la salida de ese modelo simulado;
no son una medida validada de riesgo informático real. La interfaz debe mostrar
el `academic_notice` que viene en cada predicción.
