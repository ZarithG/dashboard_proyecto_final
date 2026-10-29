# 📊 Predictor de Renovación de Pólizas

**Sistema de predicción de renovación de pólizas de seguros mediante Machine Learning**, desarrollado con Python y Streamlit. Permite analizar información financiera, características de las pólizas y variables operativas para estimar la probabilidad de renovación y facilitar la identificación de pólizas con mayor riesgo de no renovación.

El proyecto integra un modelo de clasificación basado en **Random Forest**, técnicas de balanceo de clases mediante SMOTETomek y un umbral de decisión optimizado según el F1-Score.

## 🎯 Objetivo del proyecto

Desarrollar una herramienta interactiva que permita evaluar la probabilidad de renovación de pólizas de seguros, identificar posibles riesgos y presentar los resultados mediante visualizaciones que faciliten su interpretación.

## ✨ Características principales

- **Predicción individual:** análisis de una póliza a partir de sus características financieras, comerciales y operativas.
- **Predicción por lotes:** procesamiento de múltiples pólizas mediante archivos CSV.
- **Probabilidades de predicción:** estimación de las probabilidades de renovación y no renovación.
- **Clasificación del riesgo:** categorización del riesgo en niveles bajo, medio y alto.
- **Visualizaciones interactivas:** gráficos de indicadores, distribución de probabilidades y factores de riesgo.
- **Procesamiento de archivos CSV:** carga de datos, descarga de plantillas y exportación de resultados.
- **Modo demostración:** permite explorar la interfaz sin cargar el modelo entrenado.
- **Interfaz personalizada:** dashboard desarrollado con Streamlit y estilos CSS, con gráficos interactivos de Plotly.

## 🧠 Modelo de Machine Learning

El sistema está diseñado para utilizar un modelo de clasificación basado en Random Forest.

| Componente | Descripción |
|---|---|
| Algoritmo | Random Forest Classifier |
| Número de árboles | 300, según la configuración indicada en el dashboard |
| Balanceo de clases | SMOTETomek |
| Escalado | Transformación mediante un scaler previamente entrenado |
| Variables categóricas | Codificación de ciudad y variables one-hot |
| Umbral de decisión | Umbral optimizado según F1-Score |
| Salidas | Probabilidad de renovación, probabilidad de no renovación y clasificación |

**Importante:** el dashboard necesita el archivo del modelo entrenado para realizar predicciones reales. Sin este archivo, utiliza un mecanismo heurístico de demostración que no debe interpretarse como un modelo predictivo validado.

## 🛠️ Tecnologías utilizadas

- **Python:** lenguaje principal.
- **Pandas:** manipulación y procesamiento de datos.
- **NumPy:** operaciones numéricas.
- **Scikit-learn:** componentes de procesamiento y clasificación utilizados por el modelo.
- **SMOTETomek:** técnica de balanceo de clases utilizada en el entrenamiento.
- **Streamlit:** desarrollo de la aplicación web interactiva.
- **Plotly:** creación de visualizaciones interactivas.
- **Joblib:** carga del modelo entrenado y sus componentes.

## 📋 Requisitos previos

Antes de ejecutar el proyecto, necesitas:

- Python instalado.
- Git, para clonar el repositorio.
- Un editor de código, como Visual Studio Code.
- El archivo del modelo entrenado, si deseas realizar predicciones reales.

## 🚀 Instalación y ejecución

### 1. Clonar el repositorio

```bash
git clone URL_DE_TU_REPOSITORIO
cd NOMBRE_DEL_PROYECTO
```

Reemplaza los valores anteriores por la URL y el nombre reales de tu repositorio.

### 2. Crear un entorno virtual

En Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

En Linux o macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar las dependencias

Instala las bibliotecas principales:

```bash
pip install streamlit pandas numpy plotly scikit-learn imbalanced-learn joblib
```

También puedes generar un archivo `requirements.txt` para facilitar la instalación reproducible de las dependencias:

```bash
pip freeze > requirements.txt
```

Si el proyecto ya tiene un archivo `requirements.txt`, puedes instalar sus dependencias con:

```bash
pip install -r requirements.txt
```

### 4. Ejecutar la aplicación

Si el archivo principal se llama `app.py`, ejecuta:

```bash
streamlit run app.py
```

Streamlit mostrará la dirección local desde la que puedes acceder al dashboard.

## 📈 Cómo utilizar la aplicación

### Modo 1. Predicción individual

1. Abre la aplicación.
2. Selecciona el modo **Póliza individual**.
3. Introduce la información financiera de la póliza.
4. Selecciona la compañía, el evento, el ramo y el canal.
5. Completa los datos de ubicación, línea de negocio y sucursal.
6. Introduce la fecha de aviso y selecciona las características adicionales.
7. Haz clic en **Predecir Renovación**.
8. Consulta el resultado, las probabilidades y los gráficos de riesgo.

### Modo 2. Predicción por lotes

1. Selecciona el modo **Lote (CSV)**.
2. Descarga la plantilla CSV proporcionada por la aplicación.
3. Completa el archivo con la información de las pólizas.
4. Sube el archivo CSV.
5. Ejecuta la predicción del lote.
6. Explora las métricas generales y los resultados individuales.
7. Descarga el archivo CSV con las predicciones obtenidas.

## 📁 Estructura del proyecto

Una organización posible para el repositorio es:

```text
predictor-renovacion-polizas/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   └── polizas_rf_model.pkl
│
├── notebooks/
│   └── entrenamiento_modelo.ipynb
│
└── data/
    └── README.md
```

Esta estructura es orientativa y debe adaptarse a los archivos reales del proyecto.

El archivo `polizas_rf_model.pkl` representa el bundle del modelo entrenado que la aplicación permite cargar. No es necesario incluirlo públicamente en el repositorio si contiene información confidencial o si su tamaño lo hace inadecuado para Git.

## 📄 Formato de los datos

Para realizar predicciones por lotes, el archivo CSV debe contener las siguientes columnas:

| Columna | Descripción |
|---|---|
| `Prima_Neta` | Valor de la prima de la póliza. |
| `Valor_Cuota` | Valor de la cuota. |
| `Dias_Entre_Aviso_Cierre` | Días transcurridos entre el aviso y el cierre. |
| `Riesgo_Pais` | Indicador de riesgo del país. |
| `Fecha_Aviso` | Fecha del aviso, en formato de fecha reconocible. |
| `Sin_Plan_Pago` | Indicador de ausencia de plan de pago. |
| `Garantia_Suscripcion` | Indicador de garantía de suscripción. |
| `Compania` | Compañía aseguradora. |
| `Evento` | Tipo de evento asociado a la póliza. |
| `Nombre_Ramo` | Ramo del seguro. |
| `Canal` | Canal comercial. |
| `Ciudad_y` | Ciudad asociada a la póliza. |
| `Linea_Negocio` | Línea de negocio. |
| `Sucursal` | Sucursal correspondiente. |

Se recomienda utilizar la plantilla generada por la aplicación para conservar los nombres de las columnas y los formatos esperados.

## 📊 Resultados e indicadores

El dashboard presenta información para facilitar el análisis de las predicciones:

- **Probabilidad de no renovación:** estimación de la posibilidad de que una póliza no sea renovada.
- **Probabilidad de renovación:** estimación de la posibilidad de que una póliza sea renovada.
- **Umbral de decisión:** valor utilizado para determinar la clase predicha.
- **Nivel de riesgo:** clasificación de riesgo bajo, medio o alto.
- **Distribución de resultados:** comparación entre pólizas renovadas y no renovadas.
- **Análisis por lotes:** resumen del número de pólizas procesadas y de los resultados obtenidos.

## ⚠️ Consideraciones importantes

- Las predicciones reales dependen del modelo entrenado y de sus componentes de preprocesamiento.
- El archivo del modelo debe corresponder al formato esperado por la aplicación.
- Las ciudades y categorías deben ser compatibles con las utilizadas durante el entrenamiento.
- El modo de demostración utiliza una heurística y valores aleatorios; sus resultados no representan probabilidades calibradas ni predicciones reales del modelo.
- Los resultados deben interpretarse como apoyo al análisis y no como garantía de que una póliza será renovada o no renovada.
- Antes de utilizar el sistema en un entorno real, es necesario evaluar el rendimiento del modelo con datos apropiados y verificar su comportamiento en producción.

## 🔮 Posibles mejoras futuras

- Incorporar un módulo de explicación de predicciones mediante SHAP.
- Añadir métricas de evaluación del modelo, como precisión, recall, F1-Score y ROC-AUC.
- Implementar un historial de predicciones.
- Integrar una base de datos para almacenar resultados.
- Incorporar filtros para analizar las pólizas por compañía, ciudad y ramo.
- Desplegar la aplicación en un servicio de alojamiento en la nube.
- Añadir pruebas automatizadas y validaciones de calidad de los datos.


Proyecto de Machine Learning enfocado en la aplicación de modelos predictivos, procesamiento de datos y visualización interactiva para apoyar la toma de decisiones.

---

⭐ Si este proyecto te resulta interesante, puedes explorar el código, utilizarlo como referencia para aprender sobre Machine Learning y proponer nuevas mejoras.
