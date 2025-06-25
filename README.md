# analitica-farma

Aplicación para analizar datos productivos en la industria farmacéutica y recomendar modelos de machine learning. Permite cargar datos, configurarlos, validarlos, transformarlos, evaluar modelos y generar reportes. Desarrollada con Streamlit, Python y SQLite para almacenamiento local.

## Estructura del Proyecto

```text
├── app.py                  # Punto de entrada principal de la app Streamlit y la navegación multipágina
├── requirements.txt        # Dependencias del proyecto
├── README.md               # Este archivo
├── analitica_farma.db      # Base de datos SQLite local
├── pages/                  # Páginas multipágina de Streamlit (cada funcionalidad principal)
│   ├── 00_Logueo.py        # Página de inicio de sesión
│   ├── Datos/
│   │   ├── 01_Cargar_Datos.py     # Carga de datos desde CSV o base de datos
│   │   ├── 02_Configurar_Datos.py # Configuración del problema y variables
│   │   └── 03_Validar_Datos.py    # Validación de tipos, fechas y unidades
│   ├── Machine Learning/
│   │   ├── 04_Entrenar_Modelos.py # Benchmarking automático de múltiples modelos
│   │   ├── 05_Evaluar_Modelos.py  # Evaluación detallada de los modelos entrenados
│   │   └── 06_Recomendar_Modelo.py # Recomendación del mejor modelo según criterios
│   └── Reportes/
│       ├── 07_Reporte.py
│       └── 08_Dashboard.py
├── src/                    # Código fuente modularizado
│   ├── audit/              # Auditoría y logging
│   │   └── logger.py
│   ├── config/             # Configuración centralizada
│   │   ├── __init__.py
│   │   └── workflow_steps.json    # Definición de pasos del workflow
│   ├── datos/              # Carga, limpieza y transformación de datos
│   │   ├── cargador.py     # Carga de datos desde CSV o base de datos
│   │   ├── formateador.py  # Estandarización de formatos y unidades
│   │   ├── limpiador.py    # Limpieza de datos
│   │   ├── transformador.py # Transformaciones de datos
│   │   ├── validador.py    # Validación de datos
│   │   └── mock_db.py      # Base de datos mock para desarrollo
│   ├── modelos/            # Entrenamiento, evaluación y recomendación de modelos ML
│   │   ├── configurador.py # Configuración de parámetros de modelos
│   │   ├── entrenador.py   # Entrenamiento y benchmarking de modelos
│   │   ├── evaluador.py    # Evaluación detallada de modelos
│   │   └── recomendador.py # Recomendación del mejor modelo
│   ├── reportes/           # Generación de reportes PDF/CSV
│   │   └── generador.py
│   ├── seguridad/          # Autenticación y control de acceso
│   │   └── autenticador.py
│   ├── state/              # Gestión centralizada del estado
│   │   ├── __init__.py
│   │   └── session_manager.py # Gestor de sesiones y progreso
│   └── ui/                 # Componentes de interfaz de usuario reutilizables
│       ├── __init__.py
│       └── sidebar.py      # Componentes para la barra lateral
├── logs/                   # Logs de auditoría y operaciones
│   ├── auditoria_YYYYMMDD.log
│   └── carga_datos_YYYYMMDD.log
```

- Las páginas en `pages/` están organizadas en subcarpetas por dominio funcional: Datos, Machine Learning y Reportes.
- El archivo `app.py` implementa la navegación multipágina, la barra lateral con información del dataset y una lista de comprobación del progreso.
- El código fuente en `src/` está organizado por dominio con un enfoque modular de tipo MVC.
- La base de datos SQLite (`analitica_farma.db`) almacena los datos, metadatos, usuarios y registros de auditoría.
- Nuevos módulos `state` y `ui` para gestión centralizada del estado y componentes de interfaz reutilizables.

## app.py

`app.py` es el punto de entrada de la aplicación y define:

- La configuración global de Streamlit (`st.set_page_config`).
- El control de sesión para login/logout.
- La navegación multipágina agrupada por secciones, usando `st.Page` y `st.navigation`.
- La barra lateral con información del dataset y progreso del workflow.
- El acceso a las páginas está restringido según el estado de login del usuario.

Ejemplo de navegación y barra lateral:

```python
from src.state.session_manager import SessionManager
from src.ui.sidebar import SidebarComponents

# Inicializar el estado de la sesión
SessionManager.init_session_state()

# Definir la navegación según el estado de login
if st.session_state.logged_in:
    pg = st.navigation({
        "Cuenta": [pagina_deslogueo],
        "Datos": [cargar_datos, configurar_datos, validar_datos, transformaciones],
        "Machine Learning": [entrenar_modelos, evaluar_modelos, recomendar_modelo],
        "Reportes & Dashboards": [reporte, dashboard]
    })
else:
    pg = st.navigation([pagina_logueo])

# Renderizar la barra lateral si el usuario está logueado
if st.session_state.logged_in:
    SidebarComponents.render_sidebar()

# Ejecutar la página actual
pg.run()
```

Esto permite una experiencia de usuario moderna, segura y fácil de mantener, alineada con las mejores prácticas de Streamlit.

## Flujo de trabajo

La aplicación implementa un flujo de trabajo guiado para el análisis de datos:

1. **Carga de datos**: Importación desde CSV o selección de datasets existentes en la base de datos local.
2. **Configuración de datos**: Selección del tipo de problema (regresión/clasificación), variable objetivo y predictores.
3. **Validación de datos**: Validación automática de tipos de datos, formatos de fecha y unidades de medida.
4. **Transformación de datos**: Aplicación de transformaciones para mejorar la calidad de los datos.
5. **Entrenamiento de modelos**: Configuración y entrenamiento de múltiples modelos de ML.
6. **Evaluación de modelos**: Comparación de métricas de rendimiento entre los modelos entrenados.
7. **Recomendación de modelo**: Selección automática del mejor modelo según criterios predefinidos.
8. **Generación de reportes**: Creación de informes detallados con resultados y visualizaciones.

Cada paso está representado por una página separada, y el progreso se visualiza en la barra lateral mediante una lista de comprobación dinámica.

## Flujo de Machine Learning

La aplicación implementa un enfoque integral para el entrenamiento, evaluación y recomendación de modelos de machine learning a través de tres módulos principales:

### 1. Entrenamiento de Modelos (Benchmarking)

El módulo de entrenamiento (`entrenador.py`) implementa un proceso de benchmarking automático que:

- **Detecta automáticamente** el tipo de problema (clasificación o regresión) basado en la variable objetivo
- **Prepara los datos** realizando división en conjuntos de entrenamiento/prueba y aplicando las transformaciones necesarias (escalado, codificación de etiquetas)
- **Ejecuta múltiples modelos** de forma paralela según el tipo de problema:
  - Para clasificación: LogisticRegression, DecisionTree, RandomForest, GradientBoosting, SVC, KNeighbors, GaussianNB, AdaBoost
  - Para regresión: LinearRegression, Ridge, Lasso, ElasticNet, DecisionTree, RandomForest, GradientBoosting, SVR, KNeighbors, AdaBoost
- **Maneja errores individualmente** para cada modelo, permitiendo que el benchmarking continúe incluso si algunos modelos fallan
- **Calcula métricas relevantes** según el tipo de problema:
  - Clasificación: accuracy, precision, recall, F1-score, validación cruzada
  - Regresión: R², MSE, RMSE, MAE, validación cruzada
- **Ordena los modelos** según su rendimiento y guarda los resultados en la base de datos
- **Mantiene un registro de auditoría** de las operaciones realizadas

La interfaz de usuario en `04_Entrenar_Modelos.py` proporciona:

- Selección de la variable objetivo y predictores
- Control de variables temporales y fechas para evitar problemas en el entrenamientovo
- Detección automática del tipo de problema con opción de cambio manual
- Visualización de la distribución de clases/valores
- Ajuste de parámetros avanzados (tamaño del conjunto de prueba)
- Ejecución del benchmarking con barra de progreso detallada
- Visualización de resultados comparativos con métricas clave
- Identificación del mejor modelo encontrado
- Navegación a las siguientes etapas del proceso

### 2. Evaluación Detallada de Modelos

El módulo de evaluación (`evaluador.py`) permite:

- **Analizar en detalle** los modelos entrenados en el benchmarking
- **Cargar benchmarkings anteriores** guardados en la base de datos
- **Visualizar métricas detalladas** para cada modelo
- **Comparar el rendimiento** entre diferentes modelos con gráficos
- **Generar curvas de aprendizaje** para detectar overfitting/underfitting (implementación futura)

La interfaz de usuario en `05_Evaluar_Modelos.py` ofrece:

- Selección del modelo a evaluar
- Visualización detallada de métricas por tipo de problema
- Análisis de validación cruzada
- Gráficos comparativos de métricas
- Carga de benchmarkings anteriores por ID

### 3. Recomendación de Modelos

El módulo de recomendación (`recomendador.py`) proporciona:

- **Criterios personalizables** para la selección del mejor modelo
  - Clasificación: accuracy, F1-score, precision, recall
  - Regresión: R², RMSE, MAE, MSE
- **Justificación detallada** de la recomendación realizada
- **Comparación visual** entre todos los modelos según el criterio seleccionado
- **Persistencia de la selección** del usuario en la base de datos
- **Registro de auditoría** de las decisiones tomadas

La interfaz de usuario en `06_Recomendar_Modelo.py` incluye:

- Selección del criterio de recomendación
- Visualización del modelo recomendado con sus métricas
- Justificación de la recomendación según el criterio
- Comparación con otros modelos disponibles
- Opción para aceptar la recomendación o seleccionar otro modelo
- Comentarios sobre la selección realizada

## Sistema de Validación de Datos

La aplicación implementa un robusto sistema de validación de datos para asegurar la calidad y consistencia de los datos antes de su análisis. Este sistema se encarga de detectar y corregir diversos problemas que podrían afectar el rendimiento de los modelos de machine learning.

### Arquitectura del Sistema de Validación

El sistema de validación está organizado en tres capas principales:

1. **Módulos de validación** (`src/datos/validador.py`):
   - Contiene algoritmos especializados para detectar problemas en los datos
   - Implementa validadores independientes para tipos de datos, fechas y unidades
   - Utiliza heurísticas avanzadas para identificar inconsistencias sutiles

2. **Módulos de corrección** (`src/datos/transformador.py` y `src/datos/limpiador.py`):
   - Proporciona funciones para corregir los problemas detectados
   - Implementa algoritmos de conversión entre diferentes tipos y formatos
   - Ofrece herramientas para la gestión de duplicados

3. **Interfaz de usuario** (`pages/Datos/03_Validar_Datos.py`):
   - Presenta los resultados de la validación de forma clara y procesable
   - Permite al usuario seleccionar qué correcciones aplicar
   - Proporciona retroalimentación visual sobre el estado de los datos

### Validaciones Implementadas

#### 1. Validación de Tipos de Datos (`validar_tipos_datos`)

Este validador detecta inconsistencias en los tipos de datos de las columnas:

- **Detección de variables categóricas codificadas como numéricas**
  - Identifica columnas numéricas con pocos valores únicos
  - Sugiere la conversión a tipo categórico cuando es apropiado

- **Identificación de columnas de texto que contienen valores numéricos**
  - Analiza el contenido de columnas de texto para detectar patrones numéricos
  - Recomienda conversión a tipo numérico cuando todos los valores son números

- **Reconocimiento de fechas almacenadas como texto**
  - Utiliza expresiones regulares y múltiples formatos para detectar fechas
  - Sugiere la conversión a tipo datetime para análisis temporal

#### 2. Validación de Formatos de Fecha (`validar_fechas`)

Este validador se especializa en columnas temporales:

- **Detección automática de columnas que podrían contener fechas**
  - Identifica columnas por nombre (fecha, date, time, etc.)
  - Analiza el contenido mediante patrones y heurísticas

- **Identificación de formatos de fecha inconsistentes**
  - Detecta cuando una misma columna contiene múltiples formatos (DD/MM/YYYY, MM/DD/YYYY, etc.)
  - Recomienda la estandarización a un formato común (preferentemente ISO 8601)

- **Verificación de información de zona horaria**
  - Comprueba si las columnas datetime tienen información de zona horaria
  - Sugiere agregar información de zona horaria para análisis temporales precisos

#### 3. Validación de Unidades de Medida (`validar_unidades`)

Este validador identifica inconsistencias en unidades de magnitudes físicas:

- **Detección de columnas que podrían contener unidades de medida**
  - Identifica columnas por nombre (temperatura, peso, volumen, etc.)
  - Analiza rangos de valores para inferir posibles unidades

- **Identificación de posibles mezclas de unidades**
  - Detecta valores atípicos que podrían indicar un cambio de unidad
  - Sugiere la conversión a una unidad estándar

- **Soporte para diferentes tipos de magnitudes**
  - Temperatura (Celsius, Fahrenheit, Kelvin)
  - Peso (kilogramos, gramos, libras, onzas)
  - Longitud (metros, centímetros, pulgadas, pies)
  - Volumen (litros, mililitros, galones)

### Detección y Gestión de Duplicados

El sistema también incorpora herramientas avanzadas para la detección y gestión de duplicados:

1. **Detección personalizable** (`detectar_duplicados`):
   - Permite seleccionar las columnas clave para identificar duplicados
   - Genera estadísticas detalladas sobre la cantidad y distribución de duplicados
   - Identifica grupos de registros duplicados para análisis detallado

2. **Opciones de corrección**:
   - **Eliminación selectiva** (`eliminar_duplicados`): Permite conservar la primera ocurrencia, la última, o eliminar todas
   - **Fusión inteligente** (`fusionar_duplicados`): Combina registros duplicados mediante diferentes estrategias por columna:
     - Para columnas numéricas: media, suma, mínimo, máximo, mediana
     - Para columnas categóricas: primer valor, último valor, valor más frecuente

3. **Feedback visual**:
   - Previsualización de resultados antes de aplicar cambios
   - Estadísticas comparativas pre/post corrección
   - Historial detallado de correcciones aplicadas

### Flujo de Trabajo de Validación

El proceso de validación sigue estos pasos:

1. **Detección automática**:

   ```python
   Cargar datos → Ejecutar validaciones → Presentar resultados
   ```

2. **Selección de correcciones**:

   ```python
   Revisar problemas → Seleccionar correcciones → Configurar parámetros
   ```

3. **Aplicación de cambios**:

   ```python
   Previsualizar cambios → Aplicar correcciones → Registrar historial
   ```

4. **Gestión de duplicados** (opcional):

   ```python
   Seleccionar columnas clave → Detectar duplicados → Eliminar o fusionar
   ```

### Beneficios del Sistema de Validación

- **Detección proactiva**: Identifica problemas antes de que afecten al análisis
- **Corrección guiada**: Asiste al usuario en la aplicación de las correcciones adecuadas
- **Flexibilidad**: Permite personalizar las validaciones según las necesidades específicas
- **Trazabilidad**: Mantiene un historial detallado de todas las correcciones aplicadas
- **Robustez**: Manejo de errores para evitar pérdida de datos durante las transformaciones

### Librerías Utilizadas en la Validación

- **pandas**: Para manipulación eficiente de datos y detección de tipos
- **re (expresiones regulares)**: Para la identificación de patrones en fechas y unidades
- **datetime**: Para validación y estandarización de formatos de fecha
- **streamlit**: Para la interfaz de usuario interactiva
- **logging**: Para el registro detallado de operaciones y auditoría

## Mejoras Recientes (Junio 2025)

Se ha implementado la Historia de Usuario 9 (HU9) "Evaluar múltiples modelos de ML para encontrar el más preciso", que incluye:

1. **Benchmarking automático** de múltiples modelos con manejo robusto de errores
   - Cada modelo se ejecuta de forma independiente para que el fallo de uno no afecte al resto
   - Detección automática del tipo de problema (clasificación/regresión)
   - Selección automática del conjunto apropiado de modelos según el tipo de problema

2. **Persistencia mejorada** de resultados del benchmarking
   - Guardado completo de resultados en la base de datos SQLite
   - Recuperación de benchmarkings anteriores para comparación
   - Almacenamiento de modelos entrenados y conjuntos de datos de prueba para evaluación posterior

3. **Experiencia de usuario optimizada**
   - Barra de progreso detallada durante el benchmarking
   - Visualización comparativa de resultados con gráficos interactivos
   - Sistema de caché inteligente para evitar re-entrenamiento innecesario
   - Control manual para forzar un nuevo entrenamiento cuando sea necesario

4. **Visualizaciones avanzadas de modelos** (HU10 - Comparación visual avanzada)
   - Matrices de confusión interactivas con normalización personalizable
   - Curvas ROC y Precision-Recall para clasificación
   - Gráficos de residuos y análisis de distribuciones para regresión
   - Comparación visual directa entre múltiples modelos
   - Interpretaciones automáticas de las visualizaciones
   - Exportación de gráficos en formato PNG para reportes

5. **Integración con el sistema de auditoría**
   - Registro detallado de operaciones en la tabla de auditoría
   - Trazabilidad completa del proceso de entrenamiento y evaluación
   - Captura de errores y excepciones para facilitar la depuración
   - Posibilidad de cargar benchmarkings anteriores por ID
   - Registro detallado en el log de auditoría

6. **Visualización comparativa** de modelos
   - Tabla con todas las métricas relevantes por modelo
   - Gráficos de barras comparativos para las métricas clave
   - Código de colores para facilitar la interpretación visual

7. **Navegación mejorada** entre etapas del workflow
   - Integración fluida entre entrenamiento, evaluación y recomendación
   - Conservación del estado entre páginas mediante SessionManager
   - Botones de navegación directa a las siguientes etapas del proceso

8. **Experiencia de usuario robusta**
   - Mensajes informativos en cada etapa del proceso
   - Manejo de excepciones y visualización amigable de errores
   - Barras de progreso detalladas durante el entrenamiento

9. **Soporte para conjuntos de datos desbalanceados**
   - Incorporación de métricas específicas (F1, precision, recall)
   - Visualización de la distribución de clases
   - Incorporación de imbalanced-learn como dependencia

## Gestión del estado

La aplicación utiliza un gestor centralizado de estado (`SessionManager`) que:

- Mantiene el estado de sesión entre páginas
- Registra el progreso del usuario en el flujo de trabajo
- Almacena y proporciona información del dataset y la configuración
- Ofrece métodos para restablecer el análisis cuando sea necesario

## Objetivo de los archivos principales

| Archivo/Carpeta                        | Descripción                                                                                |
|----------------------------------------|--------------------------------------------------------------------------------------------|
| app.py                                 | Punto de entrada principal de la app Streamlit. Inicializa la navegación multipágina.      |
| pages/00_Logueo.py                     | Página de inicio de sesión de la aplicación.                                               |
| pages/Datos/01_Cargar_Datos.py         | Página para cargar datos desde CSV o seleccionar datasets existentes.                      |
| pages/Datos/02_Configurar_Datos.py     | Página para configurar el tipo de problema, variable objetivo y predictores.               |
| pages/Datos/03_Validar_Datos.py        | Página para validar tipos de datos, formatos de fecha y unidades.                          |
| pages/Machine Learning/04_Entrenar_Modelos.py | Página para ejecutar benchmarking automático de múltiples modelos de ML.            |
| pages/Machine Learning/05_Evaluar_Modelos.py  | Página para visualizar y comparar en detalle los modelos entrenados.               |
| pages/Machine Learning/06_Recomendar_Modelo.py | Página que recomienda el mejor modelo según criterios seleccionables.             |
| pages/Reportes/07_Reporte.py           | Página para generar y descargar reportes en PDF o CSV.                                    |
| pages/Reportes/08_Dashboard.py         | Dashboard unificado con resumen de datasets, modelos, reportes y transformaciones.        |
| src/audit/logger.py                    | Registro de logs de auditoría: carga, transformación, entrenamiento, exportación, etc.    |
| src/config/workflow_steps.json         | Configuración de los pasos del workflow y su estado para la barra lateral.                |
| src/datos/cargador.py                  | Funciones para cargar datos desde CSV, validar y almacenar en SQLite.                     |
| src/datos/formateador.py               | Funciones para estandarizar formatos de fecha y unidades de medida.                       |
| src/datos/limpiador.py                 | Funciones para detectar y limpiar duplicados, valores nulos y problemas de calidad.       |
| src/datos/transformador.py             | Funciones para aplicar transformaciones y revertirlas si es necesario.                    |
| src/datos/validador.py                 | Funciones para validar tipos de datos, formatos y unidades.                               |
| src/datos/mock_db.py                   | Inicializa la base de datos SQLite y crea información de ejemplo.                         |
| src/modelos/configurador.py            | Configuración de parámetros para los modelos de machine learning.                         |
| src/modelos/entrenador.py              | Lógica para entrenar múltiples modelos de ML y realizar benchmarking automático.          |
| src/modelos/evaluador.py               | Funciones para evaluar modelos y visualizar métricas detalladas.                          |
| src/modelos/modelo_serializer.py       | Sistema de serialización/deserialización de modelos ML para almacenamiento y recuperación.|
| src/modelos/diagnostico_modelo.py      | Diagnóstico sobre disponibilidad y estado de objetos modelo en el benchmarking.           |
| src/modelos/recomendador.py            | Algoritmo para recomendar el mejor modelo según criterios seleccionables.                 |
| src/modelos/visualizador.py            | Generación de visualizaciones avanzadas para evaluación de modelos.                       |
| src/reportes/generador.py              | Generación de reportes PDF/CSV con resumen de análisis, transformaciones y modelos.       |
| src/seguridad/autenticador.py          | Control de roles y validación de permisos de usuario.                                     |
| src/state/session_manager.py           | Gestión centralizada del estado de la aplicación y progreso del workflow.                 |
| src/ui/sidebar.py                      | Componentes para la barra lateral y visualización del progreso.                           |

## Uso de la base de datos SQLite

La aplicación utiliza SQLite como sistema de almacenamiento de datos principal. Esto proporciona:

1. **Almacenamiento estructurado**: Tablas relacionales para datasets, transformaciones, modelos y auditoría.
2. **Persistencia de datos**: Los datos cargados y procesados se mantienen entre sesiones.
3. **Auditoría completa**: Registro de todas las operaciones realizadas por los usuarios.
4. **Facilidad de despliegue**: No requiere configuración de servidores de base de datos adicionales.

### Tablas principales relacionadas con Machine Learning

La aplicación utiliza varias tablas en la base de datos SQLite para almacenar información relacionada con el proceso de machine learning:

1. **benchmarking_modelos**: Almacena los resultados del benchmarking de modelos
   - `id`: Identificador único del benchmarking
   - `id_usuario`: Usuario que ejecutó el benchmarking
   - `tipo_problema`: Tipo de problema (clasificación/regresión)
   - `variable_objetivo`: Nombre de la variable objetivo
   - `cantidad_modelos_exitosos`: Número de modelos entrenados exitosamente
   - `cantidad_modelos_fallidos`: Número de modelos que fallaron
   - `mejor_modelo`: Nombre del mejor modelo según la métrica principal
   - `resultados_completos`: JSON con los resultados detallados de todos los modelos
   - `fecha_ejecucion`: Fecha y hora de ejecución

2. **modelos_seleccionados**: Registra las selecciones de modelos realizadas por los usuarios
   - `id`: Identificador único de la selección
   - `id_usuario`: Usuario que realizó la selección
   - `id_benchmarking`: Referencia al benchmarking
   - `nombre_modelo`: Nombre del modelo seleccionado
   - `comentarios`: Comentarios sobre la selección
   - `fecha_seleccion`: Fecha y hora de la selección

3. **auditoria**: Registro de todas las acciones realizadas
   - `id`: Identificador único del registro
   - `id_usuario`: Usuario que realizó la acción
   - `accion`: Tipo de acción (BENCHMARKING_MODELOS, EVALUACION_DETALLADA, SELECCION_MODELO, etc.)
   - `descripcion`: Descripción detallada de la acción
   - `fecha`: Fecha y hora de la acción

### Configuración y uso de la base de datos

1. **Creación de la base de datos**

   Ejecuta el script de inicialización para crear la base de datos SQLite y las tablas necesarias:

   ```bash
   python src/datos/mock_db.py
   ```

   Esto generará un archivo `analitica_farma.db` en el directorio raíz.

2. **Uso en la aplicación**

   La aplicación utiliza la base de datos para:
   - Almacenar datasets cargados desde CSV
   - Guardar configuraciones de análisis
   - Registrar metadatos de modelos entrenados
   - Mantener un registro de auditoría

3. **Ejemplo de conexión a la base de datos en Python**

```python
import sqlite3
conn = sqlite3.connect("analitica_farma.db")
cursor = conn.cursor()
cursor.execute("SELECT * FROM datasets")
resultados = cursor.fetchall()
conn.close()
```

## Componentes UI reutilizables

La aplicación utiliza componentes de UI reutilizables para mantener una experiencia de usuario consistente:

1. **Barra lateral**: Muestra información del dataset actual y el progreso del workflow
2. **Indicadores de progreso**: Visualización del estado de cada etapa del análisis
3. **Componentes de validación**: Formateo y validación unificada de datos

Estos componentes se implementan en el módulo `src/ui/` y se utilizan en todas las páginas de la aplicación.

## Gestión de estado centralizada

El módulo `src/state/session_manager.py` proporciona:

1. **Inicialización del estado**: Configura todas las variables de estado necesarias
2. **Acceso a la información del dataset**: Métodos para obtener metadatos del dataset actual
3. **Seguimiento del progreso**: Actualización y consulta del estado de las etapas del workflow
4. **Reinicio de análisis**: Funcionalidad para reiniciar un análisis completo

## Instalación y ejecución

1. **Clonar el repositorio**

   ```bash
   git clone [URL del repositorio]
   cd analitica-farma
   ```

2. **Instalar dependencias**

   ```bash
   pip install -r requirements.txt
   ```

3. **Inicializar la base de datos (primer uso)**

   ```bash
   python src/datos/mock_db.py
   ```

4. **Ejecutar la aplicación**

   ```bash
   streamlit run app.py
   ```

## Sistema de Serialización de Modelos

La aplicación incorpora un sistema robusto para la serialización y deserialización de modelos de machine learning, garantizando que los objetos modelo estén disponibles para visualizaciones avanzadas y predicciones en tiempo real.

### Arquitectura del Sistema de Serialización

El sistema utiliza un enfoque en capas para manejar la serialización:

1. **Nivel de modelo individual**:  
   - `serializar_modelo()` y `deserializar_modelo()` convierten modelos individuales entre su forma de objeto y representación serializada
   - Utiliza joblib para la serialización eficiente de objetos scikit-learn
   - Codifica en base64 para almacenamiento seguro en JSON/SQLite

2. **Nivel de benchmarking completo**:
   - `serializar_modelos_benchmarking()` procesa todos los modelos en un resultado de benchmarking
   - `deserializar_modelos_benchmarking()` reconstruye todos los modelos a partir de resultados guardados
   - Mantiene la estructura de datos original para compatibilidad con el resto de la aplicación

3. **Nivel de diagnóstico**:
   - `diagnosticar_objetos_modelo()` verifica la disponibilidad de objetos modelo
   - Proporciona información de depuración sobre el estado de serialización

### Proceso de Serialización

El flujo de trabajo de serialización sigue estos pasos:

1. **Entrenamiento**:

   ```python
   Entrenar modelos → Serializar objetos → Guardar en SQLite
   ```

2. **Carga de Benchmarking**:

   ```python
   Cargar de SQLite → Deserializar objetos → Restaurar modelos completos
   ```

3. **Evaluación y Visualización**:

   ```python
   Usar objetos modelo para → Predicciones → Visualizaciones avanzadas
   ```

### Ventajas técnicas

- **Persistencia completa**: Los modelos entrenados se conservan incluso después de cerrar la aplicación
- **Ahorro de recursos**: No es necesario reentrenar modelos para evaluaciones posteriores
- **Experiencia de usuario mejorada**: Acceso a visualizaciones avanzadas en cualquier momento
- **Compatibilidad**: Funciona con todos los modelos de scikit-learn y sus extensiones

### Tecnologías utilizadas

- **joblib**: Serialización optimizada para objetos científicos de Python, especialmente eficiente con arrays NumPy
- **base64**: Codificación segura para almacenamiento en bases de datos y formatos JSON
- **io.BytesIO**: Buffers en memoria para operaciones eficientes de serialización sin archivos temporales
- **JSON**: Formato intermedio para almacenamiento estructurado en SQLite

## Librerías y Métodos de Machine Learning

La aplicación utiliza una variedad de librerías y métodos para implementar el flujo completo de análisis de datos y machine learning, cada una con un propósito específico dentro del pipeline:

### Librerías principales

#### 1. Scikit-learn (sklearn)

Es la columna vertebral de nuestras funcionalidades de machine learning, proporcionando:

- **Preprocesamiento de datos**:
  - `train_test_split`: División de datos en conjuntos de entrenamiento y prueba
  - `StandardScaler`: Estandarización de variables numéricas
  - `LabelEncoder`: Codificación de variables categóricas

- **Modelos de clasificación**:
  - `LogisticRegression`: Modelo lineal para clasificación binaria y multiclase
  - `DecisionTreeClassifier`: Árboles de decisión para problemas no lineales
  - `RandomForestClassifier`: Conjunto de árboles para mayor robustez
  - `GradientBoostingClassifier`: Boosting de gradiente para mejorar la precisión
  - `SVC`: Máquinas de vectores de soporte para problemas complejos
  - `KNeighborsClassifier`: Clasificación basada en vecinos cercanos
  - `GaussianNB`: Clasificador bayesiano para probabilidades condicionadas
  - `AdaBoostClassifier`: Boosting adaptativo para mejorar modelos débiles

- **Modelos de regresión**:
  - `LinearRegression`: Regresión lineal simple y múltiple
  - `Ridge`, `Lasso`, `ElasticNet`: Regresiones regularizadas para evitar sobreajuste
  - `DecisionTreeRegressor`: Árboles de decisión para regresión no lineal
  - `RandomForestRegressor`: Ensamble de árboles para regresión robusta
  - `GradientBoostingRegressor`: Boosting de gradiente para regresión
  - `SVR`: Regresión con vectores de soporte
  - `KNeighborsRegressor`: Regresión basada en vecinos cercanos

- **Validación y evaluación**:
  - `cross_val_score`: Validación cruzada para evaluación robusta
  - Métricas de clasificación: `accuracy_score`, `precision_score`, `recall_score`, `f1_score`
  - Métricas de regresión: `r2_score`, `mean_squared_error`, `mean_absolute_error`

#### 2. Pandas y NumPy

- `pandas`: Manipulación y análisis de datos estructurados
  - DataFrames para almacenamiento eficiente y consultas
  - Funciones de limpieza y transformación de datos
  - Métodos para manejo de valores nulos y duplicados

- `numpy`: Computación numérica eficiente
  - Arrays multidimensionales y operaciones vectorizadas
  - Funciones matemáticas y estadísticas
  - Generación de números aleatorios controlados por semilla

#### 3. Visualización

- `matplotlib` y `seaborn`: Generación de gráficos estadísticos
  - Histogramas para visualizar distribuciones
  - Gráficos de barras para comparar métricas
  - Matrices de confusión para evaluar clasificación
  - Curvas ROC y PR para evaluación detallada

- `plotly`: Visualizaciones interactivas
  - Gráficos dinámicos para exploración de datos
  - Dashboards interactivos para la interpretación de resultados

#### 4. Extensiones específicas

- `imbalanced-learn`: Manejo de datasets desbalanceados
  - Técnicas de sobremuestreo (SMOTE, ADASYN)
  - Técnicas de submuestreo (RandomUnderSampler)
  - Combinaciones híbridas (SMOTETomek, SMOTEENN)

- `shap`: Interpretabilidad de modelos
  - Valores SHAP para explicar predicciones individuales
  - Gráficos de importancia de características
  - Análisis de dependencia para entender relaciones

- `joblib` y `base64`: Serialización robusta de modelos
  - Almacenamiento eficiente de modelos scikit-learn
  - Codificación segura para base de datos
  - Compresión integrada para optimizar espacio

- Modelos avanzados:
  - `lightgbm`: Implementación eficiente de gradient boosting
  - `xgboost`: Implementación escalable de boosting de gradiente extremo
  - `catboost`: Modelo optimizado para variables categóricas

### Métodos clave implementados

#### 1. Benchmarking automático (`entrenador.py`)

- **`ejecutar_benchmarking`**: Método principal que:
  1. Detecta automáticamente el tipo de problema (clasificación/regresión)
  2. Preprocesa los datos (división, escalado, codificación)
  3. Entrena múltiples modelos en paralelo con manejo de errores
  4. Calcula métricas relevantes para cada modelo
  5. Ordena los resultados según el rendimiento

- **`guardar_resultados_benchmarking`**: Persistencia de resultados que:
  1. Serializa las métricas y parámetros de cada modelo
  2. Almacena en la base de datos SQLite con timestamp
  3. Registra en el log de auditoría

- **`obtener_benchmarking_por_id`**: Recupera resultados anteriores para:
  1. Comparación entre diferentes ejecuciones
  2. Análisis de rendimiento a lo largo del tiempo
  3. Evaluación detallada de modelos específicos

#### 2. Sistema de Serialización de Modelos (`modelo_serializer.py`)

- **`serializar_modelo`**: Convierte modelos de scikit-learn a formato serializable
  1. Utiliza joblib para serializar eficientemente el objeto modelo completo
  2. Codifica el resultado en base64 para almacenamiento seguro en JSON/SQLite
  3. Comprime los datos para optimizar el espacio de almacenamiento

- **`deserializar_modelo`**: Reconstruye objetos modelo a partir de representaciones serializadas
  1. Decodifica la cadena base64 a su representación binaria
  2. Utiliza joblib para cargar el objeto modelo completo con sus parámetros
  3. Recupera todas las funcionalidades del modelo original (predict, predict_proba, etc.)

- **`serializar_modelos_benchmarking`**: Prepara los resultados completos del benchmarking
  1. Procesa recursivamente todos los modelos en los resultados
  2. Marca los modelos que tienen objetos serializados para seguimiento
  3. Elimina objetos no serializables manteniendo la estructura de datos

- **`deserializar_modelos_benchmarking`**: Restaura los objetos modelo en los resultados
  1. Reconstruye automáticamente todos los modelos serializados
  2. Mantiene la estructura original de datos para compatibilidad
  3. Optimiza la memoria eliminando representaciones serializadas redundantes

#### 3. Diagnóstico de Modelos (`diagnostico_modelo.py`)

- **`diagnosticar_objetos_modelo`**: Verifica la disponibilidad de objetos modelo
  1. Analiza los resultados del benchmarking para detectar modelos sin objetos
  2. Proporciona diagnóstico detallado del estado de cada modelo
  3. Genera recomendaciones sobre cómo resolver problemas de serialización

#### 4. Visualización avanzada de modelos (`visualizador.py`)

- **`generar_matriz_confusion`**: Crea matrices de confusión personalizables
- **`generar_curva_roc`**: Visualiza curvas ROC para problemas binarios y multiclase
- **`generar_curva_precision_recall`**: Grafica curvas de precisión-recall
- **`generar_grafico_residuos`**: Analiza residuos para modelos de regresión
- **`comparar_distribuciones`**: Compara valores reales vs. predichos
- **`comparar_modelos_roc`**: Contrasta rendimiento de múltiples modelos en un gráfico

#### 2. Evaluación detallada (`evaluador.py`)

- **`evaluar_modelo_detallado`**: Análisis profundo que:
  1. Extrae métricas completas del modelo seleccionado
  2. Genera visualizaciones específicas del tipo de problema
  3. Analiza el rendimiento con validación cruzada

- **`comparar_modelos`**: Comparación visual que:
  1. Genera gráficos comparativos de métricas clave
  2. Destaca fortalezas y debilidades de cada modelo
  3. Facilita la toma de decisiones basada en datos

#### 3. Recomendación inteligente (`recomendador.py`)

- **`recomendar_mejor_modelo`**: Selección basada en criterios que:
  1. Evalúa modelos según criterios personalizables
  2. Proporciona justificación detallada de la recomendación
  3. Sugiere el modelo óptimo para el problema específico

- **`guardar_seleccion_modelo`**: Persistencia de decisiones que:
  1. Registra la selección del usuario con comentarios
  2. Mantiene un historial de selecciones para análisis
  3. Facilita la auditoría y trazabilidad

### Integración y flujo de datos

El flujo de datos a través del pipeline de machine learning sigue estas etapas:

1. **Carga y preparación**:
   - Los datos se cargan desde CSV o Snowflake mediante `cargador.py`
   - Se aplican transformaciones de limpieza con `limpiador.py`
   - Se validan los tipos y formatos con `validador.py`

2. **Modelado y evaluación**:
   - Se ejecuta el benchmarking automático con `entrenador.py`
   - Se analizan los resultados detalladamente con `evaluador.py`
   - Se recomienda el mejor modelo con `recomendador.py`

3. **Visualización y reporte**:
   - Se generan visualizaciones comparativas con matplotlib/seaborn
   - Se crean dashboards interactivos con plotly
   - Se producen reportes estructurados con los hallazgos clave

Cada componente está diseñado para funcionar de manera independiente pero integrada, siguiendo principios de modularidad y responsabilidad única.

### Gestión de errores y robustez

El sistema implementa varias estrategias para garantizar la robustez:

1. **Manejo de excepciones específicas** para cada modelo en el benchmarking
2. **Registro detallado** de errores y advertencias en el log de auditoría
3. **Validación de datos** antes del entrenamiento para prevenir problemas
4. **Mecanismos de fallback** en caso de fallo de componentes individuales
5. **Persistencia transaccional** de resultados para evitar pérdida de información

Esta arquitectura asegura que el sistema pueda manejar eficientemente diferentes tipos de datos, problemas y escenarios, proporcionando resultados confiables y explicables para la toma de decisiones industriales.

## Visualizaciones Avanzadas de Modelos (HU10)

La historia de usuario 10 implementa un sistema integral de visualizaciones avanzadas para la evaluación detallada de modelos de machine learning, permitiendo un análisis profundo del rendimiento mediante técnicas de visualización interactivas.

### Características principales

#### 1. Módulo de visualización centralizado (`visualizador.py`)

Se ha implementado un módulo especializado en la generación de visualizaciones avanzadas que:

- **Estandariza el estilo gráfico** en toda la aplicación
- **Maneja errores robustamente** para evitar fallos en la interfaz
- **Genera interpretaciones automáticas** de las visualizaciones
- **Soporta exportación de gráficos** para reportes

#### 2. Visualizaciones para modelos de clasificación

Para problemas de clasificación, se implementan:

- **Matrices de confusión interactivas**
  - Opciones de normalización (por filas, columnas o total)
  - Interpretación automática con análisis de clases problemáticas
  - Identificación de patrones de confusión más comunes

- **Curvas ROC y área bajo la curva (AUC)**
  - Soporte para problemas binarios y multiclase
  - Comparación visual entre diferentes modelos
  - Interpretación automática del poder discriminativo

- **Curvas Precision-Recall**
  - Análisis detallado para conjuntos desbalanceados
  - Evaluación de compromiso entre precisión y exhaustividad
  - Personalización de puntos de corte

#### 3. Visualizaciones para modelos de regresión

Para problemas de regresión, se implementan:

- **Gráficos de residuos multi-panel**
  - Residuos vs. valores predichos
  - Histograma de distribución de residuos
  - QQ-plot para evaluar normalidad
  - Predicciones vs. valores reales

- **Comparación de distribuciones**
  - Histogramas superpuestos de valores reales y predichos
  - Estimaciones de densidad kernel (KDE)
  - Análisis visual de sesgos y varianza

#### 4. Comparación directa entre modelos

La interfaz permite:

- **Selección múltiple de modelos** para comparación directa
- **Visualización conjunta de curvas ROC o predicciones**
- **Tablas de métricas comparativas** con resaltado automático
- **Análisis de fortalezas y debilidades** de cada modelo

### Integración en la interfaz de usuario

Las visualizaciones se integran en la página `05_Evaluar_Modelos.py` mediante:

- **Sistema de pestañas** para organizar diferentes visualizaciones
- **Selectores interactivos** para personalizar las visualizaciones
- **Paneles informativos** con interpretaciones automáticas
- **Botones de descarga** para exportar gráficos en formato PNG

Esta implementación proporciona a los usuarios una herramienta poderosa para entender el comportamiento de los modelos más allá de las métricas numéricas, facilitando la toma de decisiones informadas sobre qué modelo seleccionar para su problema específico.

## Próximos pasos

Las próximas mejoras planificadas incluyen:

1. **Implementación de curvas de aprendizaje** (HU11) para evaluar el comportamiento de los modelos con diferentes tamaños de datos de entrenamiento
2. **Herramientas de interpretabilidad de modelos** (HU12) como importancia de características, SHAP values y dependencias parciales
3. **Reportes automáticos de ML** (HU13) con explicaciones en lenguaje natural y visualizaciones clave
4. **Optimización automática de hiperparámetros** (HU14) mediante búsqueda en grid y bayesiana
5. **Integración con sistemas de MLOps** (HU15) para seguimiento de experimentos y versionado de modelos

Estas funcionalidades completarán un ecosistema integral para el análisis de datos industriales y la aplicación efectiva de técnicas de machine learning en entornos de producción real.
