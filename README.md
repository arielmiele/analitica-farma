# analitica-farma

Aplicación para analizar datos productivos en la industria farmacéutica y recomendar modelos de machine learning. Permite cargar datos, validarlos, transformarlos, entrenar y comparar modelos, recomendar el mejor y generar reportes completos. Arquitectura modular, multipágina, con backend de datos SQLite para pruebas locales y Supabase para despliegue en Streamlit Community Cloud, además de logging centralizado.

---

## Tabla de Historias de Usuario Implementadas y Planificadas

| HU    | Descripción breve                                                          | Criterios de aceptación principales                | Archivos/módulos clave |
|-------|---------------------------------------------------------------------------|----------------------------------------------------|----|
| HU1   | Carga de datos desde CSV y persistencia en backend activo                 | Importar, validar y almacenar datasets             | src/datos/cargador.py, pages/Datos/01_Cargar_Datos.py |
| HU2   | Configuración del problema y variables                                    | Selección de variable objetivo y predictores       | src/datos/cargador.py, pages/Datos/04_Configurar_Datos.py |
| HU3   | Validación automática de tipos, fechas y unidades                         | Detectar y sugerir correcciones                    | src/datos/validador.py, pages/Datos/03_Validar_Datos.py |
| HU4   | Transformación y limpieza de datos                                        | Aplicar transformaciones, gestionar duplicados     | src/datos/transformador.py, src/datos/limpiador.py |
| HU5   | Análisis de calidad y resumen de datos                                    | Estadísticas, visualización de calidad             | src/datos/analizador.py, pages/Datos/04_Analizar_Calidad.py |
| HU6   | Entrenamiento automático de modelos ML                                    | Benchmarking, manejo de errores, persistencia      | src/modelos/entrenador.py, pages/Machine Learning/05_Entrenar_Modelos.py |
| HU7   | Evaluación detallada de modelos                                           | Métricas, validación cruzada, comparación          | src/modelos/evaluador.py, pages/Machine Learning/06_Evaluar_Modelos.py |
| HU8   | Recomendación automática del mejor modelo                                 | Selección por criterios, justificación             | src/modelos/recomendador.py, pages/Machine Learning/08_Recomendar_Modelo.py |
| HU9   | Visualización avanzada y comparación de modelos                           | Matriz confusión, ROC, PR, residuos, exportación   | src/modelos/visualizador.py, pages/Machine Learning/06_Evaluar_Modelos.py |
| HU10  | Validación estricta Model-View, separación lógica negocio/UI              | Lógica de negocio solo en modelos, UI solo presentación | src/modelos/evaluador.py, src/ui/, pages/ |
| HU11  | Curvas de aprendizaje y validación cruzada avanzada                       | Curvas learning, análisis de overfitting           | src/modelos/evaluador.py, src/modelos/visualizador.py |
| HU12  | Interpretabilidad avanzada (SHAP, importancia de variables)               | Explicaciones automáticas, visualización de importancia | src/modelos/explicador.py, src/modelos/visualizador.py |
| HU13  | Generación y descarga de reportes completos (PDF)                         | Reporte PDF con resultados, gráficos y recomendaciones | src/reportes/generador.py, pages/Reportes/10_Reporte.py |
| HU14  | Optimización automática de hiperparámetros                                | Búsqueda grid/bayesiana, comparación de resultados | src/modelos/entrenador.py, src/modelos/configurador.py |
| HU15  | Integración MLOps y versionado de modelos                                 | Seguimiento de experimentos, versionado, auditoría | src/modelos/modelo_serializer.py, src/audit/logger.py |

---

## Historias de Usuario: Narrativa y Cumplimiento

### HU1: Carga de datos desde CSV y persistencia por backend

Permite importar datasets desde archivos locales, validando estructura y almacenando en el backend configurado (`sqlite` para local o `supabase` en cloud). Implementado en `src/datos/cargador.py` y `pages/Datos/01_Cargar_Datos.py`.

### HU2: Configuración del problema y variables

Selección guiada de variable objetivo y predictores, con validación de tipos y sugerencias. Implementado en `src/datos/cargador.py` y `pages/Datos/04_Configurar_Datos.py`.

### HU3: Validación automática de tipos, fechas y unidades

Detección de inconsistencias y sugerencia de correcciones, con feedback visual. Implementado en `src/datos/validador.py` y `pages/Datos/03_Validar_Datos.py`.

### HU4: Transformación y limpieza de datos

Aplicación de transformaciones, gestión de duplicados y valores nulos, con historial de cambios. Implementado en `src/datos/transformador.py`, `src/datos/limpiador.py`.

### HU5: Análisis de calidad y resumen de datos

Estadísticas descriptivas, visualización de calidad y alertas de problemas. Implementado en `src/datos/analizador.py` y `pages/Datos/04_Analizar_Calidad.py`.

### HU6: Entrenamiento automático de modelos ML

Benchmarking de múltiples modelos, manejo de errores y persistencia de resultados. Implementado en `src/modelos/entrenador.py` y `pages/Machine Learning/05_Entrenar_Modelos.py`.

### HU7: Evaluación detallada de modelos

Métricas completas, validación cruzada y comparación visual. Implementado en `src/modelos/evaluador.py` y `pages/Machine Learning/06_Evaluar_Modelos.py`.

### HU8: Recomendación automática del mejor modelo

Selección basada en criterios personalizables, justificación y persistencia. Implementado en `src/modelos/recomendador.py` y `pages/Machine Learning/08_Recomendar_Modelo.py`.

### HU9: Visualización avanzada y comparación de modelos

Matriz de confusión, curvas ROC/PR, gráficos de residuos y exportación de visualizaciones. Implementado en `src/modelos/visualizador.py` y `pages/Machine Learning/06_Evaluar_Modelos.py`.

### HU10: Validación estricta Model-View

Refactorización para separar lógica de negocio (modelos, cálculos, visualizaciones) en `src/modelos/` y presentación/interacción en `src/ui/` y `pages/`. Validado en `docs-private/registros/HU11_Fase3_Validacion_Model_View_Implementacion.md`.

### HU11: Curvas de aprendizaje y validación cruzada avanzada

Implementación de curvas de aprendizaje y análisis de overfitting/underfitting. Planificado en `src/modelos/evaluador.py`, `src/modelos/visualizador.py`.

### HU12: Interpretabilidad avanzada

Explicaciones automáticas de modelos (SHAP, importancia de variables), visualización y análisis de dependencias. Planificado en `src/modelos/explicador.py`, `src/modelos/visualizador.py`.

### HU13: Reportes completos en PDF

Generación de reportes PDF con resultados, gráficos y recomendaciones, guardado en el backend activo y descarga desde la UI. Implementado en `src/reportes/generador.py` y `pages/Reportes/10_Reporte.py`.

### HU14: Optimización automática de hiperparámetros

Búsqueda de hiperparámetros (grid/bayesiana), comparación de resultados y selección óptima. Planificado en `src/modelos/entrenador.py`, `src/modelos/configurador.py`.

### HU15: Integración MLOps y versionado de modelos

Seguimiento de experimentos, versionado de modelos y auditoría avanzada. Planificado en `src/modelos/modelo_serializer.py`, `src/audit/logger.py`.

---

## Instalación y Ejecución Local

1. **Clonar el repositorio**

   ```bash
   # Clonar el repositorio y acceder al directorio
   git clone https://github.com/arielmiele/analitica-farma.git
   cd analitica-farma
   ```

2. **Crear y activar entorno virtual (recomendado)**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # En Windows
   # source .venv/bin/activate  # En Linux/Mac
   ```

3. **Instalar dependencias**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configurar variables de entorno**
  - Solicita acceso a un usuario habilitado. El acceso está restringido y la aplicación no cuenta con gestión de usuarios; solo quienes tengan permisos podrán operar la herramienta.
  - Local: no requiere credenciales externas si usas SQLite (valor por defecto).
  - Cloud (Streamlit Community Cloud): configurar Supabase con `DB_BACKEND=supabase`, `SUPABASE_URL` y `SUPABASE_KEY`.

5. **Ejecutar la aplicación**

   ```bash
   streamlit run app.py
   ```

6. **Acceso**
   - Abrir el navegador en la URL que indica Streamlit (por defecto <http://localhost:8501>)

---

## Notas de Arquitectura y Estructura

- Arquitectura modular, separación estricta Model-View: lógica de negocio en `src/modelos/`, UI en `src/ui/` y `pages/`.
- Backend de datos configurable por entorno: SQLite para pruebas locales y Supabase para despliegue en Streamlit Community Cloud.
- El acceso está restringido a usuarios habilitados; aún no existe gestión de usuarios en la aplicación.

---

## Estructura del Proyecto

```text
├── app.py                  # Punto de entrada principal de la app Streamlit y la navegación multipágina
├── requirements.txt        # Dependencias del proyecto
├── README.md               # Este archivo
├── pages/                  # Páginas multipágina de Streamlit (cada funcionalidad principal)
│   ├── 00_Logueo.py        # Página de inicio de sesión
│   ├── Datos/
│   │   ├── 01_Cargar_Datos.py     # Carga de datos desde CSV
│   │   ├── 02_Validar_Datos.py    # Validación de tipos, fechas y unidades
│   │   ├── 03_Analizar_Calidad.py # Análisis de calidad y estadísticas
│   │   └── 04_Configurar_Datos.py # Configuración del problema y variables
│   ├── Machine Learning/
│   │   ├── 05_Entrenar_Modelos.py   # Benchmarking automático de múltiples modelos
│   │   ├── 06_Evaluar_Modelos.py    # Evaluación detallada de los modelos entrenados
│   │   ├── 07_Validacion_Cruzada.py # Validación cruzada y curvas de aprendizaje
│   │   ├── 08_Recomendar_Modelo.py  # Recomendación del mejor modelo según criterios
│   │   └── 09_Explicar_Modelo.py    # Interpretabilidad del modelo
│   └── Reportes/
│       └── 10_Reporte.py
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
      "Inicio": [pagina_deslogueo],
      "Datos": [cargar_datos, validar_datos, analizar_calidad, configurar_datos],
      "Machine Learning": [entrenar_modelos, evaluar_modelos, validacion_cruzada, recomendar_modelo, explicar_modelo],
      "Reporte": [reporte]
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

1. **Carga de datos**: Importación desde CSV o selección de datasets existentes en el backend activo.
2. **Configuración de datos**: Selección del tipo de problema (regresión/clasificación), variable objetivo y predictores.
3. **Validación de datos**: Validación automática de tipos de datos, formatos de fecha y unidades de medida.
4. **Transformación de datos**: Aplicación de transformaciones para mejorar la calidad de los datos.
5. **Entrenamiento de modelos**: Configuración y entrenamiento de múltiples modelos de ML.
6. **Evaluación de modelos**: Comparación de métricas de rendimiento entre los modelos entrenados.
7. **Recomendación de modelo**: Selección automática del mejor modelo según criterios predefinidos.
8. **Generación de reportes**: Creación de informes detallados con resultados y visualizaciones.

Cada paso está representado por una página separada, y el progreso se visualiza en la barra lateral mediante una lista de comprobación dinámica.

---

## Descripción Detallada de Pantallas y Componentes

A continuación se describe el objetivo, funcionamiento y aspectos técnicos clave de cada pantalla principal de la aplicación, incluyendo el uso de librerías, funciones y el manejo de auditoría, datos y sesión.

### 00_Logueo.py

**Objetivo:**

- Permitir el acceso seguro a la aplicación mediante autenticación corporativa (SSO o usuario habilitado).

**Qué hace y cómo lo hace:**

- Presenta un formulario de login.
- Valida credenciales contra el backend activo (SQLite o Supabase).
- Al autenticar, inicializa el estado de sesión (`SessionManager`).
- Registra el evento de login en los logs de auditoría (`logger.py`).

**Paquetes y funciones:**

- `streamlit` para UI.
- `src/seguridad/autenticador.py` para validación de usuario.
- `src/state/session_manager.py` para manejo de sesión.
- `src/audit/logger.py` para registro de auditoría.

---

### 01_Cargar_Datos.py

**Objetivo:**

- Permitir la carga de datos productivos desde archivos CSV y su persistencia en el backend activo.

**Qué hace y cómo lo hace:**

- Permite seleccionar y cargar archivos locales (CSV).
- Valida la estructura y formato de los datos usando `cargador.py` y `validador.py`.
- Almacena los datos en memoria y en el backend activo (SQLite o Supabase).
- Registra la operación en los logs de auditoría.

**Paquetes y funciones:**

- `pandas` para manipulación de datos.
- `sqlite3` (local) o `supabase` (cloud) para persistencia de datos.
- `src/datos/cargador.py` para carga y validación inicial.
- `src/audit/logger.py` para auditoría.

---

### 02_Validar_Datos.py

**Objetivo:**

- Validar automáticamente tipos de datos, formatos de fecha y unidades.

**Qué hace y cómo lo hace:**

- Ejecuta validaciones automáticas sobre el dataset cargado.
- Utiliza heurísticas y expresiones regulares para detectar problemas.
- Presenta resultados y sugerencias de corrección al usuario.
- Registra los resultados y acciones en el log de auditoría.

**Paquetes y funciones:**

- `pandas`, `re`, `datetime` para validaciones.
- `src/datos/validador.py` para lógica de validación.
- `src/audit/logger.py` para registro de validaciones.

---

### 03_Analizar_Calidad.py

**Objetivo:**

- Analizar la calidad de los datos y mostrar estadísticas descriptivas.

**Qué hace y cómo lo hace:**

- Calcula métricas de calidad (nulos, duplicados, outliers).
- Genera visualizaciones de calidad y alertas.
- Permite exportar reportes de calidad.
- Registra el análisis en los logs de auditoría.

**Paquetes y funciones:**

- `pandas`, `matplotlib`, `seaborn` para análisis y visualización.
- `src/datos/analizador.py` para lógica de análisis.
- `src/audit/logger.py` para auditoría.

---

### 04_Configurar_Datos.py

**Objetivo:**

- Configurar el tipo de problema, variable objetivo y predictores.

**Qué hace y cómo lo hace:**

- Permite seleccionar la variable objetivo y las variables predictoras.
- Valida la selección y sugiere configuraciones óptimas.
- Actualiza el estado de sesión con la configuración.
- Registra la configuración en el log de auditoría.

**Paquetes y funciones:**

- `streamlit` para UI.
- `src/datos/cargador.py` para obtención de variables.
- `src/state/session_manager.py` para guardar configuración.
- `src/audit/logger.py` para registro.

---

### 05_Entrenar_Modelos.py

**Objetivo:**

- Entrenar y comparar múltiples modelos de machine learning.

**Qué hace y cómo lo hace:**

- Ejecuta benchmarking automático de modelos (clasificación/regresión).
- Utiliza scikit-learn y LazyPredict para entrenamiento y comparación.
- Maneja errores y persistencia de resultados en el backend activo (SQLite/Supabase).
- Registra el proceso y resultados en el log de auditoría.

**Paquetes y funciones:**

- `scikit-learn`, `lazypredict`, `pandas` para ML.
- `src/modelos/entrenador.py` para lógica de entrenamiento.
- `src/audit/logger.py` para auditoría.
- `src/state/session_manager.py` para estado de modelos.

---

### 06_Evaluar_Modelos.py

**Objetivo:**

- Evaluar en detalle los modelos entrenados y comparar métricas.

**Qué hace y cómo lo hace:**

- Permite seleccionar modelos entrenados y visualizar métricas avanzadas.
- Genera gráficos comparativos y tablas de métricas.
- Permite cargar evaluaciones anteriores desde el backend activo.
- Registra la evaluación en el log de auditoría.

**Paquetes y funciones:**

- `scikit-learn`, `matplotlib`, `seaborn` para métricas y visualización.
- `src/modelos/evaluador.py` para lógica de evaluación.
- `src/audit/logger.py` para auditoría.

---

### 07_Validación_Cruzada.py

**Objetivo:**

- Realizar validación cruzada y análisis de curvas de aprendizaje.

**Qué hace y cómo lo hace:**

- Ejecuta validación cruzada sobre los modelos seleccionados.
- Genera curvas de aprendizaje y análisis de overfitting/underfitting.
- Presenta resultados visuales y recomendaciones.
- Registra el proceso en el log de auditoría.

**Paquetes y funciones:**

- `scikit-learn` para validación cruzada.
- `matplotlib`, `seaborn` para visualización.
- `src/modelos/validacion_cruzada.py`, `src/modelos/visualizador.py` para lógica y gráficos.
- `src/audit/logger.py` para auditoría.

---

### 08_Recomendar_Modelo.py

**Objetivo:**

- Recomendar automáticamente el mejor modelo según criterios definidos.

**Qué hace y cómo lo hace:**

- Evalúa los modelos entrenados según criterios seleccionados (accuracy, F1, R2, etc.).
- Presenta la recomendación y justificación al usuario.
- Permite guardar la selección en el backend activo y registrar la decisión.

**Paquetes y funciones:**

- `src/modelos/recomendador.py` para lógica de recomendación.
- `src/audit/logger.py` para registro de la recomendación.
- `src/state/session_manager.py` para persistencia de la selección.

---

### 09_Explicar_Modelo.py

**Objetivo:**

- Proveer interpretabilidad avanzada de los modelos (SHAP, importancia de variables).

**Qué hace y cómo lo hace:**

- Calcula e interpreta la importancia de variables y valores SHAP.
- Genera visualizaciones explicativas.
- Presenta análisis de dependencias y justificaciones.
- Registra el análisis en el log de auditoría.

**Paquetes y funciones:**

- `shap`, `matplotlib`, `seaborn` para interpretabilidad y visualización.
- `src/modelos/explicador.py`, `src/modelos/visualizador.py` para lógica y gráficos.
- `src/audit/logger.py` para auditoría.

---

### 10_Reporte.py

**Objetivo:**

- Generar y descargar reportes completos en PDF o CSV.

**Qué hace y cómo lo hace:**

- Compila resultados, gráficos y recomendaciones en un reporte estructurado.
- Permite exportar el reporte y almacenarlo en el backend activo.
- Registra la generación y descarga en el log de auditoría.

**Paquetes y funciones:**

- `reportlab`, `pandas` para generación de PDF/CSV.
- `src/reportes/generador.py` para lógica de reporte.
- `src/audit/logger.py` para auditoría.

---

## Manejo de Auditoría, Datos y Sesión

- **Auditoría:** Todas las acciones relevantes (login, carga, validación, entrenamiento, selección, reporte) se registran mediante `src/audit/logger.py` en el backend activo, incluyendo usuario, acción, timestamp y detalles.
- **Almacenamiento/Lectura de Datos:** La persistencia y consulta de datos se realiza con backend configurable: SQLite para entorno local y Supabase para cloud.
- **Gestión de Sesión:** El estado de usuario, configuración, progreso y selección de modelos se maneja centralizadamente con `src/state/session_manager.py`, asegurando persistencia y consistencia entre pantallas.

---

## Estructura de Base de Datos y Uso en la Aplicación

La aplicación utiliza un backend seleccionable por entorno mediante `DB_BACKEND`:

- `sqlite` (default): orientado a pruebas locales y desarrollo.
- `supabase`: orientado a despliegue en Streamlit Community Cloud.

### Tablas lógicas principales

La capa de datos mantiene la misma estructura funcional en ambos backends:

- `usuarios`: usuarios habilitados, roles y hash de credenciales.
- `sesiones`: trazabilidad de sesiones de usuario.
- `auditoria`: registro de acciones relevantes del flujo.
- `datasets`: metadatos de datasets cargados.
- `configuraciones_modelo`: configuración de target/predictores/parámetros.
- `benchmarking_modelos`: resultados de entrenamiento y benchmarking.
- `reportes`: metadatos de reportes generados.
- `historial_ejecuciones`: resumen de ejecuciones para análisis posterior.

### Persistencia de datasets

- En SQLite: los datasets se guardan como archivos Parquet en `data/datasets/` y se referencian desde la tabla `datasets`.
- En Supabase: los datasets se guardan en Storage (bucket `datasets`) y se referencian desde la tabla `datasets`.

### Flujo operativo

- El backend se resuelve en tiempo de ejecución desde `st.secrets` o variables de entorno.
- El resto de módulos de negocio no cambia su API pública; la capa `src/database/` abstrae la implementación concreta.
- Esto permite correr localmente sin infraestructura externa y desplegar en cloud con persistencia remota.

---
