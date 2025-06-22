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
│   │   ├── 04_Entrenar_Modelos.py
│   │   ├── 05_Evaluar_Modelos.py
│   │   └── 06_Recomendar_Modelo.py
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
│   │   ├── entrenador.py   # Entrenamiento de modelos
│   │   ├── evaluador.py    # Evaluación de modelos
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
| pages/Machine Learning/04_Entrenar_Modelos.py | Página para seleccionar, entrenar y comparar modelos de ML.                        |
| pages/Machine Learning/05_Evaluar_Modelos.py  | Página para visualizar y comparar métricas de los modelos entrenados.              |
| pages/Machine Learning/06_Recomendar_Modelo.py | Página que recomienda el mejor modelo según desempeño y permite su aprobación.    |
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
| src/modelos/entrenador.py              | Lógica para entrenar modelos de ML y separar conjuntos de entrenamiento/prueba.           |
| src/modelos/evaluador.py               | Funciones para evaluar modelos y calcular métricas clave (accuracy, RMSE, F1, etc.).      |
| src/modelos/recomendador.py            | Algoritmo para recomendar el mejor modelo según los resultados obtenidos.                 |
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

Ventajas del uso de SQLite para desarrollo:

- Desarrollo y pruebas sin dependencias externas
- Sin costos ni riesgos de modificar datos reales
- Simulación de diferentes escenarios con datos de ejemplo
- Ejecución de pruebas automatizadas reproducibles

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
