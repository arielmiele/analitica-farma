# analitica-farma

Aplicación para analizar datos productivos en la industria farmacéutica y recomendar modelos de machine learning. Permite cargar datos, transformarlos, evaluar modelos y generar reportes. Desarrollada con Streamlit, Python y SQLite para almacenamiento local.

## Estructura del Proyecto

```text
├── app.py                  # Punto de entrada principal de la app Streamlit y la navegación multipágina
├── requirements.txt        # Dependencias del proyecto
├── README.md               # Este archivo
├── analitica_farma.db      # Base de datos SQLite local
├── pages/                  # Páginas multipágina de Streamlit (cada funcionalidad principal)
│   ├── 00_Logueo.py        # Página de inicio de sesión
│   ├── Datos/
│   │   ├── 01_Cargar_Datos.py
│   │   ├── 02_Validar_Datos.py
│   │   └── 03_Transformaciones.py
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
│   ├── datos/              # Carga, limpieza y transformación de datos
│   │   ├── cargador.py
│   │   ├── limpiador.py
│   │   ├── transformador.py
│   │   └── mock_db.py
│   ├── modelos/            # Entrenamiento, evaluación y recomendación de modelos ML
│   │   ├── entrenador.py
│   │   ├── evaluador.py
│   │   └── recomendador.py
│   ├── reportes/           # Generación de reportes PDF/CSV
│   │   └── generador.py
│   └── seguridad/          # Autenticación y control de acceso
│       └── autenticador.py
├── logs/                   # Logs de auditoría y operaciones
│   ├── auditoria_YYYYMMDD.log
│   └── carga_datos_YYYYMMDD.log
```

- Las páginas en `pages/` están organizadas en subcarpetas por dominio funcional: Datos, Machine Learning y Reportes.
- El archivo `app.py` implementa la navegación multipágina y el control de acceso (login/logout) usando `st.Page` y `st.navigation`.
- El código fuente en `src/` está organizado por dominio: datos, modelos, reportes, seguridad y auditoría.
- La base de datos SQLite (`analitica_farma.db`) almacena los datos, metadatos, usuarios y registros de auditoría.

## app.py

`app.py` es el punto de entrada de la aplicación y define:

- La configuración global de Streamlit (`st.set_page_config`).
- El control de sesión para login/logout.
- La navegación multipágina agrupada por secciones, usando `st.Page` y `st.navigation`.
- El acceso a las páginas está restringido según el estado de login del usuario.

Ejemplo de navegación:

```python
if st.session_state.logged_in:
    pg = st.navigation({
        "Cuenta": [pagina_deslogueo],
        "Datos": [cargar_datos, validar_datos, transformaciones],
        "Machine Learning": [entrenar_modelos, evaluar_modelos, recomendar_modelo],
        "Reportes & Dashboards": [reporte, dashboard]
    })
else:
    pg = st.navigation([pagina_logueo])
pg.run()
```

Esto permite una experiencia de usuario moderna, segura y fácil de mantener, alineada con las mejores prácticas de Streamlit.

## Objetivo de los archivos principales

| Archivo/Carpeta                        | Descripción                                                                                |
|----------------------------------------|--------------------------------------------------------------------------------------------|
| app.py                                 | Punto de entrada principal de la app Streamlit. Inicializa la navegación multipágina.      |
| pages/00_Logueo.py                     | Página de inicio de sesión de la aplicación.                                               |
| pages/01_Cargar_Datos.py               | Página para cargar datos desde CSV y mostrar vista previa.                                 |
| pages/02_Validar_Datos.py              | Página para validar estructura, tipos y calidad de los datos cargados.                     |
| pages/03_Transformaciones.py           | Página para aplicar transformaciones (normalización, imputación, etc.) a los datos.        |
| pages/04_Entrenar_Modelos.py           | Página para seleccionar, entrenar y comparar modelos de ML.                                |
| pages/05_Evaluar_Modelos.py            | Página para visualizar y comparar métricas de los modelos entrenados.                      |
| pages/06_Recomendar_Modelo.py          | Página que recomienda el mejor modelo según desempeño y permite su aprobación.             |
| pages/07_Reporte.py                    | Página para generar y descargar reportes en PDF o CSV.                                     |
| pages/08_Dashboard.py                  | Dashboard unificado con resumen de datasets, modelos, reportes y transformaciones.         |
| src/datos/cargador.py                  | Funciones para cargar datos desde CSV, validar y almacenar en SQLite.                      |
| src/datos/limpiador.py                 | Funciones para detectar y limpiar duplicados, valores nulos y problemas de calidad.        |
| src/datos/transformador.py             | Funciones para aplicar transformaciones y revertirlas si es necesario.                     |
| src/datos/mock_db.py                   | Inicializa la base de datos SQLite y crea información de ejemplo.                          |
| src/modelos/entrenador.py              | Lógica para entrenar modelos de ML y separar conjuntos de entrenamiento/prueba.            |
| src/modelos/evaluador.py               | Funciones para evaluar modelos y calcular métricas clave (accuracy, RMSE, F1, etc.).       |
| src/modelos/recomendador.py            | Algoritmo para recomendar el mejor modelo según los resultados obtenidos.                  |
| src/reportes/generador.py              | Generación de reportes PDF/CSV con resumen de análisis, transformaciones y modelos.        |
| src/seguridad/autenticador.py          | Control de roles y validación de permisos de usuario.                                      |
| src/audit/logger.py                    | Registro de logs de auditoría: carga, transformación, entrenamiento, exportación, etc.     |

## Uso de la base de datos SQLite

La aplicación utiliza SQLite como sistema de almacenamiento de datos principal. Esto proporciona:

1. **Almacenamiento estructurado**: Tablas relacionales para datasets, transformaciones, modelos y auditoría.
2. **Persistencia de datos**: Los datos cargados y procesados se mantienen entre sesiones.
3. **Auditoría completa**: Registro de todas las operaciones realizadas por los usuarios.
4. **Facilidad de despliegue**: No requiere configuración de servidores de base de datos adicionales.

- Desarrollar y testear sin depender de la infraestructura de la nube.
- Evitar costos y riesgos de modificar datos reales.
- Simular distintos escenarios y poblar datos de ejemplo fácilmente.
- Ejecutar pruebas automáticas y reproducibles.

En producción, la aplicación debe conectarse a Snowflake para acceder a los datos reales y cumplir con los requisitos de seguridad y auditoría.

### Configuración y uso de la base de datos mock

1. **Creación de la base de datos mock**

   Ejecuta el script de inicialización para crear la base de datos SQLite y las tablas de ejemplo:

   ```bash
   python src/datos/mock_db.py
   ```

   Esto generará un archivo `analitica_farma.db` en el directorio raíz o donde lo especifiques.

2. **Uso en la aplicación**

   La app puede ser configurada para usar la base de datos mock en modo desarrollo. Simplemente asegúrate de que el archivo `analitica_farma.db` exista y que la lógica de carga de datos apunte a este archivo cuando corresponda.

3. **Cambio a Snowflake en producción**

   Para producción, configura las variables de entorno y credenciales para conectar a Snowflake. La lógica de la app debe permitir seleccionar la fuente de datos (mock o real) según el entorno.

### Ejemplo de conexión a la base de datos mock en Python

```python
import sqlite3
conn = sqlite3.connect("analitica_farma.db")
cursor = conn.cursor()
cursor.execute("SELECT * FROM datasets")
resultados = cursor.fetchall()
conn.close()
```

### Consideraciones

- No ejecutes el script de inicialización cada vez que inicies la app, solo cuando necesites crear o reinicializar la base de datos.
- El script inserta automáticamente un usuario de ejemplo en la tabla 'usuarios' si está vacía, para facilitar pruebas y acceso inicial.
- Puedes modificar el script para agregar o poblar más tablas según tus necesidades de desarrollo.
- En producción, nunca uses la base de datos mock para análisis reales.
