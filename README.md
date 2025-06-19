# analitica-farma

Aplicación para analizar datos productivos en la industria farmacéutica y recomendar modelos de machine learning. Permite cargar datos, transformarlos, evaluar modelos y generar reportes. Desarrollada con Streamlit, Python y conexión a Snowflake.

## Estructura del Proyecto

```text
├── app.py                  # Punto de entrada principal de la app Streamlit
├── requirements.txt        # Dependencias del proyecto
├── README.md               # Este archivo
├── pages/                  # Páginas multipágina de Streamlit (cada funcionalidad principal)
│   ├── 01_Cargar_Datos.py
│   ├── 02_Validar_Datos.py
│   ├── 03_Transformaciones.py
│   ├── 04_Entrenar_Modelos.py
│   ├── 05_Evaluar_Modelos.py
│   ├── 06_Recomendar_Modelo.py
│   ├── 07_Reporte.py
│   └── 08_Dashboard.py
├── src/                    # Código fuente modularizado
│   ├── audit/              # Auditoría y logging
│   │   └── logger.py
│   ├── datos/              # Carga, limpieza y transformación de datos
│   │   ├── cargador.py
│   │   ├── limpiador.py
│   │   └── transformador.py
│   ├── modelos/            # Entrenamiento, evaluación y recomendación de modelos ML
│   │   ├── entrenador.py
│   │   ├── evaluador.py
│   │   └── recomendador.py
│   ├── reportes/           # Generación de reportes PDF/CSV
│   │   └── generador.py
│   └── seguridad/          # Autenticación y control de acceso
│       └── autenticador.py
```

- Las páginas en `pages/` definen la navegación principal de la app (multipágina).
- El código fuente en `src/` está organizado por dominio: datos, modelos, reportes, seguridad y auditoría.

Esta estructura facilita el mantenimiento, la escalabilidad y el cumplimiento de buenas prácticas para aplicaciones empresariales de análisis de datos.

## Objetivo de los archivos principales

| Archivo/Carpeta                        | Descripción                                                                                 |
|----------------------------------------|--------------------------------------------------------------------------------------------|
| app.py                                 | Punto de entrada principal de la app Streamlit. Inicializa la navegación multipágina.      |
| pages/01_Cargar_Datos.py               | Página para cargar datos desde CSV o Snowflake y mostrar vista previa.                     |
| pages/02_Validar_Datos.py              | Página para validar estructura, tipos y calidad de los datos cargados.                     |
| pages/03_Transformaciones.py           | Página para aplicar transformaciones (normalización, imputación, etc.) a los datos.        |
| pages/04_Entrenar_Modelos.py           | Página para seleccionar, entrenar y comparar modelos de ML.                                |
| pages/05_Evaluar_Modelos.py            | Página para visualizar y comparar métricas de los modelos entrenados.                      |
| pages/06_Recomendar_Modelo.py          | Página que recomienda el mejor modelo según desempeño y permite su aprobación.             |
| pages/07_Reporte.py                    | Página para generar y descargar reportes en PDF o CSV.                                     |
| pages/08_Dashboard.py                  | Dashboard unificado con resumen de datasets, modelos, reportes y transformaciones.         |
| src/datos/cargador.py                  | Funciones para cargar datos desde CSV o Snowflake, validando esquema y conexión.           |
| src/datos/limpiador.py                 | Funciones para detectar y limpiar duplicados, valores nulos y problemas de calidad.        |
| src/datos/transformador.py             | Funciones para aplicar transformaciones y revertirlas si es necesario.                     |
| src/modelos/entrenador.py              | Lógica para entrenar modelos de ML y separar conjuntos de entrenamiento/prueba.            |
| src/modelos/evaluador.py               | Funciones para evaluar modelos y calcular métricas clave (accuracy, RMSE, F1, etc.).       |
| src/modelos/recomendador.py            | Algoritmo para recomendar el mejor modelo según los resultados obtenidos.                  |
| src/reportes/generador.py              | Generación de reportes PDF/CSV con resumen de análisis, transformaciones y modelos.        |
| src/seguridad/autenticador.py          | Integración SSO, control de roles y validación de permisos de usuario.                     |
| src/audit/logger.py                    | Registro de logs de auditoría: carga, transformación, entrenamiento, exportación, etc.     |
