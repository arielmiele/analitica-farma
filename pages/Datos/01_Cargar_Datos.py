import streamlit as st
import os
import sys
from datetime import datetime

# Agregar el directorio src al path para poder importar los módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Importar módulos de la aplicación
from src.audit.logger import setup_logger, log_operation
from src.datos.cargador import cargar_datos_desde_csv, validar_dataframe_csv

# Configurar el logger con el ID de usuario de la sesión
usuario_id = st.session_state.get("usuario_id", 1)
logger = setup_logger("carga_datos", id_usuario=usuario_id)

# Inicializar session_state para flujo de trabajo y almacenamiento de datos
# Solo inicializar claves realmente necesarias para el flujo de carga y navegación
if 'df' not in st.session_state:
    st.session_state.df = None
if 'filename' not in st.session_state:
    st.session_state.filename = None
if 'metadatos' not in st.session_state:
    st.session_state.metadatos = None
if 'upload_timestamp' not in st.session_state:
    st.session_state.upload_timestamp = None

# Título y descripción de la página
st.title("📊 1. Cargar Datos")

st.write("Esta página permite cargar los datos necesarios para el análisis industrial. Puedes subir un nuevo archivo CSV o, próximamente, seleccionar un conjunto de datos disponible en Snowflake.")

# Función para cargar y procesar el CSV con caché para mejor rendimiento
@st.cache_data(ttl=3600, show_spinner="Procesando archivo CSV...")
def procesar_archivo_csv(uploaded_file, **kwargs):
    """
    Procesa el archivo CSV cargado y devuelve un DataFrame y metadatos de validación
    
    Args:
        uploaded_file: El archivo CSV cargado por el usuario
        **kwargs: Parámetros adicionales para pd.read_csv
    
    Returns:
        pd.DataFrame: DataFrame con los datos procesados
        dict: Metadatos del proceso de carga
        list: Advertencias de validación
        dict: Metadatos de validación
    """
    try:
        df, metadatos = cargar_datos_desde_csv(uploaded_file, **kwargs)
        warnings, metadatos_validacion = validar_dataframe_csv(df)
        log_operation(logger, "CARGA", f"Archivo {uploaded_file.name} procesado: {df.shape[0]} filas, {df.shape[1]} columnas", success=True, id_usuario=usuario_id)
        return df, metadatos, warnings, metadatos_validacion
    except Exception as e:
        log_operation(logger, "CARGA", f"Error al procesar archivo {uploaded_file.name}: {str(e)}", success=False, id_usuario=usuario_id)
        st.error(f"Error al procesar el archivo: {str(e)}")
        return None, None, [], {}

# Estructura tipo notebook: secciones claras y separadores

st.header("1. Selección del origen de datos")
tipo_carga = st.selectbox(
    "Selecciona el origen de los datos:",
    options=["Selecciona una opción...", "Subir archivo CSV", "Seleccionar dataset de Snowflake (próximamente)"],
    index=0,
    help="Por ahora solo está disponible la carga de archivos CSV."
)
st.divider()

if tipo_carga == "Subir archivo CSV":
    st.header("2. Subir archivo CSV")
    uploaded_file = st.file_uploader("Selecciona un archivo CSV", type=["csv"], key="file_uploader")
    cargar_btn = st.button("Cargar Datos", use_container_width=True, disabled=uploaded_file is None)
    if cargar_btn and uploaded_file is not None:
        with st.spinner("Cargando y procesando el archivo..."):
            df, metadatos, warnings, metadatos_validacion = procesar_archivo_csv(uploaded_file)
            if df is not None:
                st.session_state.df = df
                st.session_state.metadatos = metadatos
                st.session_state.filename = uploaded_file.name
                st.session_state.upload_timestamp = datetime.now()
                st.session_state.warnings = warnings
                st.session_state.metadatos_validacion = metadatos_validacion
                # Inicializar y limpiar claves de predictores/target para consistencia del workflow
                st.session_state.variables_predictoras = []
                for key in ["predictores", "target", "variable_objetivo"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.success(f"Archivo '{uploaded_file.name}' cargado exitosamente y disponible en la sesión.")
                if warnings:
                    with st.expander("Advertencias de validación", expanded=True):
                        for warning in warnings:
                            st.warning(warning)
                else:
                    st.info("Validación básica completada. Los datos parecen correctos.")
                st.rerun()
    st.divider()

elif tipo_carga == "Seleccionar dataset de Snowflake (próximamente)":
    st.header("2. Selección de dataset en Snowflake")
    st.info("Esta funcionalidad estará disponible cuando se integre Snowflake.")
    st.write("Aquí podrás seleccionar un dataset empresarial validado desde Snowflake y cargarlo para su análisis. Próximamente se mostrarán los pasos y opciones de conexión.")
    st.divider()

# 3. Vista previa del dataset cargado
if st.session_state.get("df") is not None:
    df = st.session_state.df
    if df is not None and hasattr(df, 'shape') and hasattr(df, 'head'):
        st.header("3. Vista previa del dataset cargado")
        st.markdown("""
        Se muestran las primeras filas del dataset para inspección rápida. Para validar, analizar calidad y configurar, continúa al siguiente paso.
        """)
        st.write(f"**Archivo cargado:** `{st.session_state.filename}`")
        st.write(f"- **Filas:** {df.shape[0]}  |  **Columnas:** {df.shape[1]}")
        st.dataframe(df.head(10), use_container_width=True)
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Cargar un dataset diferente", use_container_width=True):
                st.session_state.df = None
                st.session_state.filename = None
                st.session_state.metadatos = None
                st.session_state.upload_timestamp = None
                st.session_state.variables_predictoras = []
                for key in ["predictores", "target", "variable_objetivo"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        with col2:
            if st.button("➡️ Validar Datos", use_container_width=True):
                st.switch_page("pages/Datos/02_Validar_Datos.py")
