import streamlit as st
import os
import sys
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.audit.logger import setup_logger, log_audit
from src.datos.cargador import cargar_datos_desde_csv, validar_dataframe_csv
from src.database.datasets_db import guardar_dataset, listar_datasets, obtener_dataset_por_id, cargar_dataset_fisico, eliminar_dataset
from src.state.session_manager import SessionManager

# Centralizar obtención de usuario_id e id_sesion
usuario_id = st.session_state.get("usuario_id", 1)
id_sesion = SessionManager.obtener_estado("id_sesion", None)

# Configurar el logger con el ID de usuario de la sesión
logger = setup_logger("carga_datos", usuario=str(usuario_id))

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

st.write("Esta página permite cargar los datos necesarios para el análisis industrial. Podés subir un nuevo archivo CSV o seleccionar un dataset guardado localmente.")

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
        df, metadatos = cargar_datos_desde_csv(uploaded_file, id_sesion=id_sesion, **kwargs)
        warnings, metadatos_validacion = validar_dataframe_csv(df, id_sesion=id_sesion)
        log_audit(
            usuario=str(usuario_id),
            accion="CARGA_DATOS",
            entidad=uploaded_file.name,
            id_entidad="",
            detalles=f"Archivo {uploaded_file.name} procesado: {df.shape[0]} filas, {df.shape[1]} columnas",
            id_sesion=id_sesion
        )
        return df, metadatos, warnings, metadatos_validacion
    except Exception as e:
        log_audit(
            usuario=str(usuario_id),
            accion="ERROR_CARGA",
            entidad=uploaded_file.name,
            id_entidad="",
            detalles=f"Error al procesar archivo {uploaded_file.name}: {str(e)}",
            id_sesion=id_sesion
        )
        st.error(f"Error al procesar el archivo: {str(e)}")
        return None, None, [], {}

# Estructura tipo notebook: secciones claras y separadores

st.header("1. Selección del origen de datos")
tipo_carga = st.selectbox(
    "Selecciona el origen de los datos:",
    options=["Selecciona una opción...", "Subir archivo CSV", "Seleccionar dataset guardado localmente"],
    index=0,
    help="Puedes subir un archivo CSV o seleccionar un dataset guardado localmente."
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
                st.session_state.metodo_carga = 'csv'  # <-- Origen CSV
                # Inicializar y limpiar claves de predictores/target para consistencia del workflow
                st.session_state.variables_predictoras = []
                for key in ["predictores", "target", "variable_objetivo"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.success(f"Archivo '{uploaded_file.name}' cargado exitosamente y disponible en la sesión.")
                log_audit(
                    usuario=str(usuario_id),
                    accion="CARGA_EXITOSA",
                    entidad=uploaded_file.name,
                    id_entidad="",
                    detalles="Archivo cargado exitosamente en la sesión.",
                    id_sesion=id_sesion
                )
                if warnings:
                    with st.expander("Advertencias de validación", expanded=True):
                        for warning in warnings:
                            st.warning(warning)
                else:
                    st.info("Validación básica completada. Los datos parecen correctos.")
                st.rerun()
    st.divider()

elif tipo_carga == "Seleccionar dataset guardado localmente":
    # Sección de selección de datasets almacenados localmente
    st.header("2. Selección de dataset local")
    datasets = listar_datasets(id_usuario=usuario_id if isinstance(usuario_id, int) else None)
    if datasets:
        opciones = {f"{ds['nombre']} ({ds['id_dataset'][:8]}...)": ds['id_dataset'] for ds in datasets}
        seleccion = st.selectbox("Selecciona un dataset guardado:", options=list(opciones.keys()))
        if seleccion:
            if st.button("Cargar dataset seleccionado", use_container_width=True):
                dataset_id = opciones[seleccion]
                df = cargar_dataset_fisico(dataset_id)
                if df is not None:
                    st.session_state.df = df
                    st.session_state.filename = seleccion
                    st.session_state.upload_timestamp = datetime.now()
                    st.session_state.metodo_carga = 'existente'
                    st.success(f"Dataset '{seleccion}' cargado localmente.")
                    log_audit(
                        usuario=str(usuario_id),
                        accion="CARGA_LOCAL",
                        entidad=seleccion,
                        id_entidad=str(dataset_id),
                        detalles="Dataset cargado desde almacenamiento local.",
                        id_sesion=id_sesion
                    )
                    st.rerun()
                else:
                    st.error("No se pudo cargar el dataset. El archivo puede haberse eliminado.")
    else:
        st.info("No hay datasets guardados localmente para este usuario.")
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
        # Opción para guardar el dataset cargado localmente
        if tipo_carga == "Subir archivo CSV" and st.session_state.get("df") is not None:
            with st.expander("💾 Guardar dataset localmente", expanded=False):
                nombre_dataset = st.text_input("Nombre del dataset", value=st.session_state.get("filename", ""))
                descripcion_dataset = st.text_area("Descripción", value="")
                guardar_btn = st.button("Guardar localmente", use_container_width=True, disabled=not nombre_dataset)
                if guardar_btn:
                    id_usuario_creador = st.session_state.get("usuario_id", 1)
                    df = st.session_state.df
                    if df is not None:
                        ok = guardar_dataset(
                            nombre_dataset,
                            descripcion_dataset,
                            id_usuario_creador,
                            df,
                            id_sesion=id_sesion,
                            usuario=str(usuario_id)
                        )
                        if ok:
                            st.success("Dataset guardado exitosamente en almacenamiento local.")
                            log_audit(
                                usuario=str(usuario_id),
                                accion="GUARDAR_DATASET",
                                entidad=nombre_dataset,
                                id_entidad="",
                                detalles="Dataset guardado localmente.",
                                id_sesion=id_sesion
                            )
                        else:
                            st.error("Error al guardar el dataset localmente.")
                    else:
                        st.error("No hay datos cargados para guardar.")

        # Mostrar datasets disponibles localmente (listado y opción de eliminar)
        with st.expander("📚 Datasets guardados localmente", expanded=False):
            id_usuario_creador = st.session_state.get("usuario_id", None)
            datasets = listar_datasets(id_usuario=id_usuario_creador)
            if datasets:
                for ds in datasets:
                    st.write(f"- **{ds['nombre']}** — {ds.get('descripcion', '')} | Filas: {ds.get('filas', '?')} | Columnas: {ds.get('columnas', '?')}")
                    eliminar_btn = st.button(
                        f"🗑️ Eliminar '{ds['nombre']}'",
                        key=f"eliminar_{ds['id_dataset']}"
                    )
                    if eliminar_btn:
                        ok = eliminar_dataset(ds['id_dataset'])
                        if ok:
                            st.success(f"Dataset '{ds['nombre']}' eliminado.")
                            log_audit(
                                usuario=str(usuario_id),
                                accion="ELIMINAR_DATASET",
                                entidad=ds['nombre'],
                                id_entidad=str(ds['id_dataset']),
                                detalles="Dataset eliminado del almacenamiento local.",
                                id_sesion=id_sesion
                            )
                            st.rerun()
                        else:
                            st.error(f"No se pudo eliminar el dataset '{ds['nombre']}'.")
            else:
                st.info("No hay datasets guardados localmente para este usuario.")
    # Botones de acción al final de la página
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
