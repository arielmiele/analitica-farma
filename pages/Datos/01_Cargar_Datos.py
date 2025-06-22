import streamlit as st
import pandas as pd
import os
import sys
from datetime import datetime

# Agregar el directorio src al path para poder importar los módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Importar módulos de la aplicación
from src.audit.logger import setup_logger, log_operation, log_audit
from src.datos.cargador import (
    cargar_datos_desde_csv, 
    guardar_en_bd_local, 
    listar_datasets_disponibles,
    obtener_dataset
)

# Configurar el logger
logger = setup_logger("carga_datos")

st.title("📊 Cargar Datos")

st.markdown("""
Esta página permite cargar los datos necesarios para el análisis industrial.
Puedes subir archivos CSV y guardarlos en la base de datos local.
""")

# Inicializar session_state para guardar el DataFrame
if 'df' not in st.session_state:
    st.session_state.df = None
if 'filename' not in st.session_state:
    st.session_state.filename = None
if 'upload_timestamp' not in st.session_state:
    st.session_state.upload_timestamp = None
if 'mostrar_datasets' not in st.session_state:
    st.session_state.mostrar_datasets = False

# Función para cargar y procesar el CSV con caché para mejor rendimiento
@st.cache_data(ttl=3600, show_spinner="Procesando archivo CSV...")
def procesar_archivo_csv(uploaded_file, validar=True, **kwargs):
    """
    Procesa el archivo CSV cargado y devuelve un DataFrame
    
    Args:
        uploaded_file: El archivo CSV cargado por el usuario
        validar: Si se debe validar el archivo
        **kwargs: Parámetros adicionales para pd.read_csv
    
    Returns:
        pd.DataFrame: DataFrame con los datos procesados
        dict: Metadatos del proceso de carga
    """
    try:
        # Utilizar la función del módulo cargador
        df, metadatos = cargar_datos_desde_csv(uploaded_file, validar=validar, **kwargs)
        
        # Registrar la operación exitosa
        log_operation(logger, "CARGA", f"Archivo {uploaded_file.name} procesado: {df.shape[0]} filas, {df.shape[1]} columnas")
        
        return df, metadatos
    except Exception as e:
        log_operation(logger, "CARGA", f"Error al procesar archivo {uploaded_file.name}: {str(e)}", success=False)
        st.error(f"Error al procesar el archivo: {str(e)}")
        return None, None

# Mostrar datasets disponibles si el usuario lo solicita
if st.checkbox("Mostrar datasets disponibles", value=st.session_state.mostrar_datasets):
    st.session_state.mostrar_datasets = True
    
    try:
        with st.spinner("Cargando datasets disponibles..."):
            df_datasets = listar_datasets_disponibles()
            
        if df_datasets.empty:
            st.info("No hay datasets disponibles en la base de datos.")
        else:
            st.write("### Datasets disponibles")
            # Mostrar una tabla con los datasets
            st.dataframe(
                df_datasets[['id_dataset', 'nombre', 'origen', 'fecha_carga', 'usuario', 'filas', 'columnas']], 
                use_container_width=True
            )
            
            # Permitir cargar un dataset existente
            col1, col2 = st.columns(2)
            with col1:
                dataset_seleccionado = st.selectbox(
                    "Seleccionar dataset para cargar",
                    options=df_datasets['id_dataset'].tolist(),
                    format_func=lambda x: f"{df_datasets[df_datasets['id_dataset'] == x]['nombre'].values[0]} (ID: {x})"
                )
            
            with col2:
                if st.button("Cargar dataset seleccionado"):
                    with st.spinner("Cargando dataset..."):
                        df, metadatos = obtener_dataset(id_dataset=dataset_seleccionado)
                        
                        if df is not None:
                            st.session_state.df = df
                            st.session_state.metadatos = metadatos
                            st.session_state.filename = metadatos['nombre']
                            st.session_state.upload_timestamp = datetime.now()
                            
                            st.success(f"Dataset '{metadatos['nombre']}' cargado exitosamente")
                            log_operation(logger, "CARGA", f"Dataset '{metadatos['nombre']}' cargado desde la base de datos")
                            log_audit("usuario_actual", "CARGA_DATASET", metadatos['nombre'], 
                                     f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
                            
                            # Refrescar la página para mostrar el dataset
                            st.rerun()
                        else:
                            st.error(f"Error al cargar el dataset: {metadatos.get('error', 'Error desconocido')}")
    
    except Exception as e:
        st.error(f"Error al listar los datasets disponibles: {str(e)}")
else:
    st.session_state.mostrar_datasets = False

# Crear un formulario para cargar el archivo
with st.form(key="upload_form"):
    st.write("### Seleccione un archivo CSV para cargar")
    
    # Selector de archivo
    uploaded_file = st.file_uploader("Seleccione un archivo CSV", type=["csv"], 
                                     accept_multiple_files=False, key="file_uploader")
    
    # Opciones adicionales de carga
    col1, col2 = st.columns(2)
    with col1:
        guardar_en_bd = st.checkbox("Guardar en base de datos", value=True, 
                                   help="Almacena los datos en la base de datos para su uso posterior")
    with col2:
        validar_datos = st.checkbox("Validar datos automáticamente", value=True,
                                   help="Realiza una validación básica del formato de los datos")
    
    # Opciones avanzadas en un expander
    with st.expander("Opciones avanzadas de carga"):
        delimiter = st.text_input("Delimitador", ",", help="Carácter utilizado para separar los campos en el CSV")
        encoding = st.selectbox("Codificación", ["utf-8", "latin1", "iso-8859-1", "cp1252"], 
                               help="Codificación del archivo CSV")
        skip_rows = st.number_input("Saltar filas iniciales", 0, 100, 0, 
                                   help="Número de filas iniciales a omitir")
    
    # Nombre de dataset personalizado
    dataset_nombre = st.text_input("Nombre del dataset (opcional)", 
                                  help="Nombre personalizado para el dataset en la base de datos")
    
    # Botón para enviar el formulario
    submit_button = st.form_submit_button(label="Cargar Datos")

# Procesar el archivo cuando se envía el formulario
if submit_button and st.session_state.file_uploader is not None:
    with st.spinner("Cargando y procesando el archivo..."):
        # Procesar el archivo usando la función en caché
        df, metadatos = procesar_archivo_csv(
            st.session_state.file_uploader,
            validar=validar_datos,
            delimiter=delimiter,
            encoding=encoding,
            skiprows=skip_rows if skip_rows > 0 else None
        )
        
        if df is not None:
            # Guardar en session_state
            st.session_state.df = df
            st.session_state.metadatos = metadatos
            st.session_state.filename = st.session_state.file_uploader.name
            st.session_state.upload_timestamp = datetime.now()
            
            # Si se seleccionó guardar en base de datos
            if guardar_en_bd:
                try:
                    # Usar el nombre personalizado o generar uno basado en la fecha
                    if dataset_nombre and dataset_nombre.strip():
                        tabla_nombre = dataset_nombre.strip()
                    else:
                        # Crear un nombre de tabla basado en el nombre del archivo sin extensión
                        nombre_archivo_sin_ext = os.path.splitext(st.session_state.filename)[0]
                        tabla_nombre = f"{nombre_archivo_sin_ext}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    
                    # Usar la función del módulo cargador
                    dataset_id = guardar_en_bd_local(
                        df, 
                        tabla_nombre, 
                        metadatos if metadatos else {
                            'nombre_archivo': st.session_state.filename,
                            'fecha_carga': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'filas': df.shape[0],
                            'columnas': df.shape[1],
                            'origen': 'csv'
                        }
                    )
                    
                    st.success(f"Datos guardados en la base de datos con ID: {dataset_id}")
                    log_operation(logger, "ALMACENAMIENTO", f"Datos guardados en la base de datos con ID: {dataset_id}")
                except Exception as e:
                    st.error(f"Error al guardar en la base de datos: {str(e)}")
                    log_operation(logger, "ALMACENAMIENTO", f"Error al guardar en la base de datos: {str(e)}", success=False)
              # Si se seleccionó validar datos
            if validar_datos:
                if metadatos is not None and 'warnings' in metadatos and metadatos['warnings']:
                    with st.expander("Advertencias de validación"):
                        for warning in metadatos['warnings']:
                            st.warning(warning)
                else:
                    st.info("Validación básica completada. Los datos parecen correctos.")
                
                log_operation(logger, "VALIDACIÓN", f"Validación básica completada para: {st.session_state.filename}")
                    
            # Registrar acción de usuario para auditoría
            log_audit("usuario_actual", "CARGA_DATOS", st.session_state.filename, 
                     f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")

# Mostrar información del archivo cargado si existe
if st.session_state.df is not None:
    st.success(f"Archivo cargado: {st.session_state.filename}")
    
    # Mostrar fecha y hora de carga si existe
    if st.session_state.upload_timestamp:
        st.text(f"Cargado el: {st.session_state.upload_timestamp.strftime('%d/%m/%Y %H:%M:%S')}")
    
    # Información del DataFrame
    st.write("### Información del conjunto de datos")
    st.write(f"Filas: {st.session_state.df.shape[0]}, Columnas: {st.session_state.df.shape[1]}")
    
    if 'metadatos' in st.session_state and st.session_state.metadatos:
        with st.expander("Ver metadatos completos"):
            st.json(st.session_state.metadatos)
    
    # Vista previa de los datos
    st.write("### Vista previa de los datos")
    st.dataframe(st.session_state.df.head(10), use_container_width=True)
    
    # Estadísticas básicas
    st.write("### Estadísticas descriptivas")
    st.write(st.session_state.df.describe())
    
    # Información de las columnas
    st.write("### Información de las columnas")
    info_df = pd.DataFrame({
        'Columna': st.session_state.df.columns,
        'Tipo': [str(st.session_state.df[col].dtype) for col in st.session_state.df.columns],
        'Valores únicos': [st.session_state.df[col].nunique() for col in st.session_state.df.columns],
        'Valores nulos': [st.session_state.df[col].isna().sum() for col in st.session_state.df.columns],
        'Ejemplo': [str(st.session_state.df[col].iloc[0]) if not st.session_state.df[col].empty else "" for col in st.session_state.df.columns]
    })
    st.dataframe(info_df, use_container_width=True)
    
    # Botones para acciones adicionales
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Ver datos completos"):
            st.session_state.view_full = True
    with col2:
        if st.button("Descargar CSV procesado"):
            # Lógica para descargar el CSV procesado
            csv = st.session_state.df.to_csv(index=False)
            st.download_button(
                label="Descargar CSV",
                data=csv,
                file_name=f"procesado_{st.session_state.filename}",
                mime="text/csv"
            )
    with col3:
        if st.button("Continuar con validación"):
            # Redireccionar a la página de validación
            st.session_state.next_page = "validacion"
            st.info("Redirigiendo a la página de validación...")
            
            # Registrar acción de usuario para auditoría
            log_audit("usuario_actual", "NAVEGACIÓN", "validacion", 
                     f"Continuar con validación de {st.session_state.filename}")
    
    # Mostrar el DataFrame completo si se seleccionó
    if 'view_full' in st.session_state and st.session_state.view_full:
        st.write("### Datos completos")
        st.dataframe(st.session_state.df, use_container_width=True)
        if st.button("Ocultar datos completos"):
            st.session_state.view_full = False
