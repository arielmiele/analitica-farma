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
from src.state.session_manager import SessionManager

# Configurar el logger con el ID de usuario de la sesión
usuario_id = st.session_state.get("usuario_id", 1)
logger = setup_logger("carga_datos", id_usuario=usuario_id)

# Inicializar session_state para flujo de trabajo y almacenamiento de datos
if 'df' not in st.session_state:
    st.session_state.df = None
if 'filename' not in st.session_state:
    st.session_state.filename = None
if 'upload_timestamp' not in st.session_state:
    st.session_state.upload_timestamp = None
if 'paso_carga' not in st.session_state:
    # Inicializar en paso 0: selección del método de carga
    st.session_state.paso_carga = 0
if 'metodo_carga' not in st.session_state:
    # None = no seleccionado, "existente" = dataset existente, "nuevo" = nuevo CSV
    st.session_state.metodo_carga = None

# Título y descripción de la página
st.title("📊 Cargar Datos")

st.markdown("""
Esta página permite cargar los datos necesarios para el análisis industrial.
Puedes utilizar un conjunto de datos existente o subir un nuevo archivo CSV.
""")

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
        log_operation(logger, "CARGA", f"Archivo {uploaded_file.name} procesado: {df.shape[0]} filas, {df.shape[1]} columnas", 
                     success=True, id_usuario=usuario_id)
        
        return df, metadatos
    except Exception as e:
        log_operation(logger, "CARGA", f"Error al procesar archivo {uploaded_file.name}: {str(e)}", 
                     success=False, id_usuario=usuario_id)
        st.error(f"Error al procesar el archivo: {str(e)}")
        return None, None

# Función para mostrar información del dataset cargado
def mostrar_info_dataset():
    if st.session_state.df is not None:
        st.success(f"Dataset cargado: {st.session_state.filename}")
        
        # Mostrar fecha y hora de carga si existe
        if st.session_state.upload_timestamp:
            st.text(f"Cargado el: {st.session_state.upload_timestamp.strftime('%d/%m/%Y %H:%M:%S')}")
        
        # Información del DataFrame
        st.write("### Información del conjunto de datos")
        st.write(f"Filas: {st.session_state.df.shape[0]}, Columnas: {st.session_state.df.shape[1]}")
        
        if 'metadatos' in st.session_state and st.session_state.metadatos:
            with st.expander("Ver metadatos completos"):
                st.json(st.session_state.metadatos)
        
        # Estadísticas básicas
        with st.expander("Estadísticas descriptivas"):
            st.write(st.session_state.df.describe())
        
        # Información de las columnas como subtítulo
        st.write("#### Detalle de columnas")
        info_df = pd.DataFrame({
            'Columna': st.session_state.df.columns,
            'Tipo': [str(st.session_state.df[col].dtype) for col in st.session_state.df.columns],
            'Valores únicos': [st.session_state.df[col].nunique() for col in st.session_state.df.columns],
            'Valores nulos': [st.session_state.df[col].isna().sum() for col in st.session_state.df.columns],
            'Ejemplo': [str(st.session_state.df[col].iloc[0]) if not st.session_state.df[col].empty else "" for col in st.session_state.df.columns]
        })
        st.dataframe(info_df, use_container_width=True)
        
        # Vista previa de los datos
        st.write("### Vista previa de los datos")
        st.dataframe(st.session_state.df.head(10), use_container_width=True)
        
        # Separador para botones de navegación
        st.write("---")
        st.write("### Navegación")
          # Botones de navegación al mismo nivel
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Cargar un dataset diferente", use_container_width=True):
                st.session_state.paso_carga = 0
                st.session_state.metodo_carga = None
                st.session_state.df = None
                st.session_state.filename = None
                st.session_state.metadatos = None
                st.session_state.upload_timestamp = None
                st.rerun()
        with col2:
            if st.button("➡️ Continuar con configuración", use_container_width=True):
                # Redireccionar a la página de configuración
                st.session_state.next_page = "configuracion"
                st.info("Redirigiendo a la página de configuración de datos...")
                # Registrar acción de usuario para auditoría
                log_audit(usuario_id, "NAVEGACIÓN", "configuracion", 
                         f"Continuar con configuración de {st.session_state.filename}")
                st.switch_page("pages/datos/02_Configurar_Datos.py")

# PASO 0: Seleccionar método de carga
if st.session_state.paso_carga == 0:
    st.write("### Paso 1: Selecciona cómo quieres cargar los datos")
    
    # Crear un contenedor para las opciones de selección
    opciones_container = st.container()
    col1, col2 = opciones_container.columns(2)
    
    with col1:
        if st.button("📋 Cargar dataset existente", use_container_width=True):
            st.session_state.metodo_carga = "existente"
            st.session_state.paso_carga = 1
            st.rerun()
    
    with col2:
        if st.button("📤 Subir nuevo archivo CSV", use_container_width=True):
            st.session_state.metodo_carga = "nuevo"
            st.session_state.paso_carga = 1
            st.rerun()
      # Mostrar información adicional sobre cada opción
    st.write("---")
    st.write("### Información sobre las opciones de carga")
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        **Cargar dataset existente:**
        * Selecciona un conjunto de datos previamente cargado en la base de datos.
        * Ideal para continuar análisis previos o usar datos ya validados.
        * Acceso inmediato sin necesidad de procesamiento adicional.
        """)
    with col2:
        st.info("""
        **Subir nuevo archivo CSV:**
        * Carga un nuevo archivo CSV desde tu computadora.
        * Opciones para validar y configurar el formato de carga.
        * Posibilidad de guardar en la base de datos para uso futuro.
        """)

# PASO 1: Cargar datos según el método seleccionado
elif st.session_state.paso_carga == 1:
    # Mostrar botón para volver al paso anterior
    if st.button("⬅️ Volver a selección de método de carga"):
        st.session_state.paso_carga = 0
        st.session_state.metodo_carga = None
        st.rerun()
    
    # Opción 1: Cargar dataset existente
    if st.session_state.metodo_carga == "existente":
        st.write("### Paso 2: Selecciona un dataset existente")
        
        try:
            with st.spinner("Cargando datasets disponibles..."):
                df_datasets = listar_datasets_disponibles()
                
            if df_datasets.empty:
                st.warning("No hay datasets disponibles en la base de datos.")
                st.info("Por favor, vuelve atrás y selecciona la opción de subir un nuevo archivo CSV.")
            else:
                st.write("#### Datasets disponibles")
                # Mostrar una tabla con los datasets
                st.dataframe(
                    df_datasets[['id_dataset', 'nombre', 'origen', 'fecha_carga', 'usuario', 'filas', 'columnas']], 
                    use_container_width=True
                )
                
                # Permitir cargar un dataset existente
                col1, col2 = st.columns([3, 1])
                with col1:
                    dataset_seleccionado = st.selectbox(
                        "Seleccionar dataset para cargar",
                        options=df_datasets['id_dataset'].tolist(),
                        format_func=lambda x: f"{df_datasets[df_datasets['id_dataset'] == x]['nombre'].values[0]} (ID: {x})"
                    )
                
                with col2:
                    if st.button("Cargar dataset", use_container_width=True):
                        with st.spinner("Cargando dataset..."):
                            df, metadatos = obtener_dataset(id_dataset=dataset_seleccionado)
                            
                            if df is not None:
                                st.session_state.df = df
                                st.session_state.metadatos = metadatos
                                st.session_state.filename = metadatos['nombre']
                                st.session_state.upload_timestamp = datetime.now()
                                
                                st.success(f"Dataset '{metadatos['nombre']}' cargado exitosamente")
                                log_operation(logger, "CARGA", f"Dataset '{metadatos['nombre']}' cargado desde la base de datos", 
                                            id_usuario=usuario_id)
                                log_audit(usuario_id, "CARGA_DATASET", metadatos['nombre'], 
                                        f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
                                
                                # Avanzar al paso de visualización
                                st.session_state.paso_carga = 2
                                st.rerun()
                            else:
                                st.error(f"Error al cargar el dataset: {metadatos.get('error', 'Error desconocido')}")
        
        except Exception as e:
            st.error(f"Error al listar los datasets disponibles: {str(e)}")
            log_operation(logger, "ERROR", f"Error al listar datasets disponibles: {str(e)}", 
                         success=False, id_usuario=usuario_id)
    
    # Opción 2: Subir nuevo archivo CSV
    elif st.session_state.metodo_carga == "nuevo":
        st.write("### Paso 2: Sube un archivo CSV")
        
        # Crear un formulario para cargar el archivo
        with st.form(key="upload_form"):
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
        if submit_button and 'file_uploader' in st.session_state and st.session_state.file_uploader is not None:
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
                                },
                                id_usuario=usuario_id
                            )
                            
                            st.success(f"Datos guardados en la base de datos con ID: {dataset_id}")
                            log_operation(logger, "ALMACENAMIENTO", f"Datos guardados en la base de datos con ID: {dataset_id}", 
                                        id_usuario=usuario_id)
                        except Exception as e:
                            st.error(f"Error al guardar en la base de datos: {str(e)}")
                            log_operation(logger, "ALMACENAMIENTO", f"Error al guardar en la base de datos: {str(e)}", 
                                        success=False, id_usuario=usuario_id)
                    
                    # Si se seleccionó validar datos
                    if validar_datos:
                        if metadatos is not None and 'warnings' in metadatos and metadatos['warnings']:
                            with st.expander("Advertencias de validación", expanded=True):
                                for warning in metadatos['warnings']:
                                    st.warning(warning)
                        else:
                            st.info("Validación básica completada. Los datos parecen correctos.")
                        
                        log_operation(logger, "VALIDACIÓN", f"Validación básica completada para: {st.session_state.filename}", 
                                    id_usuario=usuario_id)                    
                    # Registrar acción de usuario para auditoría
                    log_audit(usuario_id, "CARGA_DATOS", st.session_state.filename, 
                            f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
                    
                    # Marcar etapa como completada en el estado de sesión usando SessionManager
                    SessionManager.update_progress("carga_datos", True)
                    
                    # Avanzar al paso de visualización
                    st.session_state.paso_carga = 2
                    st.rerun()

# PASO 2: Mostrar información del dataset cargado
elif st.session_state.paso_carga == 2:
    st.write("### Paso 3: Revisar y trabajar con el dataset")
    
    # Mostrar información del dataset
    mostrar_info_dataset()
