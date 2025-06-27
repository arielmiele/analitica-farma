"""
Página para entrenar y comparar múltiples modelos de machine learning.
"""
import streamlit as st
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Importar módulos propios
from src.modelos.entrenador import ejecutar_benchmarking, obtener_ultimo_benchmarking
from src.modelos.configurador import guardar_configuracion_modelo
from src.state.session_manager import SessionManager
from src.audit.logger import Logger

# Inicializar gestor de sesión y logger
session = SessionManager()
logger = Logger("Entrenar_Modelos")

# Función con caché para ejecutar el benchmarking sin repetirse al recargar la página
@st.cache_data(show_spinner=False, ttl=3600)  # Cache válida por 1 hora
def ejecutar_benchmarking_cached(X, y, tipo_problema, test_size, trigger_key=None):
    """
    Versión en caché de la función de benchmarking para evitar re-entrenamientos automáticos.
    
    Args:
        X: DataFrame con variables predictoras
        y: Series con variable objetivo
        tipo_problema: Tipo de problema ('clasificacion' o 'regresion')
        test_size: Tamaño del conjunto de prueba
        trigger_key: Parámetro que al cambiar invalida la caché (para control manual)
        
    Returns:
        dict: Resultados del benchmarking
    """
    return ejecutar_benchmarking(X, y, tipo_problema=tipo_problema, test_size=test_size)

def main():
    st.title("🤖 Evaluación de Modelos de Machine Learning")
    
    # Verificar si hay resultados de benchmarking y mostrar estadísticas
    if session.obtener_estado("resultados_benchmarking"):
        # Obtener estadísticas de benchmarking
        stats = session.get_benchmarking_stats()
        
        # Añadir información sobre las estadísticas de entrenamiento
        with st.expander("📈 Estadísticas de entrenamiento", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Total de entrenamientos:** {stats['total_entrenamientos']}")
                st.write(f"**Último entrenamiento:** {stats['ultimo_entrenamiento'] or 'No disponible'}")
            with col2:
                st.write(f"**Contador de trigger:** {stats['trigger_count']}")
                if stats['ultimo_benchmarking_id']:
                    st.write(f"**ID del último benchmarking:** {stats['ultimo_benchmarking_id']}")
            
            st.info("""
            Nota sobre la caché: Streamlit guarda en caché los resultados del entrenamiento para evitar cálculos
            repetidos al recargar la página. Si necesita forzar un nuevo entrenamiento, use el botón 
            "Forzar nuevo entrenamiento" que aparece debajo.
            """)
    # Inicializar tipo_problema con valor predeterminado o cargarlo de la sesión si existe
    tipo_problema = session.obtener_estado("tipo_problema", "clasificacion")
    
    # Verificar si hay datos cargados
    if not session.is_dataset_loaded():
        st.warning("⚠️ No hay datos cargados. Por favor, cargue un conjunto de datos primero.")
        st.info("👈 Vaya a la sección 'Cargar Datos' en el menú lateral.")
        return
    
    # Obtener el dataframe actual
    df = session.cargar_dataframe()
    
    # Sección de explicación
    with st.expander("ℹ️ Acerca de esta página", expanded=False):
        st.markdown("""
        ### Evaluación de Modelos
        
        Esta página le permite ejecutar un benchmarking automático de múltiples algoritmos de Machine Learning
        para encontrar el modelo que mejor se ajusta a sus datos.
        
        **Beneficios:**
        - Entrena y compara diversos modelos con un solo clic
        - Muestra métricas de rendimiento comparativas
        - Identifica el mejor modelo según el tipo de problema
        - Continúa con el proceso incluso si algunos modelos fallan
        
        **Requisitos previos:**
        - Haber cargado un conjunto de datos
        - Haber definido una variable objetivo
        - Haber limpiado y preparado los datos adecuadamente
        """)
    
    # Verificar variables predictoras y objetivo
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Información del dataset")
        if df is not None:
            st.write(f"**Filas:** {df.shape[0]}")
            st.write(f"**Columnas:** {df.shape[1]}")
            
            # Mostrar las primeras filas
            if st.checkbox("Ver primeras filas del dataset", value=False):
                st.dataframe(df.head())
        else:
            st.warning("No se pudo cargar el dataset correctamente.")
    
    with col2:
        # Selección de variable objetivo
        st.subheader("🎯 Configuración del modelo")
        
        # Cargar variable objetivo de la sesión si existe
        var_objetivo = session.obtener_estado("variable_objetivo", None)
        
        # Permitir al usuario seleccionar o cambiar la variable objetivo
        if df is not None:
            columnas = df.columns.tolist()
            var_objetivo = st.selectbox(
                "Seleccione la variable objetivo (target):",
                options=columnas,
                index=columnas.index(var_objetivo) if var_objetivo in columnas else 0
            )
            # Guardar variable objetivo en la sesión
            session.guardar_estado("variable_objetivo", var_objetivo)
        else:
            st.warning("No se pudo cargar el dataset correctamente para seleccionar la variable objetivo.")
        
        # Detectar automáticamente el tipo de problema
        if var_objetivo and df is not None:
            y = df[var_objetivo]
            
            # Reglas simples para determinar si es clasificación o regresión
            if y.dtype == 'object' or y.dtype == 'category' or y.dtype == 'bool':
                tipo_problema = "clasificacion"
            else:
                # Si tiene pocos valores únicos en relación al total, probablemente es clasificación
                unique_ratio = len(y.unique()) / len(y)
                if unique_ratio < 0.05 or len(y.unique()) < 10:
                    tipo_problema = "clasificacion"
                else:
                    tipo_problema = "regresion"
            
            # Permitir al usuario cambiar el tipo de problema detectado
            tipo_problema = st.radio(
                "Tipo de problema detectado:",
                options=["clasificacion", "regresion"],
                index=0 if tipo_problema == "clasificacion" else 1
            )
            
            # Guardar tipo de problema en la sesión
            session.guardar_estado("tipo_problema", tipo_problema)
            
            # Mostrar información adicional según el tipo de problema
            if tipo_problema == "clasificacion":
                st.write(f"**Clases únicas:** {len(y.unique())}")
                
                # Mostrar distribución de clases
                if st.checkbox("Ver distribución de clases", value=False):
                    fig, ax = plt.subplots(figsize=(8, 4))
                    y.value_counts().plot(kind='bar', ax=ax)
                    plt.title(f"Distribución de clases - {var_objetivo}")
                    plt.xlabel("Clase")
                    plt.ylabel("Conteo")
                    st.pyplot(fig)
            else:
                st.write(f"**Rango de valores:** {y.min()} a {y.max()}")
                
                # Mostrar histograma
                if st.checkbox("Ver distribución de valores", value=False):
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.histplot(y, kde=True, ax=ax)
                    plt.title(f"Distribución de valores - {var_objetivo}")
                    plt.xlabel(var_objetivo)
                    plt.ylabel("Frecuencia")
                    st.pyplot(fig)
        elif var_objetivo and df is None:
            st.warning("No se pudo cargar el dataset correctamente para analizar la variable objetivo.")    # Sección para ejecutar el benchmarking
    st.subheader("🚀 Ejecutar benchmarking de modelos")
    
    # Información sobre el comportamiento de la caché
    st.info("""
        📌 **Nota importante:** El entrenamiento de modelos solo se ejecutará cuando presione el botón "Iniciar evaluación".
        Si recarga la página, se mostrarán los resultados del último entrenamiento sin volver a ejecutar el proceso.
    """)
    
    # Opciones adicionales para el benchmarking
    with st.expander("Opciones avanzadas"):
        test_size = st.slider(
            "Porcentaje de datos para prueba (test):",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
            format="%.2f"
        )
    
    # Botón para ejecutar benchmarking
    if st.button("🔍 Iniciar evaluación de modelos", type="primary"):
        if not var_objetivo:
            st.error("⚠️ Debe seleccionar una variable objetivo.")
            return
            
        # Verificar que df no es None antes de proceder
        if df is None:
            st.error("⚠️ No hay datos cargados. Por favor, cargue un conjunto de datos primero.")
            return
        
        # Incrementar el trigger para forzar un nuevo entrenamiento
        session.incrementar_trigger_benchmarking()
        
        # Guardar configuración del modelo
        config = {
            "tipo_problema": tipo_problema,
            "variable_objetivo": var_objetivo,
            "variables_predictoras": [col for col in df.columns if col != var_objetivo],
            "test_size": test_size,
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Preparar datos para el benchmarking
        X = df.drop(columns=[var_objetivo])
        y = df[var_objetivo]
        
        # Pre-procesar columnas de fecha para evitar errores
        columnas_a_procesar = []
        for columna in X.columns:
            # Verificar si es una columna de fecha o si contiene valores de fecha en formato string
            if X[columna].dtype == 'datetime64[ns]':
                columnas_a_procesar.append(columna)
            elif X[columna].dtype == 'object':
                # Intentar detectar si parece contener fechas
                muestra = X[columna].dropna().astype(str).iloc[:5]
                patron_fecha = r'\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4}|\d{1,2}[-/]\d{1,2}[-/]\d{2}'
                if any(muestra.str.match(patron_fecha)):
                    columnas_a_procesar.append(columna)
        
        # Guardar los datos iniciales en la sesión para usarlos después
        session.guardar_estado("datos_pre_entrenamiento", {
            "X": X, 
            "y": y,
            "tipo_problema": tipo_problema,
            "test_size": test_size,
            "var_objetivo": var_objetivo,
            "config": config,
            "columnas_fecha": columnas_a_procesar
        })
        
        # Si hay columnas de fecha, mostrar advertencia y opciones, y detener el flujo
        if columnas_a_procesar:
            st.warning(f"Se detectaron posibles columnas de fecha: {', '.join(columnas_a_procesar)}")
            st.info("Antes de continuar con el entrenamiento, debe decidir cómo manejar estas columnas.")
            
            # Guardar el estado para saber que estamos en este paso
            session.guardar_estado("paso_entrenamiento", "configurar_fechas")
            st.rerun()  # Recargar la página para mostrar las opciones de fechas
        else:
            # Si no hay columnas de fecha, continuar con el entrenamiento directamente
            session.guardar_estado("paso_entrenamiento", "ejecutar_entrenamiento")
            st.rerun()  # Recargar la página para iniciar el entrenamiento
            
    # Verificar en qué paso del entrenamiento estamos
    paso_entrenamiento = session.obtener_estado("paso_entrenamiento", None)
    datos_pre_entrenamiento = session.obtener_estado("datos_pre_entrenamiento", None)
    
    # Si estamos en el paso de configurar fechas, mostrar opciones
    if paso_entrenamiento == "configurar_fechas" and datos_pre_entrenamiento:
        X = datos_pre_entrenamiento["X"]
        columnas_a_procesar = datos_pre_entrenamiento["columnas_fecha"]
        
        st.subheader("🗓️ Configuración de columnas de fecha")
        st.write("Seleccione cómo desea manejar las columnas de fecha detectadas antes de continuar con el entrenamiento.")
        
        opcion_fecha = st.radio(
            "¿Cómo desea manejar las columnas de fecha?",
            ["Convertir automáticamente", "Eliminar columnas de fecha", "Mantener como están (podría causar errores)"],
            index=0
        )
        
        # Mostrar vista previa del resultado según la opción seleccionada
        X_procesado = X.copy()
        
        if opcion_fecha == "Convertir automáticamente":
            for columna in columnas_a_procesar:
                if X_procesado[columna].dtype == 'datetime64[ns]':
                    # Para columnas ya en formato datetime
                    X_procesado[f"{columna}_año"] = X_procesado[columna].dt.year
                    X_procesado[f"{columna}_mes"] = X_procesado[columna].dt.month
                    X_procesado[f"{columna}_dia"] = X_procesado[columna].dt.day
                    X_procesado = X_procesado.drop(columns=[columna])
                else:
                    # Para columnas en formato string
                    try:
                        fechas = pd.to_datetime(X_procesado[columna], errors='coerce')
                        X_procesado[f"{columna}_año"] = fechas.dt.year
                        X_procesado[f"{columna}_mes"] = fechas.dt.month
                        X_procesado[f"{columna}_dia"] = fechas.dt.day
                        X_procesado = X_procesado.drop(columns=[columna])
                    except Exception:
                        pass  # Ignorar errores en la vista previa
        elif opcion_fecha == "Eliminar columnas de fecha":
            X_procesado = X_procesado.drop(columns=columnas_a_procesar)
        
        # Mostrar vista previa de los datos procesados
        with st.expander("👁️ Vista previa de datos procesados", expanded=False):
            st.dataframe(X_procesado.head())
            st.write(f"Columnas originales: {X.shape[1]}")
            st.write(f"Columnas después del procesamiento: {X_procesado.shape[1]}")
        
        # Botón para confirmar y continuar
        if st.button("✅ Confirmar y continuar con el entrenamiento", type="primary"):
            # Guardar la opción seleccionada
            datos_pre_entrenamiento["opcion_fecha"] = opcion_fecha
            session.guardar_estado("datos_pre_entrenamiento", datos_pre_entrenamiento)
            
            # Cambiar al paso de entrenamiento
            session.guardar_estado("paso_entrenamiento", "ejecutar_entrenamiento")
            st.rerun()  # Recargar la página para iniciar el entrenamiento
    
    # Si estamos en el paso de ejecutar el entrenamiento, proceder con el benchmarking
    elif paso_entrenamiento == "ejecutar_entrenamiento" and datos_pre_entrenamiento:
        try:
            # Recuperar datos
            X = datos_pre_entrenamiento["X"]
            y = datos_pre_entrenamiento["y"]
            tipo_problema = datos_pre_entrenamiento["tipo_problema"]
            test_size = datos_pre_entrenamiento["test_size"]
            var_objetivo = datos_pre_entrenamiento["var_objetivo"]
            config = datos_pre_entrenamiento["config"]
            columnas_a_procesar = datos_pre_entrenamiento.get("columnas_fecha", [])
            opcion_fecha = datos_pre_entrenamiento.get("opcion_fecha", "Mantener como están (podría causar errores)")
            
            # Procesar columnas de fecha según la opción seleccionada
            if columnas_a_procesar:
                if opcion_fecha == "Convertir automáticamente":
                    for columna in columnas_a_procesar:
                        if X[columna].dtype == 'datetime64[ns]':
                            # Para columnas ya en formato datetime
                            X[f"{columna}_año"] = X[columna].dt.year
                            X[f"{columna}_mes"] = X[columna].dt.month
                            X[f"{columna}_dia"] = X[columna].dt.day
                            X = X.drop(columns=[columna])
                            st.info(f"Columna {columna} convertida a componentes numéricos: año, mes, día")
                        else:
                            # Para columnas en formato string
                            try:
                                fechas = pd.to_datetime(X[columna], errors='coerce')
                                X[f"{columna}_año"] = fechas.dt.year
                                X[f"{columna}_mes"] = fechas.dt.month
                                X[f"{columna}_dia"] = fechas.dt.day
                                X = X.drop(columns=[columna])
                                st.info(f"Columna {columna} convertida a componentes numéricos: año, mes, día")
                            except Exception as e:
                                st.error(f"No se pudo convertir la columna {columna}: {str(e)}")
                elif opcion_fecha == "Eliminar columnas de fecha":
                    X = X.drop(columns=columnas_a_procesar)
                    st.info(f"Columnas eliminadas: {', '.join(columnas_a_procesar)}")
            
            # Guardar la configuración en la base de datos y obtener su ID
            guardar_configuracion_modelo(config)
            
            # Registrar en el log de auditoría
            logger.log_evento(
                "INICIAR_BENCHMARKING",
                f"Iniciando benchmarking de modelos para {var_objetivo}",
                "Entrenar_Modelos"
            )
            
            # Iniciar barra de progreso
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Mensaje de proceso
            status_text.text("Iniciando benchmarking de modelos...")
            
            # Ejecutar benchmarking con barra de progreso real
            with st.spinner("Entrenando y evaluando múltiples modelos..."):
                # Determinar total de modelos a entrenar
                if tipo_problema == "clasificacion":
                    from src.modelos.entrenador import CLASSIFICATION_MODELS
                    total_modelos = len(CLASSIFICATION_MODELS)
                else:
                    from src.modelos.entrenador import REGRESSION_MODELS
                    total_modelos = len(REGRESSION_MODELS)
                
                # Fases del benchmarking
                fases = [
                    (0.05, "Preparando datos..."),
                    (0.10, "Dividiendo conjuntos de entrenamiento y prueba..."),
                    (0.15, "Escalando características..."),
                    (0.20, "Iniciando entrenamiento de modelos...")
                ]
                
                # Mostrar progreso de fases iniciales
                for progreso, mensaje in fases:
                    progress_bar.progress(int(progreso * 100))
                    status_text.text(mensaje)
                    time.sleep(0.2)  # Breve pausa para visualizar progreso
                
                # Simular progreso durante entrenamiento de modelos
                # Reservamos 80% del progreso para entrenar modelos (del 20% al 100%)
                for i in range(20, 95, 5):
                    current_model = int((i - 20) / 75 * total_modelos) + 1
                    progress_bar.progress(i)
                    status_text.text(f"Entrenando modelos... ({current_model}/{total_modelos})")
                    time.sleep(0.1)  # Breve pausa para visualizar progreso                
                # Ejecutar el benchmarking real
                resultados = ejecutar_benchmarking_cached(
                    X, y, 
                    tipo_problema=tipo_problema,
                    test_size=test_size,
                    trigger_key=session.obtener_trigger_benchmarking()
                )
                
                # Completar la barra de progreso
                progress_bar.progress(95)
                status_text.text("Guardando resultados en la base de datos...")
                time.sleep(0.3)
                
                progress_bar.progress(100)
                status_text.text("¡Benchmarking completado con éxito!")
            
            # Guardar resultados en la sesión
            session.guardar_estado("resultados_benchmarking", resultados)
            
            # Registrar el entrenamiento en las estadísticas
            session.registrar_entrenamiento()
            
            # Notificar éxito y redirigir
            st.success("✅ Evaluación de modelos completada con éxito. Vea los resultados a continuación.")
            
            # Registrar en el log de auditoría
            logger.log_evento(
                "BENCHMARKING_COMPLETADO",
                f"Benchmarking completado con {len(resultados['modelos_exitosos'])} modelos exitosos",
                "Entrenar_Modelos"
            )
            
            # Mostrar resumen de resultados
            mostrar_resumen_resultados(resultados)
            
        except Exception as e:
            st.error(f"❌ Error al ejecutar el benchmarking: {str(e)}")
            logger.log_evento(
                "ERROR_BENCHMARKING",
                f"Error al ejecutar benchmarking: {str(e)}",
                "Entrenar_Modelos",
                tipo="error"
            )    # Cargar y mostrar resultados previos si existen
    elif session.obtener_estado("resultados_benchmarking"):
        # Mensaje informativo con marco más destacado
        st.success("🔍 Mostrando resultados del último benchmarking ejecutado.")
        st.markdown("---")
        
        # Botón para forzar un nuevo entrenamiento si ya hay resultados previos
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("🔄 Forzar nuevo entrenamiento", type="secondary", use_container_width=True):
                # Incrementar el trigger para invalidar la caché
                session.incrementar_trigger_benchmarking()
                # Limpiar el estado actual para forzar un nuevo entrenamiento completo
                session.guardar_estado("paso_entrenamiento", None)
                session.guardar_estado("datos_pre_entrenamiento", None)
                st.rerun()
        with col2:
            # Agregar instrucciones claras para el usuario
            st.caption("Use este botón si desea volver a entrenar todos los modelos con la configuración actual.")
            
        # Mostrar los resultados
        mostrar_resumen_resultados(session.obtener_estado("resultados_benchmarking"))
    
    # Si no hay resultados en la sesión, intentar cargar de la base de datos
    else:
        try:
            ultimo_benchmarking = obtener_ultimo_benchmarking()
            if ultimo_benchmarking:
                st.info("🔍 Mostrando resultados del último benchmarking guardado en la base de datos.")
                
                # Guardar en la sesión para futuros accesos
                session.guardar_estado("resultados_benchmarking", ultimo_benchmarking)
                
                # Mostrar resumen
                mostrar_resumen_resultados(ultimo_benchmarking)
        except Exception:
            st.info("ℹ️ Aún no se ha ejecutado ningún benchmarking. Presione el botón para iniciar la evaluación.")

def mostrar_resumen_resultados(resultados):
    """
    Muestra un resumen de los resultados del benchmarking.
    
    Args:
        resultados: Diccionario con los resultados del benchmarking
    """
    if not resultados:
        st.warning("No hay resultados disponibles.")
        return
    
    st.subheader("📊 Resumen de resultados")
    
    # Información general
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total de modelos evaluados", 
            f"{resultados['total_modelos']}"
        )
    
    with col2:
        st.metric(
            "Modelos exitosos", 
            f"{len(resultados['modelos_exitosos'])}"
        )
    
    with col3:
        st.metric(
            "Modelos fallidos", 
            f"{len(resultados['modelos_fallidos'])}"
        )
    
    # Mejor modelo
    if resultados['mejor_modelo']:
        st.subheader("🏆 Mejor modelo")
        
        mejor = resultados['mejor_modelo']
        
        # Información del mejor modelo
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Modelo:** {mejor['nombre']}")
            st.write(f"**Tiempo de entrenamiento:** {mejor['tiempo_entrenamiento']:.4f} segundos")
        
        with col2:
            # Mostrar métrica principal según tipo de problema
            if resultados['tipo_problema'] == 'clasificacion':
                st.metric("Accuracy", f"{mejor['metricas']['accuracy']:.4f}")
                st.metric("F1-Score", f"{mejor['metricas']['f1']:.4f}")
            else:
                st.metric("R²", f"{mejor['metricas']['r2']:.4f}")
                st.metric("RMSE", f"{mejor['metricas']['rmse']:.4f}")
    
    # Tabla comparativa de todos los modelos
    st.subheader("📋 Comparación de modelos")
    
    if resultados['modelos_exitosos']:
        # Crear DataFrame para la tabla comparativa
        modelos_df = pd.DataFrame()
        
        for modelo in resultados['modelos_exitosos']:
            # Crear fila para el modelo actual
            modelo_data = {
                'Modelo': modelo['nombre'],
                'Tiempo (s)': f"{modelo['tiempo_entrenamiento']:.4f}"
            }
            
            # Agregar métricas según tipo de problema
            if resultados['tipo_problema'] == 'clasificacion':
                modelo_data.update({
                    'Accuracy': f"{modelo['metricas']['accuracy']:.4f}",
                    'Precision': f"{modelo['metricas']['precision']:.4f}",
                    'Recall': f"{modelo['metricas']['recall']:.4f}",
                    'F1-Score': f"{modelo['metricas']['f1']:.4f}",
                    'CV Score': f"{modelo['metricas']['cv_score_media']:.4f} ± {modelo['metricas']['cv_score_std']:.4f}"
                })
            else:
                modelo_data.update({
                    'R²': f"{modelo['metricas']['r2']:.4f}",
                    'MSE': f"{modelo['metricas']['mse']:.4f}",
                    'RMSE': f"{modelo['metricas']['rmse']:.4f}",
                    'MAE': f"{modelo['metricas']['mae']:.4f}",
                    'CV Score': f"{modelo['metricas']['cv_score_media']:.4f} ± {modelo['metricas']['cv_score_std']:.4f}"
                })
            
            # Agregar fila al DataFrame
            modelos_df = pd.concat([modelos_df, pd.DataFrame([modelo_data])], ignore_index=True)
        
        # Mostrar tabla
        st.dataframe(modelos_df)
        
        # Visualización gráfica de métricas
        st.subheader("📈 Visualización de resultados")
        
        # Seleccionar métrica para visualizar
        if resultados['tipo_problema'] == 'clasificacion':
            metrica_options = ['accuracy', 'precision', 'recall', 'f1']
            metrica_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            metrica_default = 'accuracy'
        else:
            metrica_options = ['r2', 'mse', 'rmse', 'mae']
            metrica_labels = ['R²', 'MSE', 'RMSE', 'MAE']
            metrica_default = 'r2'
        
        col1, col2 = st.columns(2)
        
        with col1:
            metrica_index = metrica_options.index(metrica_default)
            metrica = st.selectbox(
                "Seleccione métrica para visualizar:",
                options=metrica_labels,
                index=metrica_index
            )
            
            # Mapear etiqueta a clave de métrica
            metrica_key = metrica_options[metrica_labels.index(metrica)]
        
        with col2:
            orden = st.radio(
                "Orden:",
                options=["Descendente", "Ascendente"],
                index=0 if metrica_key in ['accuracy', 'f1', 'r2'] else 1
            )
        
        # Crear gráfico de barras para la métrica seleccionada
        datos_grafico = []
        
        for modelo in resultados['modelos_exitosos']:
            datos_grafico.append({
                'Modelo': modelo['nombre'],
                'Valor': modelo['metricas'][metrica_key]
            })
        
        # Convertir a DataFrame
        df_grafico = pd.DataFrame(datos_grafico)
        
        # Ordenar según preferencia
        df_grafico = df_grafico.sort_values(
            by='Valor', 
            ascending=(orden == "Ascendente")
        )
        
        # Crear gráfico
        fig, ax = plt.subplots(figsize=(10, 6))        # Crear barras con colores simples
        if orden == "Descendente":
            barras = ax.barh(df_grafico['Modelo'], df_grafico['Valor'], color='green')
        else:
            barras = ax.barh(df_grafico['Modelo'], df_grafico['Valor'], color='red')
        
        # Agregar valores a las barras
        for i, barra in enumerate(barras):
            ax.text(
                barra.get_width() + 0.01, 
                barra.get_y() + barra.get_height()/2, 
                f'{df_grafico["Valor"].iloc[i]:.4f}', 
                va='center'
            )
        
        # Configurar gráfico
        plt.title(f'Comparación de modelos por {metrica}')
        plt.xlabel(metrica)
        plt.tight_layout()
        
        # Mostrar gráfico
        st.pyplot(fig)
    
    # Información sobre modelos fallidos
    if resultados['modelos_fallidos']:
        with st.expander("⚠️ Modelos con errores"):
            for modelo in resultados['modelos_fallidos']:
                st.warning(f"**{modelo['nombre']}**: {modelo['error']}")    # Opciones adicionales
    st.subheader("⏩ Próximos pasos")
    
    st.write("""
    Ahora que ha completado el benchmarking de modelos, puede:
    1. **Evaluar detalladamente** los modelos para analizar su rendimiento a fondo
    2. **Consultar la recomendación** de modelo para obtener sugerencias basadas en sus datos
    """)
    
    # Botones para navegación
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Ir a Evaluación detallada", use_container_width=True):
            # Guardar instrucción de navegación en la sesión
            session.guardar_estado("navegacion", "Evaluar_Modelos")
            # Redirigir a la página de evaluación
            st.switch_page("pages/Machine Learning/06_Evaluar_Modelos.py")
    
    with col2:
        if st.button("👑 Ir a Recomendación de modelo", use_container_width=True):
            # Guardar instrucción de navegación en la sesión
            session.guardar_estado("navegacion", "Recomendar_Modelo")
            # Redirigir a la página de recomendación
            st.switch_page("pages/Machine Learning/08_Recomendar_Modelo.py")

if __name__ == "__main__":
    main()
