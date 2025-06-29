import streamlit as st
from src.modelos.explicador import obtener_importancias_shap
from src.modelos.entrenador import cargar_modelo_entrenado, preparar_datos_para_ml
from src.datos.cargador import cargar_datos_entrada
from src.ui.explicacion import mostrar_grafico_importancias
from src.state.session_manager import SessionManager
from src.audit.logger import log_audit

st.set_page_config(page_title="Explicación del Modelo", page_icon="🧠", layout="wide")
st.title("Explicación del Modelo: Variables Influyentes")

st.markdown("""
Esta sección permite interpretar el modelo seleccionado mostrando la importancia de cada variable predictiva usando SHAP.
El modelo a explicar es el recomendado y seleccionado previamente en el flujo de la aplicación.
""")

# Obtener modelo recomendado de la sesión
session = SessionManager()
# Log de acceso a la página
log_audit(
    usuario=session.obtener_estado("usuario_id", "sistema"),
    accion="ACCESO_PAGINA",
    entidad="explicar_modelo",
    id_entidad="",
    detalles="Acceso a la página de explicación de modelo",
    id_sesion=session.obtener_estado("id_sesion", "sin_sesion")
)

modelo_recomendado = session.obtener_estado("modelo_recomendado")

if not modelo_recomendado or not isinstance(modelo_recomendado, dict) or not modelo_recomendado.get('modelo_recomendado'):
    st.warning("No hay modelo recomendado seleccionado en la sesión. Seleccione un modelo en la sección de recomendación antes de continuar.")
    st.stop()

modelo_id = modelo_recomendado['modelo_recomendado'].get('nombre')

# Cargar modelos disponibles y buscar el modelo recomendado
modelos_disponibles = cargar_modelo_entrenado(listar=True)
if not modelos_disponibles or modelo_id not in modelos_disponibles:
    st.warning("El modelo recomendado no está disponible en los modelos entrenados. Vuelva a entrenar o recomendar un modelo.")
    st.stop()

st.info(f"Modelo a explicar: **{modelo_id}**", icon="🤖")

# Obtener la columna objetivo persistida desde la configuración de datos
columna_objetivo = st.session_state.get('variable_objetivo', None)

# Cargar datos de entrada y mostrar información previa
X, y, info = None, None, None
try:
    if columna_objetivo:
        X, y = cargar_datos_entrada(columna_objetivo=columna_objetivo)
    else:
        X, y = cargar_datos_entrada()
    if X is not None and not X.empty:
        info = {
            'Filas': X.shape[0],
            'Columnas': X.shape[1],
            'Columnas disponibles': list(X.columns),
            'Tipos de datos': X.dtypes.astype(str).to_dict(),
        }
except Exception as e:
    st.warning(f"No se pudo cargar el dataset: {e}")

if info:
    st.markdown("### Información del dataset de entrada")
    st.write("**Filas:** {}  |  **Columnas:** {}".format(info['Filas'], info['Columnas']))
    with st.expander("Ver columnas disponibles", expanded=False):
        st.write(", ".join(info['Columnas disponibles']))
    with st.expander("Ver tipos de datos", expanded=False):
        st.json(info['Tipos de datos'])
    if X is not None:
        st.markdown("#### Primeras 10 filas del dataset de entrada")
        st.dataframe(X.head(10))
    if y is not None:
        st.success(f"Columna objetivo detectada: **{columna_objetivo if columna_objetivo else (y.name if hasattr(y, 'name') else str(y))}**")
    else:
        st.warning("No se detectó columna objetivo. Verifique la configuración de datos.")
else:
    st.warning("No se pudo mostrar información del dataset. Verifique la carga de datos.")

# Mensaje de importancia del análisis en un expander
with st.expander("ℹ️ ¿Por qué es importante este análisis?", expanded=False):
    st.markdown("""
    La explicación de modelos mediante SHAP permite identificar qué variables tienen mayor influencia en las predicciones del modelo. Esto ayuda a:
    - Validar que el modelo utiliza información relevante y no sesgada.
    - Comunicar resultados de manera transparente a equipos técnicos y no técnicos.
    - Detectar posibles riesgos regulatorios o de negocio asociados a variables sensibles.
    - Mejorar la confianza y la interpretabilidad de los modelos en entornos industriales y farmacéuticos.
    
    **¿Cómo interpretar los resultados?**
    - El gráfico de barras muestra la importancia media de cada variable: cuanto mayor, más impacto tiene en las predicciones.
    - El summary plot de SHAP permite ver cómo los valores altos o bajos de cada variable afectan la salida del modelo.
    - Utiliza estos resultados para discutir con tu equipo y tomar decisiones informadas sobre el uso del modelo.
    """)

if st.button("Explicar modelo"):
    with st.spinner("Calculando importancias SHAP..."):
        modelo = cargar_modelo_entrenado(modelo_id)
        # Usar X, y ya cargados arriba
        if X is None or X.empty:
            st.error("No se pudieron cargar los datos de entrada. Verifique el dataset y vuelva a intentarlo.")
            log_audit(
                usuario=session.obtener_estado("usuario_id", "sistema"),
                accion="ERROR_EXPLICACION",
                entidad="explicar_modelo",
                id_entidad=modelo_id,
                detalles="No se pudieron cargar los datos de entrada para explicación.",
                id_sesion=session.obtener_estado("id_sesion", "sin_sesion")
            )
        else:
            try:
                X_pre = preparar_datos_para_ml(
                    X,
                    id_sesion=session.obtener_estado("id_sesion", "sin_sesion"),
                    usuario=session.obtener_estado("usuario_id", "sistema")
                )
                # Filtrar X_pre para que tenga solo las columnas usadas en el entrenamiento
                feature_names = None
                # 1. Intentar obtener desde la sesión
                feature_names_session = st.session_state.get('variables_predictoras', None)
                if feature_names_session:
                    feature_names = feature_names_session
                    st.info("Variables predictoras obtenidas desde la sesión.")
                # 2. Si no están en sesión, intentar desde el modelo (solo si es un objeto válido)
                elif modelo is not None and not isinstance(modelo, dict) and hasattr(modelo, 'feature_names_in_') and getattr(modelo, 'feature_names_in_', None) is not None:
                    feature_names = list(modelo.feature_names_in_)
                    st.info("Variables predictoras obtenidas desde el modelo entrenado.")
                # 3. Si no, intentar desde la configuración guardada
                else:
                    try:
                        from src.modelos.configurador import obtener_configuracion_modelo
                        config = obtener_configuracion_modelo(
                            id_sesion=session.obtener_estado("id_sesion", "sin_sesion"),
                            usuario=session.obtener_estado("usuario_id", "sistema"),
                            id_configuracion=modelo_id
                        )
                        if config and 'variables_predictoras' in config:
                            feature_names = config['variables_predictoras']
                            st.info("Variables predictoras recuperadas desde la configuración guardada.")
                        else:
                            st.warning("No se pudo determinar las variables usadas en el entrenamiento. Verifica la serialización del modelo o la consistencia de las variables predictoras.")
                            log_audit(
                                usuario=session.obtener_estado("usuario_id", "sistema"),
                                accion="ADVERTENCIA_EXPLICACION",
                                entidad="explicar_modelo",
                                id_entidad=modelo_id,
                                detalles="No se pudo determinar las variables predictoras ni desde la sesión, el modelo ni la configuración.",
                                id_sesion=session.obtener_estado("id_sesion", "sin_sesion")
                            )
                    except Exception as e:
                        st.warning(f"Error al recuperar variables predictoras desde la configuración: {e}")
                        log_audit(
                            usuario=session.obtener_estado("usuario_id", "sistema"),
                            accion="ERROR_EXPLICACION",
                            entidad="explicar_modelo",
                            id_entidad=modelo_id,
                            detalles=f"Error al recuperar variables desde configuración: {str(e)}",
                            id_sesion=session.obtener_estado("id_sesion", "sin_sesion")
                        )
                if feature_names:
                    try:
                        X_pre = X_pre[feature_names]
                    except Exception as e:
                        st.warning(f"No se pudieron filtrar las variables predictoras: {e}. Verifica que el dataset de entrada tenga las mismas columnas que el modelo.")
                        log_audit(
                            usuario=session.obtener_estado("usuario_id", "sistema"),
                            accion="ERROR_EXPLICACION",
                            entidad="explicar_modelo",
                            id_entidad=modelo_id,
                            detalles=f"Error al filtrar variables predictoras: {str(e)}",
                            id_sesion=session.obtener_estado("id_sesion", "sin_sesion")
                        )
                        st.stop()
                else:
                    st.warning("No se pudo determinar las variables predictoras. Considera reentrenar el modelo o revisar la serialización.")
                    st.info("Puedes reentrenar el modelo para asegurar que las variables se guarden correctamente.")
                    st.stop()
                importancias, shap_values = obtener_importancias_shap(
                    modelo,
                    X_pre,
                    id_sesion=session.obtener_estado("id_sesion", "sin_sesion"),
                    usuario=session.obtener_estado("usuario_id", "sistema")
                )
                # Guardar interpretabilidad en la sesión
                interpretabilidad_data = {
                    'importancias': importancias.to_dict() if hasattr(importancias, 'to_dict') else importancias,
                    'shap_values': shap_values.tolist() if hasattr(shap_values, 'tolist') else shap_values,
                    'columnas': list(X_pre.columns),
                    'modelo_id': modelo_id,
                    'usuario': session.obtener_estado("usuario_id", "sistema"),
                    'id_sesion': session.obtener_estado("id_sesion", "sin_sesion")
                }
                session.guardar_estado("interpretabilidad", interpretabilidad_data)
                mostrar_grafico_importancias(importancias, shap_values, X_pre)
                st.success("Explicación generada correctamente.")
                log_audit(
                    usuario=session.obtener_estado("usuario_id", "sistema"),
                    accion="EXPLICACION_SHAP",
                    entidad="explicar_modelo",
                    id_entidad=modelo_id,
                    detalles=f"Explicación SHAP generada para modelo {modelo_id}",
                    id_sesion=session.obtener_estado("id_sesion", "sin_sesion")
                )
                st.info(
                    """
                    **Interpretación de los resultados:**
                    
                    - Las variables en la parte superior del gráfico son las más influyentes.
                    - El summary plot muestra cómo los valores de cada variable afectan la predicción (color azul: valores bajos, rojo: altos).
                    - Si una variable esperada no es importante, revisa la calidad de los datos o el entrenamiento.
                    - Usa esta información para validar, ajustar o comunicar el modelo.
                    """,
                    icon="🔍"
                )
            except Exception as e:
                st.error(f"Error al calcular las importancias SHAP: {e}")
                log_audit(
                    usuario=session.obtener_estado("usuario_id", "sistema"),
                    accion="ERROR_EXPLICACION",
                    entidad="explicar_modelo",
                    id_entidad=modelo_id,
                    detalles=f"Error al calcular importancias SHAP: {str(e)}",
                    id_sesion=session.obtener_estado("id_sesion", "sin_sesion")
                )
                
# Botones de navegación
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    if st.button("🔙 Volver a Recomendación de Modelo", use_container_width=True):
        st.switch_page("pages/Machine Learning/08_Recomendar_Modelo.py")
with col2:
    if st.button("📊 Ir a la Generación de Reporte", use_container_width=True):
        st.switch_page("pages/Reportes/10_Reporte.py")

# Si ocurre un error crítico en la carga, mostrar depuración y detener ejecución
if X is None or (hasattr(X, 'empty') and X.empty):
    st.warning("No se pudo cargar el dataset o está vacío. Revisa la información de depuración abajo.")
    with st.expander("🔧 Depuración (siempre visible si hay error de carga)", expanded=True):
        st.markdown("**Depuración de estado de sesión y dataset**")
        debug_data = [
            ("variable_objetivo en sesión", st.session_state.get('variable_objetivo', None)),
            ("Columnas en X", list(X.columns) if X is not None else None),
            ("y (nombre y tipo)", f"{getattr(y, 'name', None)} | {type(y)}" if y is not None else None),
            ("info", info),
            ("X shape", X.shape if X is not None else None),
            ("y shape", y.shape if y is not None else None),
            ("Modelo recomendado", modelo_id),
        ]
        st.table([[k, v if v not in [None, [], ''] else ':red[None o vacío]'] for k, v in debug_data])
        st.markdown("**Estado de sesión relevante:**")
        st.json({k: v for k, v in st.session_state.items() if k in ['df', 'filename', 'variable_objetivo', 'metodo_carga', 'upload_timestamp']})
        st.info("Si 'variable_objetivo' es None o las columnas están vacías, revisa la configuración de datos y asegúrate de completar el paso de configuración antes de explicar el modelo.")
    st.stop()