import streamlit as st
from src.modelos.explicador import obtener_importancias_shap
from src.modelos.entrenador import cargar_modelo_entrenado, preparar_datos_para_ml
from src.datos.cargador import cargar_datos_entrada
from src.ui.explicacion import mostrar_grafico_importancias
from src.state.session_manager import SessionManager

st.set_page_config(page_title="Explicación del Modelo", page_icon="🧠", layout="wide")
st.title("Explicación del Modelo: Variables Influyentes")

st.markdown("""
Esta sección permite interpretar el modelo seleccionado mostrando la importancia de cada variable predictiva usando SHAP.
Selecciona un modelo entrenado para visualizar la influencia de las variables en sus predicciones.
""")

# Cargar modelos disponibles
dic_modelos = cargar_modelo_entrenado(listar=True)
dic_modelos = dic_modelos or {}
if not isinstance(dic_modelos, dict) or not dic_modelos:
    st.warning("No hay modelos entrenados disponibles. Entrena un modelo antes de continuar.")
    st.stop()

def nombre_modelo(dic, k):
    v = dic.get(k)
    if isinstance(v, dict) and 'nombre' in v:
        return str(v['nombre'])
    return str(k)

# Obtener modelo recomendado de la sesión si existe
session = SessionManager()
modelo_recomendado = session.obtener_estado("modelo_recomendado")

if modelo_recomendado and isinstance(modelo_recomendado, dict):
    modelo_id_default = modelo_recomendado.get('modelo_recomendado', {}).get('id', None)
else:
    modelo_id_default = None

modelo_id = st.selectbox(
    "Selecciona el modelo a explicar:",
    list(dic_modelos.keys()),
    format_func=lambda k: nombre_modelo(dic_modelos, k),
    index=list(dic_modelos.keys()).index(modelo_id_default) if modelo_id_default in dic_modelos else 0
)

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
    st.write("**Columnas disponibles:** {}".format(', '.join(info['Columnas disponibles'])))
    st.write("**Tipos de datos:**")
    st.json(info['Tipos de datos'])
    if X is not None:
        st.dataframe(X.head(10))
    if y is not None:
        st.success(f"Columna objetivo detectada: **{columna_objetivo if columna_objetivo else (y.name if hasattr(y, 'name') else str(y))}**")
    else:
        st.warning("No se detectó columna objetivo. Verifique la configuración de datos.")
else:
    st.warning("No se pudo mostrar información del dataset. Verifique la carga de datos.")

st.info(
    """
    **¿Por qué es importante este análisis?**
    
    La explicación de modelos mediante SHAP permite identificar qué variables tienen mayor influencia en las predicciones del modelo. Esto ayuda a:
    - Validar que el modelo utiliza información relevante y no sesgada.
    - Comunicar resultados de manera transparente a equipos técnicos y no técnicos.
    - Detectar posibles riesgos regulatorios o de negocio asociados a variables sensibles.
    - Mejorar la confianza y la interpretabilidad de los modelos en entornos industriales y farmacéuticos.
    
    **¿Cómo interpretar los resultados?**
    - El gráfico de barras muestra la importancia media de cada variable: cuanto mayor, más impacto tiene en las predicciones.
    - El summary plot de SHAP permite ver cómo los valores altos o bajos de cada variable afectan la salida del modelo.
    - Utiliza estos resultados para discutir con tu equipo y tomar decisiones informadas sobre el uso del modelo.
    """,
    icon="ℹ️"
)

if st.button("Explicar modelo"):
    with st.spinner("Calculando importancias SHAP..."):
        modelo = cargar_modelo_entrenado(modelo_id)
        # Usar X, y ya cargados arriba
        if X is None or X.empty:
            st.error("No se pudieron cargar los datos de entrada. Verifique el dataset y vuelva a intentarlo.")
        else:
            try:
                X_pre = preparar_datos_para_ml(X)
                importancias, shap_values = obtener_importancias_shap(modelo, X_pre)
                mostrar_grafico_importancias(importancias, shap_values, X_pre)
                st.success("Explicación generada correctamente.")
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
                st.error(f"No se pudo generar la explicación: {e}")

    # Botones de navegación
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Volver a Recomendación de Modelo", use_container_width=True):
            st.switch_page("pages/Machine Learning/08_Recomendar_Modelo.py")
    with col2:
        if st.button("📊 Generar Reporte", use_container_width=True):
            st.switch_page("pages/Reportes/10_Reporte.py")

    st.info(
        """
        👉 **Siguiente paso recomendado:**
        
        Una vez interpretado el modelo, puedes generar un reporte completo para documentar los resultados y compartirlos con tu equipo o para auditoría. Recuerda que la interpretación y la trazabilidad son claves para la toma de decisiones en entornos industriales y farmacéuticos.
        """,
        icon="📄"
    )

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
            ("Modelos disponibles", list(dic_modelos.keys())),
            ("Modelo seleccionado", modelo_id),
        ]
        st.table([[k, v if v not in [None, [], ''] else ':red[None o vacío]'] for k, v in debug_data])
        st.markdown("**Estado de sesión relevante:**")
        st.json({k: v for k, v in st.session_state.items() if k in ['df', 'filename', 'variable_objetivo', 'metodo_carga', 'upload_timestamp']})
        st.info("Si 'variable_objetivo' es None o las columnas están vacías, revisa la configuración de datos y asegúrate de completar el paso de configuración antes de explicar el modelo.")
    st.stop()