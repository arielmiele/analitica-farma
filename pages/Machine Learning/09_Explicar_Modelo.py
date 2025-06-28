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
El modelo a explicar es el recomendado y seleccionado previamente en el flujo de la aplicación.
""")

# Obtener modelo recomendado de la sesión
session = SessionManager()
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
                # Mensaje genérico sobre la variable más influyente
                if importancias is not None and not importancias.empty:
                    variable_top = importancias.index[0]
                    importancia_top = importancias.iloc[0]
                    st.warning(f"""
                    La variable más influyente según SHAP es: **{variable_top}** (importancia media: {importancia_top:.3f}).
                    
                    Esto significa que el modelo utiliza principalmente esta variable para predecir el objetivo. Te recomendamos analizar si esta variable es relevante para tu análisis o si podría estar reflejando un sesgo o una relación indirecta. Si consideras que no debería ser la principal, prueba excluirla y vuelve a entrenar el modelo para comparar resultados.
                    """, icon="⚠️")
            except Exception as e:
                st.error(f"Error al generar la explicación: {e}")

# Botones de navegación
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    if st.button("🔙 Volver a Recomendación de Modelo", use_container_width=True):
        st.switch_page("pages/Machine Learning/08_Recomendar_Modelo.py")
with col2:
    if st.button("📊 Ir a Evaluación Detallada", use_container_width=True):
        st.switch_page("pages/Machine Learning/06_Evaluar_Modelos.py")

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