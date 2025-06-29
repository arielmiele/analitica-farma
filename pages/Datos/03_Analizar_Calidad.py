import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from src.audit.logger import setup_logger, log_audit
from src.state.session_manager import SessionManager
from src.datos import analizador

logger = setup_logger("analisis_calidad")
usuario_id = st.session_state.get("usuario_id", "sistema")

st.title("📊 3. Análisis Exploratorio de Datos (EDA)")
st.markdown("""
Esta página realiza un análisis exploratorio del dataset cargado y validado. El objetivo es comprender la estructura, calidad y relaciones de los datos antes de modelar.
""")

# Verificar datos cargados y validados
if 'df' not in st.session_state or st.session_state.df is None:
    st.warning("⚠️ No hay datos cargados. Por favor, carga un dataset primero en la página 'Cargar Datos'.")
    log_audit(
        usuario=usuario_id,
        accion="EDA_NO_DATOS",
        entidad="analisis_calidad",
        id_entidad="",
        detalles="Intento de acceso a EDA sin datos cargados.",
        id_sesion=st.session_state.get("id_sesion", "sin_sesion")
    )
    if st.button("Ir a Cargar Datos"):
        st.switch_page("pages/Datos/01_Cargar_Datos.py")
elif 'validacion_completa' not in st.session_state or not st.session_state.validacion_completa:
    st.warning("⚠️ Los datos no han sido validados. Por favor, valida los datos primero.")
    log_audit(
        usuario=usuario_id,
        accion="EDA_NO_VALIDADO",
        entidad="analisis_calidad",
        id_entidad="",
        detalles="Intento de acceso a EDA sin validación completa.",
        id_sesion=st.session_state.get("id_sesion", "sin_sesion")
    )
    if st.button("Ir a Validar Datos"):
        st.switch_page("pages/Datos/02_Validar_Datos.py")
else:
    df = st.session_state.df
    st.write(f"### Dataset: {st.session_state.filename}")
    st.write(f"Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
    st.write("---")
    log_audit(
        usuario=usuario_id,
        accion="EDA_INICIO",
        entidad="analisis_calidad",
        id_entidad=st.session_state.filename,
        detalles="Inicio de análisis exploratorio de datos.",
        id_sesion=st.session_state.get("id_sesion", "sin_sesion")
    )

    # === SECCIÓN 1: Vista general ===
    st.header("1. Vista general")
    st.markdown("""
    **Objetivo:** Observar las primeras filas y los tipos de variables para familiarizarse con la estructura y el contenido del dataset.
    """)
    st.dataframe(df.head(), use_container_width=True)
    st.write("**Tipos de variables:**")
    tipos = pd.DataFrame({"Columna": df.columns, "Tipo": df.dtypes.values})
    st.dataframe(tipos, use_container_width=True)
    st.caption("Variables numéricas permiten análisis estadístico y visualizaciones como histogramas y boxplots. Las categóricas son útiles para análisis de frecuencia y segmentación.")

    # === SECCIÓN 2: Estadísticas descriptivas ===
    st.header("2. Estadísticas descriptivas")
    st.markdown("""
    **Objetivo:** Resumir la tendencia central, dispersión y valores extremos de las variables. Permite detectar posibles errores, outliers y rangos inesperados.
    """)
    st.write("**Variables numéricas:**")
    st.dataframe(df.describe().T, use_container_width=True)
    st.caption("Valores atípicos en mínimo/máximo o una desviación estándar muy alta pueden indicar outliers o errores de carga.")
    st.write("**Variables categóricas:**")
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        st.dataframe(df[cat_cols].describe().T, use_container_width=True)
        st.caption("Categorías con muy baja frecuencia pueden ser errores, valores nulos o categorías poco informativas.")
    else:
        st.info("No hay variables categóricas.")

    # === SECCIÓN 3: Valores nulos y duplicados ===
    st.header("3. Valores nulos y duplicados")
    st.markdown("""
    **Objetivo:** Identificar la presencia y el patrón de valores faltantes y duplicados, que pueden afectar la calidad del análisis y los modelos.
    """)
    nulos = df.isnull().sum()
    nulos_pct = (nulos / len(df) * 100).round(2)
    nulos_df = pd.DataFrame({"Nulos": nulos, "%": nulos_pct})
    cols_con_nulos = nulos_df[nulos_df["Nulos"] > 0]
    if not cols_con_nulos.empty:
        st.dataframe(cols_con_nulos, use_container_width=True)
        st.caption("Columnas con alto porcentaje de nulos pueden requerir imputación, eliminación o revisión de la fuente de datos.")
    else:
        st.success("No se detectaron valores nulos en ninguna columna.")
    num_dupes = df.duplicated().sum()
    if num_dupes > 0:
        st.write(f"**Filas duplicadas:** {num_dupes}")
        st.caption("Duplicados pueden indicar errores de carga, registros repetidos o procesos de integración incompletos.")
    else:
        st.success("No se detectaron filas duplicadas en el dataset.")

    # === SECCIÓN 4: Distribución de variables ===
    st.header("4. Distribución de variables")
    st.markdown("""
    **Objetivo:** Visualizar la forma y dispersión de las variables para detectar asimetrías, valores extremos y patrones inusuales.
    """)
    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) > 0:
        col = st.selectbox("Selecciona una variable numérica", num_cols)
        fig = px.histogram(df, x=col, nbins=30, title=f"Histograma de {col}")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Distribuciones sesgadas o multimodales pueden requerir transformaciones o segmentación adicional.")
        # Gráfico de densidad
        fig2 = px.density_contour(df, x=col, title=f"Densidad de {col}")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No hay variables numéricas.")
    if len(cat_cols) > 0:
        col_cat = st.selectbox("Selecciona una variable categórica", cat_cols)
        vc = df[col_cat].value_counts().reset_index()
        vc.columns = [col_cat, 'count']
        fig = px.bar(vc, x=col_cat, y='count', title=f"Frecuencia de {col_cat}")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Categorías dominantes pueden indicar desbalance o necesidad de agrupar valores poco frecuentes.")
        # Gráfico de pastel
        fig2 = px.pie(df, names=col_cat, title=f"Distribución de {col_cat}")
        st.plotly_chart(fig2, use_container_width=True)

    # === SECCIÓN 5: Comparativa con la variable objetivo ===
    st.header("5. Comparativa con la variable objetivo")
    st.markdown("""
    Visualiza cómo se relacionan las variables explicativas seleccionadas con la variable objetivo. Esta sección ayuda a identificar relaciones predictivas y patrones útiles para el modelado.
    """)
    # Selección y persistencia de variable objetivo
    all_cols = list(df.columns)
    if 'target_col' not in st.session_state or st.session_state.target_col not in all_cols:
        st.session_state.target_col = all_cols[0]
    target_col = st.selectbox(
        "Selecciona la variable objetivo (target)",
        all_cols,
        index=all_cols.index(st.session_state.target_col),
        key="target_selector"
    )
    st.session_state.target_col = target_col
    st.info(f"Variable objetivo seleccionada: **{target_col}** (persistente en la sesión y visible en el sidebar)")

    with st.container():
        st.markdown("**Selecciona variables explicativas para el análisis comparativo:**")
        col_num, col_cat = st.columns(2)
        with col_num:
            st.caption(f"Variables numéricas disponibles: {len([c for c in all_cols if c in num_cols and c != target_col])}")
            if 'explicativas_num' in st.session_state:
                explicativas_num_default = []
            else:
                explicativas_num_default = []
            explicativas_num = st.multiselect(
                "Variables numéricas",
                [c for c in num_cols if c != target_col],
                default=explicativas_num_default,
                key="explicativas_num_selector"
            )
            st.session_state.explicativas_num = explicativas_num
        with col_cat:
            st.caption(f"Variables categóricas disponibles: {len([c for c in all_cols if c in cat_cols and c != target_col])}")
            if 'explicativas_cat' in st.session_state:
                explicativas_cat_default = []
            else:
                explicativas_cat_default = []
            explicativas_cat = st.multiselect(
                "Variables categóricas",
                [c for c in cat_cols if c != target_col],
                default=explicativas_cat_default,
                key="explicativas_cat_selector"
            )
            st.session_state.explicativas_cat = explicativas_cat
    
    # Unir ambas selecciones para el análisis
    explicativas_cols = st.session_state.explicativas_num + st.session_state.explicativas_cat
    st.session_state.explicativas_cols = explicativas_cols

    mostrar_graficos = False
    if len(explicativas_cols) == 0:
        st.warning("Selecciona al menos una variable explicativa para visualizar los gráficos comparativos.")
    else:
        if len(explicativas_cols) > 15:
            st.warning("Has seleccionado más de 15 variables explicativas. El procesamiento y la visualización pueden tardar y consumir muchos recursos. Considera reducir la selección para un análisis más ágil.")
        if st.button("Mostrar análisis comparativo y avanzado", use_container_width=True):
            mostrar_graficos = True

    if mostrar_graficos:
        target_is_cat = (df[target_col].dtype == 'object' or str(df[target_col].dtype).startswith('category') or df[target_col].nunique() < 10)
        if target_is_cat:
            st.markdown("### Boxplot de variables numéricas por clase de la variable objetivo")
            st.info("Estos gráficos muestran la distribución de cada variable numérica para cada clase de la variable objetivo. Observa si hay diferencias claras entre grupos, presencia de outliers o solapamiento entre clases. Diferencias marcadas pueden indicar buen poder predictivo.")
            for col in explicativas_cols:
                if col in num_cols:
                    fig = px.box(df, x=target_col, y=col, points="outliers", title=f"Boxplot de {col} según {target_col}")
                    st.plotly_chart(fig, use_container_width=True)
            st.markdown("### Distribución de variables categóricas vs objetivo")
            st.info("Estos gráficos muestran la frecuencia de cada categoría según la clase objetivo. Busca categorías asociadas a una clase específica o desbalances importantes.")
            for col in explicativas_cols:
                if col in cat_cols and col != target_col:
                    cross = pd.crosstab(df[col], df[target_col])
                    fig = px.bar(cross, barmode="group", title=f"{col} vs {target_col}")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("### Scatter plots de variables numéricas coloreados por la variable objetivo")
            st.info("Cada gráfico muestra la relación entre una variable explicativa y la variable objetivo. Busca tendencias, agrupamientos, relaciones lineales/no lineales y presencia de outliers. Una nube de puntos bien separada o con tendencia clara indica potencial predictivo.")
            scatter_cols = [col for col in explicativas_cols if col in num_cols and col != target_col]
            if scatter_cols:
                col1, col2 = st.columns(2)
                for i, col in enumerate(scatter_cols):
                    fig = px.scatter(df, x=col, y=target_col, color=target_col, title=f"{col} vs {target_col}")
                    with (col1 if i % 2 == 0 else col2):
                        st.plotly_chart(fig, use_container_width=True)
            st.markdown("### Boxplot de la variable objetivo por categorías")
            st.info("Estos gráficos muestran cómo varía la variable objetivo según cada categoría de la variable explicativa. Diferencias claras entre cajas pueden indicar que la variable categórica es relevante para predecir el objetivo.")
            cat_exp_cols = [col for col in explicativas_cols if col in cat_cols]
            if cat_exp_cols:
                for col in cat_exp_cols:
                    fig = px.box(df, x=col, y=target_col, points="outliers", title=f"Boxplot de {target_col} según {col}")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hay variables categóricas seleccionadas para comparar con la variable objetivo.")
        st.caption("Estos gráficos ayudan a identificar relaciones predictivas y patrones útiles para el modelado.")
        st.write("---")

        # === SECCIÓN: Recomendaciones y hallazgos clave ===
        st.header("Recomendaciones y hallazgos clave")
        st.markdown("""
        Resumen de los principales problemas detectados y sugerencias para mejorar la calidad y utilidad del dataset.
        """)
        recomendaciones = []
        if nulos.sum() > 0:
            recomendaciones.append("Hay columnas con valores nulos. Considera imputar o eliminar filas/columnas según el caso.")
        if df.duplicated().sum() > 0:
            recomendaciones.append("Existen filas duplicadas. Se recomienda revisar y limpiar duplicados.")
        if len(num_cols) > 0:
            for col in num_cols:
                skew = df[col].skew()
                if abs(skew) > 1:
                    recomendaciones.append(f"La variable '{col}' presenta alta asimetría (skewness={skew:.2f}). Considera transformaciones.")
        if len(recomendaciones) == 0:
            st.success("No se detectaron problemas importantes en el análisis exploratorio.")
        else:
            for rec in recomendaciones:
                st.warning(rec)
        st.write("---")
    
    # === GUARDADO DE CALIDAD DE DATOS EN SESIÓN ===
    session = SessionManager()
    id_sesion = st.session_state.get("id_sesion", "sin_sesion")
    df = st.session_state.df if 'df' in st.session_state else None

    if df is not None:
        calidad_datos = {}
        # Métricas globales
        calidad_datos['global'] = analizador.calcular_metricas_basicas(df, id_sesion)
        calidad_datos['evaluacion'] = analizador.evaluar_calidad_global(df, id_sesion)
        # Nulos por columna
        calidad_datos['nulos_por_columna'] = analizador.analizar_nulos_por_columna(df, id_sesion).to_dict(orient='records')
        # Duplicados
        calidad_datos['duplicados'] = analizador.analizar_duplicados(df, id_sesion=id_sesion)
        # Outliers
        calidad_datos['outliers'] = analizador.detectar_outliers(df, id_sesion=id_sesion)
        # Estadísticas por columna
        calidad_datos['estadisticas_columnas'] = analizador.generar_estadisticas_por_columna(df, id_sesion).to_dict(orient='records')
        # Guardar en sesión
        session.guardar_estado("calidad_datos", calidad_datos)

    # Botones de navegación al final de la página (siempre visibles)
    nav_col1, nav_col2 = st.columns(2)
    with nav_col1:
        if st.button("⬅️ Volver a Validar Datos", use_container_width=True, key="btn_volver_validar"):
            st.switch_page("pages/Datos/02_Validar_Datos.py")
    with nav_col2:
        if st.button("➡️ Configurar Datos", use_container_width=True, key="btn_ir_configurar"):
            st.switch_page("pages/Datos/04_Configurar_Datos.py")
