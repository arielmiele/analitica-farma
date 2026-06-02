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
    # Auto-ejecutar validación si los datos están cargados pero no validados
    with st.spinner("Ejecutando validación automática de datos..."):
        from src.datos.validador import validar_tipos_datos, validar_fechas
        id_sesion = st.session_state.get("id_sesion", "sin_sesion")
        errores_tipo = validar_tipos_datos(st.session_state.df, usuario=usuario_id, id_sesion=id_sesion)
        errores_fecha = validar_fechas(st.session_state.df, usuario=usuario_id, id_sesion=id_sesion)
        st.session_state.errores_tipo = errores_tipo
        st.session_state.errores_fecha = errores_fecha
        st.session_state.validacion_realizada = True
        st.session_state.validacion_completa = True
    if errores_tipo or errores_fecha:
        st.warning(f"⚠️ Se encontraron {len(errores_tipo)} problema(s) de tipo y {len(errores_fecha)} problema(s) de fecha. Puedes corregirlos en '🔍 Validar Datos'.")
    st.rerun()
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

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Vista general",
        "📊 Estadísticas",
        "🔍 Nulos y Duplicados",
        "📈 Distribuciones",
        "🎯 vs Objetivo"
    ])

    with tab1:
        # === SECCIÓN 1: Vista general ===
        st.header("Vista general")
        st.markdown("""
        **Objetivo:** Observar las primeras filas y los tipos de variables para familiarizarse con la estructura y el contenido del dataset.
        """)
        st.dataframe(df.head(), use_container_width=True)
        st.write("**Tipos de variables:**")
        tipos = pd.DataFrame({"Columna": df.columns, "Tipo": df.dtypes.values})
        st.dataframe(tipos, use_container_width=True, height=300)
        st.caption("Variables numéricas permiten análisis estadístico y visualizaciones como histogramas y boxplots. Las categóricas son útiles para análisis de frecuencia y segmentación.")

    with tab2:
        # === SECCIÓN 2: Estadísticas descriptivas ===
        st.header("Estadísticas descriptivas")
        st.markdown("""
        **Objetivo:** Resumir la tendencia central, dispersión y valores extremos de las variables. Permite detectar posibles errores, outliers y rangos inesperados.
        """)
        st.write("**Variables numéricas:**")
        st.dataframe(df.describe().T, use_container_width=True, height=300)
        st.caption("Valores atípicos en mínimo/máximo o una desviación estándar muy alta pueden indicar outliers o errores de carga.")
        st.write("**Variables categóricas:**")
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            st.dataframe(df[cat_cols].describe().T, use_container_width=True, height=300)
            st.caption("Categorías con muy baja frecuencia pueden ser errores, valores nulos o categorías poco informativas.")
        else:
            st.info("No hay variables categóricas.")

    with tab3:
        # === SECCIÓN 3: Valores nulos y duplicados ===
        st.header("Valores nulos y duplicados")
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

    with tab4:
        # === SECCIÓN 4: Distribución de variables ===
        st.header("Distribución de variables")
        st.markdown("""
        **Objetivo:** Visualizar la forma y dispersión de las variables para detectar asimetrías, valores extremos y patrones inusuales.
        """)
        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) > 0:
            col = st.selectbox("Selecciona una variable numérica", num_cols)
            fig = px.histogram(df, x=col, nbins=30, title=f"Histograma de {col}")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Distribuciones sesgadas o multimodales pueden requerir transformaciones o segmentación adicional.")
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
            fig2 = px.pie(df, names=col_cat, title=f"Distribución de {col_cat}")
            st.plotly_chart(fig2, use_container_width=True)

    with tab5:
        # === SECCIÓN 5: Comparativa con la variable objetivo ===
        st.header("Comparativa con la variable objetivo")
        st.markdown("""
        Visualiza cómo se relacionan las variables explicativas seleccionadas con la variable objetivo. Esta sección ayuda a identificar relaciones predictivas y patrones útiles para el modelado.
        """)
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
            col_num, col_cat2 = st.columns(2)
            with col_num:
                num_cols_tab5 = df.select_dtypes(include=np.number).columns
                st.caption(f"Variables numéricas disponibles: {len([c for c in all_cols if c in num_cols_tab5 and c != target_col])}")
                explicativas_num = st.multiselect(
                    "Variables numéricas",
                    [c for c in num_cols_tab5 if c != target_col],
                    default=[],
                    key="explicativas_num_selector"
                )
            with col_cat2:
                cat_cols_tab5 = df.select_dtypes(include=['object', 'category']).columns
                st.caption(f"Variables categóricas disponibles: {len([c for c in all_cols if c in cat_cols_tab5 and c != target_col])}")
                explicativas_cat = st.multiselect(
                    "Variables categóricas",
                    [c for c in cat_cols_tab5 if c != target_col],
                    default=[],
                    key="explicativas_cat_selector"
                )

        explicativas_cols = list(explicativas_num) + list(explicativas_cat)

        if len(explicativas_cols) == 0:
            st.warning("Selecciona al menos una variable explicativa para visualizar los gráficos comparativos.")
        else:
            if len(explicativas_cols) > 15:
                st.warning("Has seleccionado más de 15 variables explicativas. El procesamiento puede tardar. Considera reducir la selección.")
            if st.button("Mostrar análisis comparativo y avanzado", use_container_width=True):
                target_is_cat = (df[target_col].dtype == 'object' or str(df[target_col].dtype).startswith('category') or df[target_col].nunique() < 10)
                if target_is_cat:
                    st.markdown("### Boxplot de variables numéricas por clase de la variable objetivo")
                    for col in explicativas_cols:
                        if col in num_cols_tab5:
                            fig = px.box(df, x=target_col, y=col, points="outliers", title=f"Boxplot de {col} según {target_col}")
                            st.plotly_chart(fig, use_container_width=True)
                    st.markdown("### Distribución de variables categóricas vs objetivo")
                    for col in explicativas_cols:
                        if col in cat_cols_tab5 and col != target_col:
                            cross = pd.crosstab(df[col], df[target_col])
                            fig = px.bar(cross, barmode="group", title=f"{col} vs {target_col}")
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.markdown("### Scatter plots de variables numéricas coloreados por la variable objetivo")
                    scatter_cols = [col for col in explicativas_cols if col in num_cols_tab5 and col != target_col]
                    if scatter_cols:
                        c1, c2 = st.columns(2)
                        for i, col in enumerate(scatter_cols):
                            fig = px.scatter(df, x=col, y=target_col, color=target_col, title=f"{col} vs {target_col}")
                            with (c1 if i % 2 == 0 else c2):
                                st.plotly_chart(fig, use_container_width=True)
                    st.markdown("### Boxplot de la variable objetivo por categorías")
                    cat_exp_cols = [col for col in explicativas_cols if col in cat_cols_tab5]
                    if cat_exp_cols:
                        for col in cat_exp_cols:
                            fig = px.box(df, x=col, y=target_col, points="outliers", title=f"Boxplot de {target_col} según {col}")
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No hay variables categóricas seleccionadas para comparar con la variable objetivo.")
                st.caption("Estos gráficos ayudan a identificar relaciones predictivas y patrones útiles para el modelado.")

        # === Recomendaciones y hallazgos clave ===
        st.write("---")
        st.header("Recomendaciones y hallazgos clave")
        nulos = df.isnull().sum()
        recomendaciones = []
        if nulos.sum() > 0:
            recomendaciones.append("Hay columnas con valores nulos. Considera imputar o eliminar filas/columnas según el caso.")
        if df.duplicated().sum() > 0:
            recomendaciones.append("Existen filas duplicadas. Se recomienda revisar y limpiar duplicados.")
        num_cols_rec = df.select_dtypes(include=np.number).columns
        for col in num_cols_rec:
            skew = df[col].skew()
            if abs(skew) > 1:
                recomendaciones.append(f"La variable '{col}' presenta alta asimetría (skewness={skew:.2f}). Considera transformaciones.")
        if len(recomendaciones) == 0:
            st.success("No se detectaron problemas importantes en el análisis exploratorio.")
        else:
            for rec in recomendaciones:
                st.warning(rec)

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
