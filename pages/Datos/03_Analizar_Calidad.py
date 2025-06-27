import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.title("📊 3. Análisis Exploratorio de Datos (EDA)")
st.markdown("""
Esta página realiza un análisis exploratorio del dataset cargado y validado. El objetivo es comprender la estructura, calidad y relaciones de los datos antes de modelar.
""")

# Verificar datos cargados y validados
if 'df' not in st.session_state or st.session_state.df is None:
    st.warning("⚠️ No hay datos cargados. Por favor, carga un dataset primero en la página 'Cargar Datos'.")
    if st.button("Ir a Cargar Datos"):
        st.switch_page("pages/Datos/01_Cargar_Datos.py")
elif 'validacion_completa' not in st.session_state or not st.session_state.validacion_completa:
    st.warning("⚠️ Los datos no han sido validados. Por favor, valida los datos primero.")
    if st.button("Ir a Validar Datos"):
        st.switch_page("pages/Datos/02_Validar_Datos.py")
else:
    df = st.session_state.df
    st.write(f"### Dataset: {st.session_state.filename}")
    st.write(f"Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
    st.write("---")

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
    st.dataframe(nulos_df[nulos_df["Nulos"] > 0], use_container_width=True)
    st.caption("Columnas con alto porcentaje de nulos pueden requerir imputación, eliminación o revisión de la fuente de datos.")
    st.write(f"**Filas duplicadas:** {df.duplicated().sum()}")
    st.caption("Duplicados pueden indicar errores de carga, registros repetidos o procesos de integración incompletos.")

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
        fig = px.bar(df[col_cat].value_counts().reset_index(), x='index', y=col_cat, title=f"Frecuencia de {col_cat}")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Categorías dominantes pueden indicar desbalance o necesidad de agrupar valores poco frecuentes.")
        # Gráfico de pastel
        fig2 = px.pie(df, names=col_cat, title=f"Distribución de {col_cat}")
        st.plotly_chart(fig2, use_container_width=True)

    # === SECCIÓN 5: Correlación y relaciones ===
    if len(num_cols) > 1:
        st.header("5. Correlación entre variables numéricas")
        st.markdown("""
        **Objetivo:** Identificar relaciones lineales entre variables numéricas. Correlaciones altas pueden indicar redundancia o multicolinealidad.
        """)
        corr = df[num_cols].corr()
        fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Matriz de correlación")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Correlaciones cercanas a 1 o -1 sugieren relación fuerte; valores cercanos a 0 indican independencia.")
        # Scatter matrix
        st.markdown("**Matriz de dispersión (scatter matrix):**")
        fig2 = px.scatter_matrix(df, dimensions=num_cols, title="Matriz de dispersión de variables numéricas")
        st.plotly_chart(fig2, use_container_width=True)

    # === SECCIÓN 6: Outliers ===
    st.header("6. Detección de outliers (boxplot)")
    st.markdown("""
    **Objetivo:** Visualizar valores atípicos que pueden distorsionar el análisis y los modelos. Los outliers pueden ser errores, casos especiales o información relevante.
    """)
    if len(num_cols) > 0:
        col_out = st.selectbox("Selecciona una variable numérica para boxplot", num_cols, key="boxplot")
        fig = px.box(df, y=col_out, points="outliers", title=f"Boxplot de {col_out}")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Puntos fuera de los bigotes del boxplot son posibles outliers. Revisar si son errores o casos válidos.")
        # Swarm plot (imitado con strip)
        st.markdown("**Distribución detallada (strip plot):**")
        fig2 = px.strip(df, y=col_out, title=f"Distribución detallada de {col_out}")
        st.plotly_chart(fig2, use_container_width=True)

    # === SECCIÓN 7: Recomendaciones ===
    st.header("7. Recomendaciones y hallazgos clave")
    st.markdown("""
    **Objetivo:** Resumir los principales problemas detectados y sugerir próximos pasos para mejorar la calidad y utilidad del dataset.
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
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Volver a Validar Datos"):
            st.switch_page("pages/Datos/02_Validar_Datos.py")
    with col2:
        if st.button("➡️ Configurar Datos"):
            st.switch_page("pages/Datos/04_Configurar_Datos.py")
