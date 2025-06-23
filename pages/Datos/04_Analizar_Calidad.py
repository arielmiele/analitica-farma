import streamlit as st
import pandas as pd
import os
import sys
import plotly.express as px
import plotly.graph_objects as go

# Agregar el directorio src al path para poder importar los módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Importar módulos de la aplicación
from src.audit.logger import setup_logger, log_operation, log_audit
from src.datos.analizador import (
    analizar_nulos_por_columna,
    detectar_outliers,
    analizar_duplicados,
    generar_estadisticas_por_columna,
    evaluar_calidad_global,
    obtener_recomendaciones
)
from src.state.session_manager import SessionManager

# Configurar el logger
usuario_id = st.session_state.get("usuario_id", 1)
logger = setup_logger("calidad_datos", id_usuario=usuario_id)

# Inicializar session_state para esta página
if 'paso_calidad' not in st.session_state:
    st.session_state.paso_calidad = 0  # 0: inicio, 1: análisis detallado

# Título y descripción de la página
st.title("📊 Análisis de Calidad de Datos")

st.markdown("""
Esta página evalúa automáticamente la calidad de los datos y muestra métricas sobre valores nulos,
duplicados y outliers para ayudarte a evaluar rápidamente el estado de tus datos antes del modelado.
""")

# Verificar si hay datos cargados y validados
if 'df' not in st.session_state or st.session_state.df is None:
    st.warning("⚠️ No hay datos cargados. Por favor, carga un dataset primero en la página 'Cargar Datos'.")
    if st.button("Ir a Cargar Datos"):
        st.session_state.paso_carga = 0  # Reiniciar el paso de carga
        st.switch_page("pages/datos/01_Cargar_Datos.py")
elif 'validacion_completa' not in st.session_state or not st.session_state.validacion_completa:
    st.warning("⚠️ Los datos no han sido validados. Por favor, valida los datos primero.")
    if st.button("Ir a Validar Datos"):
        st.switch_page("pages/datos/03_Validar_Datos.py")
else:
    # Mostrar información del dataset cargado
    st.write(f"### Dataset analizado: {st.session_state.filename}")
    st.write(f"Dimensiones: {st.session_state.df.shape[0]} filas × {st.session_state.df.shape[1]} columnas")
    
    # Mostrar configuración seleccionada
    st.write("### Configuración actual")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Tipo de problema:** {st.session_state.tipo_problema.capitalize()}")
        st.write(f"**Variable objetivo:** {st.session_state.variable_objetivo}")
    with col2:
        n_predictoras = len(st.session_state.variables_predictoras)
        st.write(f"**Variables predictoras:** {n_predictoras} seleccionadas")
    
    # PASO 0: Dashboard general de calidad
    if st.session_state.paso_calidad == 0:
        # Evaluación general de calidad
        with st.spinner("Analizando calidad de datos..."):
            df = st.session_state.df
            
            # Registrar inicio de análisis
            log_operation(logger, "INICIO_ANALISIS", 
                         f"Iniciando análisis de calidad para {st.session_state.filename}", 
                         id_usuario=usuario_id)
            
            # Obtener evaluación global
            evaluacion = evaluar_calidad_global(df)
            
            if 'error' in evaluacion:
                st.error(f"Error al evaluar calidad: {evaluacion['error']}")
            else:
                # Mostrar calificación global
                calificacion = evaluacion['calificacion']
                puntaje = evaluacion['puntaje']
                
                # Usar diferentes colores según la calificación
                if calificacion == 'Excelente':
                    color = 'green'
                elif calificacion == 'Buena':
                    color = 'blue'
                elif calificacion == 'Regular':
                    color = 'orange'
                else:
                    color = 'red'
                
                # Crear tarjeta de calificación
                st.write("## Calificación Global de Calidad")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Mostrar calificación con estilo
                    st.markdown(
                        f"""
                        <div style="padding: 20px; 
                                   border-radius: 10px; 
                                   background-color: {color}; 
                                   color: white; 
                                   text-align: center;
                                   font-size: 24px;
                                   font-weight: bold;">
                            {calificacion}
                            <br>
                            <span style="font-size: 36px;">{puntaje:.1f}/100</span>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                
                with col2:
                    # Mostrar desglose de puntuación
                    fig = go.Figure()
                    
                    categorias = ['Completitud', 'Duplicados', 'Outliers']
                    valores = [
                        evaluacion['puntaje_completitud'],
                        evaluacion['puntaje_duplicados'],
                        evaluacion['puntaje_outliers']
                    ]
                    
                    # Usar colores según el valor (verde para alto, rojo para bajo)
                    colores = ['green' if v >= 0.75 * max_val else 'orange' if v >= 0.5 * max_val else 'red' 
                              for v, max_val in zip(valores, [40, 30, 30])]
                    
                    fig.add_trace(go.Bar(
                        x=categorias,
                        y=valores,
                        marker_color=colores,
                        text=[f"{v:.1f}" for v in valores],
                        textposition='auto'
                    ))
                    
                    fig.update_layout(
                        title="Desglose de Puntuación",
                        xaxis_title="Categoría",
                        yaxis_title="Puntos",
                        yaxis=dict(range=[0, 40]),
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Mostrar recomendaciones
                st.write("## Recomendaciones")
                
                recomendaciones = obtener_recomendaciones(df)
                
                for rec in recomendaciones:
                    tipo = rec['tipo']
                    mensaje = rec['mensaje']
                    
                    if tipo == 'exito':
                        st.success(mensaje)
                    elif tipo == 'informacion':
                        st.info(mensaje)
                    elif tipo == 'advertencia':
                        st.warning(mensaje)
                    elif tipo == 'error':
                        st.error(mensaje)
                
                # Mostrar métricas básicas
                st.write("## Métricas Básicas")
                
                metricas = evaluacion['metricas']
                
                # Crear tres columnas para mostrar métricas
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Completitud", 
                        f"{metricas['completitud']:.1f}%",
                        delta=None
                    )
                    st.metric(
                        "Valores Nulos", 
                        f"{metricas['nulos_totales']} ({metricas['porcentaje_nulos']:.1f}%)",
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        "Duplicados", 
                        f"{metricas['duplicados']} ({metricas['porcentaje_duplicados']:.1f}%)",
                        delta=None
                    )
                    st.metric(
                        "Valores Únicos (Promedio)", 
                        f"{metricas['valores_unicos_promedio']:.1f}",
                        delta=None
                    )
                
                with col3:
                    # Métricas adicionales específicas del dataset
                    n_columnas_criticas = len(analizar_nulos_por_columna(df)[analizar_nulos_por_columna(df)['porcentaje'] > 20])
                    
                    st.metric(
                        "Columnas con >20% Nulos", 
                        f"{n_columnas_criticas}",
                        delta=None
                    )
                    
                    # Calcular columnas con outliers
                    outliers_info = detectar_outliers(df)
                    cols_con_outliers = sum(1 for info in outliers_info.values() if info['porcentaje'] > 5)
                    
                    st.metric(
                        "Columnas con Outliers", 
                        f"{cols_con_outliers}",
                        delta=None
                    )
                
                # Gráfico de barras para nulos por columna (top 10)
                st.write("## Valores Nulos por Columna")
                
                nulos_df = analizar_nulos_por_columna(df)
                
                if not nulos_df.empty:
                    # Tomar las 10 columnas con más nulos
                    top_nulos = nulos_df.nlargest(10, 'porcentaje')
                    
                    fig = px.bar(
                        top_nulos,
                        x='columna',
                        y='porcentaje',
                        color='clasificacion',
                        color_discrete_map={
                            'Excelente': 'green',
                            'Buena': 'blue',
                            'Regular': 'orange',
                            'Crítica': 'red'
                        },
                        title="Top 10 Columnas con Valores Nulos (%)",
                        labels={'columna': 'Columna', 'porcentaje': 'Porcentaje de Nulos (%)'}
                    )
                    
                    fig.update_layout(xaxis={'categoryorder': 'total descending'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No se pudieron analizar valores nulos en el dataset.")
                
                # Botón para ver análisis detallado
                if st.button("Ver Análisis Detallado", key="btn_detalle"):
                    st.session_state.paso_calidad = 1
                    st.rerun()
                
                # Marcar esta etapa como completada
                SessionManager.update_progress("analisis_calidad", True)
                
                # Botón para continuar al siguiente paso
                st.write("---")
                st.write("### Navegación")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("⬅️ Volver a Validación", use_container_width=True):
                        st.switch_page("pages/datos/03_Validar_Datos.py")
                
                with col2:
                    if st.button("➡️ Continuar con Transformaciones", use_container_width=True):
                        # Registrar acción en el log
                        log_audit(usuario_id, "NAVEGACIÓN", "transformaciones", 
                                "Continuando a transformaciones desde análisis de calidad")
                        
                        # Aquí iría la redirección a la página de transformaciones
                        st.info("Próximamente: Página de transformaciones en desarrollo")
    
    # PASO 1: Análisis detallado
    elif st.session_state.paso_calidad == 1:
        st.write("## Análisis Detallado de Calidad")
        
        # Crear pestañas para las diferentes secciones del análisis
        tab1, tab2, tab3, tab4 = st.tabs([
            "📉 Estadísticas por Columna", 
            "🔍 Valores Nulos", 
            "🔄 Duplicados", 
            "⚠️ Outliers"
        ])
        
        with tab1:
            st.write("### Estadísticas Descriptivas por Columna")
            
            with st.spinner("Generando estadísticas..."):
                estadisticas = generar_estadisticas_por_columna(st.session_state.df)
                
                if not estadisticas.empty:
                    # Opciones de filtrado
                    tipos_datos = estadisticas['tipo'].unique().tolist()
                    tipo_seleccionado = st.multiselect(
                        "Filtrar por tipo de dato",
                        options=tipos_datos,
                        default=tipos_datos
                    )
                    
                    # Aplicar filtro
                    if tipo_seleccionado:
                        estadisticas_filtradas = estadisticas[estadisticas['tipo'].isin(tipo_seleccionado)]
                    else:
                        estadisticas_filtradas = estadisticas
                    
                    # Mostrar tabla con estadísticas
                    st.dataframe(
                        estadisticas_filtradas,
                        use_container_width=True,
                        height=400
                    )
                    
                    # Visualizaciones específicas según tipo de dato
                    st.write("### Visualizaciones por Tipo de Dato")
                    
                    # Para columnas numéricas: distribución
                    columnas_numericas = estadisticas[pd.api.types.is_numeric_dtype(estadisticas['tipo'])]['columna'].tolist()
                    
                    if columnas_numericas:
                        col_numerica = st.selectbox(
                            "Seleccionar columna numérica para visualizar distribución",
                            options=columnas_numericas
                        )
                        
                        if col_numerica:
                            fig = px.histogram(
                                st.session_state.df, 
                                x=col_numerica,
                                marginal="box",
                                title=f"Distribución de {col_numerica}"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Para columnas categóricas: frecuencia
                    columnas_categoricas = estadisticas[
                        (pd.api.types.is_string_dtype(estadisticas['tipo'])) | 
                        (pd.api.types.is_object_dtype(estadisticas['tipo']))
                    ]['columna'].tolist()
                    
                    if columnas_categoricas:
                        col_categorica = st.selectbox(
                            "Seleccionar columna categórica para visualizar frecuencia",
                            options=columnas_categoricas
                        )
                        
                        if col_categorica:
                            # Calcular frecuencias
                            freq = st.session_state.df[col_categorica].value_counts().reset_index()
                            freq.columns = ['valor', 'frecuencia']
                            
                            # Limitar a los 20 valores más frecuentes
                            if len(freq) > 20:
                                freq = freq.head(20)
                                titulo = f"Top 20 valores más frecuentes en {col_categorica}"
                            else:
                                titulo = f"Frecuencia de valores en {col_categorica}"
                            
                            fig = px.bar(
                                freq,
                                x='valor',
                                y='frecuencia',
                                title=titulo
                            )
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("No se pudieron generar estadísticas para este dataset.")
        
        with tab2:
            st.write("### Análisis de Valores Nulos")
            
            with st.spinner("Analizando valores nulos..."):
                nulos_df = analizar_nulos_por_columna(st.session_state.df)
                
                if not nulos_df.empty:
                    # Mostrar tabla con detalles
                    st.dataframe(
                        nulos_df,
                        use_container_width=True,
                        height=300
                    )
                    
                    # Heatmap de valores nulos
                    st.write("### Mapa de Calor de Valores Nulos")
                    
                    # Crear matriz para el heatmap
                    df_nulos = st.session_state.df.isna()
                    
                    # Limitar a una muestra si el dataset es grande
                    if len(df_nulos) > 100:
                        df_muestra = df_nulos.sample(n=100, random_state=42)
                        titulo = "Mapa de Calor de Valores Nulos (muestra de 100 filas)"
                    else:
                        df_muestra = df_nulos
                        titulo = "Mapa de Calor de Valores Nulos (dataset completo)"
                    
                    fig = px.imshow(
                        df_muestra.T,
                        color_continuous_scale=[[0, 'white'], [1, 'red']],
                        title=titulo,
                        labels=dict(x="Fila", y="Columna", color="Es nulo")
                    )
                    
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Correlación entre nulos
                    st.write("### Correlación entre Valores Nulos")
                    st.info("Esta visualización muestra si los valores nulos en diferentes columnas aparecen juntos, "
                          "lo que puede indicar un patrón en los datos faltantes.")
                    
                    # Calcular matriz de correlación de nulos
                    corr_nulos = df_nulos.corr()
                    
                    fig = px.imshow(
                        corr_nulos,
                        color_continuous_scale='RdBu_r',
                        title="Correlación entre Valores Nulos por Columna",
                        labels=dict(x="Columna", y="Columna", color="Correlación")
                    )
                    
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No se encontraron valores nulos en el dataset.")
        
        with tab3:
            st.write("### Análisis de Duplicados")
            
            with st.spinner("Analizando duplicados..."):
                # Permitir seleccionar columnas para buscar duplicados
                todas_columnas = st.session_state.df.columns.tolist()
                
                cols_seleccionadas = st.multiselect(
                    "Seleccionar columnas para detectar duplicados",
                    options=todas_columnas,
                    default=todas_columnas
                )
                
                if not cols_seleccionadas:
                    st.warning("Por favor, selecciona al menos una columna.")
                else:
                    # Analizar duplicados en las columnas seleccionadas
                    info_duplicados = analizar_duplicados(st.session_state.df, cols_seleccionadas)
                    
                    if 'error' in info_duplicados:
                        st.error(f"Error al analizar duplicados: {info_duplicados['error']}")
                    else:
                        # Mostrar resultados
                        st.write(f"**Filas duplicadas:** {info_duplicados['cantidad']} ({info_duplicados['porcentaje']:.1f}%)")
                        
                        if info_duplicados['cantidad'] > 0:
                            # Mostrar grupos de duplicados
                            st.write("### Grupos de Valores Duplicados")
                            
                            if not info_duplicados['grupos_duplicados'].empty:
                                st.dataframe(
                                    info_duplicados['grupos_duplicados'],
                                    use_container_width=True,
                                    height=300
                                )
                                
                                # Gráfico de barras para grupos más frecuentes
                                if len(info_duplicados['grupos_duplicados']) > 0:
                                    top_grupos = info_duplicados['grupos_duplicados'].nlargest(10, 'conteo')
                                    
                                    # Crear un identificador para cada grupo
                                    top_grupos['grupo_id'] = [f"Grupo {i+1}" for i in range(len(top_grupos))]
                                    
                                    fig = px.bar(
                                        top_grupos,
                                        x='grupo_id',
                                        y='conteo',
                                        title="Top 10 Grupos con Más Duplicados",
                                        labels={'grupo_id': 'Grupo', 'conteo': 'Cantidad de Duplicados'}
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("No se pudieron identificar grupos específicos de duplicados.")
                        else:
                            st.success("No se encontraron duplicados en las columnas seleccionadas.")
        
        with tab4:
            st.write("### Detección de Outliers")
            
            # Selección de método y umbral
            col1, col2 = st.columns(2)
            
            with col1:
                metodo = st.selectbox(
                    "Método de detección",
                    options=['iqr', 'zscore', 'desviacion'],
                    format_func=lambda x: {
                        'iqr': 'Rango Intercuartil (IQR)',
                        'zscore': 'Z-Score',
                        'desviacion': 'Desviación Estándar'
                    }[x]
                )
            
            with col2:
                if metodo == 'iqr':
                    umbral = st.slider("Factor IQR", 0.5, 3.0, 1.5, 0.1)
                elif metodo == 'zscore':
                    umbral = st.slider("Umbral Z-Score", 1.0, 5.0, 3.0, 0.1)
                else:  # desviacion
                    umbral = st.slider("Factor de Desviación", 1.0, 5.0, 3.0, 0.1)
            
            with st.spinner("Detectando outliers..."):
                # Detectar outliers con el método seleccionado
                outliers = detectar_outliers(st.session_state.df, metodo=metodo, umbral=umbral)
                
                if not outliers:
                    st.info("No se pudieron detectar outliers en este dataset.")
                else:
                    # Crear DataFrame con resultados
                    resultados = []
                    
                    for columna, info in outliers.items():
                        resultados.append({
                            'columna': columna,
                            'cantidad_outliers': info['cantidad'],
                            'porcentaje': info['porcentaje']
                        })
                    
                    resultados_df = pd.DataFrame(resultados)
                    
                    if not resultados_df.empty:
                        # Ordenar por porcentaje descendente
                        resultados_df = resultados_df.sort_values('porcentaje', ascending=False)
                        
                        # Mostrar tabla
                        st.dataframe(
                            resultados_df,
                            use_container_width=True,
                            height=300
                        )
                        
                        # Gráfico de barras para columnas con más outliers
                        fig = px.bar(
                            resultados_df,
                            x='columna',
                            y='porcentaje',
                            title=f"Porcentaje de Outliers por Columna (Método: {metodo}, Umbral: {umbral})",
                            labels={'columna': 'Columna', 'porcentaje': 'Porcentaje de Outliers (%)'}
                        )
                        
                        fig.update_layout(xaxis={'categoryorder': 'total descending'})
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Seleccionar columna para visualizar outliers
                        columnas_con_outliers = resultados_df[resultados_df['cantidad_outliers'] > 0]['columna'].tolist()
                        
                        if columnas_con_outliers:
                            col_seleccionada = st.selectbox(
                                "Seleccionar columna para visualizar outliers",
                                options=columnas_con_outliers
                            )
                            
                            if col_seleccionada:
                                # Obtener límites del método seleccionado
                                info_columna = outliers[col_seleccionada]
                                metrica = info_columna['metrica']
                                
                                if metodo == 'iqr':
                                    limite_inf = metrica['limite_inferior']
                                    limite_sup = metrica['limite_superior']
                                    titulo = f"Box Plot con Outliers para {col_seleccionada} (IQR × {umbral})"
                                elif metodo == 'zscore':
                                    # Para z-score, calculamos límites equivalentes
                                    media = metrica['media']
                                    desv = metrica['desv_std']
                                    limite_inf = media - umbral * desv
                                    limite_sup = media + umbral * desv
                                    titulo = f"Box Plot con Outliers para {col_seleccionada} (Z-Score {umbral})"
                                else:  # desviacion
                                    limite_inf = metrica['limite_inferior']
                                    limite_sup = metrica['limite_superior']
                                    titulo = f"Box Plot con Outliers para {col_seleccionada} (Desv. Std × {umbral})"
                                
                                # Crear gráfico
                                fig = px.box(
                                    st.session_state.df,
                                    y=col_seleccionada,
                                    title=titulo
                                )
                                
                                # Añadir líneas para los límites
                                fig.add_shape(
                                    type="line",
                                    x0=-0.5,
                                    x1=0.5,
                                    y0=limite_inf,
                                    y1=limite_inf,
                                    line=dict(color="red", width=2, dash="dash")
                                )
                                
                                fig.add_shape(
                                    type="line",
                                    x0=-0.5,
                                    x1=0.5,
                                    y0=limite_sup,
                                    y1=limite_sup,
                                    line=dict(color="red", width=2, dash="dash")
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Histograma con outliers marcados
                                valores = st.session_state.df[col_seleccionada]
                                es_outlier = (valores < limite_inf) | (valores > limite_sup)
                                
                                fig = px.histogram(
                                    st.session_state.df,
                                    x=col_seleccionada,
                                    color=es_outlier,
                                    color_discrete_map={True: 'red', False: 'blue'},
                                    title=f"Distribución con Outliers Marcados para {col_seleccionada}",
                                    labels={True: 'Outlier', False: 'Normal'}
                                )
                                
                                # Añadir líneas para los límites
                                fig.add_vline(x=limite_inf, line_dash="dash", line_color="red")
                                fig.add_vline(x=limite_sup, line_dash="dash", line_color="red")
                                
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No se encontraron columnas con outliers según los criterios seleccionados.")
                    else:
                        st.info("No se encontraron outliers según los criterios seleccionados.")
        
        # Botón para volver al resumen
        if st.button("⬅️ Volver al Resumen", key="btn_volver"):
            st.session_state.paso_calidad = 0
            st.rerun()
