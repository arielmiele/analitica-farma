"""
Módulo de visualización para Validación Cruzada - Analítica Farma
Contiene funciones para mostrar resultados, diagnósticos y visualizaciones interactivas.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def mostrar_resultados_analisis(resultados_curvas, modelo, resultados_benchmarking):
    """Muestra los resultados del análisis de validación."""
    
    # Análisis de diagnóstico
    diagnostico = resultados_curvas.get('diagnostico', {})
    interpretacion = resultados_curvas.get('interpretacion', {})
    
    # 1. Mostrar diagnóstico principal
    mostrar_diagnostico_principal(diagnostico, modelo['nombre'])
    
    # 2. Mostrar curvas de aprendizaje
    mostrar_curvas_aprendizaje_interactivas(resultados_curvas, modelo['nombre'])
    
    # 3. Mostrar interpretación detallada
    mostrar_interpretacion_detallada(interpretacion, diagnostico)
    
    # 4. Mostrar recomendaciones
    from .recomendaciones import mostrar_recomendaciones_mejora
    mostrar_recomendaciones_mejora(diagnostico, modelo, resultados_benchmarking.get('tipo_problema', 'clasificacion'))


def mostrar_diagnostico_principal(diagnostico, nombre_modelo):
    """Muestra el diagnóstico principal del modelo."""
    st.subheader(f"🔍 Diagnóstico de {nombre_modelo}")
    
    tipo_problema = diagnostico.get('tipo', 'balanceado')
    severidad = diagnostico.get('severidad', 0.0)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if tipo_problema == 'overfitting':
            st.error("🚨 **OVERFITTING DETECTADO**")
            st.markdown("""
            El modelo muestra **sobreajuste** a los datos de entrenamiento. 
            Esto significa que ha memorizado patrones específicos del conjunto 
            de entrenamiento que no se generalizan bien a datos nuevos.
            """)
        elif tipo_problema == 'underfitting':
            st.warning("⚠️ **UNDERFITTING DETECTADO**")
            st.markdown("""
            El modelo muestra **subajuste** y no está capturando adecuadamente 
            los patrones en los datos. El modelo es demasiado simple para 
            el problema que está tratando de resolver.
            """)
        else:
            st.success("✅ **MODELO BALANCEADO**")
            st.markdown("""
            El modelo muestra un **buen balance** entre sesgo y varianza. 
            Está generalizando adecuadamente sin sobreajustarse a los datos 
            de entrenamiento.
            """)
    
    with col2:
        st.metric(
            "**Severidad**", 
            f"{severidad:.2f}/1.0",
            help="Indica qué tan severo es el problema detectado"
        )
        
        # Mostrar gap train-test si está disponible
        if 'detalles' in diagnostico and 'gap_train_test' in diagnostico['detalles']:
            gap = diagnostico['detalles']['gap_train_test']
            st.metric(
                "**Gap Train-Test**", 
                f"{gap:.3f}",
                help="Diferencia entre rendimiento en entrenamiento y validación"
            )
    
    with col3:
        # Mostrar indicador visual de calidad
        if tipo_problema == 'overfitting':
            color = "🔴"
            nivel = "Crítico"
        elif tipo_problema == 'underfitting':
            color = "🟡"
            nivel = "Moderado"
        else:
            color = "🟢"
            nivel = "Bueno"
        
        st.metric("**Estado**", f"{color} {nivel}")


def mostrar_curvas_aprendizaje_interactivas(resultados_curvas, nombre_modelo):
    """Muestra las curvas de aprendizaje con visualización interactiva."""
    st.subheader("📈 Curvas de Aprendizaje")
    
    # Extraer datos
    train_sizes = np.array(resultados_curvas['train_sizes'])
    train_scores = np.array(resultados_curvas['train_scores'])
    test_scores = np.array(resultados_curvas['test_scores'])
    
    # Calcular medias y desviaciones estándar
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    # Crear gráfico interactivo con Plotly
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            'Curvas de Aprendizaje', 
            'Gap entre Entrenamiento y Validación'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Curva de entrenamiento
    fig.add_trace(
        go.Scatter(
            x=train_sizes,
            y=train_scores_mean,
            mode='lines+markers',
            name='Entrenamiento',
            line=dict(color='blue', width=2),
            error_y=dict(
                type='data',
                array=train_scores_std,
                visible=True,
                color='lightblue'
            ),
            hovertemplate='<b>Entrenamiento</b><br>' +
                         'Tamaño: %{x}<br>' +
                         'Score: %{y:.3f}<br>' +
                         '<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Curva de validación
    fig.add_trace(
        go.Scatter(
            x=train_sizes,
            y=test_scores_mean,
            mode='lines+markers',
            name='Validación',
            line=dict(color='red', width=2),
            error_y=dict(
                type='data',
                array=test_scores_std,
                visible=True,
                color='lightcoral'
            ),
            hovertemplate='<b>Validación</b><br>' +
                         'Tamaño: %{x}<br>' +
                         'Score: %{y:.3f}<br>' +
                         '<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Gap de overfitting
    gap = train_scores_mean - test_scores_mean
    fig.add_trace(
        go.Scatter(
            x=train_sizes,
            y=gap,
            mode='lines+markers',
            name='Gap (Overfitting)',
            line=dict(color='orange', width=2),
            fill='tonexty',
            fillcolor='rgba(255, 165, 0, 0.2)',
            hovertemplate='<b>Gap</b><br>' +
                         'Tamaño: %{x}<br>' +
                         'Gap: %{y:.3f}<br>' +
                         '<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Configurar layout
    fig.update_layout(
        title=f"Análisis de Curvas de Aprendizaje - {nombre_modelo}",
        height=500,
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Tamaño del Conjunto de Entrenamiento", row=1, col=1)
    fig.update_xaxes(title_text="Tamaño del Conjunto de Entrenamiento", row=1, col=2)
    fig.update_yaxes(title_text="Score de Validación", row=1, col=1)
    fig.update_yaxes(title_text="Diferencia (Train - Validation)", row=1, col=2)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Opción para descargar
    col1, col2 = st.columns([3, 1])
    with col2:
        # Convertir a imagen para descarga
        img_bytes = fig.to_image(format="png", width=1200, height=600, scale=2)
        st.download_button(
            label="📥 Descargar Gráfico",
            data=img_bytes,
            file_name=f"curvas_aprendizaje_{nombre_modelo}.png",
            mime="image/png"
        )


def mostrar_interpretacion_detallada(interpretacion, diagnostico):
    """Muestra la interpretación detallada de los resultados."""
    st.subheader("🧠 Interpretación de Resultados")
    
    # Métricas finales
    rendimiento = interpretacion.get('rendimiento_final', {})
    tendencia = interpretacion.get('tendencia', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Métricas Finales")
        
        final_train = rendimiento.get('train', 0)
        final_test = rendimiento.get('test', 0)
        final_gap = rendimiento.get('gap', 0)
        
        st.metric("Score Final (Entrenamiento)", f"{final_train:.4f}")
        st.metric("Score Final (Validación)", f"{final_test:.4f}")
        st.metric("Gap Final", f"{final_gap:.4f}")
    
    with col2:
        st.markdown("### 📈 Análisis de Tendencias")
        
        train_improving = tendencia.get('train_improving', False)
        test_improving = tendencia.get('test_improving', False)
        train_slope = tendencia.get('train_slope', 0)
        test_slope = tendencia.get('test_slope', 0)
        
        st.write(f"**Mejora en Entrenamiento:** {'✅ Sí' if train_improving else '❌ No'}")
        st.write(f"**Mejora en Validación:** {'✅ Sí' if test_improving else '❌ No'}")
        st.write(f"**Pendiente Train:** {train_slope:.4f}")
        st.write(f"**Pendiente Test:** {test_slope:.4f}")
    
    # Insights en lenguaje natural
    insights = interpretacion.get('insights', [])
    if insights:
        st.markdown("### 💡 Insights Automáticos")
        for i, insight in enumerate(insights, 1):
            st.info(f"**{i}.** {insight}")
