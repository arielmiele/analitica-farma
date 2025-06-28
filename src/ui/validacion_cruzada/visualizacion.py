"""
Módulo de visualización para Validación Cruzada - Analítica Farma
Contiene funciones para mostrar resultados, diagnósticos y visualizaciones interactivas.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def mostrar_resultados_analisis(resultados_curvas, modelo, resultados_benchmarking):
    """Muestra los resultados del análisis de validación."""
    
    # Verificar si hay error en los resultados
    if 'error' in resultados_curvas:
        st.error(f"❌ {resultados_curvas['error']}")
        if 'solucion' in resultados_curvas:
            st.info(f"💡 **Sugerencia:** {resultados_curvas['solucion']}")
        return
    
    # Análisis de diagnóstico
    diagnostico = resultados_curvas.get('diagnostico', {})
    metricas = resultados_curvas.get('metricas_principales', {})
    cv_results_completos = resultados_curvas.get('cv_results_completos', {})
    
    st.success("✅ Análisis de validación cruzada completado")
    
    # 1. Mostrar diagnóstico principal
    mostrar_diagnostico_principal(diagnostico, modelo['nombre'])
    
    # 2. Mostrar curvas de aprendizaje (nueva funcionalidad)
    mostrar_curvas_aprendizaje_interactivas(resultados_curvas, modelo['nombre'])
    
    # 3. Mostrar métricas y puntuaciones CV mejoradas
    mostrar_metricas_validacion_mejoradas(metricas, cv_results_completos, resultados_curvas.get('tipo_problema'))
    
    # 4. Mostrar información de datos
    mostrar_informacion_datos(resultados_curvas.get('datos_disponibles', {}))
    
    # 5. Mostrar recomendaciones
    from .recomendaciones import mostrar_recomendaciones_mejora
    mostrar_recomendaciones_mejora(diagnostico, modelo, resultados_benchmarking.get('tipo_problema', 'clasificacion'))


def mostrar_metricas_validacion(metricas, cv_scores, tipo_problema):
    """Muestra las métricas de validación y puntuaciones CV."""
    if not metricas and not cv_scores:
        st.info("📊 No hay métricas de validación disponibles")
        return
    
    st.subheader("📈 Métricas de Validación")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Métricas principales:**")
        for metrica, valor in metricas.items():
            if isinstance(valor, (int, float)):
                st.metric(metrica.replace('_', ' ').title(), f"{valor:.4f}")
            else:
                st.write(f"- **{metrica.replace('_', ' ').title()}:** {valor}")
    
    with col2:
        if cv_scores:
            st.write("**Validación Cruzada:**")
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            st.metric("Media CV", f"{cv_mean:.4f}")
            st.metric("Desviación Estándar CV", f"{cv_std:.4f}")
            
            # Mostrar distribución de scores
            with st.expander("📊 Distribución de puntuaciones CV"):
                st.bar_chart(cv_scores)


def mostrar_informacion_datos(datos_disponibles):
    """Muestra información sobre los datos utilizados."""
    if not datos_disponibles:
        return
    
    with st.expander("📊 Información de datos", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Forma de X_test:** {datos_disponibles.get('X_test_shape', 'N/A')}")
            st.write(f"**Forma de y_test:** {datos_disponibles.get('y_test_shape', 'N/A')}")
        
        with col2:
            st.write(f"**Total de filas:** {datos_disponibles.get('total_filas', 'N/A')}")
            st.write(f"**% datos de prueba:** {datos_disponibles.get('porcentaje_test', 'N/A')}%")


def mostrar_diagnostico_principal(diagnostico, nombre_modelo):
    """Muestra el diagnóstico principal del modelo."""
    st.subheader(f"🔍 Diagnóstico de {nombre_modelo}")
    
    # Obtener información del diagnóstico
    overfitting = diagnostico.get('overfitting', 'desconocido')
    underfitting = diagnostico.get('underfitting', 'desconocido')
    varianza_cv = diagnostico.get('varianza_cv', 0)
    mensaje = diagnostico.get('mensaje', 'Análisis no disponible')
    nivel_confianza = diagnostico.get('nivel_confianza', 'bajo')
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Mostrar diagnóstico principal
        if overfitting == 'posible':
            st.warning("⚠️ **POSIBLE OVERFITTING**")
            st.markdown("""
            El modelo podría estar sobreajustado a los datos de entrenamiento. 
            Esto significa que puede haber memorizado patrones específicos que 
            no se generalizan bien a datos nuevos.
            """)
        elif underfitting == 'posible':
            st.warning("⚠️ **POSIBLE UNDERFITTING**")
            st.markdown("""
            El modelo parece ser demasiado simple para capturar los patrones 
            en los datos. Considere aumentar la complejidad del modelo.
            """)
        elif overfitting == 'improbable' and underfitting == 'improbable':
            st.success("✅ **MODELO BALANCEADO**")
            st.markdown("""
            El modelo muestra un comportamiento equilibrado sin signos evidentes 
            de overfitting o underfitting.
            """)
        else:
            st.info("📊 **ANÁLISIS INCOMPLETO**")
            st.markdown("""
            No hay suficiente información para determinar con certeza el 
            comportamiento del modelo.
            """)
        
        # Mostrar mensaje detallado
        st.write(f"**Análisis:** {mensaje}")
    
    with col2:
        st.metric("Varianza CV", f"{varianza_cv:.4f}")
        
        # Interpretación de la varianza
        if varianza_cv > 0.1:
            st.write("🔴 Alta varianza")
        elif varianza_cv < 0.03:
            st.write("🟢 Baja varianza")
        else:
            st.write("🟡 Varianza normal")
    
    with col3:
        st.metric("Confianza", nivel_confianza.title())
        
        # Código de colores para confianza
        if nivel_confianza == 'alto':
            st.write("🟢 Diagnóstico fiable")
        elif nivel_confianza == 'medio':
            st.write("🟡 Diagnóstico parcial")
        else:
            st.write("🔴 Diagnóstico limitado")
    
    # Mostrar información adicional en expander
    with st.expander("ℹ️ Detalles técnicos", expanded=False):
        st.write("**Estado del diagnóstico:**")
        st.write(f"- Overfitting: {overfitting}")
        st.write(f"- Underfitting: {underfitting}")
        st.write(f"- Varianza en validación cruzada: {varianza_cv:.6f}")
        st.write(f"- Nivel de confianza: {nivel_confianza}")
        
        if varianza_cv > 0:
            st.write("**Interpretación de varianza:**")
            if varianza_cv > 0.1:
                st.write("La alta varianza sugiere que el modelo es inconsistente entre diferentes subconjuntos de datos.")
            elif varianza_cv < 0.03:
                st.write("La baja varianza indica que el modelo es consistente entre diferentes subconjuntos de datos.")
            else:
                st.write("La varianza está en un rango normal, indicando un comportamiento estable del modelo.")


def mostrar_curvas_aprendizaje_interactivas(resultados_curvas, nombre_modelo):
    """Muestra las curvas de aprendizaje con visualización interactiva."""
    st.subheader("📈 Análisis de Curvas de Aprendizaje")
    
    # Verificar si hay curvas de aprendizaje reales disponibles
    learning_curves = resultados_curvas.get('learning_curves', {})
    cv_scores = resultados_curvas.get('cv_scores', [])
    
    if learning_curves and not learning_curves.get('error'):
        st.success("✅ Curvas de aprendizaje generadas con scikit-learn")
        
        # Mostrar métricas principales de las curvas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            final_gap = learning_curves.get('final_gap', 0)
            st.metric("Gap Final", f"{final_gap:.4f}")
            if final_gap > 0.1:
                st.write("🔴 Alto overfitting")
            elif final_gap < 0.03:
                st.write("🟢 Buen ajuste")
            else:
                st.write("🟡 Overfitting moderado")
        
        with col2:
            max_gap = learning_curves.get('max_gap', 0)
            st.metric("Gap Máximo", f"{max_gap:.4f}")
        
        with col3:
            gap_trend = learning_curves.get('gap_trend', 'estable')
            st.metric("Tendencia", gap_trend.title())
        
        with col4:
            scoring_metric = learning_curves.get('scoring_metric', 'N/A')
            st.metric("Métrica", scoring_metric.upper())
        
        # Crear gráfico de curvas de aprendizaje
        crear_grafico_curvas_aprendizaje(learning_curves, nombre_modelo)
        
        # Interpretación del gap de overfitting
        mostrar_interpretacion_gap(learning_curves)
        
    elif cv_scores:
        st.info("📊 Análisis basado en validación cruzada disponible")
        
        # Mostrar estadísticas de CV
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Media CV", f"{np.mean(cv_scores):.4f}")
        with col2:
            st.metric("Desv. Estándar CV", f"{np.std(cv_scores):.4f}")
        with col3:
            st.metric("Rango CV", f"{np.max(cv_scores) - np.min(cv_scores):.4f}")
        
        # Gráfico de barras de puntuaciones CV
        st.bar_chart(cv_scores)
        
    else:
        st.warning("⚠️ No hay datos de validación cruzada disponibles para mostrar curvas de aprendizaje")
        st.info("💡 Ejecute el análisis de validación cruzada para ver las curvas completas")


def mostrar_interpretacion_detallada(interpretacion, diagnostico):
    """Muestra interpretación detallada del análisis."""
    if not interpretacion and not diagnostico:
        return
        
    st.subheader("📋 Interpretación Detallada")
    
    # Mostrar resumen del análisis
    mensaje = diagnostico.get('mensaje', 'No hay mensaje de diagnóstico disponible')
    st.info(f"**Resumen:** {mensaje}")
    
    # Mostrar recomendaciones si están disponibles
    if 'recomendaciones' in interpretacion:
        st.write("**Recomendaciones técnicas:**")
        for rec in interpretacion['recomendaciones']:
            st.write(f"• {rec}")
    
    # Detalles técnicos en expander
    with st.expander("🔍 Detalles técnicos avanzados", expanded=False):
        st.write("**Información del diagnóstico:**")
        for clave, valor in diagnostico.items():
            if clave != 'mensaje':
                st.write(f"- **{clave.replace('_', ' ').title()}:** {valor}")


def crear_grafico_distribucion_cv(cv_scores, nombre_modelo):
    """Crea un gráfico de distribución de puntuaciones CV."""
    if not cv_scores:
        return None
    
    fig = go.Figure()
    
    # Agregar histograma
    fig.add_trace(go.Histogram(
        x=cv_scores,
        nbinsx=min(10, len(cv_scores)),
        name='Distribución CV',
        marker_color='lightblue',
        opacity=0.7
    ))
    
    # Línea de la media
    media = np.mean(cv_scores)
    fig.add_vline(
        x=media, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Media: {media:.3f}"
    )
    
    fig.update_layout(
        title=f"Distribución de Puntuaciones CV - {nombre_modelo}",
        xaxis_title="Puntuación",
        yaxis_title="Frecuencia",
        showlegend=False
    )
    
    return fig


def mostrar_comparacion_modelos(resultados_comparacion):
    """Muestra comparación entre múltiples modelos."""
    if not resultados_comparacion:
        return
        
    st.subheader("⚖️ Comparación de Modelos")
    
    # Crear tabla comparativa
    df_comparacion = pd.DataFrame(resultados_comparacion)
    st.dataframe(df_comparacion, use_container_width=True)
    
    # Gráfico de comparación
    if len(df_comparacion) > 1:
        metric_cols = [col for col in df_comparacion.columns if col not in ['Modelo', 'Tipo']]
        
        for metric in metric_cols:
            if df_comparacion[metric].dtype in ['float64', 'int64']:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=df_comparacion['Modelo'],
                    y=df_comparacion[metric],
                    name=metric,
                    marker_color='lightgreen'
                ))
                
                fig.update_layout(
                    title=f"Comparación: {metric}",
                    xaxis_title="Modelo",
                    yaxis_title=metric,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)


def crear_grafico_curvas_aprendizaje(learning_curves, nombre_modelo):
    """Crea un gráfico interactivo de curvas de aprendizaje."""
    # Extraer datos de las curvas
    train_sizes = learning_curves.get('train_sizes', [])
    train_scores_mean = learning_curves.get('train_scores_mean', [])
    train_scores_std = learning_curves.get('train_scores_std', [])
    validation_scores_mean = learning_curves.get('validation_scores_mean', [])
    validation_scores_std = learning_curves.get('validation_scores_std', [])
    overfitting_gap = learning_curves.get('overfitting_gap', [])
    
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        if not train_sizes or not train_scores_mean:
            st.warning("⚠️ Datos de curvas de aprendizaje incompletos")
            return
        
        # Crear subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                '<b>Curvas de Aprendizaje</b><br><span style="font-size:12px; color:gray;">Entrenamiento vs Validación</span>',
                '<b>Gap de Overfitting (Entrenamiento - Validación)</b>'
            ),
            vertical_spacing=0.18  # Aumenta el espacio vertical entre los gráficos
        )
        
        # Gráfico 1: Curvas de aprendizaje
        # Curva de entrenamiento
        fig.add_trace(
            go.Scatter(
                x=train_sizes,
                y=train_scores_mean,
                mode='lines+markers',
                name='Entrenamiento',
                line=dict(color='blue', width=2),
                marker=dict(size=6),
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Banda de confianza entrenamiento
        if train_scores_std:
            train_upper = [m + s for m, s in zip(train_scores_mean, train_scores_std)]
            train_lower = [m - s for m, s in zip(train_scores_mean, train_scores_std)]
            
            fig.add_trace(
                go.Scatter(
                    x=train_sizes + train_sizes[::-1],
                    y=train_upper + train_lower[::-1],
                    fill='toself',
                    fillcolor='rgba(0,100,200,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False,
                    name='Banda entrenamiento'
                ),
                row=1, col=1
            )
        
        # Curva de validación
        fig.add_trace(
            go.Scatter(
                x=train_sizes,
                y=validation_scores_mean,
                mode='lines+markers',
                name='Validación',
                line=dict(color='red', width=2),
                marker=dict(size=6),
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Banda de confianza validación
        if validation_scores_std:
            val_upper = [m + s for m, s in zip(validation_scores_mean, validation_scores_std)]
            val_lower = [m - s for m, s in zip(validation_scores_mean, validation_scores_std)]
            
            fig.add_trace(
                go.Scatter(
                    x=train_sizes + train_sizes[::-1],
                    y=val_upper + val_lower[::-1],
                    fill='toself',
                    fillcolor='rgba(200,50,50,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False,
                    name='Banda validación'
                ),
                row=1, col=1
            )
        
        # Gráfico 2: Gap de overfitting
        if overfitting_gap:
            fig.add_trace(
                go.Scatter(
                    x=train_sizes,
                    y=overfitting_gap,
                    mode='lines+markers',
                    name='Gap Overfitting',
                    line=dict(color='orange', width=2),
                    marker=dict(size=6),
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # Línea de referencia en 0.1 (umbral de overfitting)
            fig.add_trace(
                go.Scatter(
                    x=[min(train_sizes), max(train_sizes)],
                    y=[0.1, 0.1],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Umbral overfitting (0.1)',
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Configurar layout
        fig.update_layout(
            title=f"Análisis de Curvas de Aprendizaje - {nombre_modelo}",
            height=700,
            showlegend=True,
            template="plotly_white"
        )
        
        # Etiquetas de ejes
        fig.update_xaxes(title_text="Tamaño del conjunto de entrenamiento", row=1, col=1)
        fig.update_xaxes(title_text="Tamaño del conjunto de entrenamiento", row=2, col=1)
        fig.update_yaxes(title_text="Puntuación", row=1, col=1)
        fig.update_yaxes(title_text="Gap (Train - Val)", row=2, col=1)
        
        # Mostrar gráfico
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error al crear gráfico de curvas de aprendizaje: {str(e)}")
        st.info("💡 Mostrando datos tabulares como alternativa")
        
        # Mostrar datos en formato tabular como fallback
        if train_sizes and train_scores_mean and validation_scores_mean:
            df_curves = pd.DataFrame({
                'Tamaño_Entrenamiento': train_sizes,
                'Score_Entrenamiento': train_scores_mean,
                'Score_Validacion': validation_scores_mean,
                'Gap_Overfitting': overfitting_gap if overfitting_gap else [0] * len(train_sizes)
            })
            st.dataframe(df_curves, use_container_width=True)


def mostrar_interpretacion_gap(learning_curves):
    """Muestra interpretación detallada del gap de overfitting."""
    st.subheader("🔍 Interpretación del Gap de Overfitting")
    
    final_gap = learning_curves.get('final_gap', 0)
    max_gap = learning_curves.get('max_gap', 0)
    gap_trend = learning_curves.get('gap_trend', 'estable')
    
    # Interpretación principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if final_gap > 0.15:
            st.error("🚨 **OVERFITTING SEVERO DETECTADO**")
            st.markdown("""
            **Diagnóstico:** El modelo muestra signos claros de sobreajuste.
            
            **Implicaciones:**
            - El modelo memoriza los datos de entrenamiento en lugar de aprender patrones generalizables
            - El rendimiento en datos nuevos será significativamente menor
            - Existe una brecha considerable entre el rendimiento de entrenamiento y validación
            """)
            
        elif final_gap > 0.1:
            st.warning("⚠️ **OVERFITTING MODERADO**")
            st.markdown("""
            **Diagnóstico:** El modelo muestra tendencias de sobreajuste que requieren atención.
            
            **Implicaciones:**
            - Hay espacio para mejorar la generalización del modelo
            - El rendimiento en datos nuevos podría ser sub-óptimo
            - Se recomienda aplicar técnicas de regularización
            """)
            
        elif final_gap > 0.05:
            st.info("ℹ️ **AJUSTE NORMAL**")
            st.markdown("""
            **Diagnóstico:** El modelo muestra un comportamiento normal con un gap mínimo aceptable.
            
            **Implicaciones:**
            - El modelo generaliza razonablemente bien
            - La diferencia entre entrenamiento y validación está en un rango aceptable
            - El modelo está funcionando como se espera
            """)
            
        else:
            st.success("✅ **EXCELENTE GENERALIZACIÓN**")
            st.markdown("""
            **Diagnóstico:** El modelo muestra una excelente capacidad de generalización.
            
            **Implicaciones:**
            - Muy poca diferencia entre rendimiento de entrenamiento y validación
            - El modelo debería funcionar bien en datos nuevos
            - Posible candidato para el modelo final
            """)
    
    with col2:
        st.metric("Gap Final", f"{final_gap:.4f}")
        st.metric("Gap Máximo", f"{max_gap:.4f}")
        st.metric("Tendencia", gap_trend.title())
        
        # Medidor visual del gap
        if final_gap > 0.15:
            st.markdown("🔴🔴🔴🔴🔴")
        elif final_gap > 0.1:
            st.markdown("🟠🟠🟠🟠⚪")
        elif final_gap > 0.05:
            st.markdown("🟡🟡🟡⚪⚪")
        else:
            st.markdown("🟢🟢🟢🟢🟢")
    
    # Recomendaciones específicas basadas en el gap
    with st.expander("💡 Recomendaciones técnicas", expanded=False):
        if final_gap > 0.1:
            st.markdown("""
            **Estrategias para reducir overfitting:**
            
            1. **Regularización:**
               - Aumentar parámetros de regularización (L1, L2)
               - Usar dropout en redes neuronales
               
            2. **Datos:**
               - Aumentar el tamaño del conjunto de datos
               - Aplicar técnicas de data augmentation
               
            3. **Modelo:**
               - Reducir la complejidad del modelo
               - Usar early stopping durante el entrenamiento
               
            4. **Validación:**
               - Implementar validación cruzada más robusta
               - Usar ensemble methods
            """)
        else:
            st.markdown("""
            **El modelo muestra buen comportamiento. Consideraciones adicionales:**
            
            1. **Optimización:**
               - Evaluar si se puede mejorar el rendimiento general
               - Considerar técnicas de ensemble
               
            2. **Validación:**
               - Probar en un conjunto de datos completamente independiente
               - Validar en datos de diferentes períodos de tiempo
               
            3. **Monitoreo:**
               - Establecer alertas para detectar drift del modelo
               - Monitorear rendimiento en producción
            """)


def mostrar_metricas_validacion_mejoradas(metricas, cv_results_completos, tipo_problema):
    """Muestra las métricas de validación mejoradas con resultados completos de CV."""
    if not metricas and not cv_results_completos:
        st.info("📊 No hay métricas de validación disponibles")
        return
    
    st.subheader("📈 Métricas de Validación Detalladas")
    
    # Métricas principales en columnas
    col1, col2, col3 = st.columns(3)
    
    # Métricas originales del modelo
    with col1:
        st.write("**Métricas del Modelo:**")
        for metrica, valor in metricas.items():
            if isinstance(valor, (int, float)):
                st.metric(metrica.replace('_', ' ').title(), f"{valor:.4f}")
            else:
                st.write(f"- **{metrica.replace('_', ' ').title()}:** {valor}")
    
    # Resultados de validación cruzada
    with col2:
        if cv_results_completos:
            st.write("**Validación Cruzada:**")
            mean_score = cv_results_completos.get('mean_score', 0)
            std_score = cv_results_completos.get('std_score', 0)
            cv_folds = cv_results_completos.get('cv_folds', 5)
            scoring_metric = cv_results_completos.get('scoring_metric', 'N/A')
            
            st.metric("Media CV", f"{mean_score:.4f}")
            st.metric("Desviación CV", f"{std_score:.4f}")
            st.write(f"- **Folds:** {cv_folds}")
            st.write(f"- **Métrica:** {scoring_metric.upper()}")
    
    # Estadísticas adicionales
    with col3:
        if cv_results_completos:
            st.write("**Estadísticas CV:**")
            min_score = cv_results_completos.get('min_score', 0)
            max_score = cv_results_completos.get('max_score', 0)
            variance = cv_results_completos.get('variance', 0)
            
            st.metric("Mínimo CV", f"{min_score:.4f}")
            st.metric("Máximo CV", f"{max_score:.4f}")
            st.metric("Varianza CV", f"{variance:.6f}")
    
    # Gráfico de distribución de puntuaciones CV
    if cv_results_completos and cv_results_completos.get('cv_scores'):
        cv_scores = cv_results_completos['cv_scores']
        
        with st.expander("📊 Distribución de Puntuaciones CV", expanded=False):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Gráfico de barras
                df_cv = pd.DataFrame({
                    'Fold': [f'Fold {i+1}' for i in range(len(cv_scores))],
                    'Puntuación': cv_scores
                })
                st.bar_chart(df_cv.set_index('Fold'))
            
            with col2:
                # Estadísticas resumidas
                st.write("**Resumen estadístico:**")
                st.write(f"• **Media:** {np.mean(cv_scores):.4f}")
                st.write(f"• **Mediana:** {np.median(cv_scores):.4f}")
                st.write(f"• **Desv. Std:** {np.std(cv_scores):.4f}")
                st.write(f"• **Rango:** {np.max(cv_scores) - np.min(cv_scores):.4f}")
                
                # Interpretación de consistencia
                if np.std(cv_scores) < 0.02:
                    st.success("🟢 Muy consistente")
                elif np.std(cv_scores) < 0.05:
                    st.info("🟡 Consistencia normal")
                else:
                    st.warning("🔴 Inconsistente")
