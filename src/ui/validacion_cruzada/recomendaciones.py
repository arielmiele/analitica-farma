"""
Módulo de recomendaciones para Validación Cruzada - Analítica Farma
Contiene funciones para mostrar recomendaciones de mejora específicas para modelos y la industria farmacéutica.
"""

import streamlit as st


def mostrar_recomendaciones_mejora(diagnostico, modelo, tipo_problema):
    """Muestra recomendaciones específicas para mejorar el modelo."""
    st.subheader("💡 Recomendaciones para Mejora")
    
    # Obtener recomendaciones del resultado del análisis (desde el evaluador)
    recomendaciones_del_analisis = modelo.get('recomendaciones', [])
    
    # También obtener las del diagnóstico si están disponibles
    recomendaciones_diagnostico = diagnostico.get('recomendaciones', [])
    
    # Combinar todas las recomendaciones disponibles
    todas_recomendaciones = recomendaciones_del_analisis + recomendaciones_diagnostico
    
    # Determinar el tipo de problema basado en el diagnóstico
    overfitting = diagnostico.get('overfitting', 'desconocido')
    underfitting = diagnostico.get('underfitting', 'desconocido')
    
    # Mostrar recomendaciones en categorías
    if overfitting == 'posible':
        st.markdown("### 🔥 Estrategias para Reducir Overfitting")
        tipo_diag = 'overfitting'
    elif underfitting == 'posible':
        st.markdown("### 📈 Estrategias para Reducir Underfitting")
        tipo_diag = 'underfitting'
    else:
        st.markdown("### ✨ Estrategias para Optimización Adicional")
        tipo_diag = 'balanceado'
    
    # Mostrar recomendaciones básicas si están disponibles
    if todas_recomendaciones:
        st.markdown("#### Recomendaciones específicas:")
        for i, recomendacion in enumerate(todas_recomendaciones, 1):
            with st.expander(f"💡 Recomendación {i}", expanded=i <= 3):
                st.markdown(recomendacion)
    else:
        st.info("📋 No hay recomendaciones específicas disponibles")
        # Mostrar recomendaciones genéricas basadas en el diagnóstico
        mostrar_recomendaciones_genericas(tipo_diag)
    
    # Recomendaciones adicionales para la industria farmacéutica
    mostrar_recomendaciones_industria(tipo_diag)


def mostrar_recomendaciones_genericas(tipo_diag):
    """Muestra recomendaciones genéricas basadas en el tipo de diagnóstico."""
    if tipo_diag == 'overfitting':
        recomendaciones_genericas = [
            "🔄 Considere usar regularización (L1/L2)",
            "📊 Aumente el tamaño del dataset de entrenamiento", 
            "🌳 Reduzca la complejidad del modelo",
            "✂️ Aplique técnicas de feature selection"
        ]
    elif tipo_diag == 'underfitting':
        recomendaciones_genericas = [
            "🔧 Aumente la complejidad del modelo",
            "🎯 Agregue más características relevantes",
            "⚙️ Ajuste los hiperparámetros", 
            "🔍 Verifique la calidad de los datos"
        ]
    else:
        recomendaciones_genericas = [
            "✅ El modelo muestra un comportamiento balanceado",
            "🔍 Considere realizar ajuste fino de hiperparámetros",
            "📈 Monitoree el rendimiento en producción",
            "🎯 Evalúe la adición de características adicionales"
        ]
    
    st.markdown("#### Recomendaciones generales:")
    for i, rec in enumerate(recomendaciones_genericas, 1):
        with st.expander(f"🎯 Recomendación general {i}", expanded=False):
            st.markdown(rec)


def mostrar_recomendaciones_industria(tipo_problema):
    """Muestra recomendaciones específicas para la industria farmacéutica."""
    with st.expander("🏭 Consideraciones para la Industria Farmacéutica", expanded=False):
        st.markdown("""
        ### Aplicación en Procesos Biotecnológicos
        
        #### Para Overfitting:
        - **📋 Documentación**: Registre todas las correcciones en el batch record
        - **🔄 Validación cruzada**: Implemente validación con datos de múltiples lotes
        - **📊 Monitoreo continuo**: Establezca alertas para drift del modelo en producción
        - **👥 Revisión por pares**: Involucre a QA en la validación del modelo
        
        #### Para Underfitting:
        - **🔬 Revisión de variables**: Incluya más CPPs (Critical Process Parameters)
        - **📈 Aumento de datos**: Considere datos históricos de sitios similares
        - **🧪 Experimentos dirigidos**: Planifique experimentos para llenar gaps de datos
        - **🎯 Refinamiento de objetivos**: Revise si los KPIs están bien definidos
        
        #### Para Modelos Balanceados:
        - **✅ Validación final**: Proceda con validación en lotes piloto
        - **📝 Documentación GMP**: Prepare documentación para transferencia
        - **🔍 Monitoreo de performance**: Implemente sistema de seguimiento continuo
        - **🎓 Entrenamiento**: Capacite al personal en el uso del modelo
        """)
