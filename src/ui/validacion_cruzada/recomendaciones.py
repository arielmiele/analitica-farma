"""
Módulo de recomendaciones para Validación Cruzada - Analítica Farma
Contiene funciones para mostrar recomendaciones de mejora específicas para modelos y la industria farmacéutica.
"""

import streamlit as st


def mostrar_recomendaciones_mejora(diagnostico, modelo, tipo_problema):
    """Muestra recomendaciones específicas para mejorar el modelo."""
    st.subheader("💡 Recomendaciones para Mejora")
    
    # Obtener recomendaciones básicas del diagnóstico
    recomendaciones_basicas = diagnostico.get('recomendaciones', [])
    
    # Mostrar recomendaciones en categorías
    tipo_diag = diagnostico.get('tipo', 'balanceado')
    
    if tipo_diag == 'overfitting':
        st.markdown("### 🔥 Estrategias para Reducir Overfitting")
    elif tipo_diag == 'underfitting':
        st.markdown("### 📈 Estrategias para Reducir Underfitting")
    else:
        st.markdown("### ✨ Estrategias para Optimización Adicional")
    
    # Mostrar recomendaciones básicas
    for i, recomendacion in enumerate(recomendaciones_basicas, 1):
        with st.expander(f"💡 Recomendación {i}", expanded=i <= 3):
            st.markdown(recomendacion)
    
    # Recomendaciones adicionales para la industria farmacéutica
    mostrar_recomendaciones_industria(tipo_diag)


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
