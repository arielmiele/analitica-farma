"""
Módulo de presentación de recomendaciones para Validación Cruzada - Analítica Farma
Contiene únicamente funciones de UI para mostrar recomendaciones generadas por la capa de modelos.
"""

import streamlit as st
from src.modelos.recomendador import generar_recomendaciones_completas


def mostrar_recomendaciones_mejora(diagnostico, modelo, tipo_problema):
    """Muestra recomendaciones específicas para mejorar el modelo."""
    st.subheader("💡 Recomendaciones para Mejora")
    
    # Generar todas las recomendaciones usando la capa de modelos
    recomendaciones_completas = generar_recomendaciones_completas(diagnostico, modelo, tipo_problema)
    
    # Verificar si hay error
    if 'error' in recomendaciones_completas:
        st.error(f"❌ {recomendaciones_completas['error']}")
        return
    
    # Obtener información del diagnóstico
    tipo_diag = recomendaciones_completas.get('tipo_diagnostico', 'balanceado')
    
    # Mostrar título según el tipo de diagnóstico
    if tipo_diag == 'overfitting':
        st.markdown("### 🔥 Estrategias para Reducir Overfitting")
    elif tipo_diag == 'underfitting':
        st.markdown("### 📈 Estrategias para Reducir Underfitting")
    else:
        st.markdown("### ✨ Estrategias para Optimización Adicional")
    
    # Mostrar recomendaciones específicas del análisis
    recomendaciones_especificas = recomendaciones_completas.get('recomendaciones_especificas', [])
    if recomendaciones_especificas:
        st.markdown("#### Recomendaciones específicas del análisis:")
        for i, recomendacion in enumerate(recomendaciones_especificas, 1):
            with st.expander(f"💡 Recomendación específica {i}", expanded=i <= 3):
                st.markdown(recomendacion)
    
    # Mostrar recomendaciones genéricas
    recomendaciones_genericas = recomendaciones_completas.get('recomendaciones_genericas', [])
    if recomendaciones_genericas:
        st.markdown("#### Recomendaciones generales:")
        for i, rec in enumerate(recomendaciones_genericas, 1):
            with st.expander(f"🎯 Recomendación general {i}", expanded=False):
                st.markdown(rec)
    
    # Mostrar recomendaciones para la industria farmacéutica
    mostrar_recomendaciones_industria_ui(recomendaciones_completas)


def mostrar_recomendaciones_industria_ui(recomendaciones_completas):
    """Presenta las recomendaciones específicas para la industria farmacéutica."""
    recomendaciones_industria = recomendaciones_completas.get('recomendaciones_industria', {})
    tipo_diag = recomendaciones_completas.get('tipo_diagnostico', 'general')
    
    with st.expander("🏭 Consideraciones para la Industria Farmacéutica", expanded=False):
        st.markdown("### Aplicación en Procesos Biotecnológicos")
        
        # Mostrar recomendaciones específicas del tipo de diagnóstico
        if tipo_diag in recomendaciones_industria:
            st.markdown(f"#### Para {tipo_diag.title()}:")
            recomendaciones_especificas = recomendaciones_industria[tipo_diag]
            for rec in recomendaciones_especificas:
                st.markdown(f"- {rec}")
        
        # Mostrar recomendaciones generales de la industria
        if 'general' in recomendaciones_industria:
            st.markdown("#### Consideraciones Generales:")
            recomendaciones_generales = recomendaciones_industria['general']
            for rec in recomendaciones_generales:
                st.markdown(f"- {rec}")
        
        # Información adicional para compliance
        st.markdown("""
        ---
        #### � Checklist de Compliance:
        - ✅ Documentación completa en batch records
        - ✅ Validación según estándares FDA/EMA  
        - ✅ Trazabilidad completa de cambios
        - ✅ Revisión y aprobación por QA
        - ✅ Plan de mantenimiento del modelo
        - ✅ Procedimientos de respaldo y recuperación
        """)
