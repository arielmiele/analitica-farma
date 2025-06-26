"""
Página de Validación Cruzada - Analítica Farma
Análisis avanzado de overfitting/underfitting mediante curvas de aprendizaje y validación cruzada.

Esta página utiliza módulos especializados para una mejor organización del código:
- educativo: Contenido educativo sobre overfitting/underfitting
- configuracion: Selección de modelos y configuración de parámetros
- analisis: Ejecución del análisis de validación cruzada
- visualizacion: Presentación de resultados y gráficos interactivos
- recomendaciones: Sugerencias de mejora específicas para la industria farmacéutica
"""

import sys
import os
import streamlit as st

# Agregar el directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# Importar módulos modularizados
from src.ui.validacion_cruzada.educativo import (
    mostrar_introduccion,
    mostrar_importancia_validacion
)
from src.ui.validacion_cruzada.configuracion import (
    seleccionar_modelo,
    configurar_validacion
)
from src.ui.validacion_cruzada.analisis import (
    ejecutar_analisis_completo
)


def main():
    """Función principal de la página."""
    st.title("🔬 Validación Cruzada y Detección de Overfitting")
    
    # Mostrar introducción teórica
    mostrar_introduccion()
    
    # Mostrar importancia del análisis
    mostrar_importancia_validacion()
    
    st.markdown("---")
    
    # Selección de modelo
    modelo, resultados_benchmarking = seleccionar_modelo()
    
    if modelo and resultados_benchmarking:
        st.markdown("---")
        
        # Configuración de parámetros
        configuracion = configurar_validacion()
        
        st.markdown("---")
        
        # Ejecutar análisis
        ejecutar_analisis_completo(modelo, configuracion, resultados_benchmarking)
        
        # Navegación entre páginas
        st.markdown("---")
        st.subheader("⏩ Navegación")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔙 Volver a Evaluar Modelos", use_container_width=True):
                st.switch_page("pages/Machine Learning/06_Evaluar_Modelos.py")
        
        with col2:
            if st.button("👑 Ir a Recomendación de Modelo", use_container_width=True):
                st.switch_page("pages/Machine Learning/08_Recomendar_Modelo.py")
    
    # Footer informativo
    with st.expander("📚 Referencias y Documentación", expanded=False):
        st.markdown("""
        ### Recursos Adicionales
        
        - **Validación de Modelos en GMP**: ICH Q8, Q9, Q10
        - **Machine Learning en Farmacéutica**: FDA Guidance on Software as Medical Device
        - **Documentación Técnica**: 21 CFR Part 11 para sistemas electrónicos
        - **Mejores Prácticas**: GAMP 5 para sistemas computarizados
        
        ### Métricas de Evaluación
        
        - **Accuracy**: Proporción de predicciones correctas
        - **F1-Score**: Media armónica entre precisión y recall
        - **ROC-AUC**: Área bajo la curva ROC para clasificación
        - **Cross-Validation**: Técnica para estimar el rendimiento del modelo
        """)


if __name__ == "__main__":
    main()
