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
from src.audit.logger import log_audit


def main():
    """Función principal de la página."""
    st.title("🔬 Validación Cruzada y Detección de Overfitting")
    # Log de acceso a la página
    log_audit(
        usuario=st.session_state.get("usuario_id", "sistema"),
        accion="ACCESO_PAGINA",
        entidad="validacion_cruzada",
        id_entidad="",
        detalles="Acceso a la página de Validación Cruzada",
        id_sesion=st.session_state.get("id_sesion", "sin_sesion")
    )
    
    # Mostrar introducción teórica
    mostrar_introduccion()
    
    # Mostrar importancia del análisis
    mostrar_importancia_validacion()
    
    st.markdown("---")
    
    # Selección de modelo
    modelo, resultados_benchmarking = seleccionar_modelo()

    if modelo and resultados_benchmarking:
        st.markdown("---")
        
        # Explicación sobre la configuración de validación cruzada
        with st.expander("ℹ️ ¿Cómo configurar la validación cruzada?", expanded=True):
            st.markdown("""
            **¿Qué es la validación cruzada?**
            
            La validación cruzada permite estimar el rendimiento real de un modelo dividiendo los datos en varios subconjuntos ("folds") y repitiendo el entrenamiento y evaluación múltiples veces.
            
            **¿Cómo configurar los parámetros?**
            - **Número de folds para CV:** Define en cuántas partes se divide el dataset para la validación cruzada. Más folds (por ejemplo, 10) dan una estimación más robusta pero requieren más tiempo de cómputo.
            - **Puntos en curva de aprendizaje:** Determina cuántos tamaños de muestra se usarán para construir la curva de aprendizaje. Más puntos permiten ver mejor la evolución del rendimiento.
            - **Métrica de evaluación:** Selecciona la métrica principal para comparar el desempeño del modelo (accuracy, f1, precision, recall, roc_auc).
            - **Semilla aleatoria:** Fija el valor para garantizar resultados reproducibles.
            - **Procesamiento:** Permite elegir entre ejecución secuencial o paralela (usando todos los núcleos disponibles).
            
            > Ajuste estos parámetros según el tamaño de su dataset y el tiempo disponible. En datasets pequeños, usar más folds y más puntos puede ayudar a obtener una mejor estimación del rendimiento.
            """)
        
        # Configuración de parámetros
        configuracion = configurar_validacion()
        
        st.markdown("---")
        # Log de inicio de análisis de validación cruzada
        log_audit(
            usuario=st.session_state.get("usuario_id", "sistema"),
            accion="INICIO_ANALISIS_VALIDACION_CRUZADA",
            entidad="validacion_cruzada",
            id_entidad=str(modelo),
            detalles=f"Inicio de análisis de validación cruzada para modelo: {getattr(modelo, 'nombre', str(modelo))}",
            id_sesion=st.session_state.get("id_sesion", "sin_sesion")
        )
        # Ejecutar análisis
        ejecutar_analisis_completo(
            modelo,
            configuracion,
            resultados_benchmarking,
            id_sesion=st.session_state.get("id_sesion", "sin_sesion"),
            usuario=st.session_state.get("usuario_id", "sistema")
        )
        
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
    
if __name__ == "__main__":
    main()
