"""
Módulo de análisis para Validación Cruzada - Analítica Farma
Contiene funciones para ejecutar y procesar análisis de validación cruzada.
"""

import streamlit as st
from src.modelos.evaluador import generar_curvas_aprendizaje
from src.state.session_manager import SessionManager
from src.audit.logger import Logger

# Instancias de servicios
session = SessionManager()
logger = Logger("Validacion_Cruzada")


def ejecutar_analisis_completo(modelo, configuracion, resultados_benchmarking):
    """Ejecuta el análisis completo de validación cruzada."""
    st.subheader("🚀 Análisis de Validación Cruzada")
    
    # Verificar disponibilidad de datos
    if not ('X_train' in resultados_benchmarking and 'y_train' in resultados_benchmarking):
        st.error("❌ Datos de entrenamiento no disponibles para este análisis")
        st.info("💡 Los datos de entrenamiento son necesarios para generar curvas de aprendizaje")
        return
    
    # Verificar modelo objeto
    if 'modelo_objeto' not in modelo:
        st.error("❌ Objeto del modelo no disponible para re-entrenamiento")
        st.info("💡 Intente ejecutar un nuevo benchmarking o cargar uno con modelos serializados")
        return
    
    # Botón para ejecutar análisis
    if st.button("🔬 **Ejecutar Análisis Completo de Validación**", type="primary"):
        realizar_analisis_validacion(modelo, configuracion, resultados_benchmarking)


def realizar_analisis_validacion(modelo, config, resultados_benchmarking):
    """Realiza el análisis completo de validación."""
    try:
        with st.spinner("🔄 Ejecutando validación cruzada y generando curvas de aprendizaje..."):
            
            # Generar curvas de aprendizaje
            resultados_curvas = generar_curvas_aprendizaje(
                id_benchmarking=int(resultados_benchmarking.get('id_benchmarking', 0)),
                nombre_modelo=str(modelo['nombre'])
            )
            
            if 'error' in resultados_curvas:
                st.error(f"❌ Error al generar curvas de aprendizaje: {resultados_curvas['error']}")
                if 'solucion' in resultados_curvas:
                    st.info(f"💡 **Sugerencia:** {resultados_curvas['solucion']}")
                return
            
            # Guardar resultados en sesión para uso posterior
            session.guardar_estado(f"validacion_cv_{modelo['nombre']}", resultados_curvas)
            
            # Importar dinámicamente la función de visualización para evitar dependencias circulares
            from .visualizacion import mostrar_resultados_analisis
            
            # Mostrar resultados
            mostrar_resultados_analisis(resultados_curvas, modelo, resultados_benchmarking)
            
    except Exception as e:
        st.error(f"❌ Error durante el análisis: {str(e)}")
        logger.log_evento(
            "ERROR_VALIDACION_CRUZADA", 
            f"Error en análisis de {modelo['nombre']}: {str(e)}", 
            "06_Validacion_Cruzada"
        )
