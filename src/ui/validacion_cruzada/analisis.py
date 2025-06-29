"""
Módulo de análisis UI para Validación Cruzada - Analítica Farma
Contiene funciones de UI para ejecutar y mostrar análisis de validación cruzada.
La lógica de cálculos está en src.modelos.validacion_cruzada
"""

import streamlit as st
from src.state.session_manager import SessionManager
from src.modelos.validacion_cruzada import (
    verificar_datos_para_validacion,
    generar_analisis_completo_validacion_cruzada
)
from src.audit.logger import log_audit

# Instancias de servicios
session = SessionManager()


def verificar_datos_disponibles(resultados_benchmarking, id_sesion: str, usuario: str):
    """
    Verifica si los datos necesarios están disponibles para la validación cruzada.
    Delegando a la lógica de negocio.
    """
    return verificar_datos_para_validacion(resultados_benchmarking, id_sesion, usuario)


def ejecutar_analisis_completo(modelo, configuracion, resultados_benchmarking, id_sesion: str, usuario: str):
    """Ejecuta el análisis completo de validación cruzada."""
    st.subheader("🚀 Análisis de Validación Cruzada")
    
    # Verificar datos básicos necesarios
    datos_disponibles = verificar_datos_disponibles(resultados_benchmarking, id_sesion, usuario)
    if not datos_disponibles['datos_ok']:
        st.error(f"❌ {datos_disponibles['mensaje']}")
        st.info("💡 " + datos_disponibles['solucion'])
        return
    
    # Verificar modelo objeto
    if 'modelo_objeto' not in modelo:
        st.error("❌ Objeto del modelo no disponible para re-entrenamiento")
        st.info("💡 Intente ejecutar un nuevo benchmarking o cargar uno con modelos serializados")
        return
    
    # Mostrar información de datos disponibles
    with st.expander("📊 Información de datos disponibles", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Variable objetivo:** {resultados_benchmarking.get('variable_objetivo', 'N/A')}")
            st.write(f"**Total de filas:** {resultados_benchmarking.get('total_filas', 'N/A')}")
            st.write(f"**Tipo de problema:** {resultados_benchmarking.get('tipo_problema', 'N/A')}")
        with col2:
            st.write(f"**% datos de prueba:** {resultados_benchmarking.get('porcentaje_test', 'N/A')}%")
            st.write(f"**Columnas originales:** {len(resultados_benchmarking.get('columnas_originales', []))}")
            if resultados_benchmarking.get('tiene_label_encoder'):
                st.write("**Codificación:** Aplicada (clasificación)")
    
    # Botón para ejecutar análisis
    if st.button("🔬 **Ejecutar Análisis Completo de Validación**", type="primary"):
        realizar_analisis_validacion(modelo, configuracion, resultados_benchmarking, id_sesion, usuario)


def realizar_analisis_validacion(modelo, config, resultados_benchmarking, id_sesion: str, usuario: str):
    """Realiza el análisis completo de validación usando la lógica de negocio."""
    try:
        with st.spinner("🔄 Ejecutando validación cruzada y generando curvas de aprendizaje..."):
            
            # Usar la función de lógica de negocio del módulo de modelos
            resultados_curvas = generar_analisis_completo_validacion_cruzada(modelo, resultados_benchmarking, id_sesion, usuario)
            
            if 'error' in resultados_curvas:
                st.error(f"❌ Error al generar curvas de aprendizaje: {resultados_curvas['error']}")
                if 'solucion' in resultados_curvas:
                    st.info(f"💡 **Sugerencia:** {resultados_curvas['solucion']}")
                return
            
            # Agregar las recomendaciones al contexto del modelo para que estén disponibles
            modelo_con_recomendaciones = modelo.copy()
            modelo_con_recomendaciones['recomendaciones'] = resultados_curvas.get('recomendaciones', [])
            
            # Guardar resultados en sesión para uso posterior
            session.guardar_estado(f"validacion_cv_{modelo['nombre']}", resultados_curvas)
            
            # Importar dinámicamente la función de visualización para evitar dependencias circulares
            from .visualizacion import mostrar_resultados_analisis
            
            # Mostrar resultados
            mostrar_resultados_analisis(resultados_curvas, modelo_con_recomendaciones, resultados_benchmarking)
            
    except Exception as e:
        st.error(f"❌ Error durante el análisis: {str(e)}")
        log_audit(id_sesion, usuario, "ERROR_VALIDACION_CRUZADA", "validacion_cruzada_ui", str(e))
