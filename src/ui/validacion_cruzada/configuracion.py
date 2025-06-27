"""
Módulo de configuración para Validación Cruzada - Analítica Farma
Contiene funciones para selección de modelos y configuración de parámetros de validación cruzada.
"""

import streamlit as st
from src.state.session_manager import SessionManager

# Instancia del gestor de sesión
session = SessionManager()


def seleccionar_modelo():
    """Permite seleccionar un modelo para análisis."""
    resultados_benchmarking = session.obtener_estado("resultados_benchmarking")
    
    if not resultados_benchmarking or not resultados_benchmarking.get('modelos_exitosos'):
        st.warning("⚠️ No hay modelos entrenados disponibles para análisis.")
        st.info("👈 Vaya a la sección **'Entrenar Modelos'** para ejecutar un benchmarking primero.")
        
        with st.expander("📋 Requisitos para usar esta página", expanded=True):
            st.markdown("""
            ### Para utilizar esta funcionalidad necesita:
            
            1. **✅ Datos cargados**: Un dataset CSV válido cargado en la aplicación
            2. **✅ Modelos entrenados**: Al menos un benchmarking ejecutado exitosamente
            3. **✅ Objetos modelo**: Los modelos deben estar disponibles para re-entrenamiento
            
            ### Una vez cumplidos los requisitos, podrá:
            
            - 📈 **Generar curvas de aprendizaje** interactivas
            - 🔍 **Detectar automáticamente** problemas de overfitting/underfitting
            - 💡 **Recibir recomendaciones** personalizadas para mejorar sus modelos
            - 📊 **Comparar** el comportamiento de diferentes algoritmos
            - 📝 **Documentar** los resultados para cumplimiento regulatorio
            """)
        
        return None, None
    
    # Mostrar información del benchmarking actual
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Selección de Modelo para Análisis")
        
        modelos_disponibles = resultados_benchmarking['modelos_exitosos']
        nombres_modelos = [modelo['nombre'] for modelo in modelos_disponibles]
        
        modelo_seleccionado = st.selectbox(
            "**Seleccione un modelo para análisis de validación cruzada:**",
            options=nombres_modelos,
            help="Elija el modelo que desea analizar en profundidad"
        )
        
        # Encontrar el modelo completo
        modelo = None
        for m in modelos_disponibles:
            if m['nombre'] == modelo_seleccionado:
                modelo = m
                break
    
    with col2:
        st.subheader("ℹ️ Información del Benchmarking")
        st.write(f"**Tipo de problema:** {resultados_benchmarking.get('tipo_problema', 'N/A').title()}")
        st.write(f"**Variable objetivo:** {resultados_benchmarking.get('variable_objetivo', 'N/A')}")
        st.write(f"**Modelos exitosos:** {len(modelos_disponibles)}")
        st.write(f"**Fecha:** {resultados_benchmarking.get('timestamp', 'N/A')}")
    
    return modelo, resultados_benchmarking


def configurar_validacion():
    """Configura los parámetros para la validación cruzada."""
    st.subheader("⚙️ Configuración de Validación Cruzada")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cv_folds = st.slider(
            "**Número de folds para CV:**", 
            min_value=3, max_value=10, value=5,
            help="Más folds = mayor precisión pero más tiempo de cómputo"
        )
    
    with col2:
        n_puntos = st.slider(
            "**Puntos en curva de aprendizaje:**", 
            min_value=5, max_value=20, value=10,
            help="Más puntos = curva más suave pero mayor tiempo de cómputo"
        )
    
    with col3:
        scoring = st.selectbox(
            "**Métrica de evaluación:**",
            options=['accuracy', 'f1', 'precision', 'recall', 'roc_auc'],
            help="Métrica utilizada para evaluar el rendimiento del modelo"
        )
    
    # Opciones avanzadas
    with st.expander("🔧 Configuración Avanzada", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            random_state = st.number_input(
                "**Semilla aleatoria:**", 
                value=42, min_value=1, max_value=999,
                help="Para reproducibilidad de resultados"
            )
        
        with col2:
            n_jobs = st.selectbox(
                "**Procesamiento:**",
                options=[1, -1],
                format_func=lambda x: "Secuencial" if x == 1 else "Paralelo (todos los cores)",
                help="Procesamiento paralelo acelera el cálculo"
            )
    
    return {
        'cv_folds': cv_folds,
        'n_puntos': n_puntos,
        'scoring': scoring,
        'random_state': random_state,
        'n_jobs': n_jobs
    }
