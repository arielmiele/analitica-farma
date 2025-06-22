import streamlit as st
import json
import os
from typing import Dict, Any, Optional
import datetime

from src.state.session_manager import SessionManager

class SidebarComponents:
    """
    Clase para gestionar los componentes de UI del sidebar.
    Contiene métodos para renderizar diferentes partes del sidebar.
    """
    
    @staticmethod
    def load_workflow_steps() -> Dict[str, Any]:
        """
        Carga la configuración de etapas del workflow desde un archivo JSON
        
        Returns:
            Dict[str, Any]: Configuración de las etapas
        """
        try:
            config_path = os.path.join("src", "config", "workflow_steps.json")
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            # Si hay un error, devolver un diccionario con configuración por defecto
            print(f"Error al cargar configuración: {str(e)}")
            return {
                "etapas": [
                    {"id": "carga_datos", "nombre": "Carga de datos", "icono": "1️⃣"},
                    {"id": "configuracion", "nombre": "Configuración", "icono": "2️⃣"},
                    {"id": "validacion", "nombre": "Validación", "icono": "3️⃣"},
                    {"id": "transformacion", "nombre": "Transformaciones", "icono": "4️⃣"},
                    {"id": "entrenamiento", "nombre": "Entrenamiento", "icono": "5️⃣"},
                    {"id": "evaluacion", "nombre": "Evaluación", "icono": "6️⃣"},
                    {"id": "recomendacion", "nombre": "Recomendación", "icono": "7️⃣"}
                ]
            }
    
    @staticmethod
    def render_dataset_info() -> None:
        """
        Renderiza la información del dataset en el sidebar
        """
        with st.expander("📊 Información del Dataset", expanded=True):
            dataset_info = SessionManager.get_dataset_info()
            
            if dataset_info:
                # Mostrar nombre con estilo más compacto
                st.success(f"**{dataset_info['nombre']}**")
                
                # Información compacta del dataset en 2 columnas
                col1, col2 = st.columns(2)
                
                with col1:
                    # Mostrar origen
                    st.write(f"**Origen:** {dataset_info['origen']}")
                    
                    # Fecha más compacta
                    if dataset_info.get('fecha_carga'):
                        st.write(f"**Fecha:** {dataset_info['fecha_carga']}")
                
                with col2:
                    # Dimensiones del dataframe
                    st.write(f"**Filas:** {dataset_info['filas']}")
                    st.write(f"**Cols:** {dataset_info['columnas']}")
                
                # Mostrar información de configuración si está disponible en formato compacto
                if dataset_info.get('variable_objetivo'):
                    st.write("---")
                    # Mostrar en formato más compacto
                    tipo_problema = dataset_info.get('tipo_problema', '').capitalize()
                    var_obj = dataset_info.get('variable_objetivo')
                    
                    st.write(f"**Problema:** {tipo_problema} → **Objetivo:** {var_obj}")
                    
                    # Mostrar predictores en forma compacta
                    if dataset_info.get('num_predictores', 0) > 0:
                        st.write(f"**Predictores:** {dataset_info['num_predictores']} variables")
            else:
                st.info("No hay dataset cargado")
                
                # Si ya tenemos un usuario logueado, mostrar un botón de acceso rápido a carga
                if SessionManager.is_logged_in():
                    if st.button("📊 Ir a Cargar Datos", key="btn_ir_cargar"):
                        st.switch_page("pages/Datos/01_Cargar_Datos.py")
    
    @staticmethod
    def render_progress_checklist() -> None:
        """
        Renderiza el checklist de progreso del workflow
        """
        if not SessionManager.is_dataset_loaded():
            return
            
        with st.expander("✅ Progreso del Análisis", expanded=True):
            # Cargar las etapas desde el archivo de configuración
            workflow_config = SidebarComponents.load_workflow_steps()
            progress_status = SessionManager.get_progress_status()
            
            # Crear checklist con el estado actual en formato compacto
            for etapa in workflow_config["etapas"]:
                etapa_id = etapa["id"]
                etapa_nombre = f"{etapa['icono']} {etapa['nombre']}"
                completada = progress_status.get(etapa_id, False)
                
                icono = "✅" if completada else "⬜"
                st.write(f"{icono} {etapa_nombre}")
    
    @staticmethod
    def render_reset_button() -> None:
        """
        Renderiza el botón para reiniciar el análisis
        """
        if not SessionManager.is_dataset_loaded():
            return
            
        # Verificar si hay al menos un paso completado
        progress_status = SessionManager.get_progress_status()
        if any(progress_status.values()):
            if st.button("🔄 Reiniciar Análisis", key="btn_reiniciar"):
                SessionManager.reset_analysis()
                st.rerun()
    
    @staticmethod
    def render_sidebar() -> None:
        """
        Renderiza el sidebar completo
        """
        with st.sidebar:
            # Renderizar componentes del sidebar
            SidebarComponents.render_dataset_info()
            SidebarComponents.render_progress_checklist()
            SidebarComponents.render_reset_button()
