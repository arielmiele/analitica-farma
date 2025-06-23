import streamlit as st
from typing import Dict, Any
import datetime

class SessionManager:
    """
    Clase para gestionar el estado de la sesión de Streamlit de manera centralizada.
    Proporciona métodos para acceder y modificar el estado de la sesión.
    """
    
    @staticmethod
    def init_session_state():
        """Inicializa todas las variables de estado necesarias si no existen"""
        if "logged_in" not in st.session_state:
            st.session_state.logged_in = False
            
        if "etapas_completadas" not in st.session_state:
            st.session_state.etapas_completadas = {
                "carga_datos": False,
                "configuracion": False,
                "validacion": False,
                "analisis_calidad": False,
                "transformacion": False,
                "entrenamiento": False,
                "evaluacion": False,
                "recomendacion": False
            }
            
        # Variables para el manejo del dataset
        if 'df' not in st.session_state:
            st.session_state.df = None
        if 'filename' not in st.session_state:
            st.session_state.filename = None
        if 'upload_timestamp' not in st.session_state:
            st.session_state.upload_timestamp = None
        if 'paso_carga' not in st.session_state:
            st.session_state.paso_carga = 0
        if 'metodo_carga' not in st.session_state:
            st.session_state.metodo_carga = None
    
    @staticmethod
    def get_dataset_info() -> Dict[str, Any]:
        """
        Obtiene la información del dataset actualmente cargado
        
        Returns:
            Dict[str, Any]: Diccionario con información del dataset o None si no hay dataset
        """
        if 'df' not in st.session_state or st.session_state.df is None:
            return {}
            
        info = {
            "nombre": st.session_state.get('filename', 'Sin nombre'),
            "filas": st.session_state.df.shape[0] if st.session_state.df is not None else 0,
            "columnas": st.session_state.df.shape[1] if st.session_state.df is not None else 0,
            "fecha_carga": None,
            "origen": None,
            "variable_objetivo": st.session_state.get('variable_objetivo', None),
            "tipo_problema": st.session_state.get('tipo_problema', ''),
            "num_predictores": len(st.session_state.get('predictores', [])) if 'predictores' in st.session_state else 0
        }
        
        # Procesar origen
        origen = st.session_state.get('metodo_carga', 'Nuevo archivo')
        if origen == 'existente':
            info["origen"] = 'BD local'
        elif origen == 'nuevo':
            info["origen"] = 'CSV'
        else:
            info["origen"] = origen
            
        # Procesar fecha
        upload_timestamp = st.session_state.get('upload_timestamp', '')
        if upload_timestamp:
            if isinstance(upload_timestamp, datetime.datetime):
                info["fecha_carga"] = upload_timestamp.strftime('%d/%m/%y')
            else:
                info["fecha_carga"] = str(upload_timestamp)
                
        return info
    
    @staticmethod
    def get_progress_status() -> Dict[str, bool]:
        """
        Obtiene el estado actual de las etapas del workflow
        
        Returns:
            Dict[str, bool]: Diccionario con el estado de cada etapa
        """
        return st.session_state.etapas_completadas.copy()
    
    @staticmethod
    def update_progress(etapa_id: str, completada: bool = True) -> None:
        """
        Actualiza el estado de una etapa del workflow
        
        Args:
            etapa_id (str): Identificador de la etapa
            completada (bool): Estado de la etapa (True = completada, False = pendiente)
        """
        if etapa_id in st.session_state.etapas_completadas:
            st.session_state.etapas_completadas[etapa_id] = completada
    
    @staticmethod
    def reset_analysis() -> None:
        """
        Reinicia todas las variables relacionadas con el análisis
        pero mantiene el estado de login
        """
        # Reiniciar todas las etapas
        for etapa in st.session_state.etapas_completadas:
            st.session_state.etapas_completadas[etapa] = False
            
        # Reiniciar variables del dataset
        st.session_state.df = None
        st.session_state.filename = None
        st.session_state.upload_timestamp = None
        st.session_state.paso_carga = 0
        st.session_state.metodo_carga = None
        
        # Eliminar variables de configuración
        if 'variable_objetivo' in st.session_state:
            del st.session_state.variable_objetivo
        if 'predictores' in st.session_state:
            del st.session_state.predictores
        if 'tipo_problema' in st.session_state:
            del st.session_state.tipo_problema
    
    @staticmethod
    def is_dataset_loaded() -> bool:
        """
        Verifica si hay un dataset cargado
        
        Returns:
            bool: True si hay un dataset cargado, False en caso contrario
        """
        return 'df' in st.session_state and st.session_state.df is not None
    
    @staticmethod
    def is_logged_in() -> bool:
        """
        Verifica si el usuario está logueado
        
        Returns:
            bool: True si está logueado, False en caso contrario
        """
        return st.session_state.get("logged_in", False)
