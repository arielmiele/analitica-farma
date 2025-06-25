import streamlit as st
from typing import Dict, Any
import datetime
import pandas as pd

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
    
    @staticmethod
    def guardar_estado(key: str, valor: Any):
        """
        Guarda un valor en el estado de la sesión
        
        Args:
            key (str): Nombre de la variable de estado
            valor (Any): Valor a guardar
        """
        st.session_state[key] = valor
    
    @staticmethod
    def obtener_estado(key: str, default: Any = None) -> Any:
        """
        Obtiene un valor del estado de la sesión
        
        Args:
            key (str): Nombre de la variable de estado
            default (Any, opcional): Valor por defecto si no existe la clave
            
        Returns:
            Any: Valor de la variable de estado o el default
        """
        return st.session_state.get(key, default)
    
    @staticmethod
    def cargar_dataframe():
        """
        Carga el dataframe actual desde la sesión
        
        Returns:
            pd.DataFrame: Dataframe actual o None si no hay ninguno
        """
        return st.session_state.get('df', None)
    
    @staticmethod
    def obtener_trigger_benchmarking():
        """
        Obtiene el valor actual del trigger para invalidar la caché de benchmarking.
        Se incrementa cada vez que se fuerza un re-entrenamiento.
        
        Returns:
            int: Valor actual del trigger
        """
        if 'benchmarking_trigger' not in st.session_state:
            st.session_state.benchmarking_trigger = 0
        return st.session_state.benchmarking_trigger
    
    @staticmethod
    def incrementar_trigger_benchmarking():
        """
        Incrementa el contador del trigger para forzar un nuevo entrenamiento.
        Invalida la caché de benchmarking al cambiar este valor.
        """
        if 'benchmarking_trigger' not in st.session_state:
            st.session_state.benchmarking_trigger = 0
        else:
            st.session_state.benchmarking_trigger += 1
        return st.session_state.benchmarking_trigger
    
    @staticmethod
    def get_benchmarking_stats():
        """
        Obtiene estadísticas sobre el benchmarking actual y los entrenamientos realizados.
        
        Returns:
            dict: Diccionario con estadísticas del benchmarking
        """
        stats = {
            "trigger_count": st.session_state.get("benchmarking_trigger", 0),
            "ultimo_entrenamiento": st.session_state.get("ultimo_entrenamiento", None),
            "total_entrenamientos": st.session_state.get("total_entrenamientos", 0),
            "ultimo_benchmarking_id": None
        }
          # Obtener ID del último benchmarking si existe
        resultados = SessionManager.obtener_estado("resultados_benchmarking", None)
        if resultados and "id" in resultados:
            stats["ultimo_benchmarking_id"] = resultados["id"]
            
        return stats
    
    @staticmethod
    def registrar_entrenamiento():
        """
        Registra un nuevo entrenamiento en las estadísticas.
        """
        # Incrementar contador total
        if "total_entrenamientos" not in st.session_state:
            st.session_state.total_entrenamientos = 1
        else:
            st.session_state.total_entrenamientos += 1
            
        # Registrar timestamp
        st.session_state.ultimo_entrenamiento = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def logout() -> None:
        """
        Cierra la sesión del usuario actual y limpia los datos de sesión
        """
        for key in list(st.session_state.keys()):
            if (isinstance(key, str) and key.startswith('usuario_')) or key in ['authenticated', 'current_user']:
                del st.session_state[key]
        
        # Aseguramos que logged_in sea False
        st.session_state.logged_in = False
        
        # Redirigir a la página de login
        try:
            st.switch_page("pages/00_Logueo.py")
        except Exception:
            # Si hay un error al redirigir, simplemente continuamos
            pass