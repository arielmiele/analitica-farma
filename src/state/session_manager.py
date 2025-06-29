import streamlit as st
from typing import Dict, Any
import datetime
import pandas as pd
import uuid
from src.snowflake.snowflake_conn import get_native_snowflake_connection

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
            "num_predictores": len(st.session_state.get('variables_predictoras', st.session_state.get('predictores', [])))
        }
        # Añadir la lista de predictores para el sidebar (máxima compatibilidad)
        info["lista_predictores"] = st.session_state.get('variables_predictoras', st.session_state.get('predictores', []))
        
        # Procesar origen
        origen = st.session_state.get('metodo_carga', 'CSV')
        if origen == 'existente' or origen == 'snowflake':
            info["origen"] = 'Snowflake'
        else:
            info["origen"] = 'CSV'
            
        # Procesar fecha
        upload_timestamp = st.session_state.get('upload_timestamp', '')
        if upload_timestamp:
            if isinstance(upload_timestamp, datetime.datetime):
                info["fecha_carga"] = upload_timestamp.strftime('%d/%m/%y')
            else:
                info["fecha_carga"] = str(upload_timestamp)
                
        return info
    
    @staticmethod
    def reset_analysis() -> None:
        """
        Reinicia solo variables esenciales del análisis (dataset, configuración y selección de variables).
        """
        st.session_state.df = None
        st.session_state.filename = None
        st.session_state.upload_timestamp = None
        st.session_state.paso_carga = 0
        st.session_state.metodo_carga = None
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
    
    @staticmethod
    def set_user(usuario_id: int, usuario_nombre: str, usuario_rol: str, usuario_email: str):
        """
        Guarda los datos mínimos del usuario en la sesión.
        """
        st.session_state.logged_in = True
        st.session_state.usuario_id = usuario_id
        st.session_state.usuario_nombre = usuario_nombre
        st.session_state.usuario_rol = usuario_rol
        st.session_state.usuario_email = usuario_email

    @staticmethod
    def get_user_info() -> dict:
        """
        Devuelve los datos persistentes del usuario en sesión.
        """
        return {
            "usuario_id": st.session_state.get("usuario_id", None),
            "usuario_nombre": st.session_state.get("usuario_nombre", None),
            "usuario_rol": st.session_state.get("usuario_rol", None),
            "usuario_email": st.session_state.get("usuario_email", None),
        }

    @staticmethod
    def clear_user():
        """
        Limpia solo los datos de usuario de la sesión.
        """
        for key in ["usuario_id", "usuario_nombre", "usuario_rol", "usuario_email", "logged_in"]:
            if key in st.session_state:
                del st.session_state[key]
    
    @staticmethod
    def crear_sesion(usuario_id: str) -> str:
        """
        Genera un ID de sesión único, lo registra en la tabla SESIONES de Snowflake y lo almacena en st.session_state.
        
        Args:
            usuario_id (str): ID del usuario autenticado
        
        Returns:
            str: ID de sesión generado
        """
        if "id_sesion" not in st.session_state:
            id_sesion = str(uuid.uuid4())
            st.session_state["id_sesion"] = id_sesion
            # Registrar la sesión en Snowflake
            conn = get_native_snowflake_connection()
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO SESIONES (ID_SESION, USUARIO, FECHA_INICIO, ESTADO)
                       VALUES (%s, %s, CURRENT_TIMESTAMP(), %s)""",
                (id_sesion, usuario_id, 'ACTIVA')
            )
            conn.commit()
            cursor.close()
            conn.close()
        return st.session_state["id_sesion"]