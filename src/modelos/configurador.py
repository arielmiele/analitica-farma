"""
Módulo para gestionar la configuración de modelos de machine learning.
"""
import json
import sqlite3
import os
import logging
from datetime import datetime

# Obtener el logger
logger = logging.getLogger("configurador")

def guardar_configuracion_modelo(configuracion, id_usuario=1, db_path=None):
    """
    Guarda la configuración del modelo en la base de datos.
    
    Args:
        configuracion (dict): Diccionario con la configuración del modelo
        id_usuario (int): ID del usuario que realiza la acción
        db_path (str): Ruta a la base de datos SQLite (por defecto, analitica_farma.db)
    
    Returns:
        int: ID de la configuración guardada
    """
    conn = None
    try:
        # Determinar la ruta de la base de datos
        if db_path is None:
            db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'analitica_farma.db')
        
        # Conectar a la base de datos
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Verificar si existe la tabla de configuraciones
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS configuraciones_modelo (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                id_usuario INTEGER NOT NULL,
                tipo_problema TEXT NOT NULL,
                variable_objetivo TEXT NOT NULL,
                variables_predictoras TEXT NOT NULL,
                configuracion_completa TEXT NOT NULL,
                fecha_creacion TIMESTAMP NOT NULL
            )
        """)
        # Verificar si existe la tabla de auditoría
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS auditoria (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                id_usuario INTEGER NOT NULL,
                accion TEXT NOT NULL,
                descripcion TEXT,
                fecha TIMESTAMP NOT NULL
            )
        """)
        
        # Preparar datos para inserción
        variables_predictoras_json = json.dumps(configuracion.get('variables_predictoras', []))
        configuracion_json = json.dumps(configuracion)
        
        # Insertar la configuración
        cursor.execute("""
            INSERT INTO configuraciones_modelo (
                id_usuario, tipo_problema, variable_objetivo, 
                variables_predictoras, configuracion_completa, fecha_creacion
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            id_usuario,
            configuracion.get('tipo_problema', ''),
            configuracion.get('variable_objetivo', ''),
            variables_predictoras_json,
            configuracion_json,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ))
        
        # Obtener el ID de la configuración insertada
        config_id = cursor.lastrowid
        
        # Registrar en la tabla de auditoría
        cursor.execute("""
            INSERT INTO auditoria (
                id_usuario, accion, descripcion, fecha
            ) VALUES (?, ?, ?, ?)
        """, (
            id_usuario,
            "GUARDAR_CONFIGURACION",
            f"Configuración de modelo guardada: {configuracion.get('tipo_problema')} - {configuracion.get('variable_objetivo')}",
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ))
        
        # Confirmar cambios
        conn.commit()
        
        logger.info(f"Configuración guardada con ID {config_id}")
        
        return config_id
    
    except Exception as e:
        logger.error(f"Error al guardar configuración: {str(e)}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


def obtener_configuracion_modelo(id_configuracion=None, id_usuario=None, db_path=None):
    """
    Obtiene una configuración de modelo guardada.
    
    Args:
        id_configuracion (int): ID de la configuración a recuperar
        id_usuario (int): ID del usuario para filtrar configuraciones
        db_path (str): Ruta a la base de datos SQLite
    
    Returns:
        dict: Configuración del modelo o None si no se encuentra
    """
    conn = None
    try:
        # Determinar la ruta de la base de datos
        if db_path is None:
            db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'analitica_farma.db')
        
        # Conectar a la base de datos
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # Para acceder a columnas por nombre
        cursor = conn.cursor()
        
        # Construir la consulta según los parámetros
        query = "SELECT * FROM configuraciones_modelo WHERE 1=1"
        params = []
        
        if id_configuracion:
            query += " AND id = ?"
            params.append(id_configuracion)
        
        if id_usuario:
            query += " AND id_usuario = ?"
            params.append(id_usuario)
        
        query += " ORDER BY fecha_creacion DESC LIMIT 1"
        
        # Ejecutar la consulta
        cursor.execute(query, params)
        resultado = cursor.fetchone()
        
        if resultado:
            # Convertir a diccionario
            config = json.loads(resultado['configuracion_completa'])
            return config
        
        return None
    
    except Exception as e:
        logger.error(f"Error al obtener configuración: {str(e)}")
        return None
    finally:
        if conn:
            conn.close()
