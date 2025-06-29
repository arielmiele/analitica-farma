"""
Módulo para gestionar la configuración de modelos de machine learning.
"""
import json
from datetime import datetime
from src.audit.logger import log_audit
from src.snowflake.snowflake_conn import get_native_snowflake_connection

def guardar_configuracion_modelo(configuracion, id_usuario, id_sesion, usuario):
    """
    Guarda la configuración del modelo en la base de datos Snowflake.
    Args:
        configuracion (dict): Configuración del modelo
        id_usuario (int): ID del usuario
        id_sesion (str): ID de sesión para trazabilidad
        usuario (str): Usuario que ejecuta la acción
    Returns:
        int: ID de la configuración guardada
    """
    conn = None
    try:
        conn = get_native_snowflake_connection()
        cursor = conn.cursor()

        # Sanitizar y preparar datos
        tipo_problema = str(configuracion.get('tipo_problema', ''))
        variable_objetivo = str(configuracion.get('variable_objetivo', ''))
        variables_predictoras_json = json.dumps(configuracion.get('variables_predictoras', []))
        configuracion_json = json.dumps(configuracion)
        fecha_creacion = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Insertar la configuración (ahora incluye ID_SESION)
        insert_query = """
            INSERT INTO CONFIGURACIONES_MODELO (
                ID_USUARIO, ID_SESION, TIPO_PROBLEMA, VARIABLE_OBJETIVO, 
                VARIABLES_PREDICTORAS, CONFIGURACION_COMPLETA, FECHA_CREACION
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(insert_query, (
            id_usuario,
            id_sesion,
            tipo_problema,
            variable_objetivo,
            variables_predictoras_json,
            configuracion_json,
            fecha_creacion
        ))
        # Obtener el ID de la configuración insertada (por usuario, sesión y timestamp)
        cursor.execute(
            "SELECT MAX(ID) FROM CONFIGURACIONES_MODELO WHERE ID_USUARIO = %s AND ID_SESION = %s AND FECHA_CREACION = %s",
            (id_usuario, id_sesion, fecha_creacion)
        )
        result = cursor.fetchone()
        config_id = result[0] if result and result[0] is not None else None

        log_audit(
            usuario=usuario,
            accion="INFO_GUARDAR_CONFIGURACION",
            entidad="configurador",
            id_entidad=str(config_id) if config_id else "N/A",
            detalles=f"Configuración guardada en Snowflake con ID {config_id}",
            id_sesion=id_sesion
        )
        conn.commit()
        return config_id
    except Exception as e:
        log_audit(
            usuario=usuario,
            accion="ERROR_GUARDAR_CONFIGURACION",
            entidad="configurador",
            id_entidad="N/A",
            detalles=f"Error al guardar configuración en Snowflake: {str(e)}",
            id_sesion=id_sesion
        )
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

def obtener_configuracion_modelo(id_sesion, usuario, id_configuracion=None, id_usuario=None):
    """
    Obtiene una configuración de modelo guardada desde Snowflake.
    Args:
        id_sesion (str): ID de sesión para trazabilidad
        usuario (str): Usuario que ejecuta la acción
        id_configuracion (int, opcional): ID de la configuración
        id_usuario (int, opcional): ID del usuario
    Returns:
        dict or None: Configuración del modelo o None si no se encuentra
    """
    conn = None
    try:
        conn = get_native_snowflake_connection()
        cursor = conn.cursor()
        # Construir la consulta según los parámetros
        query = "SELECT CONFIGURACION_COMPLETA FROM CONFIGURACIONES_MODELO WHERE 1=1"
        params = []
        if id_configuracion:
            query += " AND ID = %s"
            params.append(id_configuracion)
        if id_usuario:
            query += " AND ID_USUARIO = %s"
            params.append(id_usuario)
        query += " ORDER BY FECHA_CREACION DESC LIMIT 1"
        cursor.execute(query, tuple(params))
        resultado = cursor.fetchone()
        if resultado:
            config = json.loads(resultado[0])
            return config
        return None
    except Exception as e:
        log_audit(
            usuario=usuario,
            accion="ERROR_OBTENER_CONFIGURACION",
            entidad="configurador",
            id_entidad=str(id_configuracion) if id_configuracion else "N/A",
            detalles=f"Error al obtener configuración de Snowflake: {str(e)}",
            id_sesion=id_sesion
        )
        return None
    finally:
        if conn:
            conn.close()
