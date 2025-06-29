"""
Módulo para configurar y gestionar el logging de la aplicación.
Proporciona una forma unificada de registro en la base de datos Snowflake
para cumplir con requisitos de auditoría y trazabilidad.
"""
import logging
from datetime import datetime
import traceback
import sys
from src.snowflake.snowflake_conn import get_native_snowflake_connection


class SnowflakeHandler(logging.Handler):
    """
    Handler personalizado que guarda los logs en la tabla de auditoría de Snowflake.
    """
    def __init__(self, usuario="", entidad="", id_entidad=""):
        super().__init__()
        self.usuario = usuario
        self.entidad = entidad
        self.id_entidad = id_entidad

    def emit(self, record):
        """
        Guarda el registro de log en la tabla de auditoría de Snowflake.
        
        Args:
            record (LogRecord): Registro de log a guardar
        """
        try:
            # Formatear el mensaje
            message = self.format(record)
            
            # Determinar el tipo de acción según el nivel del log
            accion = "INFO"
            if record.levelno >= logging.ERROR:
                accion = "ERROR"
            elif record.levelno >= logging.WARNING:
                accion = "ADVERTENCIA"
            
            # Si el mensaje tiene formato específico [TIPO] [STATUS], extraer la acción
            if "[" in message and "]" in message:
                parts = message.split("]")
                if len(parts) > 1:
                    accion = parts[0].strip("[")
            
            # Conectar a Snowflake y guardar el log
            conn = get_native_snowflake_connection()
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO AUDITORIA (USUARIO, ACCION, ENTIDAD, ID_ENTIDAD, DETALLES, FECHA)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                self.usuario,
                accion,
                self.entidad,
                self.id_entidad,
                message,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ))
            
            conn.commit()
            cur.close()
            conn.close()
        
        except Exception as e:
            # En caso de error, intentar escribir a stderr como fallback
            formatted_message = self.format(record)
            sys.stderr.write(f"Error al guardar log en Snowflake: {str(e)}\n")
            sys.stderr.write(f"Mensaje original: {formatted_message}\n")
            traceback.print_exc(file=sys.stderr)


def setup_logger(logger_name, log_level=logging.INFO, usuario="", entidad="", id_entidad=""):
    """
    Configura y devuelve un logger con almacenamiento en Snowflake.
    
    Args:
        logger_name (str): Nombre del logger
        log_level (int): Nivel de logging (default: logging.INFO)
        usuario (str): Usuario para los registros de auditoría
        entidad (str): Entidad afectada por la acción
        id_entidad (str): ID de la entidad afectada
    
    Returns:
        logging.Logger: Logger configurado
    """
    # Configurar el logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    
    # Eliminar handlers existentes para evitar duplicados
    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
    
    # Crear un formatter para los logs
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Configurar el handler para Snowflake
    snowflake_handler = SnowflakeHandler(usuario, entidad, id_entidad)
    snowflake_handler.setFormatter(formatter)
    logger.addHandler(snowflake_handler)
    
    # Configurar el handler para consola (desarrollo/depuración)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def log_audit(usuario, accion, entidad, id_entidad, detalles, id_sesion=None, logger_name="audit"):  # logger_name opcional para flexibilidad
    """
    Registra una acción directamente en la tabla de auditoría de Snowflake y en el logger estándar.
    
    Args:
        usuario (str): Usuario que realizó la acción
        accion (str): Acción realizada (ej: CARGA_DATOS, TRANSFORMACION, LOGIN)
        entidad (str): Entidad afectada (ej: nombre del dataset, modelo, etc.)
        id_entidad (str): ID de la entidad afectada
        detalles (str): Detalles adicionales
        id_sesion (str, opcional): ID de la sesión activa
        logger_name (str): Nombre del logger a usar para consola (por defecto 'audit')
    """
    try:
        # Insertar directamente en la base de datos
        conn = get_native_snowflake_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO AUDITORIA (USUARIO, ACCION, ENTIDAD, ID_ENTIDAD, DETALLES, FECHA, ID_SESION)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            usuario,
            accion,
            entidad,
            id_entidad,
            detalles,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            id_sesion
        ))
        conn.commit()
        cur.close()
        conn.close()
        # Loguear en consola usando el logger estándar
        logger = logging.getLogger(logger_name)
        logger.info(f"[AUDIT] Usuario {usuario} | {accion} | {entidad} | {id_entidad} | {detalles} | Sesion: {id_sesion}")
    except Exception as e:
        logger = logging.getLogger(logger_name)
        logger.error(f"Error al registrar auditoría en Snowflake: {str(e)} | Usuario: {usuario} | Acción: {accion} | Entidad: {entidad} | ID: {id_entidad} | Detalles: {detalles} | Sesion: {id_sesion}")
        traceback.print_exc()