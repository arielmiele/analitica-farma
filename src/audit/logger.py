"""
Módulo para configurar y gestionar el logging de la aplicación.
Proporciona una forma unificada de registro en la base de datos SQLite
para cumplir con requisitos de auditoría y trazabilidad.
"""
import logging
import os
import sqlite3
from datetime import datetime
import traceback
import sys


class SQLiteHandler(logging.Handler):
    """
    Handler personalizado que guarda los logs en la tabla de auditoría de SQLite.
    """
    def __init__(self, db_path, id_usuario=1):
        """
        Inicializa el handler con la ruta a la base de datos.
        
        Args:
            db_path (str): Ruta a la base de datos SQLite
            id_usuario (int): ID del usuario para registrar en la auditoría
        """
        super().__init__()
        self.db_path = db_path
        self.id_usuario = id_usuario
    
    def emit(self, record):
        """
        Guarda el registro de log en la tabla de auditoría.
        
        Args:
            record (LogRecord): Registro de log a guardar
        """
        try:
            # Formatear el mensaje
            message = self.format(record)
            
            # Determinar el tipo de acción según el nivel del log
            if record.levelno >= logging.ERROR:
                accion = "ERROR"
            elif record.levelno >= logging.WARNING:
                accion = "ADVERTENCIA"
            else:
                accion = "INFO"
            
            # Si el mensaje tiene formato específico [TIPO] [STATUS], extraer la acción
            if "[" in message and "]" in message:
                parts = message.split("]")
                if len(parts) > 1:
                    accion = parts[0].strip("[")
            
            # Conectar a la base de datos
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insertar el log en la tabla de auditoría
            cursor.execute("""
                INSERT INTO auditoria (id_usuario, accion, descripcion, fecha)
                VALUES (?, ?, ?, ?)
            """, (
                self.id_usuario,
                accion,
                message,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ))
            
            # Confirmar y cerrar
            conn.commit()
            conn.close()
        
        except Exception as e:
            # En caso de error, intentar escribir a stderr como fallback
            formatted_message = self.format(record)
            sys.stderr.write(f"Error al guardar log en SQLite: {str(e)}\n")
            sys.stderr.write(f"Mensaje original: {formatted_message}\n")
            traceback.print_exc(file=sys.stderr)


def setup_logger(logger_name, log_level=logging.INFO, id_usuario=1, db_path=None):
    """
    Configura y devuelve un logger con almacenamiento en SQLite.
    
    Args:
        logger_name (str): Nombre del logger
        log_level (int): Nivel de logging (default: logging.INFO)
        id_usuario (int): ID del usuario para los registros de auditoría
        db_path (str): Ruta a la base de datos SQLite (opcional)
    
    Returns:
        logging.Logger: Logger configurado
    """
    # Determinar la ruta de la base de datos
    if db_path is None:
        # Obtener la ruta relativa a la raíz del proyecto
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'analitica_farma.db')
    
    # Configurar el logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    
    # Eliminar handlers existentes para evitar duplicados
    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
    
    # Crear un formatter para los logs
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Configurar el handler para SQLite
    sqlite_handler = SQLiteHandler(db_path, id_usuario)
    sqlite_handler.setFormatter(formatter)
    logger.addHandler(sqlite_handler)
    
    # Configurar el handler para consola (desarrollo/depuración)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def log_operation(logger, operation_type, details, success=True, id_usuario=1):
    """
    Registra una operación en el log con un formato estandarizado.
    
    Args:
        logger (logging.Logger): Logger a utilizar
        operation_type (str): Tipo de operación (carga, transformación, etc.)
        details (str): Detalles de la operación
        success (bool): Indica si la operación fue exitosa
        id_usuario (int): ID del usuario que realiza la operación
    """
    # Actualizar el ID de usuario en el handler de SQLite
    for handler in logger.handlers:
        if isinstance(handler, SQLiteHandler):
            handler.id_usuario = id_usuario
    
    status = "ÉXITO" if success else "ERROR"
    message = f"[{operation_type}] [{status}] {details}"
    
    if success:
        logger.info(message)
    else:
        logger.error(message)


def log_audit(usuario_id, accion, recurso, detalles="", db_path=None):
    """
    Registra una acción directamente en la tabla de auditoría.
    
    Args:
        usuario_id (int): ID del usuario que realizó la acción
        accion (str): Acción realizada (ej: CARGA_DATOS, TRANSFORMACION, LOGIN)
        recurso (str): Recurso afectado (ej: nombre del dataset, modelo, etc.)
        detalles (str): Detalles adicionales
        db_path (str): Ruta a la base de datos SQLite (opcional)
    """
    try:
        # Determinar la ruta de la base de datos
        if db_path is None:
            db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'analitica_farma.db')
        
        # Crear el mensaje completo
        mensaje = f"RECURSO: {recurso}"
        if detalles:
            mensaje += f" | DETALLES: {detalles}"
        
        # Insertar directamente en la base de datos
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO auditoria (id_usuario, accion, descripcion, fecha)
            VALUES (?, ?, ?, ?)
        """, (
            usuario_id,
            accion,
            mensaje,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ))
        
        conn.commit()
        conn.close()
        
        # También mostrar en consola para desarrollo
        print(f"[AUDIT] Usuario {usuario_id} | {accion} | {mensaje}")
        
    except Exception as e:
        print(f"Error al registrar auditoría: {str(e)}")
        traceback.print_exc()


def obtener_logs_auditoria(desde=None, hasta=None, usuario_id=None, accion=None, db_path=None):
    """
    Obtiene los logs de auditoría filtrados por diversos criterios.
    
    Args:
        desde (str): Fecha de inicio (formato: 'YYYY-MM-DD')
        hasta (str): Fecha de fin (formato: 'YYYY-MM-DD')
        usuario_id (int): ID del usuario para filtrar
        accion (str): Tipo de acción para filtrar
        db_path (str): Ruta a la base de datos SQLite (opcional)
    
    Returns:
        list: Lista de diccionarios con los logs de auditoría
    """
    try:
        # Determinar la ruta de la base de datos
        if db_path is None:
            db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'analitica_farma.db')
        
        # Construir la consulta SQL con los filtros
        query = """
            SELECT a.id_log, a.id_usuario, u.nombre, a.accion, a.descripcion, a.fecha
            FROM auditoria a
            JOIN usuarios u ON a.id_usuario = u.id_usuario
            WHERE 1=1
        """
        params = []
        
        if desde:
            query += " AND fecha >= ?"
            params.append(desde)
        
        if hasta:
            query += " AND fecha <= ?"
            params.append(hasta)
        
        if usuario_id:
            query += " AND a.id_usuario = ?"
            params.append(usuario_id)
        
        if accion:
            query += " AND accion = ?"
            params.append(accion)
        
        query += " ORDER BY fecha DESC"
        
        # Ejecutar la consulta
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # Para obtener resultados como diccionarios
        cursor = conn.cursor()
        
        cursor.execute(query, params)
        logs = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        return logs
        
    except Exception as e:
        print(f"Error al obtener logs de auditoría: {str(e)}")
        traceback.print_exc()
        return []


def update_user_id(logger, id_usuario):
    """
    Actualiza el ID de usuario en todos los handlers SQLite de un logger.
    
    Args:
        logger (logging.Logger): Logger a actualizar
        id_usuario (int): Nuevo ID de usuario
    """
    for handler in logger.handlers:
        if isinstance(handler, SQLiteHandler):
            handler.id_usuario = id_usuario
    return logger