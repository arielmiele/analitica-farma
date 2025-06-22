"""
Módulo para configurar y gestionar el logging de la aplicación.
Proporciona una forma unificada de registro para cumplir con requisitos de auditoría.
"""
import logging
import os
from datetime import datetime


def setup_logger(logger_name, log_level=logging.INFO):
    """
    Configura y devuelve un logger con el nombre especificado.
    
    Args:
        logger_name (str): Nombre del logger
        log_level (int): Nivel de logging (default: logging.INFO)
    
    Returns:
        logging.Logger: Logger configurado
    """
    # Crear el directorio de logs si no existe
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Configurar el logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    
    # Verificar si ya tiene handlers para evitar duplicados
    if not logger.handlers:
        # Crear un formatter para los logs
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Configurar el handler para archivo
        log_file = os.path.join(log_dir, f'{logger_name}_{datetime.now().strftime("%Y%m%d")}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Configurar el handler para consola
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def log_operation(logger, operation_type, details, success=True):
    """
    Registra una operación en el log con un formato estandarizado.
    
    Args:
        logger (logging.Logger): Logger a utilizar
        operation_type (str): Tipo de operación (carga, transformación, etc.)
        details (str): Detalles de la operación
        success (bool): Indica si la operación fue exitosa
    """
    status = "ÉXITO" if success else "ERROR"
    message = f"[{operation_type}] [{status}] {details}"
    
    if success:
        logger.info(message)
    else:
        logger.error(message)


def log_audit(user, action, resource, details=""):
    """
    Registra una acción de usuario para auditoría.
    
    Args:
        user (str): Usuario que realizó la acción
        action (str): Acción realizada
        resource (str): Recurso afectado
        details (str): Detalles adicionales
    """
    audit_logger = setup_logger("auditoria")
    message = f"USUARIO: {user} | ACCIÓN: {action} | RECURSO: {resource}"
    if details:
        message += f" | DETALLES: {details}"
    
    audit_logger.info(message)
