"""
Módulo para gestionar la configuración de modelos de machine learning.
"""
from src.audit.logger import log_audit
from src.database.modelos_db import (
    guardar_configuracion_modelo as _guardar,
    obtener_configuracion_modelo as _obtener,
)


def guardar_configuracion_modelo(configuracion, id_usuario, id_sesion, usuario):
    """
    Guarda la configuración del modelo en SQLite.
    Devuelve el ID de la configuración guardada.
    """
    try:
        config_id = _guardar(configuracion, id_usuario, id_sesion, usuario)
        log_audit(
            usuario=usuario,
            accion="INFO_GUARDAR_CONFIGURACION",
            entidad="configurador",
            id_entidad=str(config_id) if config_id else "N/A",
            detalles=f"Configuración guardada en SQLite con ID {config_id}",
            id_sesion=id_sesion,
        )
        return config_id
    except Exception as e:
        log_audit(
            usuario=usuario,
            accion="ERROR_GUARDAR_CONFIGURACION",
            entidad="configurador",
            id_entidad="N/A",
            detalles=f"Error al guardar configuración: {str(e)}",
            id_sesion=id_sesion,
        )
        raise


def obtener_configuracion_modelo(id_sesion, usuario, id_configuracion=None, id_usuario=None):
    """Obtiene la configuración más reciente desde SQLite."""
    return _obtener(id_sesion, usuario, id_configuracion, id_usuario)

