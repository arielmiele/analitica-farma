"""
Módulo de logging y auditoría local.
Escribe en la tabla auditoria de SQLite y en consola.
"""
import logging
from datetime import datetime
from typing import Optional


class SQLiteHandler(logging.Handler):
    """Handler que persiste logs en la tabla auditoria de SQLite."""

    def __init__(self, usuario: str = "", entidad: str = "", id_entidad: str = ""):
        super().__init__()
        self.usuario = usuario
        self.entidad = entidad
        self.id_entidad = id_entidad

    def emit(self, record):
        try:
            from src.database.auditoria_db import registrar_auditoria
            accion = "INFO"
            if record.levelno >= logging.ERROR:
                accion = "ERROR"
            elif record.levelno >= logging.WARNING:
                accion = "ADVERTENCIA"
            message = self.format(record)
            if "[" in message and "]" in message:
                partes = message.split("]")
                if len(partes) > 1:
                    accion = partes[0].strip("[")
            registrar_auditoria(
                usuario=self.usuario,
                accion=accion,
                entidad=self.entidad,
                id_entidad=self.id_entidad,
                detalles=message,
            )
        except Exception:
            pass


def setup_logger(
    logger_name: str,
    log_level: int = logging.INFO,
    usuario: str = "",
    entidad: str = "",
    id_entidad: str = "",
) -> logging.Logger:
    """Configura y devuelve un logger con almacenamiento en SQLite y consola."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    sqlite_handler = SQLiteHandler(usuario, entidad, id_entidad)
    sqlite_handler.setFormatter(formatter)
    logger.addHandler(sqlite_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def log_audit(
    usuario,
    accion: str,
    entidad: str,
    id_entidad: str,
    detalles: str,
    id_sesion: Optional[str] = None,
    logger_name: str = "audit",
) -> None:
    """
    Registra una acción en la tabla auditoria de SQLite y en el logger estándar.
    """
    try:
        from src.database.auditoria_db import registrar_auditoria
        registrar_auditoria(
            usuario=str(usuario),
            accion=accion,
            entidad=entidad,
            id_entidad=str(id_entidad),
            detalles=detalles,
            id_sesion=id_sesion,
        )
    except Exception as e:
        pass
    logger = logging.getLogger(logger_name)
    logger.info(
        f"[AUDIT] Usuario {usuario} | {accion} | {entidad} | {id_entidad} | {detalles} | Sesion: {id_sesion}"
    )
