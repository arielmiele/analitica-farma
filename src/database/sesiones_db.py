"""
Operaciones sobre la tabla sesiones en SQLite.
"""
import uuid
from datetime import datetime
from typing import Optional, Dict
from src.database.sqlite_conn import get_connection


def crear_sesion(id_usuario: int) -> str:
    """
    Genera un UUID de sesión, lo registra en SQLite y lo devuelve.
    Si ya existe una sesión activa para ese usuario, la reutiliza.
    """
    id_sesion = str(uuid.uuid4())
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO sesiones (id_sesion, id_usuario, fecha_inicio, estado)
               VALUES (?, ?, ?, 'ACTIVA')""",
            (id_sesion, id_usuario, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
        conn.commit()
    finally:
        conn.close()
    return id_sesion


def cerrar_sesion(id_sesion: str) -> None:
    """Marca la sesión como INACTIVA y registra la fecha de cierre."""
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE sesiones SET estado = 'INACTIVA', fecha_fin = ? WHERE id_sesion = ?",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), id_sesion),
        )
        conn.commit()
    finally:
        conn.close()


def obtener_sesion(id_sesion: str) -> Optional[Dict]:
    """Devuelve los datos de la sesión o None si no existe."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id_sesion, id_usuario, fecha_inicio, fecha_fin, estado FROM sesiones WHERE id_sesion = ?",
            (id_sesion,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()
