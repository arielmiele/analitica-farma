"""
Operaciones de escritura en la tabla auditoria en SQLite.
"""
from datetime import datetime
from typing import Optional
from src.database.sqlite_conn import get_connection


def registrar_auditoria(
    usuario: str,
    accion: str,
    entidad: str,
    id_entidad: str,
    detalles: str,
    id_sesion: Optional[str] = None,
) -> None:
    """Inserta un registro de auditoría en SQLite."""
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO auditoria (usuario, accion, entidad, id_entidad, detalles, fecha, id_sesion)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                str(usuario),
                str(accion),
                str(entidad),
                str(id_entidad),
                str(detalles),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                str(id_sesion) if id_sesion else None,
            ),
        )
        conn.commit()
    except Exception:
        pass  # Auditoría no debe romper el flujo principal
    finally:
        conn.close()
