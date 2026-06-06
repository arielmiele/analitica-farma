"""
Operaciones de escritura en la tabla auditoria.
Soporta SQLite (local) y Supabase (cloud) según DB_BACKEND.
"""
from datetime import datetime
from typing import Optional
from src.database.backend import get_backend


def registrar_auditoria(
    usuario: str,
    accion: str,
    entidad: str,
    id_entidad: str,
    detalles: str,
    id_sesion: Optional[str] = None,
) -> None:
    """Inserta un registro de auditoría."""
    try:
        if get_backend() == "supabase":
            _registrar_auditoria_supabase(usuario, accion, entidad, id_entidad, detalles, id_sesion)
        else:
            _registrar_auditoria_sqlite(usuario, accion, entidad, id_entidad, detalles, id_sesion)
    except Exception:
        pass  # Auditoría no debe romper el flujo principal


# ── Implementación SQLite ─────────────────────────────────────────────────────

def _registrar_auditoria_sqlite(usuario, accion, entidad, id_entidad, detalles, id_sesion):
    from src.database.sqlite_conn import get_connection
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO auditoria (usuario, accion, entidad, id_entidad, detalles, fecha, id_sesion)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                str(usuario), str(accion), str(entidad), str(id_entidad), str(detalles),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                str(id_sesion) if id_sesion else None,
            ),
        )
        conn.commit()
    finally:
        conn.close()


# ── Implementación Supabase ───────────────────────────────────────────────────

def _registrar_auditoria_supabase(usuario, accion, entidad, id_entidad, detalles, id_sesion):
    from src.database.supabase_conn import get_client
    client = get_client()
    client.table("auditoria").insert({
        "usuario": str(usuario),
        "accion": str(accion),
        "entidad": str(entidad),
        "id_entidad": str(id_entidad),
        "detalles": str(detalles),
        "id_sesion": str(id_sesion) if id_sesion else None,
    }).execute()
