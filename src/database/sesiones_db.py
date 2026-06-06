"""
Operaciones sobre la tabla sesiones.
Soporta SQLite (local) y Supabase (cloud) según DB_BACKEND.
"""
import uuid
from datetime import datetime
from typing import Optional, Dict
from src.database.backend import get_backend


def crear_sesion(id_usuario: int) -> str:
    """Genera un UUID de sesión, lo registra y lo devuelve."""
    id_sesion = str(uuid.uuid4())
    if get_backend() == "supabase":
        _crear_sesion_supabase(id_sesion, id_usuario)
    else:
        _crear_sesion_sqlite(id_sesion, id_usuario)
    return id_sesion


def cerrar_sesion(id_sesion: str) -> None:
    """Marca la sesión como INACTIVA y registra la fecha de cierre."""
    if get_backend() == "supabase":
        _cerrar_sesion_supabase(id_sesion)
    else:
        _cerrar_sesion_sqlite(id_sesion)


def obtener_sesion(id_sesion: str) -> Optional[Dict]:
    """Devuelve los datos de la sesión o None si no existe."""
    if get_backend() == "supabase":
        return _obtener_sesion_supabase(id_sesion)
    return _obtener_sesion_sqlite(id_sesion)


# ── Implementación SQLite ─────────────────────────────────────────────────────

def _crear_sesion_sqlite(id_sesion: str, id_usuario: int) -> None:
    from src.database.sqlite_conn import get_connection
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


def _cerrar_sesion_sqlite(id_sesion: str) -> None:
    from src.database.sqlite_conn import get_connection
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE sesiones SET estado = 'INACTIVA', fecha_fin = ? WHERE id_sesion = ?",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), id_sesion),
        )
        conn.commit()
    finally:
        conn.close()


def _obtener_sesion_sqlite(id_sesion: str) -> Optional[Dict]:
    from src.database.sqlite_conn import get_connection
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id_sesion, id_usuario, fecha_inicio, fecha_fin, estado FROM sesiones WHERE id_sesion = ?",
            (id_sesion,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# ── Implementación Supabase ───────────────────────────────────────────────────

def _crear_sesion_supabase(id_sesion: str, id_usuario: int) -> None:
    from src.database.supabase_conn import get_client
    client = get_client()
    client.table("sesiones").insert({
        "id_sesion": id_sesion,
        "id_usuario": id_usuario,
        "estado": "ACTIVA",
    }).execute()


def _cerrar_sesion_supabase(id_sesion: str) -> None:
    from src.database.supabase_conn import get_client
    client = get_client()
    client.table("sesiones").update({
        "estado": "INACTIVA",
        "fecha_fin": datetime.now().isoformat(),
    }).eq("id_sesion", id_sesion).execute()


def _obtener_sesion_supabase(id_sesion: str) -> Optional[Dict]:
    from src.database.supabase_conn import get_client
    client = get_client()
    result = client.table("sesiones") \
        .select("id_sesion, id_usuario, fecha_inicio, fecha_fin, estado") \
        .eq("id_sesion", id_sesion).limit(1).execute()
    if result.data:
        return result.data[0]
    return None
