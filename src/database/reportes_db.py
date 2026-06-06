"""
Operaciones sobre la tabla reportes.
Soporta SQLite (local) y Supabase (cloud) según DB_BACKEND.
"""
import json
import uuid
from datetime import datetime
from typing import Optional, Dict
from src.database.backend import get_backend


def guardar_reporte(
    nombre_archivo: str,
    tipo: str,
    id_usuario,
    id_sesion: str,
    id_benchmarking: Optional[int],
    id_dataset: Optional[str],
    resultados: Dict,
) -> str:
    """
    Guarda los metadatos y resultados estructurados del reporte.
    El PDF binario no se almacena; solo la metadata tabular.
    Devuelve el id_reporte generado.
    """
    id_reporte = str(uuid.uuid4())
    if get_backend() == "supabase":
        _guardar_reporte_supabase(id_reporte, nombre_archivo, tipo, id_usuario,
                                   id_sesion, id_benchmarking, id_dataset, resultados)
    else:
        _guardar_reporte_sqlite(id_reporte, nombre_archivo, tipo, id_usuario,
                                id_sesion, id_benchmarking, id_dataset, resultados)
    return id_reporte


def obtener_reporte_por_id(id_reporte: str) -> Optional[Dict]:
    """Devuelve el reporte con ese ID o None si no existe."""
    if get_backend() == "supabase":
        return _obtener_reporte_supabase(id_reporte)
    return _obtener_reporte_sqlite(id_reporte)


# ── Implementación SQLite ─────────────────────────────────────────────────────

def _guardar_reporte_sqlite(id_reporte, nombre_archivo, tipo, id_usuario,
                             id_sesion, id_benchmarking, id_dataset, resultados):
    from src.database.sqlite_conn import get_connection
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO reportes
               (id_reporte, nombre, tipo, fecha, id_usuario, id_sesion, id_benchmarking, id_dataset, contenido)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                id_reporte, nombre_archivo, tipo,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                id_usuario, id_sesion, id_benchmarking, id_dataset,
                json.dumps(resultados, default=str),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _obtener_reporte_sqlite(id_reporte: str) -> Optional[Dict]:
    from src.database.sqlite_conn import get_connection
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM reportes WHERE id_reporte = ?", (id_reporte,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# ── Implementación Supabase ───────────────────────────────────────────────────

def _guardar_reporte_supabase(id_reporte, nombre_archivo, tipo, id_usuario,
                               id_sesion, id_benchmarking, id_dataset, resultados):
    from src.database.supabase_conn import get_client
    client = get_client()
    client.table("reportes").insert({
        "id_reporte": id_reporte,
        "nombre": nombre_archivo,
        "tipo": tipo,
        "id_usuario": id_usuario,
        "id_sesion": id_sesion,
        "id_benchmarking": id_benchmarking,
        "id_dataset": id_dataset,
        "contenido": resultados,
    }).execute()


def _obtener_reporte_supabase(id_reporte: str) -> Optional[Dict]:
    from src.database.supabase_conn import get_client
    client = get_client()
    result = client.table("reportes").select("*") \
        .eq("id_reporte", id_reporte).limit(1).execute()
    if result.data:
        return result.data[0]
    return None
