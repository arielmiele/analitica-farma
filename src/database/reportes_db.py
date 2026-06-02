"""
Operaciones sobre la tabla reportes en SQLite.
"""
import json
import uuid
from datetime import datetime
from typing import Optional, Dict
from src.database.sqlite_conn import get_connection


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
    Guarda los metadatos y resultados estructurados del reporte en SQLite.
    El PDF binario no se almacena; solo la metadata tabular.
    Devuelve el id_reporte generado.
    """
    id_reporte = str(uuid.uuid4())
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO reportes
               (id_reporte, nombre, tipo, fecha, id_usuario, id_sesion, id_benchmarking, id_dataset, contenido)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                id_reporte,
                nombre_archivo,
                tipo,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                id_usuario,
                id_sesion,
                id_benchmarking,
                id_dataset,
                json.dumps(resultados, default=str),
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return id_reporte


def obtener_reporte_por_id(id_reporte: str) -> Optional[Dict]:
    """Devuelve el reporte con ese ID o None si no existe."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM reportes WHERE id_reporte = ?", (id_reporte,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()
