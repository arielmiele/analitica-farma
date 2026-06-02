"""
Operaciones CRUD sobre la tabla datasets en SQLite.
Los datos físicos se almacenan como archivos Parquet en data/datasets/.
"""
import uuid
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
from src.database.sqlite_conn import get_connection, DB_PATH

_DATASETS_DIR = DB_PATH.parent / "datasets"


def guardar_dataset(
    nombre: str,
    descripcion: str,
    id_usuario: int,
    df: pd.DataFrame,
    id_sesion: str,
    usuario: str,
) -> Optional[str]:
    """
    Persiste el DataFrame como Parquet en data/datasets/ y registra sus metadatos en SQLite.
    Devuelve el id_dataset generado, o None si hubo error.
    """
    _DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    id_dataset = str(uuid.uuid4())
    ruta = _DATASETS_DIR / f"{id_dataset}.parquet"
    try:
        df.to_parquet(str(ruta), index=False)
    except Exception as e:
        return None

    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO datasets (id_dataset, nombre, descripcion, fecha_creacion, id_usuario, ruta_archivo, filas, columnas)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                id_dataset, nombre, descripcion,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                id_usuario, str(ruta), df.shape[0], df.shape[1],
            ),
        )
        conn.commit()
    except Exception:
        ruta.unlink(missing_ok=True)
        conn.close()
        return None
    finally:
        conn.close()
    return id_dataset


def listar_datasets(id_usuario: Optional[int] = None) -> List[Dict]:
    """Lista los datasets, opcionalmente filtrados por usuario."""
    conn = get_connection()
    try:
        if id_usuario is not None:
            rows = conn.execute(
                "SELECT id_dataset, nombre, descripcion, fecha_creacion, id_usuario, filas, columnas FROM datasets WHERE id_usuario = ? ORDER BY fecha_creacion DESC",
                (id_usuario,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id_dataset, nombre, descripcion, fecha_creacion, id_usuario, filas, columnas FROM datasets ORDER BY fecha_creacion DESC"
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def obtener_dataset_por_id(id_dataset: str) -> Optional[Dict]:
    """Devuelve los metadatos del dataset o None si no existe."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM datasets WHERE id_dataset = ?", (id_dataset,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def cargar_dataset_fisico(id_dataset: str) -> Optional[pd.DataFrame]:
    """Carga el archivo Parquet asociado al dataset y devuelve el DataFrame."""
    meta = obtener_dataset_por_id(id_dataset)
    if not meta or not meta.get("ruta_archivo"):
        return None
    ruta = Path(meta["ruta_archivo"])
    if not ruta.exists():
        return None
    return pd.read_parquet(str(ruta))


def eliminar_dataset(id_dataset: str) -> bool:
    """Elimina el archivo Parquet y el registro de metadatos. Devuelve True si tuvo éxito."""
    meta = obtener_dataset_por_id(id_dataset)
    if not meta:
        return False
    ruta = Path(meta.get("ruta_archivo", ""))
    ruta.unlink(missing_ok=True)

    conn = get_connection()
    try:
        conn.execute("DELETE FROM datasets WHERE id_dataset = ?", (id_dataset,))
        conn.commit()
    finally:
        conn.close()
    return True
