"""
Operaciones CRUD sobre la tabla datasets.
Soporta SQLite (local, archivos Parquet en disco) y Supabase (cloud, Storage bucket).
"""
import uuid
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
from src.database.backend import get_backend
from src.database.sqlite_conn import DB_PATH

_DATASETS_DIR = DB_PATH.parent / "datasets"

# Nombre del bucket en Supabase Storage
_SUPABASE_BUCKET = "datasets"


def guardar_dataset(
    nombre: str,
    descripcion: str,
    id_usuario: int,
    df: pd.DataFrame,
    id_sesion: str,
    usuario: str,
) -> Optional[str]:
    """
    Persiste el DataFrame y registra sus metadatos.
    Devuelve el id_dataset generado, o None si hubo error.
    """
    if get_backend() == "supabase":
        return _guardar_dataset_supabase(nombre, descripcion, id_usuario, df, id_sesion)
    return _guardar_dataset_sqlite(nombre, descripcion, id_usuario, df)


def listar_datasets(id_usuario: Optional[int] = None) -> List[Dict]:
    """Lista los datasets, opcionalmente filtrados por usuario."""
    if get_backend() == "supabase":
        return _listar_datasets_supabase(id_usuario)
    return _listar_datasets_sqlite(id_usuario)


def obtener_dataset_por_id(id_dataset: str) -> Optional[Dict]:
    """Devuelve los metadatos del dataset o None si no existe."""
    if get_backend() == "supabase":
        return _obtener_dataset_supabase(id_dataset)
    return _obtener_dataset_sqlite(id_dataset)


def cargar_dataset_fisico(id_dataset: str) -> Optional[pd.DataFrame]:
    """Carga el archivo Parquet asociado al dataset y devuelve el DataFrame."""
    if get_backend() == "supabase":
        return _cargar_dataset_fisico_supabase(id_dataset)
    return _cargar_dataset_fisico_sqlite(id_dataset)


def eliminar_dataset(id_dataset: str) -> bool:
    """Elimina el archivo y el registro de metadatos. Devuelve True si tuvo éxito."""
    if get_backend() == "supabase":
        return _eliminar_dataset_supabase(id_dataset)
    return _eliminar_dataset_sqlite(id_dataset)


# ── Implementación SQLite ─────────────────────────────────────────────────────

def _guardar_dataset_sqlite(nombre, descripcion, id_usuario, df) -> Optional[str]:
    from src.database.sqlite_conn import get_connection
    _DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    id_dataset = str(uuid.uuid4())
    ruta = _DATASETS_DIR / f"{id_dataset}.parquet"
    try:
        df.to_parquet(str(ruta), index=False)
    except Exception:
        return None

    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO datasets (id_dataset, nombre, descripcion, fecha_creacion, id_usuario, ruta_archivo, filas, columnas)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (id_dataset, nombre, descripcion,
             datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             id_usuario, str(ruta), df.shape[0], df.shape[1]),
        )
        conn.commit()
    except Exception:
        ruta.unlink(missing_ok=True)
        conn.close()
        return None
    finally:
        conn.close()
    return id_dataset


def _listar_datasets_sqlite(id_usuario) -> List[Dict]:
    from src.database.sqlite_conn import get_connection
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


def _obtener_dataset_sqlite(id_dataset) -> Optional[Dict]:
    from src.database.sqlite_conn import get_connection
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM datasets WHERE id_dataset = ?", (id_dataset,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def _cargar_dataset_fisico_sqlite(id_dataset) -> Optional[pd.DataFrame]:
    meta = _obtener_dataset_sqlite(id_dataset)
    if not meta or not meta.get("ruta_archivo"):
        return None
    ruta = Path(meta["ruta_archivo"])
    if not ruta.exists():
        return None
    return pd.read_parquet(str(ruta))


def _eliminar_dataset_sqlite(id_dataset) -> bool:
    meta = _obtener_dataset_sqlite(id_dataset)
    if not meta:
        return False
    ruta = Path(meta.get("ruta_archivo", ""))
    ruta.unlink(missing_ok=True)

    from src.database.sqlite_conn import get_connection
    conn = get_connection()
    try:
        conn.execute("DELETE FROM datasets WHERE id_dataset = ?", (id_dataset,))
        conn.commit()
    finally:
        conn.close()
    return True


# ── Implementación Supabase ───────────────────────────────────────────────────

def _guardar_dataset_supabase(nombre, descripcion, id_usuario, df, id_sesion) -> Optional[str]:
    import io
    from src.database.supabase_conn import get_client, get_storage

    id_dataset = str(uuid.uuid4())
    ruta_storage = f"{id_dataset}.parquet"

    # Serializar DataFrame a bytes
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)

    try:
        storage = get_storage()
        storage.from_(_SUPABASE_BUCKET).upload(ruta_storage, buf.getvalue(),
                                                file_options={"content-type": "application/octet-stream"})
    except Exception:
        return None

    try:
        client = get_client()
        client.table("datasets").insert({
            "id_dataset": id_dataset,
            "nombre": nombre,
            "descripcion": descripcion,
            "id_usuario": id_usuario,
            "ruta_archivo": ruta_storage,
            "filas": int(df.shape[0]),
            "columnas": int(df.shape[1]),
        }).execute()
    except Exception:
        # Rollback: borrar archivo subido
        try:
            get_storage().from_(_SUPABASE_BUCKET).remove([ruta_storage])
        except Exception:
            pass
        return None

    return id_dataset


def _listar_datasets_supabase(id_usuario) -> List[Dict]:
    from src.database.supabase_conn import get_client
    client = get_client()
    query = client.table("datasets") \
        .select("id_dataset, nombre, descripcion, fecha_creacion, id_usuario, filas, columnas") \
        .order("fecha_creacion", desc=True)
    if id_usuario is not None:
        query = query.eq("id_usuario", id_usuario)
    result = query.execute()
    return result.data if result.data else []


def _obtener_dataset_supabase(id_dataset) -> Optional[Dict]:
    from src.database.supabase_conn import get_client
    client = get_client()
    result = client.table("datasets").select("*") \
        .eq("id_dataset", id_dataset).limit(1).execute()
    if result.data:
        return result.data[0]
    return None


def _cargar_dataset_fisico_supabase(id_dataset) -> Optional[pd.DataFrame]:
    import io
    from src.database.supabase_conn import get_storage

    meta = _obtener_dataset_supabase(id_dataset)
    if not meta or not meta.get("ruta_archivo"):
        return None

    try:
        storage = get_storage()
        data = storage.from_(_SUPABASE_BUCKET).download(meta["ruta_archivo"])
        buf = io.BytesIO(data)
        return pd.read_parquet(buf)
    except Exception:
        return None


def _eliminar_dataset_supabase(id_dataset) -> bool:
    from src.database.supabase_conn import get_client, get_storage

    meta = _obtener_dataset_supabase(id_dataset)
    if not meta:
        return False

    # Borrar archivo de Storage
    try:
        storage = get_storage()
        storage.from_(_SUPABASE_BUCKET).remove([meta.get("ruta_archivo", "")])
    except Exception:
        pass

    # Borrar registro
    try:
        client = get_client()
        client.table("datasets").delete().eq("id_dataset", id_dataset).execute()
    except Exception:
        return False
    return True
