"""
Operaciones sobre benchmarking, historial y configuraciones de modelos.
Soporta SQLite (local) y Supabase (cloud) según DB_BACKEND.
"""
import json
from datetime import datetime
from typing import Optional, Dict
from src.database.backend import get_backend


# ── Benchmarking ──────────────────────────────────────────────────────────────

def insertar_benchmarking_modelos(
    resultados: Dict,
    id_usuario: int,
    id_sesion: str,
    usuario: str,
) -> int:
    """Inserta los resultados del benchmarking. Devuelve el ID generado."""
    if get_backend() == "supabase":
        return _insertar_benchmarking_supabase(resultados, id_usuario, id_sesion)
    return _insertar_benchmarking_sqlite(resultados, id_usuario, id_sesion)


def obtener_ultimo_benchmarking(id_usuario: Optional[int] = None) -> Optional[Dict]:
    """Devuelve los resultados del último benchmarking."""
    if get_backend() == "supabase":
        return _obtener_ultimo_benchmarking_supabase(id_usuario)
    return _obtener_ultimo_benchmarking_sqlite(id_usuario)


def obtener_benchmarking_por_id(benchmarking_id: int) -> Optional[Dict]:
    """Devuelve los resultados de un benchmarking específico por su ID."""
    if get_backend() == "supabase":
        return _obtener_benchmarking_por_id_supabase(benchmarking_id)
    return _obtener_benchmarking_por_id_sqlite(benchmarking_id)


# ── Historial de ejecuciones ──────────────────────────────────────────────────

def insertar_historial_ejecucion(
    resultados: Dict,
    id_usuario: int,
    id_sesion: str,
    dataset_nombre: str = "",
    duracion_segundos: float = 0.0,
) -> int:
    """Inserta un registro en historial_ejecuciones."""
    if get_backend() == "supabase":
        return _insertar_historial_supabase(resultados, id_usuario, id_sesion, dataset_nombre, duracion_segundos)
    return _insertar_historial_sqlite(resultados, id_usuario, id_sesion, dataset_nombre, duracion_segundos)


def obtener_historial_ejecuciones(id_usuario: Optional[int] = None, limit: int = 50) -> list:
    """Devuelve el historial de ejecuciones para el dashboard de auditoría."""
    if get_backend() == "supabase":
        return _obtener_historial_supabase(id_usuario, limit)
    return _obtener_historial_sqlite(id_usuario, limit)


# ── Configuraciones de modelo ─────────────────────────────────────────────────

def guardar_configuracion_modelo(
    configuracion: Dict,
    id_usuario: int,
    id_sesion: str,
    usuario: str,
) -> int:
    """Guarda la configuración del modelo y devuelve el ID generado."""
    if get_backend() == "supabase":
        return _guardar_config_supabase(configuracion, id_usuario, id_sesion)
    return _guardar_config_sqlite(configuracion, id_usuario, id_sesion)


def obtener_configuracion_modelo(
    id_sesion: str,
    usuario: str,
    id_configuracion: Optional[int] = None,
    id_usuario: Optional[int] = None,
) -> Optional[Dict]:
    """Obtiene la configuración más reciente que coincida con los filtros dados."""
    if get_backend() == "supabase":
        return _obtener_config_supabase(id_sesion, id_configuracion, id_usuario)
    return _obtener_config_sqlite(id_sesion, id_configuracion, id_usuario)


# ══════════════════════════════════════════════════════════════════════════════
# IMPLEMENTACIÓN SQLITE
# ══════════════════════════════════════════════════════════════════════════════

def _insertar_benchmarking_sqlite(resultados, id_usuario, id_sesion) -> int:
    from src.database.sqlite_conn import get_connection
    resultados_json = json.dumps(resultados, default=str)
    mejor_modelo = ""
    if resultados.get("mejor_modelo"):
        mejor_modelo = resultados["mejor_modelo"].get("nombre", "")

    conn = get_connection()
    try:
        cursor = conn.execute(
            """INSERT INTO benchmarking_modelos
               (id_usuario, id_sesion, tipo_problema, variable_objetivo,
                mejor_modelo, cant_exitosos, cant_fallidos, resultados_completos, fecha_ejecucion)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                id_usuario, id_sesion,
                resultados.get("tipo_problema", ""),
                resultados.get("variable_objetivo", ""),
                mejor_modelo,
                len(resultados.get("modelos_exitosos", [])),
                len(resultados.get("modelos_fallidos", [])),
                resultados_json,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def _obtener_ultimo_benchmarking_sqlite(id_usuario) -> Optional[Dict]:
    from src.database.sqlite_conn import get_connection
    conn = get_connection()
    try:
        if id_usuario:
            row = conn.execute(
                "SELECT resultados_completos FROM benchmarking_modelos WHERE id_usuario = ? ORDER BY fecha_ejecucion DESC LIMIT 1",
                (id_usuario,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT resultados_completos FROM benchmarking_modelos ORDER BY fecha_ejecucion DESC LIMIT 1"
            ).fetchone()
        if row and row["resultados_completos"]:
            return json.loads(row["resultados_completos"])
        return None
    finally:
        conn.close()


def _obtener_benchmarking_por_id_sqlite(benchmarking_id) -> Optional[Dict]:
    from src.database.sqlite_conn import get_connection
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT resultados_completos FROM benchmarking_modelos WHERE id = ?",
            (benchmarking_id,),
        ).fetchone()
        if row and row["resultados_completos"]:
            return json.loads(row["resultados_completos"])
        return None
    finally:
        conn.close()


def _extraer_metrica_principal(resultados: Dict):
    """Extrae nombre y valor de la métrica principal del mejor modelo."""
    mejor = resultados.get("mejor_modelo") or {}
    metricas = mejor.get("metricas", {})
    tipo = resultados.get("tipo_problema", "")
    if tipo == "clasificacion":
        return "accuracy", metricas.get("accuracy")
    elif tipo == "regresion":
        return "r2", metricas.get("r2")
    return "", None


def _insertar_historial_sqlite(resultados, id_usuario, id_sesion, dataset_nombre, duracion_segundos) -> int:
    from src.database.sqlite_conn import get_connection
    mejor = resultados.get("mejor_modelo") or {}
    metrica_nombre, metrica_valor = _extraer_metrica_principal(resultados)

    conn = get_connection()
    try:
        cursor = conn.execute(
            """INSERT INTO historial_ejecuciones
               (id_usuario, id_sesion, dataset_nombre, tipo_problema, variable_objetivo,
                modelo_ganador, metrica_nombre, metrica_valor, total_modelos,
                modelos_exitosos, duracion_segundos, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                id_usuario, id_sesion, dataset_nombre,
                resultados.get("tipo_problema", ""),
                resultados.get("variable_objetivo", ""),
                mejor.get("nombre", ""),
                metrica_nombre, metrica_valor,
                resultados.get("total_modelos", 0),
                len(resultados.get("modelos_exitosos", [])),
                duracion_segundos,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def _obtener_historial_sqlite(id_usuario, limit) -> list:
    from src.database.sqlite_conn import get_connection
    conn = get_connection()
    try:
        if id_usuario:
            rows = conn.execute(
                """SELECT id, dataset_nombre, tipo_problema, variable_objetivo, modelo_ganador,
                          metrica_nombre, metrica_valor, modelos_exitosos, total_modelos,
                          duracion_segundos, timestamp
                   FROM historial_ejecuciones
                   WHERE id_usuario = ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (id_usuario, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT id, dataset_nombre, tipo_problema, variable_objetivo, modelo_ganador,
                          metrica_nombre, metrica_valor, modelos_exitosos, total_modelos,
                          duracion_segundos, timestamp
                   FROM historial_ejecuciones
                   ORDER BY timestamp DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _guardar_config_sqlite(configuracion, id_usuario, id_sesion) -> int:
    from src.database.sqlite_conn import get_connection
    conn = get_connection()
    try:
        cursor = conn.execute(
            """INSERT INTO configuraciones_modelo
               (id_usuario, id_sesion, tipo_problema, variable_objetivo,
                variables_predictoras, configuracion_completa, fecha_creacion)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                id_usuario, id_sesion,
                str(configuracion.get("tipo_problema", "")),
                str(configuracion.get("variable_objetivo", "")),
                json.dumps(configuracion.get("variables_predictoras", [])),
                json.dumps(configuracion),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def _obtener_config_sqlite(id_sesion, id_configuracion, id_usuario) -> Optional[Dict]:
    from src.database.sqlite_conn import get_connection
    conn = get_connection()
    try:
        query = "SELECT configuracion_completa FROM configuraciones_modelo WHERE 1=1"
        params = []
        if id_configuracion:
            query += " AND id = ?"
            params.append(id_configuracion)
        if id_usuario:
            query += " AND id_usuario = ?"
            params.append(id_usuario)
        query += " ORDER BY fecha_creacion DESC LIMIT 1"
        row = conn.execute(query, params).fetchone()
        if row and row["configuracion_completa"]:
            return json.loads(row["configuracion_completa"])
        return None
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# IMPLEMENTACIÓN SUPABASE
# ══════════════════════════════════════════════════════════════════════════════

def _insertar_benchmarking_supabase(resultados, id_usuario, id_sesion) -> int:
    from src.database.supabase_conn import get_client
    client = get_client()
    mejor_modelo = ""
    if resultados.get("mejor_modelo"):
        mejor_modelo = resultados["mejor_modelo"].get("nombre", "")

    result = client.table("benchmarking_modelos").insert({
        "id_usuario": id_usuario,
        "id_sesion": id_sesion,
        "tipo_problema": resultados.get("tipo_problema", ""),
        "variable_objetivo": resultados.get("variable_objetivo", ""),
        "mejor_modelo": mejor_modelo,
        "cant_exitosos": len(resultados.get("modelos_exitosos", [])),
        "cant_fallidos": len(resultados.get("modelos_fallidos", [])),
        "resultados_completos": resultados,
    }).execute()
    if result.data:
        return result.data[0].get("id", 0)
    return 0


def _obtener_ultimo_benchmarking_supabase(id_usuario) -> Optional[Dict]:
    from src.database.supabase_conn import get_client
    client = get_client()
    query = client.table("benchmarking_modelos") \
        .select("resultados_completos") \
        .order("fecha_ejecucion", desc=True).limit(1)
    if id_usuario:
        query = query.eq("id_usuario", id_usuario)
    result = query.execute()
    if result.data and result.data[0].get("resultados_completos"):
        data = result.data[0]["resultados_completos"]
        return data if isinstance(data, dict) else json.loads(data)
    return None


def _obtener_benchmarking_por_id_supabase(benchmarking_id) -> Optional[Dict]:
    from src.database.supabase_conn import get_client
    client = get_client()
    result = client.table("benchmarking_modelos") \
        .select("resultados_completos").eq("id", benchmarking_id).limit(1).execute()
    if result.data and result.data[0].get("resultados_completos"):
        data = result.data[0]["resultados_completos"]
        return data if isinstance(data, dict) else json.loads(data)
    return None


def _insertar_historial_supabase(resultados, id_usuario, id_sesion, dataset_nombre, duracion_segundos) -> int:
    from src.database.supabase_conn import get_client
    client = get_client()
    mejor = resultados.get("mejor_modelo") or {}
    metrica_nombre, metrica_valor = _extraer_metrica_principal(resultados)

    result = client.table("historial_ejecuciones").insert({
        "id_usuario": id_usuario,
        "id_sesion": id_sesion,
        "dataset_nombre": dataset_nombre,
        "tipo_problema": resultados.get("tipo_problema", ""),
        "variable_objetivo": resultados.get("variable_objetivo", ""),
        "modelo_ganador": mejor.get("nombre", ""),
        "metrica_nombre": metrica_nombre,
        "metrica_valor": metrica_valor,
        "total_modelos": resultados.get("total_modelos", 0),
        "modelos_exitosos": len(resultados.get("modelos_exitosos", [])),
        "duracion_segundos": duracion_segundos,
    }).execute()
    if result.data:
        return result.data[0].get("id", 0)
    return 0


def _obtener_historial_supabase(id_usuario, limit) -> list:
    from src.database.supabase_conn import get_client
    client = get_client()
    query = client.table("historial_ejecuciones") \
        .select("id, dataset_nombre, tipo_problema, variable_objetivo, modelo_ganador, "
                "metrica_nombre, metrica_valor, modelos_exitosos, total_modelos, "
                "duracion_segundos, timestamp") \
        .order("timestamp", desc=True).limit(limit)
    if id_usuario:
        query = query.eq("id_usuario", id_usuario)
    result = query.execute()
    return result.data if result.data else []


def _guardar_config_supabase(configuracion, id_usuario, id_sesion) -> int:
    from src.database.supabase_conn import get_client
    client = get_client()
    result = client.table("configuraciones_modelo").insert({
        "id_usuario": id_usuario,
        "id_sesion": id_sesion,
        "tipo_problema": str(configuracion.get("tipo_problema", "")),
        "variable_objetivo": str(configuracion.get("variable_objetivo", "")),
        "variables_predictoras": configuracion.get("variables_predictoras", []),
        "configuracion_completa": configuracion,
    }).execute()
    if result.data:
        return result.data[0].get("id", 0)
    return 0


def _obtener_config_supabase(id_sesion, id_configuracion, id_usuario) -> Optional[Dict]:
    from src.database.supabase_conn import get_client
    client = get_client()
    query = client.table("configuraciones_modelo") \
        .select("configuracion_completa") \
        .order("fecha_creacion", desc=True).limit(1)
    if id_configuracion:
        query = query.eq("id", id_configuracion)
    if id_usuario:
        query = query.eq("id_usuario", id_usuario)
    result = query.execute()
    if result.data and result.data[0].get("configuracion_completa"):
        data = result.data[0]["configuracion_completa"]
        return data if isinstance(data, dict) else json.loads(data)
    return None
