"""
Operaciones sobre benchmarking y configuraciones de modelos en SQLite.
"""
import json
from datetime import datetime
from typing import Optional, Dict
from src.database.sqlite_conn import get_connection


# ── Benchmarking ──────────────────────────────────────────────────────────────

def insertar_benchmarking_modelos(
    resultados: Dict,
    id_usuario: int,
    id_sesion: str,
    usuario: str,
) -> int:
    """
    Inserta los resultados del benchmarking en SQLite.
    Devuelve el ID autoincremental generado.
    """
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
                id_usuario,
                id_sesion,
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


def obtener_ultimo_benchmarking(id_usuario: Optional[int] = None) -> Optional[Dict]:
    """Devuelve los resultados del último benchmarking (filtrado por usuario si se indica)."""
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


def obtener_benchmarking_por_id(benchmarking_id: int) -> Optional[Dict]:
    """Devuelve los resultados de un benchmarking específico por su ID."""
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


# ── Historial de ejecuciones ──────────────────────────────────────────────────

def insertar_historial_ejecucion(
    resultados: Dict,
    id_usuario: int,
    id_sesion: str,
    dataset_nombre: str = "",
    duracion_segundos: float = 0.0,
) -> int:
    """
    Inserta un registro en historial_ejecuciones con los datos del benchmarking finalizado.
    Extrae el mejor modelo y su métrica principal para consultas rápidas de dashboard.
    """
    mejor = resultados.get("mejor_modelo") or {}
    metrica_nombre = ""
    metrica_valor = None
    metricas = mejor.get("metricas", {})
    tipo = resultados.get("tipo_problema", "")
    if tipo == "clasificacion":
        metrica_nombre = "accuracy"
        metrica_valor = metricas.get("accuracy")
    elif tipo == "regresion":
        metrica_nombre = "r2"
        metrica_valor = metricas.get("r2")

    conn = get_connection()
    try:
        cursor = conn.execute(
            """INSERT INTO historial_ejecuciones
               (id_usuario, id_sesion, dataset_nombre, tipo_problema, variable_objetivo,
                modelo_ganador, metrica_nombre, metrica_valor, total_modelos,
                modelos_exitosos, duracion_segundos, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                id_usuario,
                id_sesion,
                dataset_nombre,
                tipo,
                resultados.get("variable_objetivo", ""),
                mejor.get("nombre", ""),
                metrica_nombre,
                metrica_valor,
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


def obtener_historial_ejecuciones(id_usuario: Optional[int] = None, limit: int = 50) -> list:
    """
    Devuelve el historial de ejecuciones para el dashboard de auditoría.
    Si id_usuario es None, devuelve registros de todos los usuarios.
    """
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


# ── Configuraciones de modelo ─────────────────────────────────────────────────

def guardar_configuracion_modelo(
    configuracion: Dict,
    id_usuario: int,
    id_sesion: str,
    usuario: str,
) -> int:
    """Guarda la configuración del modelo en SQLite y devuelve el ID generado."""
    conn = get_connection()
    try:
        cursor = conn.execute(
            """INSERT INTO configuraciones_modelo
               (id_usuario, id_sesion, tipo_problema, variable_objetivo,
                variables_predictoras, configuracion_completa, fecha_creacion)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                id_usuario,
                id_sesion,
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


def obtener_configuracion_modelo(
    id_sesion: str,
    usuario: str,
    id_configuracion: Optional[int] = None,
    id_usuario: Optional[int] = None,
) -> Optional[Dict]:
    """Obtiene la configuración más reciente que coincida con los filtros dados."""
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
