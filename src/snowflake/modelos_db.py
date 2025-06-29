"""
Operaciones CRUD y utilitarias sobre la tabla MODELOS en Snowflake.
Incluye funciones para registrar, consultar y actualizar modelos entrenados.
"""
from src.snowflake.snowflake_conn import get_native_snowflake_connection
from typing import Optional, Dict
from src.audit.logger import log_audit


def obtener_modelo_por_id(id_modelo: str, id_sesion: str, usuario: str) -> Optional[Dict]:
    conn = get_native_snowflake_connection()
    if not conn:
        log_audit(id_sesion, usuario, "ERROR_CONEXION", "modelos_db", "No se pudo obtener la conexión a Snowflake.")
        return None
    cur = None
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT ID_MODELO, NOMBRE, TIPO, FECHA_ENTRENAMIENTO, USUARIO_CREADOR, METADATOS, MODELO_SERIALIZADO, ID_DATASET
            FROM MODELOS WHERE ID_MODELO = %s
        """, (id_modelo,))
        row = cur.fetchone()
        if row:
            log_audit(id_sesion, usuario, "OBTENER_MODELO_OK", "modelos_db", f"Modelo {id_modelo} obtenido correctamente.")
            return dict(zip([col[0] for col in cur.description], row))
        log_audit(id_sesion, usuario, "MODELO_NO_ENCONTRADO", "modelos_db", f"No se encontró el modelo {id_modelo}.")
        return None
    except Exception as e:
        log_audit(id_sesion, usuario, "ERROR_OBTENER_MODELO", "modelos_db", f"Error al obtener modelo: {e}")
        return None
    finally:
        if cur is not None:
            cur.close()
        conn.close()


def insertar_benchmarking_modelos(resultados: Dict, id_usuario: int, id_sesion: str, usuario: str) -> int:
    """
    Inserta los resultados del benchmarking en la tabla BENCHMARKING_MODELOS de Snowflake.
    Retorna el ID autoincremental generado.
    """
    conn = get_native_snowflake_connection()
    if not conn:
        log_audit(
            usuario=usuario,
            accion="ERROR_CONEXION",
            entidad="modelos_db",
            id_entidad="N/A",
            detalles="No se pudo obtener la conexión a Snowflake.",
            id_sesion=id_sesion
        )
        raise Exception("No se pudo conectar a Snowflake")
    cur = None
    try:
        cur = conn.cursor()
        import json
        resultados_json = json.dumps(resultados)
        sql = """
            INSERT INTO BENCHMARKING_MODELOS (
                ID_USUARIO, ID_SESION, USUARIO, TIPO_PROBLEMA, VARIABLE_OBJETIVO,
                CANTIDAD_MODELOS_EXITOSOS, CANTIDAD_MODELOS_FALLIDOS, MEJOR_MODELO,
                RESULTADOS_COMPLETOS, FECHA_EJECUCION
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
        """
        cur.execute(sql, (
            id_usuario,
            id_sesion,
            usuario,
            resultados['tipo_problema'],
            resultados['variable_objetivo'],
            len(resultados['modelos_exitosos']),
            len(resultados['modelos_fallidos']),
            resultados['mejor_modelo']['nombre'] if resultados['mejor_modelo'] else '',
            resultados_json
        ))
        # Obtener el ID generado (Snowflake: LAST_QUERY_ID y luego buscar el ID insertado)
        cur.execute("SELECT MAX(ID) FROM BENCHMARKING_MODELOS WHERE ID_USUARIO = %s AND ID_SESION = %s", (id_usuario, id_sesion))
        row = cur.fetchone()
        benchmarking_id = row[0] if row and row[0] is not None else None
        log_audit(
            usuario=usuario,
            accion="GUARDAR_BENCHMARKING",
            entidad="modelos_db",
            id_entidad=str(benchmarking_id),
            detalles=f"Resultados de benchmarking guardados en Snowflake con ID {benchmarking_id}",
            id_sesion=id_sesion
        )
        return int(benchmarking_id) if benchmarking_id is not None else 0
    except Exception as e:
        log_audit(
            usuario=usuario,
            accion="ERROR_GUARDAR_BENCHMARKING",
            entidad="modelos_db",
            id_entidad="N/A",
            detalles=f"Error al guardar benchmarking en Snowflake: {str(e)}",
            id_sesion=id_sesion
        )
        raise
    finally:
        if cur is not None:
            cur.close()
        conn.close()
