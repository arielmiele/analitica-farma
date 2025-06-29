"""
Operaciones CRUD y utilitarias sobre la tabla REPORTES en Snowflake.
Incluye funciones para registrar y consultar reportes generados.
"""
from src.snowflake.snowflake_conn import get_native_snowflake_connection
from typing import Optional, Dict
from src.audit.logger import log_audit


def obtener_reporte_por_id(id_reporte: str, id_sesion: str, usuario: str) -> Optional[Dict]:
    conn = get_native_snowflake_connection()
    if not conn:
        log_audit(id_sesion, usuario, "ERROR_CONEXION", "reportes_db", "No se pudo obtener la conexión a Snowflake.")
        return None
    cur = None
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT ID_REPORTE, NOMBRE, TIPO, FECHA, USUARIO, ID_MODELO, ID_DATASET, REPORTE
            FROM REPORTES WHERE ID_REPORTE = %s
        """, (id_reporte,))
        row = cur.fetchone()
        if row:
            log_audit(id_sesion, usuario, "OBTENER_REPORTE_OK", "reportes_db", f"Reporte {id_reporte} obtenido correctamente.")
            return dict(zip([col[0] for col in cur.description], row))
        log_audit(id_sesion, usuario, "REPORTE_NO_ENCONTRADO", "reportes_db", f"No se encontró el reporte {id_reporte}.")
        return None
    except Exception as e:
        log_audit(id_sesion, usuario, "ERROR_OBTENER_REPORTE", "reportes_db", f"Error al obtener reporte: {e}")
        return None
    finally:
        if cur is not None:
            cur.close()
        conn.close()
