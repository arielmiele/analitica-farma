"""
Operaciones CRUD y utilitarias sobre la tabla SESIONES en Snowflake.
Incluye funciones para registrar y consultar sesiones activas de usuarios.
"""
from src.snowflake.snowflake_conn import get_native_snowflake_connection
from typing import Optional, Dict
from src.audit.logger import log_audit


def obtener_sesion_por_id(id_sesion: str, usuario: str) -> Optional[Dict]:
    conn = get_native_snowflake_connection()
    if not conn:
        log_audit(id_sesion, usuario, "ERROR_CONEXION", "sesiones_db", "No se pudo obtener la conexión a Snowflake.")
        return None
    cur = None
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT ID_SESION, USUARIO, FECHA_INICIO, FECHA_FIN, ESTADO
            FROM SESIONES WHERE ID_SESION = %s
        """, (id_sesion,))
        row = cur.fetchone()
        if row:
            log_audit(id_sesion, usuario, "OBTENER_SESION_OK", "sesiones_db", f"Sesión {id_sesion} obtenida correctamente.")
            return dict(zip([col[0] for col in cur.description], row))
        log_audit(id_sesion, usuario, "SESION_NO_ENCONTRADA", "sesiones_db", f"No se encontró la sesión {id_sesion}.")
        return None
    except Exception as e:
        log_audit(id_sesion, usuario, "ERROR_OBTENER_SESION", "sesiones_db", f"Error al obtener sesión: {e}")
        return None
    finally:
        if cur is not None:
            cur.close()
        conn.close()
