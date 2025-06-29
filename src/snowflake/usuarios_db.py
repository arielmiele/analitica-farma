"""
Operaciones CRUD y utilitarias sobre la tabla USUARIOS en Snowflake.
Incluye funciones para crear, leer, actualizar y eliminar usuarios, así como validación de login.
"""
from src.snowflake.snowflake_conn import get_native_snowflake_connection
from typing import Optional, Dict
from src.audit.logger import log_audit


def obtener_usuario_por_email(email: str, id_sesion: str, usuario: str) -> Optional[Dict]:
    conn = get_native_snowflake_connection()
    if not conn:
        log_audit(id_sesion, usuario, "ERROR_CONEXION", "usuarios_db", "No se pudo obtener la conexión a Snowflake.")
        return None
    cur = None
    try:
        cur = conn.cursor()
        # Sanitizar el email para evitar inyección
        email_safe = email.replace("'", "")
        cur.execute(f"""
            SELECT ID_USUARIO, NOMBRE, EMAIL, HASH_PASSWORD, ROL, ACTIVO, FECHA_CREACION
            FROM USUARIOS WHERE EMAIL = '{email_safe}'
        """)
        row = cur.fetchone()
        if row:
            log_audit(id_sesion, usuario, "OBTENER_USUARIO_OK", "usuarios_db", f"Usuario {email_safe} obtenido correctamente.")
            return dict(zip([col[0] for col in cur.description], row))
        log_audit(id_sesion, usuario, "USUARIO_NO_ENCONTRADO", "usuarios_db", f"No se encontró el usuario {email_safe}.")
        return None
    except Exception as e:
        log_audit(id_sesion, usuario, "ERROR_OBTENER_USUARIO", "usuarios_db", f"Error al obtener usuario: {e}")
        return None
    finally:
        if cur is not None:
            cur.close()
