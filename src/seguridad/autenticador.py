"""
Módulo de autenticación para validar usuarios contra la tabla USUARIOS en Snowflake.
"""
import hashlib
from src.datos.snowflake_conn import get_snowflake_connection

def validar_usuario(usuario: str, password: str):
    """Valida usuario y contraseña contra la tabla USUARIOS en Snowflake.
    Retorna dict con datos del usuario si es válido, o None si no lo es."""
    usuario = usuario.strip().lower()
    password = password.strip()
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    try:
        conn = get_snowflake_connection()
        query = (
            "SELECT ID, USUARIO, EMAIL, ROL "
            "FROM ANALITICA_FARMA.PUBLIC.USUARIOS "
            "WHERE LOWER(USUARIO) = ? AND PASSWORD_HASH = ? AND ACTIVO = TRUE"
        )
        cur = conn.cursor()
        cur.execute(query, (usuario, password_hash))
        result = cur.fetchone()
        if result:
            return {
                "id": result[0],
                "usuario": result[1],
                "email": result[2],
                "rol": result[3]
            }
        else:
            return None
    except Exception:
        # Aquí podrías loguear el error
        return None
