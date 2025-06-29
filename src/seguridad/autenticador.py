"""
Módulo de autenticación para validar usuarios contra la tabla USUARIOS en Snowflake.
Utiliza el módulo src.datos.usuarios_db para la consulta.
"""
import hashlib
from typing import Optional
from src.snowflake.usuarios_db import obtener_usuario_por_email
from src.audit.logger import log_audit

def validar_usuario(usuario: str, password: str, id_sesion: Optional[str] = None):
    """
    Valida usuario y contraseña contra la tabla USUARIOS en Snowflake.
    Retorna dict con datos del usuario si es válido, o None si no lo es.
    Registra auditoría de intento de login.
    """
    usuario_limpio = usuario.strip().lower()
    password = password.strip()
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    user_data = obtener_usuario_por_email(
        usuario_limpio,
        id_sesion=id_sesion if id_sesion is not None else "",
        usuario=usuario_limpio
    )
    if user_data and user_data.get("HASH_PASSWORD") == password_hash and user_data.get("ACTIVO"):
        log_audit(
            usuario=user_data["ID_USUARIO"],
            accion="LOGIN_EXITOSO",
            entidad="USUARIOS",
            id_entidad=str(user_data["ID_USUARIO"]),
            detalles=f"Login exitoso para usuario {usuario_limpio}",
            id_sesion=id_sesion
        )
        return {
            "id": user_data["ID_USUARIO"],
            "usuario": user_data["NOMBRE"],
            "email": user_data["EMAIL"],
            "rol": user_data["ROL"]
        }
    else:
        log_audit(
            usuario=usuario_limpio,
            accion="LOGIN_FALLIDO",
            entidad="USUARIOS",
            id_entidad="",
            detalles=f"Login fallido para usuario {usuario_limpio}",
            id_sesion=id_sesion
        )
    return None
