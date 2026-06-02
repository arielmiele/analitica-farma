"""
Módulo de autenticación local.
Valida usuarios contra la tabla usuarios en SQLite usando bcrypt.
"""
from typing import Optional
from src.database.usuarios_db import validar_credenciales
from src.audit.logger import log_audit


def validar_usuario(usuario: str, password: str, id_sesion: Optional[str] = None):
    """
    Valida usuario y contraseña contra la tabla usuarios en SQLite.
    Retorna dict con datos del usuario si es válido, o None si no lo es.
    """
    email = usuario.strip().lower()
    user_data = validar_credenciales(email, password.strip())

    if user_data:
        log_audit(
            usuario=str(user_data["id"]),
            accion="LOGIN_EXITOSO",
            entidad="USUARIOS",
            id_entidad=str(user_data["id"]),
            detalles=f"Login exitoso para {email}",
            id_sesion=id_sesion,
        )
        return {
            "id": user_data["id"],
            "usuario": user_data["nombre"],
            "email": user_data["email"],
            "rol": user_data["rol"],
        }

    log_audit(
        usuario=email,
        accion="LOGIN_FALLIDO",
        entidad="USUARIOS",
        id_entidad="",
        detalles=f"Login fallido para {email}",
        id_sesion=id_sesion,
    )
    return None

