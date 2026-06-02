"""
Operaciones CRUD sobre la tabla usuarios en SQLite.
"""
import bcrypt
from datetime import datetime
from typing import Optional, Dict
from src.database.sqlite_conn import get_connection


def obtener_usuario_por_email(email: str) -> Optional[Dict]:
    """Devuelve los datos del usuario activo con ese email, o None si no existe."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id, nombre, email, hash_password, rol, activo FROM usuarios WHERE email = ? AND activo = 1",
            (email.strip().lower(),),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def validar_credenciales(email: str, password: str) -> Optional[Dict]:
    """
    Valida email y contraseña usando bcrypt.
    Devuelve dict con datos del usuario si es válido, None en caso contrario.
    """
    user = obtener_usuario_por_email(email)
    if not user:
        return None
    if bcrypt.checkpw(password.encode(), user["hash_password"].encode()):
        return user
    return None


def cargar_credentials_para_auth() -> Dict:
    """
    Carga usuarios activos en el formato que espera streamlit-authenticator.
    Usa el email como clave de usuario (username).
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT nombre, email, hash_password FROM usuarios WHERE activo = 1"
        ).fetchall()
    finally:
        conn.close()

    usernames = {}
    for row in rows:
        usernames[row["email"]] = {
            "email": row["email"],
            "name": row["nombre"],
            "password": row["hash_password"],
        }
    return {"usernames": usernames}


def crear_usuario(nombre: str, email: str, password: str, rol: str = "usuario") -> int:
    """
    Crea un nuevo usuario con contraseña hasheada en bcrypt.
    Devuelve el ID insertado.
    """
    hash_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    conn = get_connection()
    try:
        cursor = conn.execute(
            """INSERT INTO usuarios (nombre, email, hash_password, rol, activo, fecha_creacion)
               VALUES (?, ?, ?, ?, 1, ?)""",
            (nombre, email.strip().lower(), hash_pw, rol,
             datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()
