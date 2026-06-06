"""
Operaciones CRUD sobre la tabla usuarios.
Soporta SQLite (local) y Supabase (cloud) según DB_BACKEND.
"""
import bcrypt
from datetime import datetime
from typing import Optional, Dict
from src.database.backend import get_backend


def obtener_usuario_por_email(email: str) -> Optional[Dict]:
    """Devuelve los datos del usuario activo con ese email, o None si no existe."""
    if get_backend() == "supabase":
        return _obtener_usuario_por_email_supabase(email)
    return _obtener_usuario_por_email_sqlite(email)


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
    if get_backend() == "supabase":
        return _cargar_credentials_supabase()
    return _cargar_credentials_sqlite()


def crear_usuario(nombre: str, email: str, password: str, rol: str = "usuario") -> int:
    """
    Crea un nuevo usuario con contraseña hasheada en bcrypt.
    Devuelve el ID insertado.
    """
    hash_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    if get_backend() == "supabase":
        return _crear_usuario_supabase(nombre, email, hash_pw, rol)
    return _crear_usuario_sqlite(nombre, email, hash_pw, rol)


# ── Implementación SQLite ─────────────────────────────────────────────────────

def _obtener_usuario_por_email_sqlite(email: str) -> Optional[Dict]:
    from src.database.sqlite_conn import get_connection
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id, nombre, email, hash_password, rol, activo FROM usuarios WHERE email = ? AND activo = 1",
            (email.strip().lower(),),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def _cargar_credentials_sqlite() -> Dict:
    from src.database.sqlite_conn import get_connection
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


def _crear_usuario_sqlite(nombre: str, email: str, hash_pw: str, rol: str) -> int:
    from src.database.sqlite_conn import get_connection
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


# ── Implementación Supabase ───────────────────────────────────────────────────

def _obtener_usuario_por_email_supabase(email: str) -> Optional[Dict]:
    from src.database.supabase_conn import get_client
    client = get_client()
    result = client.table("usuarios").select("id, nombre, email, hash_password, rol, activo") \
        .eq("email", email.strip().lower()).eq("activo", 1).limit(1).execute()
    if result.data:
        return result.data[0]
    return None


def _cargar_credentials_supabase() -> Dict:
    from src.database.supabase_conn import get_client
    client = get_client()
    result = client.table("usuarios").select("nombre, email, hash_password") \
        .eq("activo", 1).execute()
    usernames = {}
    for row in result.data:
        usernames[row["email"]] = {
            "email": row["email"],
            "name": row["nombre"],
            "password": row["hash_password"],
        }
    return {"usernames": usernames}


def _crear_usuario_supabase(nombre: str, email: str, hash_pw: str, rol: str) -> int:
    from src.database.supabase_conn import get_client
    client = get_client()
    result = client.table("usuarios").insert({
        "nombre": nombre,
        "email": email.strip().lower(),
        "hash_password": hash_pw,
        "rol": rol,
        "activo": 1,
    }).execute()
    if result.data:
        return result.data[0]["id"]
    return 0
