"""
Conexión centralizada a Supabase (REST API via supabase-py).
Se activa cuando DB_BACKEND=supabase.

Requiere credenciales en st.secrets o variables de entorno:
- SUPABASE_URL
- SUPABASE_KEY (service_role key para acceso sin RLS)
"""
import os
from functools import lru_cache


@lru_cache(maxsize=1)
def get_client():
    """
    Devuelve un cliente Supabase singleton.
    Lee credenciales de st.secrets (Streamlit Cloud) o env vars.
    """
    from supabase import create_client

    url = None
    key = None

    # Intentar leer de Streamlit secrets
    try:
        import streamlit as st
        supabase_secrets = st.secrets.get("supabase", {})
        url = supabase_secrets.get("url", "")
        key = supabase_secrets.get("key", "")
    except Exception:
        pass

    # Fallback a variables de entorno
    if not url:
        url = os.getenv("SUPABASE_URL", "")
    if not key:
        key = os.getenv("SUPABASE_KEY", "")

    if not url or not key:
        raise RuntimeError(
            "Supabase no configurado. Definir SUPABASE_URL y SUPABASE_KEY "
            "en st.secrets o como variables de entorno."
        )

    return create_client(url, key)


def get_storage():
    """Devuelve el cliente de Storage de Supabase para manejar archivos (datasets)."""
    client = get_client()
    return client.storage
