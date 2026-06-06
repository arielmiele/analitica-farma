"""
Selector de backend de base de datos.
Soporta 'sqlite' (local, por defecto) y 'supabase' (cloud/producción).

Configuración:
- Variable de entorno DB_BACKEND=sqlite|supabase
- O en .streamlit/secrets.toml: [general] DB_BACKEND = "supabase"
"""
import os


def get_backend() -> str:
    """
    Determina el backend activo. Orden de prioridad:
    1. st.secrets (para Streamlit Cloud)
    2. Variable de entorno DB_BACKEND
    3. Default: 'sqlite'
    """
    # Intentar leer de Streamlit secrets (disponible en Cloud)
    try:
        import streamlit as st
        backend = st.secrets.get("general", {}).get("DB_BACKEND", "")
        if backend:
            return backend.lower()
    except Exception:
        pass

    return os.getenv("DB_BACKEND", "sqlite").lower()
