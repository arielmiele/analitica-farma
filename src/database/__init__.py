"""
Módulo de base de datos para Analítica Farma.
Soporta SQLite (local) y Supabase (cloud) según DB_BACKEND.
"""
from src.database.init_db import init_db
from src.database.backend import get_backend

__all__ = ["init_db", "get_backend"]
