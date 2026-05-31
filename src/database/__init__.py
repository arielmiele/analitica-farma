"""
Módulo de base de datos local SQLite para Analítica Farma.
Reemplaza la capa de Snowflake por una arquitectura 100% local.
"""
from src.database.init_db import init_db

__all__ = ["init_db"]
