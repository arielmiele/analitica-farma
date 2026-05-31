"""
Conexión centralizada a la base de datos SQLite local.
"""
import sqlite3
from pathlib import Path

# Ruta al archivo de base de datos (relativa a la raíz del proyecto)
_ROOT = Path(__file__).parent.parent.parent
DB_PATH = _ROOT / "data" / "analitica_farma.db"


def get_connection() -> sqlite3.Connection:
    """
    Devuelve una conexión SQLite con foreign keys activadas y row_factory configurada.
    Usa WAL para mejor concurrencia en Streamlit (multi-thread).
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn
