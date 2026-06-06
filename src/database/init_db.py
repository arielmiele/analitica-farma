"""
Inicialización del esquema y datos iniciales.
Soporta SQLite (local) y Supabase (cloud) según DB_BACKEND.
"""
import bcrypt
from datetime import datetime
from src.database.backend import get_backend

_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS usuarios (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre          TEXT    NOT NULL,
    email           TEXT    UNIQUE NOT NULL,
    hash_password   TEXT    NOT NULL,
    rol             TEXT    NOT NULL DEFAULT 'usuario',
    activo          INTEGER NOT NULL DEFAULT 1,
    fecha_creacion  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS sesiones (
    id_sesion       TEXT    PRIMARY KEY,
    id_usuario      INTEGER NOT NULL,
    fecha_inicio    TEXT    NOT NULL,
    fecha_fin       TEXT,
    estado          TEXT    NOT NULL DEFAULT 'ACTIVA',
    FOREIGN KEY (id_usuario) REFERENCES usuarios(id)
);

CREATE TABLE IF NOT EXISTS datasets (
    id_dataset      TEXT    PRIMARY KEY,
    nombre          TEXT    NOT NULL,
    descripcion     TEXT,
    fecha_creacion  TEXT    NOT NULL,
    id_usuario      INTEGER NOT NULL,
    ruta_archivo    TEXT    NOT NULL,
    filas           INTEGER,
    columnas        INTEGER,
    FOREIGN KEY (id_usuario) REFERENCES usuarios(id)
);

CREATE TABLE IF NOT EXISTS configuraciones_modelo (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    id_usuario              INTEGER NOT NULL,
    id_sesion               TEXT    NOT NULL,
    tipo_problema           TEXT,
    variable_objetivo       TEXT,
    variables_predictoras   TEXT,
    configuracion_completa  TEXT,
    fecha_creacion          TEXT    NOT NULL,
    FOREIGN KEY (id_usuario) REFERENCES usuarios(id)
);

CREATE TABLE IF NOT EXISTS benchmarking_modelos (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    id_usuario          INTEGER NOT NULL,
    id_sesion           TEXT    NOT NULL,
    id_dataset          TEXT,
    tipo_problema       TEXT,
    variable_objetivo   TEXT,
    mejor_modelo        TEXT,
    cant_exitosos       INTEGER,
    cant_fallidos       INTEGER,
    resultados_completos TEXT,
    fecha_ejecucion     TEXT    NOT NULL,
    FOREIGN KEY (id_usuario) REFERENCES usuarios(id)
);

CREATE TABLE IF NOT EXISTS reportes (
    id_reporte      TEXT    PRIMARY KEY,
    nombre          TEXT    NOT NULL,
    tipo            TEXT,
    fecha           TEXT    NOT NULL,
    id_usuario      INTEGER,
    id_sesion       TEXT,
    id_benchmarking INTEGER,
    id_dataset      TEXT,
    contenido       TEXT,
    FOREIGN KEY (id_usuario) REFERENCES usuarios(id)
);

CREATE TABLE IF NOT EXISTS auditoria (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    usuario     TEXT,
    accion      TEXT,
    entidad     TEXT,
    id_entidad  TEXT,
    detalles    TEXT,
    fecha       TEXT,
    id_sesion   TEXT
);
CREATE TABLE IF NOT EXISTS historial_ejecuciones (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    id_usuario          INTEGER NOT NULL,
    id_sesion           TEXT    NOT NULL,
    dataset_nombre      TEXT,
    tipo_problema       TEXT,
    variable_objetivo   TEXT,
    modelo_ganador      TEXT,
    metrica_nombre      TEXT,
    metrica_valor       REAL,
    total_modelos       INTEGER,
    modelos_exitosos    INTEGER,
    duracion_segundos   REAL,
    timestamp           TEXT    NOT NULL,
    FOREIGN KEY (id_usuario) REFERENCES usuarios(id)
);
"""


def init_db() -> None:
    """Crea las tablas y el usuario admin por defecto si no existen."""
    if get_backend() == "supabase":
        _init_supabase()
    else:
        _init_sqlite()


def _init_sqlite() -> None:
    """Inicialización SQLite: crea tablas y usuario admin."""
    from src.database.sqlite_conn import get_connection
    conn = get_connection()
    try:
        conn.executescript(_CREATE_TABLES)
        conn.commit()
        _crear_usuario_admin_sqlite(conn)
    finally:
        conn.close()


def _init_supabase() -> None:
    """Inicialización Supabase: verifica conexión y crea usuario admin si tabla vacía."""
    from src.database.supabase_conn import get_client
    client = get_client()
    # Verificar si hay usuarios; si no, crear admin
    result = client.table("usuarios").select("id", count="exact").limit(1).execute()
    if result.count == 0:
        hash_pw = bcrypt.hashpw("admin123".encode(), bcrypt.gensalt()).decode()
        client.table("usuarios").insert({
            "nombre": "Administrador",
            "email": "admin@analitica-farma.com",
            "hash_password": hash_pw,
            "rol": "admin",
            "activo": 1,
        }).execute()


def _crear_usuario_admin_sqlite(conn) -> None:
    """Crea el usuario administrador inicial si la tabla está vacía."""
    row = conn.execute("SELECT COUNT(*) as cnt FROM usuarios").fetchone()
    if row["cnt"] == 0:
        hash_pw = bcrypt.hashpw("admin123".encode(), bcrypt.gensalt()).decode()
        conn.execute(
            """INSERT INTO usuarios (nombre, email, hash_password, rol, activo, fecha_creacion)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("Administrador", "admin@analitica-farma.com", hash_pw, "admin", 1,
             datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
        conn.commit()
