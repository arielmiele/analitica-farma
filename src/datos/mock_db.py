import sqlite3

# Script para crear una base de datos mock (SQLite) para desarrollo y pruebas de la aplicación analitica-farma.
# Define la estructura de tablas principales, relaciones y restricciones para simular el backend de datos.
# Además, inserta un usuario de ejemplo automáticamente si la tabla 'usuarios' está vacía.

def crear_mock_db(db_path="analitica_farma.db"):
    # Conexión a la base de datos SQLite (se crea el archivo si no existe)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Habilitar claves foráneas en SQLite para mantener integridad referencial
    cursor.execute("PRAGMA foreign_keys = ON;")

    # Sentencias de creación de tablas principales del modelo de datos
    tablas_sql = [
        # Tabla de usuarios: almacena usuarios de la app, con roles y correo único
        """CREATE TABLE IF NOT EXISTS usuarios (
            id_usuario INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre TEXT NOT NULL,
            correo TEXT UNIQUE NOT NULL,
            rol TEXT NOT NULL CHECK (rol IN ('analista', 'administrador')),
            fecha_creacion DATETIME DEFAULT CURRENT_TIMESTAMP
        );""",

        # Tabla de datasets: almacena los datasets cargados por los usuarios
        """CREATE TABLE IF NOT EXISTS datasets (
            id_dataset INTEGER PRIMARY KEY AUTOINCREMENT,
            id_usuario INTEGER NOT NULL,
            nombre TEXT NOT NULL,
            origen TEXT CHECK (origen IN ('csv', 'snowflake')),
            fecha_carga DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (id_usuario) REFERENCES usuarios(id_usuario)
        );""",

        # Tabla de transformaciones: historial de transformaciones aplicadas a cada dataset
        """CREATE TABLE IF NOT EXISTS transformaciones (
            id_transformacion INTEGER PRIMARY KEY AUTOINCREMENT,
            id_dataset INTEGER NOT NULL,
            tipo TEXT NOT NULL,
            descripcion TEXT,
            fecha_aplicacion DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (id_dataset) REFERENCES datasets(id_dataset)
        );""",

        # Tabla de modelos: catálogo de modelos de ML disponibles
        """CREATE TABLE IF NOT EXISTS modelos (
            id_modelo INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre_modelo TEXT NOT NULL,
            libreria TEXT NOT NULL,
            tipo TEXT NOT NULL CHECK (tipo IN ('clasificacion', 'regresion'))
        );""",

        # Tabla de ejecuciones: resultados de ejecuciones de modelos sobre datasets
        """CREATE TABLE IF NOT EXISTS ejecuciones (
            id_ejecucion INTEGER PRIMARY KEY AUTOINCREMENT,
            id_dataset INTEGER NOT NULL,
            id_modelo INTEGER NOT NULL,
            metrica TEXT NOT NULL,
            valor_metrica REAL NOT NULL,
            es_recomendado BOOLEAN DEFAULT 0,
            fecha_ejecucion DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (id_dataset) REFERENCES datasets(id_dataset),
            FOREIGN KEY (id_modelo) REFERENCES modelos(id_modelo)
        );""",

        # Tabla de reportes: reportes generados a partir de los análisis
        """CREATE TABLE IF NOT EXISTS reportes (
            id_reporte INTEGER PRIMARY KEY AUTOINCREMENT,
            id_dataset INTEGER NOT NULL,
            nombre_archivo TEXT NOT NULL,
            formato TEXT CHECK (formato IN ('pdf', 'csv')),
            fecha_generacion DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (id_dataset) REFERENCES datasets(id_dataset)
        );""",

        # Tabla de auditoría: registro de acciones relevantes para trazabilidad
        """CREATE TABLE IF NOT EXISTS auditoria (
            id_log INTEGER PRIMARY KEY AUTOINCREMENT,
            id_usuario INTEGER NOT NULL,
            accion TEXT NOT NULL,
            descripcion TEXT,
            fecha DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (id_usuario) REFERENCES usuarios(id_usuario)
        );"""
    ]

    # Ejecutar todas las sentencias SQL para crear las tablas
    for sql in tablas_sql:
        cursor.execute(sql)

    # Insertar un usuario de ejemplo si la tabla está vacía
    # Esto facilita el acceso inicial y las pruebas de la app
    cursor.execute("SELECT COUNT(*) FROM usuarios")
    if cursor.fetchone()[0] == 0:
        cursor.execute(
            """
            INSERT INTO usuarios (nombre, correo, rol)
            VALUES (?, ?, ?)
            """,
            ("usuario", "usuario@empresa.com", "analista")
        )
        print("Usuario de ejemplo insertado en la tabla 'usuarios'.")
    
    conn.commit()
    conn.close()
    print(f"Base de datos '{db_path}' creada con éxito.")

# Si el script se ejecuta directamente, crea la base de datos mock y el usuario de ejemplo
if __name__ == "__main__":
    crear_mock_db()