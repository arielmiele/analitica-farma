import sqlite3

def crear_mock_db(db_path="mock_data.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Habilitar claves foráneas en SQLite
    cursor.execute("PRAGMA foreign_keys = ON;")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analisis (
            analisis_id INTEGER PRIMARY KEY AUTOINCREMENT,
            usuario_id INTEGER NOT NULL,
            fecha_inicio TEXT NOT NULL,
            fecha_fin TEXT NOT NULL,
            estado TEXT NOT NULL,
            nombre_personalizado TEXT,
            FOREIGN KEY (usuario_id) REFERENCES usuarios(id_usuario) ON DELETE CASCADE
        );
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dataset (
            dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
            analisis_id INTEGER NOT NULL,
            nombre_archivo TEXT NOT NULL,
            origen TEXT NOT NULL,
            fecha_carga TEXT NOT NULL,
            cantidad_filas INTEGER NOT NULL,
            cantidad_columnas INTEGER NOT NULL,
            tipo_problema TEXT NOT NULL,
            variable_objetivo TEXT NOT NULL,
            esquema_validado BOOLEAN NOT NULL,                  
            FOREIGN KEY (analisis_id) REFERENCES analisis(analisis_id) ON DELETE CASCADE
        );
    """)

    # Falta finalizar la creación de las tablas restantes
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    crear_mock_db()