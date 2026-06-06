-- ============================================================================
-- Schema PostgreSQL para Supabase — Analítica Farma
-- Ejecutar en: Supabase Dashboard → SQL Editor → New Query → Run
-- ============================================================================

-- Tabla de usuarios
CREATE TABLE IF NOT EXISTS usuarios (
    id              SERIAL PRIMARY KEY,
    nombre          TEXT NOT NULL,
    email           TEXT UNIQUE NOT NULL,
    hash_password   TEXT NOT NULL,
    rol             TEXT NOT NULL DEFAULT 'usuario',
    activo          INTEGER NOT NULL DEFAULT 1,
    fecha_creacion  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Tabla de sesiones
CREATE TABLE IF NOT EXISTS sesiones (
    id_sesion       TEXT PRIMARY KEY,
    id_usuario      INTEGER NOT NULL REFERENCES usuarios(id),
    fecha_inicio    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    fecha_fin       TIMESTAMPTZ,
    estado          TEXT NOT NULL DEFAULT 'ACTIVA'
);

-- Tabla de datasets (metadatos; archivos en Storage bucket)
CREATE TABLE IF NOT EXISTS datasets (
    id_dataset      TEXT PRIMARY KEY,
    nombre          TEXT NOT NULL,
    descripcion     TEXT,
    fecha_creacion  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    id_usuario      INTEGER NOT NULL REFERENCES usuarios(id),
    ruta_archivo    TEXT NOT NULL,
    filas           INTEGER,
    columnas        INTEGER
);

-- Configuraciones de modelo
CREATE TABLE IF NOT EXISTS configuraciones_modelo (
    id                      SERIAL PRIMARY KEY,
    id_usuario              INTEGER NOT NULL REFERENCES usuarios(id),
    id_sesion               TEXT NOT NULL,
    tipo_problema           TEXT,
    variable_objetivo       TEXT,
    variables_predictoras   JSONB,
    configuracion_completa  JSONB,
    fecha_creacion          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Resultados de benchmarking
CREATE TABLE IF NOT EXISTS benchmarking_modelos (
    id                  SERIAL PRIMARY KEY,
    id_usuario          INTEGER NOT NULL REFERENCES usuarios(id),
    id_sesion           TEXT NOT NULL,
    id_dataset          TEXT,
    tipo_problema       TEXT,
    variable_objetivo   TEXT,
    mejor_modelo        TEXT,
    cant_exitosos       INTEGER,
    cant_fallidos       INTEGER,
    resultados_completos JSONB,
    fecha_ejecucion     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Reportes generados
CREATE TABLE IF NOT EXISTS reportes (
    id_reporte      TEXT PRIMARY KEY,
    nombre          TEXT NOT NULL,
    tipo            TEXT,
    fecha           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    id_usuario      INTEGER,
    id_sesion       TEXT,
    id_benchmarking INTEGER,
    id_dataset      TEXT,
    contenido       JSONB
);

-- Auditoría
CREATE TABLE IF NOT EXISTS auditoria (
    id          SERIAL PRIMARY KEY,
    usuario     TEXT,
    accion      TEXT,
    entidad     TEXT,
    id_entidad  TEXT,
    detalles    TEXT,
    fecha       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    id_sesion   TEXT
);

-- Historial de ejecuciones (dashboard)
CREATE TABLE IF NOT EXISTS historial_ejecuciones (
    id                  SERIAL PRIMARY KEY,
    id_usuario          INTEGER NOT NULL REFERENCES usuarios(id),
    id_sesion           TEXT NOT NULL,
    dataset_nombre      TEXT,
    tipo_problema       TEXT,
    variable_objetivo   TEXT,
    modelo_ganador      TEXT,
    metrica_nombre      TEXT,
    metrica_valor       DOUBLE PRECISION,
    total_modelos       INTEGER,
    modelos_exitosos    INTEGER,
    duracion_segundos   DOUBLE PRECISION,
    timestamp           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Crear bucket de Storage para datasets (ejecutar aparte si es necesario)
-- INSERT INTO storage.buckets (id, name, public) VALUES ('datasets', 'datasets', false);
