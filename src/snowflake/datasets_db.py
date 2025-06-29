from src.snowflake.snowflake_conn import get_native_snowflake_connection
from snowflake.connector.pandas_tools import write_pandas
from typing import Optional, Dict
import pandas as pd
import uuid
from datetime import datetime
from sqlalchemy.engine import Engine
from snowflake.sqlalchemy import URL
from sqlalchemy import create_engine
import streamlit as st
from src.audit.logger import log_audit

"""
Operaciones CRUD y utilitarias sobre la tabla DATASETS en Snowflake.
Incluye funciones para crear, leer, actualizar y eliminar datasets.
"""

def obtener_dataset_por_id(id_dataset: str, id_sesion: str, usuario: str) -> Optional[Dict]:
    conn = get_native_snowflake_connection()
    if not conn:
        log_audit(id_sesion, usuario, "ERROR_CONEXION", "datasets_db", "No se pudo obtener la conexión a Snowflake.")
        return None
    cur = None
    try:
        cur = conn.cursor()
        # Sanitizar el id para evitar inyección
        id_safe = id_dataset.replace("'", "")
        query = f"""
            SELECT ID_DATASET, NOMBRE, DESCRIPCION, FECHA_CREACION, USUARIO_CREADOR, ESQUEMA_FISICO, TABLA_FISICA
            FROM DATASETS WHERE ID_DATASET = '{id_safe}'
        """
        cur.execute(query)
        row = cur.fetchone()
        if row:
            log_audit(id_sesion, usuario, "OBTENER_DATASET_OK", "datasets_db", f"Dataset {id_safe} obtenido correctamente.")
            return dict(zip([col[0] for col in cur.description], row))
        log_audit(id_sesion, usuario, "DATASET_NO_ENCONTRADO", "datasets_db", f"No se encontró el dataset {id_safe}.")
        return None
    except Exception as e:
        log_audit(id_sesion, usuario, "ERROR_OBTENER_DATASET", "datasets_db", f"Error al obtener dataset: {e}")
        return None
    finally:
        if cur is not None:
            cur.close()

def get_snowflake_sqlalchemy_engine(id_sesion: str, usuario: str) -> Engine:
    """
    Devuelve un engine SQLAlchemy para Snowflake usando st.secrets.
    Lanza ValueError si faltan variables requeridas.
    """
    conn_params = st.secrets["connections"]["snowflake"]
    required = ["user", "password", "account", "warehouse", "database", "schema"]
    if not all(k in conn_params and conn_params[k] for k in required):
        log_audit(id_sesion, usuario, "ERROR_SQLALCHEMY_PARAMS", "datasets_db", "Faltan parámetros requeridos en st.secrets para SQLAlchemy.")
        raise ValueError("Faltan parámetros requeridos en st.secrets para la conexión SQLAlchemy a Snowflake.")
    engine = create_engine(
        URL(
            user=conn_params["user"],
            password=conn_params["password"],
            account=conn_params["account"],
            warehouse=conn_params["warehouse"],
            database=conn_params["database"],
            schema=conn_params["schema"]
        )
    )
    log_audit(id_sesion, usuario, "SQLALCHEMY_ENGINE_OK", "datasets_db", "Engine SQLAlchemy creado correctamente.")
    return engine


def guardar_dataset(nombre: str, descripcion: str, usuario_creador: str, df: pd.DataFrame, id_sesion: str, usuario: str) -> bool:
    """
    Guarda un nuevo dataset en Snowflake creando una tabla física en el esquema configurado para datos y registrando los metadatos.
    """
    # 1. Obtener parámetros de conexión y configuración desde st.secrets
    conn_params = st.secrets["connections"]["snowflake"]
    datos_db = conn_params.get("database")
    datos_schema = conn_params.get("datasets_schema", conn_params.get("schema"))
    if not datos_db or not datos_schema:
        log_audit(id_sesion, usuario, "ERROR_PARAMETROS_DB", "datasets_db", "Faltan parámetros de base de datos o esquema de los datasets físicos en st.secrets.")
        return False
    # 2. Obtener conexión nativa a Snowflake en el esquema de datos
    conn = get_native_snowflake_connection(schema=datos_schema)
    if not conn:
        log_audit(id_sesion, usuario, "ERROR_CONEXION", "datasets_db", "No se pudo obtener la conexión a Snowflake.")
        return False
    # 3. Generar identificador único y sanitizar inputs
    id_dataset = str(uuid.uuid4())
    nombre_safe = nombre.replace("'", "")
    descripcion_safe = descripcion.replace("'", "")
    usuario_creador_safe = usuario_creador.replace("'", "")
    fecha_creacion = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # 4. Definir el nombre de la tabla física (solo nombre, sin base/esquema)
    nombre_tabla_fisica = f"DATASET_{id_dataset.replace('-', '_')}"
    # CORRECTO: solo pasar el nombre de la tabla a write_pandas, no el nombre completo con base/esquema
    cur = None
    try:
        # 5. Escribir el DataFrame como tabla física en Snowflake (en el esquema correcto)
        success, nchunks, nrows, _ = write_pandas(conn, df, nombre_tabla_fisica, auto_create_table=True)
        if not success:
            log_audit(id_sesion, usuario, "ERROR_WRITE_PANDAS", "datasets_db", f"Error al guardar DataFrame en tabla {datos_schema}.{nombre_tabla_fisica} de Snowflake.")
            return False
        # 6. Registrar los metadatos del dataset en la tabla DATASETS (en el esquema de metadatos)
        cur = conn.cursor()
        query = f"""
            INSERT INTO PUBLIC.DATASETS (ID_DATASET, NOMBRE, DESCRIPCION, FECHA_CREACION, USUARIO_CREADOR, ESQUEMA_FISICO, TABLA_FISICA)
            VALUES ('{id_dataset}', '{nombre_safe}', '{descripcion_safe}', '{fecha_creacion}', '{usuario_creador_safe}', '{datos_schema}', '{nombre_tabla_fisica}')
        """
        cur.execute(query)
        conn.commit()
        log_audit(id_sesion, usuario, "GUARDAR_DATASET_OK", "datasets_db", f"Dataset guardado: id={id_dataset}, tabla={datos_schema}.{nombre_tabla_fisica}, filas={nrows}, usuario={usuario_creador_safe}")
        return True
    except Exception as e:
        log_audit(id_sesion, usuario, "ERROR_GUARDAR_DATASET", "datasets_db", f"Error general al guardar dataset: {e} | id={id_dataset}, tabla={datos_schema}.{nombre_tabla_fisica}, usuario={usuario_creador_safe}")
        return False
    finally:
        # 7. Cerrar cursor y conexión
        if cur is not None:
            cur.close()
        conn.close()

def listar_datasets(id_sesion: str, usuario: str, usuario_creador: Optional[str] = None):
    """
    Lista los datasets disponibles, opcionalmente filtrando por usuario.
    """
    conn = get_native_snowflake_connection(schema="DATASETS")
    if not conn:
        log_audit(id_sesion, usuario, "ERROR_CONEXION", "datasets_db", "No se pudo obtener la conexión a Snowflake.")
        return []
    cur = None
    try:
        cur = conn.cursor()
        if usuario_creador:
            usuario_creador_safe = usuario_creador.replace("'", "")
            query = f"SELECT ID_DATASET, NOMBRE, DESCRIPCION, FECHA_CREACION, USUARIO_CREADOR FROM DATASETS WHERE USUARIO_CREADOR = '{usuario_creador_safe}' ORDER BY FECHA_CREACION DESC"
            cur.execute(query)
        else:
            cur.execute(
                "SELECT ID_DATASET, NOMBRE, DESCRIPCION, FECHA_CREACION, USUARIO_CREADOR FROM DATASETS ORDER BY FECHA_CREACION DESC"
            )
        rows = cur.fetchall()
        log_audit(id_sesion, usuario, "LISTAR_DATASETS_OK", "datasets_db", f"Listados {len(rows)} datasets.")
        return [dict(zip([col[0] for col in cur.description], row)) for row in rows]
    except Exception as e:
        log_audit(id_sesion, usuario, "ERROR_LISTAR_DATASETS", "datasets_db", f"Error al listar datasets: {e}")
        return []
    finally:
        if cur is not None:
            cur.close()

def cargar_dataset_fisico_por_id(id_dataset: str, id_sesion: str, usuario: str) -> Optional[pd.DataFrame]:
    """
    Carga el DataFrame de la tabla física asociada a un dataset por su ID usando el conector nativo de Snowflake.
    Devuelve el DataFrame o None si hay error.
    """
    # Conexión a Snowflake en el esquema DATASETS (donde están las tablas físicas)
    conn = get_native_snowflake_connection()
    if not conn:
        log_audit(id_sesion, usuario, "ERROR_CONEXION", "datasets_db", "No se pudo obtener la conexión a Snowflake.")
        return None
    cur = None
    try:
        cur = conn.cursor()
        # Sanitizar el id para evitar inyección SQL
        id_safe = id_dataset.replace("'", "")
        # Buscar en la tabla de metadatos el esquema y nombre de la tabla física
        query = f"""
            SELECT ESQUEMA_FISICO, TABLA_FISICA FROM DATASETS WHERE ID_DATASET = '{id_safe}'
        """
        cur.execute(query)
        row = cur.fetchone()
        if not row or not row[0] or not row[1]:
            # Si no se encuentra el dataset, registrar error y salir
            log_audit(id_sesion, usuario, "DATASET_FISICO_NO_ENCONTRADO", "datasets_db", f"No se encontró el dataset físico con id {id_safe}.")
            return None
        esquema_fisico, tabla_fisica = row
        # Cambiar el esquema de la conexión si es necesario
        if esquema_fisico:
            try:
                cur.close()
                conn.close()
                conn = get_native_snowflake_connection(schema=esquema_fisico)
                if not conn:
                    log_audit(id_sesion, usuario, "ERROR_CONEXION", "datasets_db", "No se pudo obtener la conexión a Snowflake.")
                    return None
                cur = None
                cur = conn.cursor()
            except Exception as e:
                log_audit(id_sesion, usuario, "ERROR_CAMBIO_ESQUEMA", "datasets_db", f"No se pudo cambiar al esquema {esquema_fisico}: {e}")
                return None
        # Leer la tabla física como DataFrame usando fetch_pandas_all()
        try:
            cur.execute(f'SELECT * FROM "{tabla_fisica}"')
            df = cur.fetch_pandas_all()
            log_audit(id_sesion, usuario, "CARGAR_DATASET_FISICO_OK", "datasets_db", f"Dataset físico {tabla_fisica} cargado correctamente.")
            return df
        except Exception as e:
            log_audit(id_sesion, usuario, "ERROR_LEER_TABLA_FISICA", "datasets_db", f"Error al leer la tabla física {tabla_fisica}: {e}")
            return None
    except Exception as e:
        log_audit(id_sesion, usuario, "ERROR_BUSCAR_TABLA_FISICA", "datasets_db", f"Error al buscar nombre de tabla física: {e}")
        return None
    finally:
        # Cerrar el cursor si fue abierto
        if cur is not None:
            cur.close()
        conn.close()

def eliminar_tabla_fisica(nombre_tabla: str, id_sesion: str, usuario: str, esquema: Optional[str] = None) -> bool:
    """
    Elimina una tabla física (dataset) en Snowflake.
    - nombre_tabla: solo el nombre de la tabla (sin base/esquema).
    - esquema: esquema donde está la tabla (por defecto usa datasets_schema de st.secrets).
    Devuelve True si se eliminó correctamente, False si hubo error.
    """
    conn = None
    cur = None
    try:
        conn_params = st.secrets["connections"]["snowflake"]
        datos_schema = esquema or conn_params.get("datasets_schema", conn_params.get("schema"))
        if not datos_schema:
            log_audit(id_sesion, usuario, "ERROR_PARAMETROS_DB", "datasets_db", "Faltan parámetros de esquema en st.secrets.")
            return False
        conn = get_native_snowflake_connection(schema=datos_schema)
        cur = conn.cursor()
        nombre_tabla_completo = f'{nombre_tabla}'
        query = f'DROP TABLE IF EXISTS {nombre_tabla_completo}'
        cur.execute(query)
        conn.commit()
        log_audit(id_sesion, usuario, "ELIMINAR_TABLA_OK", "datasets_db", f"Tabla eliminada: {nombre_tabla_completo}")
        return True
    except Exception as e:
        log_audit(id_sesion, usuario, "ERROR_ELIMINAR_TABLA", "datasets_db", f"Error al eliminar tabla {nombre_tabla}: {e}")
        return False
    finally:
        if cur is not None:
            try:
                cur.close()
            except Exception:
                pass
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
