"""
Módulo para la carga de datos desde diferentes fuentes.
Proporciona funciones para cargar datos desde archivos CSV y almacenarlos en SQLite.
"""
import pandas as pd
import os
import sqlite3
from datetime import datetime
import sys
from typing import Optional

# Importar el módulo de logging
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from audit.logger import setup_logger

# Configurar el logger específico para este módulo
logger = setup_logger("cargador")


def cargar_datos_desde_csv(archivo, **kwargs):
    """
    Carga datos desde un archivo CSV y devuelve el DataFrame junto con metadatos básicos.
    
    Args:
        archivo: Objeto de archivo o ruta al archivo CSV
        **kwargs: Argumentos adicionales para pd.read_csv
    
    Returns:
        pd.DataFrame: DataFrame con los datos cargados
        dict: Metadatos de la carga
    """
    try:
        # Determinar el nombre del archivo
        if hasattr(archivo, 'name'):
            nombre_archivo = archivo.name
        else:
            nombre_archivo = os.path.basename(archivo)
        
        # Cargar el archivo
        df = pd.read_csv(archivo, **kwargs)
        
        # Validar que el DataFrame no esté vacío
        if df.empty:
            logger.warning(f"El archivo {nombre_archivo} está vacío o no contiene datos válidos")
            return df, {"error": "El archivo está vacío", "nombre_archivo": nombre_archivo, "origen": "csv"}
        
        # Metadatos básicos
        metadatos = {
            'nombre_archivo': nombre_archivo,
            'fecha_carga': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'filas': df.shape[0],
            'columnas': df.shape[1],
            'columnas_nombres': list(df.columns),
            'tipos_datos': {col: str(df[col].dtype) for col in df.columns},
            'origen': 'csv'
        }
        
        logger.info(f"Archivo CSV cargado exitosamente: {nombre_archivo}, {df.shape[0]} filas, {df.shape[1]} columnas")
        
        return df, metadatos
    
    except Exception as e:
        error_msg = f"Error al cargar el archivo CSV: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)

def validar_dataframe_csv(df):
    """
    Realiza validaciones básicas sobre un DataFrame cargado desde CSV.
    Args:
        df (pd.DataFrame): DataFrame a validar
    Returns:
        list: Lista de advertencias encontradas
        dict: Metadatos de validación
    """
    warnings = []
    if df.empty:
        warnings.append("El DataFrame está vacío.")
        return warnings, {}
    # Verificar filas completamente vacías
    filas_vacias = df.isna().all(axis=1).sum()
    if filas_vacias > 0:
        warning_msg = f"El archivo contiene {filas_vacias} filas completamente vacías"
        warnings.append(warning_msg)
        logger.warning(warning_msg)
    # Verificar columnas con valores faltantes
    columnas_con_nulos = {col: df[col].isna().sum() for col in df.columns if df[col].isna().any()}
    if columnas_con_nulos:
        warning_msg = f"Columnas con valores faltantes: {columnas_con_nulos}"
        warnings.append(warning_msg)
        logger.warning(warning_msg)
    # Verificar duplicados
    duplicados = df.duplicated().sum()
    if duplicados > 0:
        warning_msg = f"Se detectaron {duplicados} filas duplicadas"
        warnings.append(warning_msg)
        logger.warning(warning_msg)
    # Tipos de datos
    tipos_inferidos = {col: str(df[col].dtype) for col in df.columns}
    logger.info(f"Tipos de datos inferidos: {tipos_inferidos}")
    metadatos_validacion = {
        'filas_vacias': filas_vacias,
        'columnas_con_nulos': columnas_con_nulos,
        'duplicados': duplicados,
        'tipos_datos': tipos_inferidos
    }
    return warnings, metadatos_validacion


def guardar_en_bd_local(df, nombre_tabla, metadatos=None, db_path=None, id_usuario=1):
    """
    Guarda un DataFrame en la base de datos SQLite local.
    
    Args:
        df (pd.DataFrame): DataFrame a guardar
        nombre_tabla (str): Nombre base para la tabla en la que se guardarán los datos
        metadatos (dict): Metadatos asociados al DataFrame
        db_path (str): Ruta a la base de datos SQLite (por defecto, analitica_farma.db)
        id_usuario (int): ID del usuario que está guardando los datos
    
    Returns:
        int: ID del dataset guardado
    """
    conn = None
    try:
        # Determinar la ruta de la base de datos
        if db_path is None:
            db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'analitica_farma.db')
        
        # Limpiar el nombre de la tabla para evitar problemas con SQLite
        # Eliminar caracteres especiales y espacios
        nombre_tabla_limpio = ''.join(c if c.isalnum() else '_' for c in nombre_tabla)
        
        # Asegurar que el nombre empieza con una letra
        if not nombre_tabla_limpio[0].isalpha():
            nombre_tabla_limpio = 'data_' + nombre_tabla_limpio
        
        # Conectar a la base de datos
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Verificar si la tabla ya existe y agregar un sufijo si es necesario
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (f"data_{nombre_tabla_limpio}",))
        if cursor.fetchone():
            # Si la tabla ya existe, agregar un timestamp como sufijo
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            nombre_tabla_limpio = f"{nombre_tabla_limpio}_{timestamp}"
        
        # Nombre final de la tabla de datos
        tabla_datos = f"data_{nombre_tabla_limpio}"
        
        # Guardar el DataFrame en una tabla dedicada para los datos
        df.to_sql(tabla_datos, conn, if_exists='replace', index=False)
        
        # Obtener el origen desde los metadatos si están disponibles
        origen = metadatos.get('origen', 'csv') if metadatos else 'csv'
        
        # Insertar el dataset en la tabla datasets
        cursor.execute("""
            INSERT INTO datasets (id_usuario, nombre, origen, fecha_carga)
            VALUES (?, ?, ?, ?)
        """, (
            id_usuario,
            nombre_tabla_limpio,  # Usar el nombre limpio
            origen,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ))
        
        # Obtener el ID del dataset insertado
        dataset_id = cursor.lastrowid
        
        # Registrar la acción en la tabla de auditoría
        descripcion_auditoria = (
            f"Carga de dataset '{nombre_tabla_limpio}' con {df.shape[0]} filas y {df.shape[1]} columnas. "
            f"Origen: {origen}. Tabla de datos: {tabla_datos}"
        )
        
        cursor.execute("""
            INSERT INTO auditoria (id_usuario, accion, descripcion, fecha)
            VALUES (?, ?, ?, ?)
        """, (
            id_usuario,
            "CARGA_DATOS",
            descripcion_auditoria,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ))
        
        # Confirmar los cambios
        conn.commit()
        
        logger.info(f"DataFrame guardado exitosamente como '{tabla_datos}' con ID de dataset {dataset_id}")
        
        return dataset_id
    
    except Exception as e:
        error_msg = f"Error al guardar el DataFrame en la base de datos: {str(e)}"
        logger.error(error_msg)
        if conn:
            conn.rollback()
        raise Exception(error_msg)
    finally:
        if conn is not None:
            conn.close()


def listar_datasets_disponibles(db_path=None):
    """
    Lista todos los datasets disponibles en la base de datos con información detallada.
    
    Args:
        db_path (str): Ruta a la base de datos SQLite (por defecto, analitica_farma.db)
    
    Returns:
        pd.DataFrame: DataFrame con la información de los datasets
    """
    conn = None
    try:
        # Determinar la ruta de la base de datos
        if db_path is None:
            db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'analitica_farma.db')
        
        # Verificar que el archivo de base de datos existe
        if not os.path.exists(db_path):
            logger.warning(f"La base de datos no existe en la ruta: {db_path}")
            return pd.DataFrame(columns=['id_dataset', 'nombre', 'origen', 'fecha_carga', 'usuario', 'correo', 'filas', 'columnas'])
        
        # Conectar a la base de datos
        conn = sqlite3.connect(db_path)
        
        # Consultar los datasets con información del usuario
        query = """
            SELECT 
                d.id_dataset, 
                d.nombre, 
                d.origen, 
                d.fecha_carga, 
                u.nombre as usuario, 
                u.correo,
                u.rol
            FROM datasets d
            JOIN usuarios u ON d.id_usuario = u.id_usuario
            ORDER BY d.fecha_carga DESC
        """
        df_datasets = pd.read_sql_query(query, conn)
        
        # Si hay datasets, agregar información sobre el número de filas y columnas
        if not df_datasets.empty:
            # Crear columnas para filas y columnas
            df_datasets['filas'] = 0
            df_datasets['columnas'] = 0
            
            # Obtener información de las tablas de datos
            for idx, row in df_datasets.iterrows():
                try:
                    # Nombre de la tabla de datos
                    tabla_datos = f"data_{row['nombre']}"
                    
                    # Verificar si la tabla existe
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (tabla_datos,))
                    if cursor.fetchone():
                        # Obtener el número de filas
                        cursor.execute(f"SELECT COUNT(*) FROM [{tabla_datos}]")
                        filas = cursor.fetchone()[0]
                        df_datasets.at[idx, 'filas'] = filas
                        
                        # Obtener el número de columnas
                        cursor.execute(f"PRAGMA table_info([{tabla_datos}])")
                        columnas = len(cursor.fetchall())
                        df_datasets.at[idx, 'columnas'] = columnas
                except Exception as e:
                    logger.warning(f"Error al obtener información de la tabla data_{row['nombre']}: {str(e)}")
        
        return df_datasets
    
    except Exception as e:
        error_msg = f"Error al listar datasets disponibles: {str(e)}"
        logger.error(error_msg)
        # Devolver un DataFrame vacío con las columnas esperadas
        return pd.DataFrame(columns=['id_dataset', 'nombre', 'origen', 'fecha_carga', 'usuario', 'correo', 'filas', 'columnas'])
    finally:
        if conn is not None:
            conn.close()

def obtener_dataset(id_dataset=None, nombre_dataset=None, db_path=None):
    """
    Obtiene un dataset específico desde la base de datos.
    
    Args:
        id_dataset (int): ID del dataset a obtener
        nombre_dataset (str): Nombre del dataset a obtener (alternativa a id_dataset)
        db_path (str): Ruta a la base de datos SQLite (por defecto, analitica_farma.db)
    
    Returns:
        pd.DataFrame: DataFrame con los datos del dataset
        dict: Metadatos del dataset
    """
    conn = None
    try:
        # Verificar que se proporcionó al menos un parámetro de búsqueda
        if id_dataset is None and nombre_dataset is None:
            raise ValueError("Debe proporcionar id_dataset o nombre_dataset")
        
        # Determinar la ruta de la base de datos
        if db_path is None:
            db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'analitica_farma.db')
        
        # Conectar a la base de datos
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Buscar el dataset por ID o nombre
        if id_dataset is not None:
            cursor.execute("SELECT id_dataset, nombre, origen, fecha_carga FROM datasets WHERE id_dataset = ?", (id_dataset,))
        else:
            cursor.execute("SELECT id_dataset, nombre, origen, fecha_carga FROM datasets WHERE nombre = ?", (nombre_dataset,))
            
        dataset_info = cursor.fetchone()
        
        if not dataset_info:
            error_msg = f"No se encontró el dataset con {'ID ' + str(id_dataset) if id_dataset is not None else 'nombre ' + str(nombre_dataset)}"
            logger.warning(error_msg)
            return None, {"error": error_msg}
        
        # Extraer información del dataset
        dataset_id, nombre, origen, fecha_carga = dataset_info
        
        # Nombre de la tabla de datos
        tabla_datos = f"data_{nombre}"
        
        # Verificar si la tabla existe
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (tabla_datos,))
        if not cursor.fetchone():
            error_msg = f"La tabla de datos '{tabla_datos}' no existe en la base de datos"
            logger.warning(error_msg)
            return None, {"error": error_msg}
        
        # Obtener los datos del dataset
        df = pd.read_sql_query(f"SELECT * FROM [{tabla_datos}]", conn)
        
        # Obtener metadatos adicionales
        cursor.execute("""
            SELECT u.nombre, u.correo, u.rol
            FROM datasets d
            JOIN usuarios u ON d.id_usuario = u.id_usuario
            WHERE d.id_dataset = ?
        """, (dataset_id,))
        
        usuario_info = cursor.fetchone()
        nombre_usuario, correo_usuario, rol_usuario = usuario_info if usuario_info else ("Desconocido", "", "")
        
        # Crear metadatos completos
        metadatos = {
            'id_dataset': dataset_id,
            'nombre': nombre,
            'origen': origen,
            'fecha_carga': fecha_carga,
            'usuario': nombre_usuario,
            'correo_usuario': correo_usuario,
            'rol_usuario': rol_usuario,
            'filas': df.shape[0],
            'columnas': df.shape[1],
            'columnas_nombres': list(df.columns),
            'tipos_datos': {col: str(df[col].dtype) for col in df.columns}
        }
        
        logger.info(f"Dataset '{nombre}' (ID: {dataset_id}) obtenido exitosamente")
        
        return df, metadatos
    
    except Exception as e:
        error_msg = f"Error al obtener el dataset: {str(e)}"
        logger.error(error_msg)
        return None, {"error": error_msg}
    finally:
        if conn is not None:
            conn.close()

def eliminar_dataset(id_dataset, db_path=None, id_usuario=1):
    """
    Elimina un dataset específico de la base de datos.
    
    Args:
        id_dataset (int): ID del dataset a eliminar
        db_path (str): Ruta a la base de datos SQLite (por defecto, analitica_farma.db)
        id_usuario (int): ID del usuario que está eliminando el dataset
    
    Returns:
        bool: True si se eliminó correctamente, False en caso contrario
    """
    conn = None
    try:
        # Determinar la ruta de la base de datos
        if db_path is None:
            db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'analitica_farma.db')
        
        # Conectar a la base de datos
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Verificar que el dataset existe
        cursor.execute("SELECT nombre FROM datasets WHERE id_dataset = ?", (id_dataset,))
        dataset_info = cursor.fetchone()
        
        if not dataset_info:
            error_msg = f"No se encontró el dataset con ID {id_dataset}"
            logger.warning(error_msg)
            return False
        
        nombre_dataset = dataset_info[0]
        tabla_datos = f"data_{nombre_dataset}"
        
        # Verificar si la tabla existe
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (tabla_datos,))
        if cursor.fetchone():
            # Eliminar la tabla de datos
            cursor.execute(f"DROP TABLE [{tabla_datos}]")
        
        # Eliminar las transformaciones asociadas
        cursor.execute("DELETE FROM transformaciones WHERE id_dataset = ?", (id_dataset,))
        
        # Eliminar las ejecuciones de modelos asociadas
        cursor.execute("DELETE FROM ejecuciones WHERE id_dataset = ?", (id_dataset,))
        
        # Eliminar los reportes asociados
        cursor.execute("DELETE FROM reportes WHERE id_dataset = ?", (id_dataset,))
        
        # Eliminar el dataset
        cursor.execute("DELETE FROM datasets WHERE id_dataset = ?", (id_dataset,))
        
        # Registrar la acción en la tabla de auditoría
        cursor.execute("""
            INSERT INTO auditoria (id_usuario, accion, descripcion, fecha)
            VALUES (?, ?, ?, ?)
        """, (
            id_usuario,
            "ELIMINAR_DATASET",
            f"Eliminación del dataset '{nombre_dataset}' con ID {id_dataset}",
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ))
        
        # Confirmar los cambios
        conn.commit()
        
        logger.info(f"Dataset '{nombre_dataset}' (ID: {id_dataset}) eliminado exitosamente")
        
        return True
    
    except Exception as e:
        error_msg = f"Error al eliminar el dataset: {str(e)}"
        logger.error(error_msg)
        if conn:
            conn.rollback()
        return False
    finally:
        if conn is not None:
            conn.close()

def cargar_datos_entrada(id_dataset: Optional[int] = None, nombre_dataset: Optional[str] = None, columna_objetivo: Optional[str] = None, db_path: Optional[str] = None):
    """
    Carga los datos de entrada (X, y) para explicación de modelos.
    Args:
        id_dataset (int): ID del dataset a cargar (opcional)
        nombre_dataset (str): Nombre del dataset (opcional)
        columna_objetivo (str): Nombre de la variable objetivo (si no se especifica, se intenta inferir)
        db_path (str): Ruta a la base de datos (opcional)
    Returns:
        X (pd.DataFrame): Variables predictoras
        y (pd.Series): Variable objetivo
    """
    try:
        df, metadatos = obtener_dataset(id_dataset=id_dataset, nombre_dataset=nombre_dataset, db_path=db_path)
        if df is None:
            raise ValueError(metadatos.get('error', 'No se pudo cargar el dataset.'))
        # Inferir columna objetivo si no se especifica
        if not columna_objetivo:
            posibles_objetivo = [col for col in df.columns if col.lower() in ['target', 'objetivo', 'y', 'clase']]
            if posibles_objetivo:
                columna_objetivo = posibles_objetivo[0]
            else:
                raise ValueError("No se pudo inferir la columna objetivo. Especifíquela explícitamente.")
        if columna_objetivo not in df.columns:
            raise ValueError(f"La columna objetivo '{columna_objetivo}' no existe en el dataset.")
        X = df.drop(columns=[columna_objetivo])
        y = df[columna_objetivo]
        return X, y
    except Exception as e:
        logger.error(f"Error en cargar_datos_entrada: {str(e)}")
        return None, None
