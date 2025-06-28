"""
Módulo para la carga de datos desde diferentes fuentes.
Proporciona funciones para cargar datos desde archivos CSV y almacenarlos en SQLite.
"""
import os
import pandas as pd
import streamlit as st
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


def cargar_datos_entrada(columna_objetivo: Optional[str] = None):
    """
    Carga los datos de entrada (X, y) para explicación de modelos desde la sesión de Streamlit.
    Args:
        columna_objetivo (str): Nombre de la variable objetivo (si no se especifica, se intenta inferir)
    Returns:
        X (pd.DataFrame): Variables predictoras
        y (pd.Series): Variable objetivo
    """
    try:
        df = st.session_state.get('df', None)
        if df is None or df.empty:
            raise ValueError("No hay dataset cargado en la sesión o está vacío.")
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
        logger.error(f"Error en cargar_datos_entrada (sesión): {str(e)}")
        return None, None
