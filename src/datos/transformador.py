"""
Módulo para la transformación de datos.

Este módulo contiene funciones para realizar transformaciones en DataFrames,
especialmente para corregir problemas detectados durante la validación:
- Corrección de tipos de datos
- Estandarización de formatos de fechas
- Conversión de unidades de medida
"""

import pandas as pd
import numpy as np
import re
from src.audit.logger import log_audit

def corregir_tipo_datos(df, columna, tipo_destino, id_sesion, usuario, metodo='auto'):
    """
    Corrige el tipo de datos de una columna.
    
    Args:
        df (pd.DataFrame): DataFrame a transformar
        columna (str): Nombre de la columna a transformar
        tipo_destino (str): Tipo de dato destino ('int', 'float', 'str', 'bool', 'datetime')
        id_sesion (str): ID de sesión para trazabilidad
        usuario (str): Usuario que ejecuta la acción
        metodo (str): Método de conversión ('auto', 'forzar', 'inferir')
        
    Returns:
        pd.DataFrame: DataFrame con la columna transformada
    
    Raises:
        ValueError: Si el tipo de dato no es válido o la columna no existe
    """
    if columna not in df.columns:
        raise ValueError(f"La columna {columna} no existe en el DataFrame")
    
    df_result = df.copy()
    
    log_audit(
        usuario=usuario,
        accion="INICIO_TRANSFORMACION",
        entidad="transformador",
        id_entidad=columna,
        detalles=f"Iniciando corrección de tipo de dato en columna {columna} a {tipo_destino}",
        id_sesion=id_sesion
    )
    
    try:
        # Conversión según el tipo destino
        if tipo_destino == 'int':
            if metodo == 'forzar':
                # Eliminar caracteres no numéricos y convertir
                df_result[columna] = df_result[columna].astype(str).str.extract(r'(\d+)').astype(float).astype('Int64')
            else:
                # Usar tipo Int64 de pandas que permite NaN
                df_result[columna] = pd.to_numeric(df_result[columna], errors='coerce').astype('Int64')
                
        elif tipo_destino == 'float':
            if metodo == 'forzar':
                # Extraer números, incluyendo decimales
                df_result[columna] = df_result[columna].astype(str).str.extract(r'([-+]?\d*\.\d+|\d+)')[0].astype(float)
            else:
                df_result[columna] = pd.to_numeric(df_result[columna], errors='coerce')
                
        elif tipo_destino == 'str':
            df_result[columna] = df_result[columna].astype(str)
            
        elif tipo_destino == 'bool':
            if metodo == 'inferir':
                # Intentar inferir valores booleanos de manera más inteligente
                true_values = ['true', 'yes', 'si', 'verdadero', 'y', 's', 't', '1']
                false_values = ['false', 'no', 'falso', 'n', 'f', '0']
                
                def to_bool(val):
                    if pd.isna(val):
                        return np.nan
                    val_str = str(val).lower().strip()
                    if val_str in true_values:
                        return True
                    elif val_str in false_values:
                        return False
                    else:
                        return np.nan
                
                df_result[columna] = df_result[columna].apply(to_bool)
            else:
                # Método estándar
                df_result[columna] = df_result[columna].astype(bool)
                
        elif tipo_destino == 'datetime':
            # Para fechas, delegamos a la función especializada
            df_result = estandarizar_fechas(df_result, columna, id_sesion, usuario, formato_destino='ISO')
            
        else:
            raise ValueError(f"Tipo de dato {tipo_destino} no soportado")
            
        log_audit(
            usuario=usuario,
            accion="TRANSFORMACION_EXITOSA",
            entidad="transformador",
            id_entidad=columna,
            detalles=f"Corrección de tipo exitosa para columna {columna} a {tipo_destino}",
            id_sesion=id_sesion
        )
        
    except Exception as e:
        log_audit(
            usuario=usuario,
            accion="ERROR_TRANSFORMACION",
            entidad="transformador",
            id_entidad=columna,
            detalles=f"Error al corregir tipo de dato en columna {columna}: {str(e)}",
            id_sesion=id_sesion
        )
        raise
        
    return df_result


def estandarizar_fechas(df, columna, id_sesion, usuario, formato_destino='ISO'):
    """
    Estandariza el formato de fechas en una columna.
    
    Args:
        df (pd.DataFrame): DataFrame a transformar
        columna (str): Nombre de la columna con fechas
        id_sesion (str): ID de sesión para trazabilidad
        usuario (str): Usuario que ejecuta la acción
        formato_destino (str): Formato de fecha destino 
                               ('ISO', 'DMY', 'MDY', 'YMD', 'datetime')
    
    Returns:
        pd.DataFrame: DataFrame con fechas estandarizadas
    """
    if columna not in df.columns:
        raise ValueError(f"La columna {columna} no existe en el DataFrame")
    
    df_result = df.copy()
    
    log_audit(
        usuario=usuario,
        accion="INICIO_ESTANDARIZACION",
        entidad="transformador",
        id_entidad=columna,
        detalles=f"Iniciando estandarización de fechas en columna {columna} a formato {formato_destino}",
        id_sesion=id_sesion
    )
    
    try:
        # Convertir a datetime primero (pandas intentará inferir el formato)
        series_fecha = pd.to_datetime(df[columna], errors='coerce')
        
        # Si muchas fechas no se pudieron convertir, intentar con formatos comunes
        if series_fecha.isna().sum() > len(df) * 0.3:  # Si más del 30% son NaN
            formatos_comunes = [
                '%d/%m/%Y', '%d-%m-%Y', '%Y-%m-%d', '%m/%d/%Y', 
                '%d/%m/%y', '%Y/%m/%d', '%d.%m.%Y',
                '%d/%m/%Y %H:%M:%S', '%Y-%m-%d %H:%M:%S'
            ]
            
            for formato in formatos_comunes:
                try:
                    series_fecha = pd.to_datetime(df[columna], format=formato, errors='coerce')
                    if series_fecha.isna().sum() < len(df) * 0.3:  # Si menos del 30% son NaN
                        break
                except Exception:
                    continue
        
        # Aplicar el formato destino
        if formato_destino == 'ISO':
            # Formato ISO 8601: YYYY-MM-DD
            df_result[columna] = series_fecha.dt.strftime('%Y-%m-%d')
        elif formato_destino == 'DMY':
            # Formato día/mes/año
            df_result[columna] = series_fecha.dt.strftime('%d/%m/%Y')
        elif formato_destino == 'MDY':
            # Formato mes/día/año
            df_result[columna] = series_fecha.dt.strftime('%m/%d/%Y')
        elif formato_destino == 'YMD':
            # Formato año/mes/día
            df_result[columna] = series_fecha.dt.strftime('%Y/%m/%d')
        elif formato_destino == 'datetime':
            # Mantener como objeto datetime de pandas
            df_result[columna] = series_fecha
        else:
            # Formato personalizado
            df_result[columna] = series_fecha.dt.strftime(formato_destino)
        
        log_audit(
            usuario=usuario,
            accion="ESTANDARIZACION_EXITOSA",
            entidad="transformador",
            id_entidad=columna,
            detalles=f"Estandarización de fechas exitosa para columna {columna} a formato {formato_destino}",
            id_sesion=id_sesion
        )
        
    except Exception as e:
        log_audit(
            usuario=usuario,
            accion="ERROR_ESTANDARIZACION",
            entidad="transformador",
            id_entidad=columna,
            detalles=f"Error al estandarizar fechas en columna {columna}: {str(e)}",
            id_sesion=id_sesion
        )
        raise
        
    return df_result


def convertir_unidades(df, columna, unidad_destino, id_sesion, usuario, unidad_origen=None):
    """
    Convierte valores de una columna a una unidad de medida estándar.
    
    Args:
        df (pd.DataFrame): DataFrame a transformar
        columna (str): Nombre de la columna con unidades
        unidad_destino (str): Unidad de medida destino
        id_sesion (str): ID de sesión para trazabilidad
        usuario (str): Usuario que ejecuta la acción
        unidad_origen (str, optional): Unidad de origen (si no se proporciona, se intenta detectar)
    
    Returns:
        pd.DataFrame: DataFrame con unidades convertidas
    """
    if columna not in df.columns:
        raise ValueError(f"La columna {columna} no existe en el DataFrame")
    
    df_result = df.copy()
    
    log_audit(
        usuario=usuario,
        accion="INICIO_CONVERSION",
        entidad="transformador",
        id_entidad=columna,
        detalles=f"Iniciando conversión de unidades en columna {columna} a {unidad_destino}",
        id_sesion=id_sesion
    )
    
    # Diccionario de factores de conversión para unidades comunes
    # Estructura: {(unidad_origen, unidad_destino): factor}
    factores_conversion = {
        # Longitud
        ('mm', 'cm'): 0.1,
        ('cm', 'mm'): 10,
        ('cm', 'm'): 0.01,
        ('m', 'cm'): 100,
        ('m', 'km'): 0.001,
        ('km', 'm'): 1000,
        ('in', 'cm'): 2.54,
        ('cm', 'in'): 0.394,
        ('ft', 'm'): 0.305,
        ('m', 'ft'): 3.281,
        
        # Masa/Peso
        ('mg', 'g'): 0.001,
        ('g', 'mg'): 1000,
        ('g', 'kg'): 0.001,
        ('kg', 'g'): 1000,
        ('lb', 'kg'): 0.454,
        ('kg', 'lb'): 2.205,
        ('oz', 'g'): 28.35,
        ('g', 'oz'): 0.035,
        
        # Volumen
        ('ml', 'l'): 0.001,
        ('l', 'ml'): 1000,
        ('l', 'm3'): 0.001,
        ('m3', 'l'): 1000,
        ('gal', 'l'): 3.785,
        ('l', 'gal'): 0.264,
        
        # Temperatura
        ('C', 'F'): lambda x: x * 9/5 + 32,
        ('F', 'C'): lambda x: (x - 32) * 5/9,
        ('C', 'K'): lambda x: x + 273.15,
        ('K', 'C'): lambda x: x - 273.15,
        
        # Tiempo
        ('s', 'min'): 1/60,
        ('min', 's'): 60,
        ('min', 'h'): 1/60,
        ('h', 'min'): 60,
        ('h', 'd'): 1/24,
        ('d', 'h'): 24,
        
        # Presión
        ('Pa', 'kPa'): 0.001,
        ('kPa', 'Pa'): 1000,
        ('kPa', 'MPa'): 0.001,
        ('MPa', 'kPa'): 1000,
        ('bar', 'kPa'): 100,
        ('kPa', 'bar'): 0.01,
        ('psi', 'kPa'): 6.895,
        ('kPa', 'psi'): 0.145,
    }
    
    try:
        # Si no se proporciona unidad de origen, tratar de detectarla mediante patrones
        if unidad_origen is None:
            # Expresión regular para extraer números y unidades
            patron = r'(-?\d+\.?\d*)\s*([a-zA-Z°]+)'
            
            # Extraer unidades detectadas
            unidades_detectadas = []
            for valor in df[columna].astype(str):
                match = re.search(patron, valor)
                if match:
                    unidades_detectadas.append(match.group(2))
            
            # Determinar la unidad más común
            if unidades_detectadas:
                from collections import Counter
                contador = Counter(unidades_detectadas)
                unidad_origen = contador.most_common(1)[0][0]
            else:
                raise ValueError("No se pudo detectar la unidad de origen automáticamente")
        
        # Convertir valores según el par de unidades
        par_conversion = (unidad_origen, unidad_destino)
        
        if par_conversion in factores_conversion:
            factor = factores_conversion[par_conversion]
            
            # Extraer valores numéricos
            patron_numero = r'(-?\d+\.?\d*)'
            valores_numericos = df[columna].astype(str).str.extract(patron_numero).astype(float)
            
            # Aplicar factor de conversión
            if callable(factor):
                # Si el factor es una función (como para conversiones no lineales)
                valores_convertidos = valores_numericos.applymap(factor)
            else:
                # Para conversiones lineales
                valores_convertidos = valores_numericos * factor
            
            # Formatear con la nueva unidad
            df_result[columna] = valores_convertidos.astype(str) + ' ' + unidad_destino
            
            log_audit(
                usuario=usuario,
                accion="CONVERSION_EXITOSA",
                entidad="transformador",
                id_entidad=columna,
                detalles=f"Conversión de unidades exitosa para columna {columna} de {unidad_origen} a {unidad_destino}",
                id_sesion=id_sesion
            )
        else:
            raise ValueError(f"No hay factor de conversión definido para {unidad_origen} a {unidad_destino}")
            
    except Exception as e:
        log_audit(
            usuario=usuario,
            accion="ERROR_CONVERSION",
            entidad="transformador",
            id_entidad=columna,
            detalles=f"Error al convertir unidades en columna {columna}: {str(e)}",
            id_sesion=id_sesion
        )
        raise
        
    return df_result

def extraer_variables_fecha(df, columna, id_sesion, usuario, variables=None):
    """
    Extrae variables derivadas de una columna de fecha: año, mes, día, día_semana, día_año, semana, etc.
    Args:
        df (pd.DataFrame): DataFrame de entrada
        columna (str): Nombre de la columna de fecha
        id_sesion (str): ID de sesión para trazabilidad
        usuario (str): Usuario que ejecuta la acción
        variables (list, opcional): Lista de variables a extraer. Si None, extrae todas.
    Returns:
        pd.DataFrame: DataFrame con nuevas columnas agregadas
    """
    if columna not in df.columns:
        raise ValueError(f"La columna {columna} no existe en el DataFrame")
    df_result = df.copy()
    fechas = pd.to_datetime(df_result[columna], errors='coerce')
    todas = {
        'anio': fechas.dt.year,
        'mes': fechas.dt.month,
        'dia': fechas.dt.day,
        'dia_semana': fechas.dt.weekday,
        'nombre_dia': fechas.dt.day_name(),
        'nombre_mes': fechas.dt.month_name(),
        'dia_anio': fechas.dt.dayofyear,
        'semana': fechas.dt.isocalendar().week,
        'es_fin_de_semana': fechas.dt.weekday >= 5,
        'fecha_epoch': fechas.astype('int64') // 10**9
    }
    if variables is None:
        variables = list(todas.keys())
    for var in variables:
        if var in todas:
            df_result[f"{columna}_{var}"] = todas[var]
    log_audit(
        usuario=usuario,
        accion="EXTRACCION_FECHA",
        entidad="transformador",
        id_entidad=columna,
        detalles=f"Extraídas variables {variables} de la columna {columna}",
        id_sesion=id_sesion
    )
    return df_result
