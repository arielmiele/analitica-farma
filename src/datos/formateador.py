"""
Módulo para estandarizar formatos y unidades en los datos.
"""
import pandas as pd
import re
import logging

# Obtener el logger
logger = logging.getLogger("formateador")

def estandarizar_fechas(df, columna, formato_destino='ISO'):
    """
    Estandariza los formatos de fecha en una columna.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos
        columna (str): Nombre de la columna a estandarizar
        formato_destino (str): Formato destino ('ISO', 'DD/MM/YYYY', etc.)
    
    Returns:
        pd.DataFrame: DataFrame con la columna estandarizada
    """
    df_resultado = df.copy()
    
    try:
        # Si la columna ya es datetime, simplemente reformatear
        if pd.api.types.is_datetime64_dtype(df[columna]):
            # Mapeo de formatos destino a formatos de strftime
            formato_strftime = {
                'ISO 8601 (YYYY-MM-DD)': '%Y-%m-%d',
                'DD/MM/YYYY': '%d/%m/%Y',
                'MM/DD/YYYY': '%m/%d/%Y',
                'YYYY/MM/DD': '%Y/%m/%d'
            }
            
            # Si el formato_destino es uno de los predefinidos, usar el formato correspondiente
            if formato_destino in formato_strftime:
                formato = formato_strftime[formato_destino]
                # Convertir a string con el formato especificado
                df_resultado[columna] = df[columna].dt.strftime(formato)
                # Volver a convertir a datetime para mantener el tipo
                df_resultado[columna] = pd.to_datetime(df_resultado[columna], format=formato)
            
            logger.info(f"Columna {columna} estandarizada al formato {formato_destino}")
            return df_resultado
        
        # Si no es datetime, intentar convertir detectando formatos
        else:
            # Convertir a string primero
            valores_str = df[columna].astype(str)
            
            # Detectar formatos comunes
            formatos_comunes = {
                r'\d{4}-\d{2}-\d{2}': '%Y-%m-%d',
                r'\d{2}/\d{2}/\d{4}': '%d/%m/%Y',  # Asumimos DD/MM/YYYY por defecto
                r'\d{4}/\d{2}/\d{2}': '%Y/%m/%d',
                r'\d{2}-\d{2}-\d{4}': '%d-%m-%Y',
                r'\d{4}\.\d{2}\.\d{2}': '%Y.%m.%d'
            }
            
            # Intentar convertir con distintos formatos
            datetime_convertida = None
            
            # Probar cada formato común
            for patron, formato in formatos_comunes.items():
                # Si al menos el 90% de los valores no nulos coinciden con el patrón, usar ese formato
                valores_no_nulos = valores_str.dropna()
                coincidencias = sum(1 for valor in valores_no_nulos if re.match(patron, valor))
                
                if coincidencias >= 0.9 * len(valores_no_nulos):
                    try:
                        datetime_convertida = pd.to_datetime(valores_str, format=formato, errors='coerce')
                        break
                    except Exception as e:
                        logger.warning(f"Error al convertir con formato {formato}: {str(e)}")
            
            # Si no se encontró un formato común, intentar con pandas por defecto
            if datetime_convertida is None:
                datetime_convertida = pd.to_datetime(valores_str, errors='coerce')
            
            # Verificar si la conversión fue exitosa
            porcentaje_validos = (~datetime_convertida.isna()).mean() * 100
            if porcentaje_validos < 70:
                logger.error(f"Solo se pudo convertir el {porcentaje_validos:.1f}% de los valores a fecha.")
                return df_resultado
            
            # Aplicar el formato de destino
            formato_strftime = {
                'ISO 8601 (YYYY-MM-DD)': '%Y-%m-%d',
                'DD/MM/YYYY': '%d/%m/%Y',
                'MM/DD/YYYY': '%m/%d/%Y',
                'YYYY/MM/DD': '%Y/%m/%d'
            }
            
            if formato_destino in formato_strftime:
                formato = formato_strftime[formato_destino]
                # Guardar como datetime
                df_resultado[columna] = datetime_convertida
            else:
                # Si no se especificó un formato válido, usar ISO por defecto
                df_resultado[columna] = datetime_convertida
            
            logger.info(f"Columna {columna} convertida a datetime y estandarizada a {formato_destino}")
            return df_resultado
    
    except Exception as e:
        logger.error(f"Error al estandarizar fechas en {columna}: {str(e)}")
        return df_resultado


def convertir_unidades(df, columna, unidad_destino):
    """
    Convierte las unidades de medida en una columna.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos
        columna (str): Nombre de la columna a convertir
        unidad_destino (str): Unidad de destino
    
    Returns:
        pd.DataFrame: DataFrame con la columna convertida
    """
    df_resultado = df.copy()
    
    try:
        # Verificar que la columna sea numérica
        if not pd.api.types.is_numeric_dtype(df[columna]):
            logger.error(f"La columna {columna} no es numérica, no se puede convertir unidades")
            return df_resultado
        
        # Definir factores de conversión para diferentes unidades
        conversiones = {
            # Temperatura
            'celsius_a_fahrenheit': lambda x: x * 9/5 + 32,
            'fahrenheit_a_celsius': lambda x: (x - 32) * 5/9,
            'celsius_a_kelvin': lambda x: x + 273.15,
            'kelvin_a_celsius': lambda x: x - 273.15,
            'fahrenheit_a_kelvin': lambda x: (x - 32) * 5/9 + 273.15,
            'kelvin_a_fahrenheit': lambda x: (x - 273.15) * 9/5 + 32,
            
            # Peso
            'kilogramos_a_gramos': lambda x: x * 1000,
            'gramos_a_kilogramos': lambda x: x / 1000,
            'kilogramos_a_libras': lambda x: x * 2.20462,
            'libras_a_kilogramos': lambda x: x / 2.20462,
            'libras_a_onzas': lambda x: x * 16,
            'onzas_a_libras': lambda x: x / 16,
            
            # Longitud
            'metros_a_centimetros': lambda x: x * 100,
            'centimetros_a_metros': lambda x: x / 100,
            'metros_a_pulgadas': lambda x: x * 39.3701,
            'pulgadas_a_metros': lambda x: x / 39.3701,
            'pulgadas_a_pies': lambda x: x / 12,
            'pies_a_pulgadas': lambda x: x * 12,
            
            # Volumen
            'litros_a_mililitros': lambda x: x * 1000,
            'mililitros_a_litros': lambda x: x / 1000,
            'litros_a_galones': lambda x: x * 0.264172,
            'galones_a_litros': lambda x: x / 0.264172
        }
        
        # Determinar la unidad actual (por el nombre de la columna o por los valores)
        nombre_lower = columna.lower()
        unidad_actual = None
        
        # Detectar unidad actual basada en el nombre o rango de valores
        if any(palabra in nombre_lower for palabra in ['temp', 'temperatura', 'temperature']):
            # Estimar la unidad actual basada en el rango de valores
            min_val, max_val = df[columna].min(), df[columna].max()
            
            if 0 <= min_val <= 50 and max_val <= 50:
                unidad_actual = 'celsius'
            elif 32 <= min_val <= 100 and max_val <= 110:
                unidad_actual = 'fahrenheit'
            elif 273 <= min_val <= 373:
                unidad_actual = 'kelvin'
        
        # Definir la conversión según las unidades actual y destino
        conversion_key = f"{unidad_actual}_a_{unidad_destino}"
        
        if conversion_key in conversiones:
            # Aplicar la conversión
            df_resultado[columna] = conversiones[conversion_key](df[columna])
            logger.info(f"Columna {columna} convertida de {unidad_actual} a {unidad_destino}")
        else:
            logger.warning(f"No se pudo determinar la conversión de {unidad_actual} a {unidad_destino}")
        
        return df_resultado
    
    except Exception as e:
        logger.error(f"Error al convertir unidades en {columna}: {str(e)}")
        return df_resultado


def corregir_tipos_datos(df, columna, tipo_destino):
    """
    Corrige el tipo de datos de una columna.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos
        columna (str): Nombre de la columna a corregir
        tipo_destino (str): Tipo de destino ('numerico', 'categorico', 'fecha', etc.)
    
    Returns:
        pd.DataFrame: DataFrame con la columna corregida
    """
    df_resultado = df.copy()
    
    try:
        # Convertir según el tipo destino
        if tipo_destino == 'numerico':
            df_resultado[columna] = pd.to_numeric(df[columna], errors='coerce')
            logger.info(f"Columna {columna} convertida a tipo numérico")
        
        elif tipo_destino == 'categorico':
            df_resultado[columna] = df[columna].astype('category')
            logger.info(f"Columna {columna} convertida a tipo categórico")
        
        elif tipo_destino == 'fecha':
            df_resultado[columna] = pd.to_datetime(df[columna], errors='coerce')
            logger.info(f"Columna {columna} convertida a tipo fecha")
        
        elif tipo_destino == 'texto':
            df_resultado[columna] = df[columna].astype(str)
            logger.info(f"Columna {columna} convertida a tipo texto")
        
        else:
            logger.warning(f"Tipo de destino '{tipo_destino}' no reconocido")
        
        return df_resultado
    
    except Exception as e:
        logger.error(f"Error al corregir tipo de datos en {columna}: {str(e)}")
        return df_resultado
