"""
Módulo para estandarizar formatos y unidades en los datos.
"""
import pandas as pd
import re
import streamlit as st
from src.audit.logger import log_audit

# El logger local se elimina, se usa log_audit centralizado

def estandarizar_fechas(df, columna, id_sesion, usuario, formato_destino='ISO'):
    """
    Estandariza los formatos de fecha en una columna.
    Args:
        df (pd.DataFrame): DataFrame con los datos
        columna (str): Nombre de la columna a estandarizar
        id_sesion (str): ID de sesión para trazabilidad
        usuario (str): Usuario que ejecuta la acción
        formato_destino (str): Formato destino ('ISO', 'DD/MM/YYYY', etc.)
    Returns:
        pd.DataFrame: DataFrame con la columna estandarizada
    """
    df_resultado = df.copy()
    try:
        # Si la columna ya es datetime, simplemente reformatear
        if pd.api.types.is_datetime64_dtype(df[columna]):
            formato_strftime = {
                'ISO 8601 (YYYY-MM-DD)': '%Y-%m-%d',
                'DD/MM/YYYY': '%d/%m/%Y',
                'MM/DD/YYYY': '%m/%d/%Y',
                'YYYY/MM/DD': '%Y/%m/%d'
            }
            if formato_destino in formato_strftime:
                formato = formato_strftime[formato_destino]
                df_resultado[columna] = df[columna].dt.strftime(formato)
                df_resultado[columna] = pd.to_datetime(df_resultado[columna], format=formato)
            log_audit(
                accion="estandarizar_fechas",
                entidad="formateador",
                id_entidad=columna,
                detalles=f"Columna {columna} estandarizada al formato {formato_destino}",
                id_sesion=id_sesion,
                usuario=usuario
            )
            return df_resultado
        else:
            valores_str = df[columna].astype(str)
            formatos_comunes = {
                r'\d{4}-\d{2}-\d{2}': '%Y-%m-%d',
                r'\d{2}/\d{2}/\d{4}': '%d/%m/%Y',
                r'\d{4}/\d{2}/\d{2}': '%Y/%m/%d',
                r'\d{2}-\d{2}-\d{4}': '%d-%m-%Y',
                r'\d{4}\.\d{2}\.\d{2}': '%Y.%m.%d'
            }
            datetime_convertida = None
            for patron, formato in formatos_comunes.items():
                valores_no_nulos = valores_str.dropna()
                coincidencias = sum(1 for valor in valores_no_nulos if re.match(patron, valor))
                if coincidencias >= 0.9 * len(valores_no_nulos):
                    try:
                        datetime_convertida = pd.to_datetime(valores_str, format=formato, errors='coerce')
                        break
                    except Exception as e:
                        log_audit(
                            accion="warning_conversion_formato",
                            entidad="formateador",
                            id_entidad=columna,
                            detalles=f"Error al convertir con formato {formato}: {str(e)}",
                            id_sesion=id_sesion,
                            usuario=usuario
                        )
            if datetime_convertida is None:
                datetime_convertida = pd.to_datetime(valores_str, errors='coerce')
            porcentaje_validos = (~datetime_convertida.isna()).mean() * 100
            if porcentaje_validos < 70:
                log_audit(
                    accion="error_conversion_fecha",
                    entidad="formateador",
                    id_entidad=columna,
                    detalles=f"Solo se pudo convertir el {porcentaje_validos:.1f}% de los valores a fecha.",
                    id_sesion=id_sesion,
                    usuario=usuario
                )
                return df_resultado
            formato_strftime = {
                'ISO 8601 (YYYY-MM-DD)': '%Y-%m-%d',
                'DD/MM/YYYY': '%d/%m/%Y',
                'MM/DD/YYYY': '%m/%d/%Y',
                'YYYY/MM/DD': '%Y/%m/%d'
            }
            if formato_destino in formato_strftime:
                formato = formato_strftime[formato_destino]
                df_resultado[columna] = datetime_convertida
            else:
                df_resultado[columna] = datetime_convertida
            log_audit(
                accion="estandarizar_fechas",
                entidad="formateador",
                id_entidad=columna,
                detalles=f"Columna {columna} convertida a datetime y estandarizada a {formato_destino}",
                id_sesion=id_sesion,
                usuario=usuario
            )
            return df_resultado
    except Exception as e:
        log_audit(
            accion="error_estandarizar_fechas",
            entidad="formateador",
            id_entidad=columna,
            detalles=f"Error al estandarizar fechas en {columna}: {str(e)}",
            id_sesion=id_sesion,
            usuario=usuario
        )
        return df_resultado


def convertir_unidades(df, columna, unidad_destino, id_sesion, usuario):
    """
    Convierte las unidades de medida en una columna.
    Args:
        df (pd.DataFrame): DataFrame con los datos
        columna (str): Nombre de la columna a convertir
        unidad_destino (str): Unidad de destino
        id_sesion (str): ID de sesión para trazabilidad
        usuario (str): Usuario que ejecuta la acción
    Returns:
        pd.DataFrame: DataFrame con la columna convertida
    """
    df_resultado = df.copy()
    try:
        if not pd.api.types.is_numeric_dtype(df[columna]):
            log_audit(
                accion="error_conversion_unidades",
                entidad="formateador",
                id_entidad=columna,
                detalles=f"La columna {columna} no es numérica, no se puede convertir unidades",
                id_sesion=id_sesion,
                usuario=usuario
            )
            return df_resultado
        conversiones = {
            'celsius_a_fahrenheit': lambda x: x * 9/5 + 32,
            'fahrenheit_a_celsius': lambda x: (x - 32) * 5/9,
            'celsius_a_kelvin': lambda x: x + 273.15,
            'kelvin_a_celsius': lambda x: x - 273.15,
            'fahrenheit_a_kelvin': lambda x: (x - 32) * 5/9 + 273.15,
            'kelvin_a_fahrenheit': lambda x: (x - 273.15) * 9/5 + 32,
            'kilogramos_a_gramos': lambda x: x * 1000,
            'gramos_a_kilogramos': lambda x: x / 1000,
            'kilogramos_a_libras': lambda x: x * 2.20462,
            'libras_a_kilogramos': lambda x: x / 2.20462,
            'libras_a_onzas': lambda x: x * 16,
            'onzas_a_libras': lambda x: x / 16,
            'metros_a_centimetros': lambda x: x * 100,
            'centimetros_a_metros': lambda x: x / 100,
            'metros_a_pulgadas': lambda x: x * 39.3701,
            'pulgadas_a_metros': lambda x: x / 39.3701,
            'pulgadas_a_pies': lambda x: x / 12,
            'pies_a_pulgadas': lambda x: x * 12,
            'litros_a_mililitros': lambda x: x * 1000,
            'mililitros_a_litros': lambda x: x / 1000,
            'litros_a_galones': lambda x: x * 0.264172,
            'galones_a_litros': lambda x: x / 0.264172
        }
        nombre_lower = columna.lower()
        unidad_actual = None
        if any(palabra in nombre_lower for palabra in ['temp', 'temperatura', 'temperature']):
            min_val, max_val = df[columna].min(), df[columna].max()
            if 0 <= min_val <= 50 and max_val <= 50:
                unidad_actual = 'celsius'
            elif 32 <= min_val <= 100 and max_val <= 110:
                unidad_actual = 'fahrenheit'
            elif 273 <= min_val <= 373:
                unidad_actual = 'kelvin'
        conversion_key = f"{unidad_actual}_a_{unidad_destino}"
        if conversion_key in conversiones:
            df_resultado[columna] = conversiones[conversion_key](df[columna])
            log_audit(
                accion="convertir_unidades",
                entidad="formateador",
                id_entidad=columna,
                detalles=f"Columna {columna} convertida de {unidad_actual} a {unidad_destino}",
                id_sesion=id_sesion,
                usuario=usuario
            )
        else:
            log_audit(
                accion="warning_conversion_unidades",
                entidad="formateador",
                id_entidad=columna,
                detalles=f"No se pudo determinar la conversión de {unidad_actual} a {unidad_destino}",
                id_sesion=id_sesion,
                usuario=usuario
            )
        return df_resultado
    except Exception as e:
        log_audit(
            accion="error_conversion_unidades",
            entidad="formateador",
            id_entidad=columna,
            detalles=f"Error al convertir unidades en {columna}: {str(e)}",
            id_sesion=id_sesion,
            usuario=usuario
        )
        return df_resultado


def corregir_tipos_datos(df, columna, tipo_destino, id_sesion, usuario):
    """
    Corrige el tipo de datos de una columna.
    Args:
        df (pd.DataFrame): DataFrame con los datos
        columna (str): Nombre de la columna a corregir
        tipo_destino (str): Tipo de destino ('numerico', 'categorico', 'fecha', etc.)
        id_sesion (str): ID de sesión para trazabilidad
        usuario (str): Usuario que ejecuta la acción
    Returns:
        pd.DataFrame: DataFrame con la columna corregida
    """
    df_resultado = df.copy()
    try:
        if tipo_destino == 'numerico':
            df_resultado[columna] = pd.to_numeric(df[columna], errors='coerce')
            log_audit(
                accion="corregir_tipo_dato",
                entidad="formateador",
                id_entidad=columna,
                detalles=f"Columna {columna} convertida a tipo numérico",
                id_sesion=id_sesion,
                usuario=usuario
            )
        elif tipo_destino == 'categorico':
            df_resultado[columna] = df[columna].astype('category')
            log_audit(
                accion="corregir_tipo_dato",
                entidad="formateador",
                id_entidad=columna,
                detalles=f"Columna {columna} convertida a tipo categórico",
                id_sesion=id_sesion,
                usuario=usuario
            )
        elif tipo_destino == 'fecha':
            df_resultado[columna] = pd.to_datetime(df[columna], errors='coerce')
            log_audit(
                accion="corregir_tipo_dato",
                entidad="formateador",
                id_entidad=columna,
                detalles=f"Columna {columna} convertida a tipo fecha",
                id_sesion=id_sesion,
                usuario=usuario
            )
        elif tipo_destino == 'texto':
            df_resultado[columna] = df[columna].astype(str)
            log_audit(
                accion="corregir_tipo_dato",
                entidad="formateador",
                id_entidad=columna,
                detalles=f"Columna {columna} convertida a tipo texto",
                id_sesion=id_sesion,
                usuario=usuario
            )
        else:
            log_audit(
                accion="warning_tipo_dato",
                entidad="formateador",
                id_entidad=columna,
                detalles=f"Tipo de destino '{tipo_destino}' no reconocido",
                id_sesion=id_sesion,
                usuario=usuario
            )
        return df_resultado
    except Exception as e:
        log_audit(
            accion="error_corregir_tipo_dato",
            entidad="formateador",
            id_entidad=columna,
            detalles=f"Error al corregir tipo de datos en {columna}: {str(e)}",
            id_sesion=id_sesion,
            usuario=usuario
        )
        return df_resultado

def persistir_dataframe(df, id_sesion, usuario):
    """
    Actualiza el DataFrame en memoria en session_state para que toda la app use la versión más reciente.
    Args:
        df (pd.DataFrame): DataFrame actualizado
        id_sesion (str): ID de sesión para trazabilidad
        usuario (str): Usuario que ejecuta la acción
    Returns:
        dict: {'success': bool, 'message': str, 'error': str (opcional)}
    """
    try:
        st.session_state['df'] = df
        log_audit(
            accion="persistir_dataframe",
            entidad="formateador",
            id_entidad=None,
            detalles="DataFrame actualizado en memoria (session_state['df']) correctamente.",
            id_sesion=id_sesion,
            usuario=usuario
        )
        return {
            'success': True,
            'message': 'DataFrame actualizado en memoria correctamente.'
        }
    except Exception as e:
        log_audit(
            accion="error_persistir_dataframe",
            entidad="formateador",
            id_entidad=None,
            detalles=f"Error al actualizar el DataFrame en memoria: {str(e)}",
            id_sesion=id_sesion,
            usuario=usuario
        )
        return {
            'success': False,
            'message': 'Error al actualizar el DataFrame en memoria.',
            'error': str(e)
        }
