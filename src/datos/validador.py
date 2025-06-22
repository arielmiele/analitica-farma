"""
Módulo para validar la estructura de datos y variables objetivo.
"""
import pandas as pd
import logging
import re
from datetime import datetime

# Obtener el logger
logger = logging.getLogger("validador")

def validar_variable_objetivo(df, variable_objetivo, tipo_problema):
    """
    Valida que la variable objetivo sea adecuada para el tipo de problema.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos
        variable_objetivo (str): Nombre de la columna objetivo
        tipo_problema (str): 'regresion' o 'clasificacion'
    
    Returns:
        tuple: (bool, str) indicando si es válida y un mensaje
    """
    try:
        # Verificar que la variable existe
        if variable_objetivo not in df.columns:
            return False, f"La variable '{variable_objetivo}' no existe en el dataset"
        
        # Verificar si hay valores nulos
        nulos = df[variable_objetivo].isna().sum()
        if nulos > 0:
            porcentaje = (nulos / len(df)) * 100
            if porcentaje > 20:
                return False, f"La variable objetivo tiene {nulos} valores nulos ({porcentaje:.1f}%). Se recomienda seleccionar otra variable o imputar los valores faltantes."
            else:
                logger.warning(f"La variable objetivo tiene {nulos} valores nulos ({porcentaje:.1f}%)")
        
        # Validaciones específicas según el tipo de problema
        if tipo_problema == "regresion":
            # Para regresión, verificar que sea numérica
            if not pd.api.types.is_numeric_dtype(df[variable_objetivo]):
                return False, f"La variable '{variable_objetivo}' no es numérica. Para regresión, la variable objetivo debe ser numérica."
            
            # Verificar variabilidad (no constante)
            if df[variable_objetivo].nunique() <= 1:
                return False, f"La variable '{variable_objetivo}' es constante. Una variable objetivo debe tener variabilidad."
                
        elif tipo_problema == "clasificacion":
            # Para clasificación, verificar número de clases
            n_clases = df[variable_objetivo].nunique()
            
            if n_clases <= 1:
                return False, f"La variable '{variable_objetivo}' solo tiene una clase. Para clasificación, se necesitan al menos dos clases."
            
            if n_clases > 10:
                return False, f"La variable '{variable_objetivo}' tiene {n_clases} clases. Se recomienda usar una variable con menos clases para clasificación."
            
            # Verificar balance de clases
            clase_counts = df[variable_objetivo].value_counts(normalize=True)
            clase_min = clase_counts.min()
            
            if clase_min < 0.01:  # Menos del 1% de los datos
                return False, f"Hay clases con muy pocos ejemplos ({clase_min*100:.2f}% del total). Esto puede afectar el rendimiento del modelo."
        
        return True, "Variable objetivo válida"
        
    except Exception as e:
        logger.error(f"Error al validar variable objetivo: {str(e)}")
        return False, f"Error al validar la variable objetivo: {str(e)}"


def validar_estructura(df, configuracion):
    """
    Valida la estructura completa del dataset para el problema especificado.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos
        configuracion (dict): Diccionario con la configuración del modelo
    
    Returns:
        tuple: (bool, str) indicando si es válida y un mensaje
    """
    try:
        tipo_problema = configuracion.get("tipo_problema")
        variable_objetivo = configuracion.get("variable_objetivo")
        variables_predictoras = configuracion.get("variables_predictoras", [])
        
        # Validar que existan todos los elementos necesarios
        if not tipo_problema or not variable_objetivo or not variables_predictoras:
            return False, "La configuración está incompleta"
        
        # Validar variable objetivo
        objetivo_valida, mensaje = validar_variable_objetivo(df, variable_objetivo, tipo_problema)
        if not objetivo_valida:
            return False, mensaje
        
        # Validar que las variables predictoras existan
        for var in variables_predictoras:
            if var not in df.columns:
                return False, f"La variable predictora '{var}' no existe en el dataset"
        
        # Validar multicolinealidad entre predictores numéricos
        predictores_numericos = [col for col in variables_predictoras 
                            if pd.api.types.is_numeric_dtype(df[col])]
        
        if len(predictores_numericos) > 1:
            # Aquí se podría implementar detección de multicolinealidad
            # Por ejemplo, calculando la matriz de correlación
            pass
        
        # Si todo está bien, retornar éxito
        return True, "La estructura de datos es válida para el problema especificado"
    
    except Exception as e:
        logger.error(f"Error al validar estructura: {str(e)}")
        return False, f"Error al validar la estructura: {str(e)}"


def validar_tipos_datos(df):
    """
    Valida los tipos de datos de cada columna y detecta inconsistencias.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos
        
    Returns:
        list: Lista de diccionarios con errores detectados
    """
    errores = []
    
    try:
        # Revisar cada columna del DataFrame
        for columna in df.columns:
            # Obtener el tipo de datos actual
            tipo_actual = df[columna].dtype
            
            # Verificar columnas numéricas que podrían ser categóricas
            if pd.api.types.is_numeric_dtype(df[columna]):
                # Si tiene pocos valores únicos en relación al total, podría ser categórica
                n_unicos = df[columna].nunique()
                if 1 < n_unicos <= 10 and n_unicos / len(df) < 0.05:
                    errores.append({
                        'columna': columna,
                        'mensaje': f"Posible variable categórica codificada como numérica. Tiene {n_unicos} valores únicos.",
                        'sugerencia': "Considerar convertir a tipo categórico",
                        'opciones': ['Mantener como numérica', 'Convertir a categórica']
                    })
            
            # Verificar columnas de texto que podrían ser numéricas
            elif pd.api.types.is_string_dtype(df[columna]):
                # Comprobar si todos los valores no nulos podrían ser números
                valores_no_nulos = df[columna].dropna()
                if len(valores_no_nulos) > 0:
                    # Intentar convertir a numérico
                    try:
                        pd.to_numeric(valores_no_nulos)
                        errores.append({
                            'columna': columna,
                            'mensaje': "Columna de texto contiene solo valores numéricos",
                            'sugerencia': "Convertir a tipo numérico para análisis cuantitativo",
                            'opciones': ['Mantener como texto', 'Convertir a numérico']
                        })
                    except:
                        pass
                    
                    # Verificar si podría ser una fecha
                    muestra = valores_no_nulos.sample(min(5, len(valores_no_nulos))).tolist()
                    if all(es_posible_fecha(valor) for valor in muestra):
                        errores.append({
                            'columna': columna,
                            'mensaje': "Columna de texto podría contener fechas",
                            'sugerencia': "Convertir a tipo datetime para análisis temporal",
                            'opciones': ['Mantener como texto', 'Convertir a fecha']
                        })
            
            # Verificar columnas object que podrían ser fechas
            elif pd.api.types.is_object_dtype(df[columna]):
                # Comprobar si podría ser fecha
                muestra = df[columna].dropna().sample(min(5, len(df[columna].dropna()))).tolist()
                if len(muestra) > 0 and all(es_posible_fecha(str(valor)) for valor in muestra):
                    errores.append({
                        'columna': columna,
                        'mensaje': "Columna object podría contener fechas",
                        'sugerencia': "Convertir a tipo datetime para análisis temporal",
                        'opciones': ['Mantener como object', 'Convertir a fecha']
                    })
        
        # Registrar resultados
        logger.info(f"Validación de tipos de datos completada: {len(errores)} problemas encontrados")
        return errores
        
    except Exception as e:
        logger.error(f"Error al validar tipos de datos: {str(e)}")
        # Agregar el error a la lista
        errores.append({
            'columna': 'general',
            'mensaje': f"Error al validar tipos de datos: {str(e)}",
            'sugerencia': "Revisar el formato del archivo de entrada"
        })
        return errores


def es_posible_fecha(texto):
    """
    Verifica si un texto podría ser una fecha.
    
    Args:
        texto (str): Texto a verificar
        
    Returns:
        bool: True si podría ser una fecha, False en caso contrario
    """
    # Patrones comunes de fecha
    patrones = [
        r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}',  # DD/MM/YYYY, MM/DD/YYYY
        r'\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}',    # YYYY/MM/DD
        r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}',  # DD Mon YYYY
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}'  # Mon DD, YYYY
    ]
    
    # Verificar si coincide con algún patrón
    for patron in patrones:
        if re.match(patron, texto, re.IGNORECASE):
            return True
    
    # Intentar parsear con datetime
    try:
        # Probar diferentes formatos comunes
        formatos = [
            '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d',
            '%d-%m-%Y', '%m-%d-%Y', '%Y.%m.%d', '%d.%m.%Y',
            '%d %b %Y', '%d %B %Y', '%b %d %Y', '%B %d %Y'
        ]
        
        for formato in formatos:
            try:
                datetime.strptime(texto, formato)
                return True
            except:
                continue
                
        return False
    except:
        return False


def validar_fechas(df):
    """
    Valida las columnas de tipo fecha y detecta formatos inconsistentes.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos
        
    Returns:
        list: Lista de diccionarios con errores detectados
    """
    errores = []
    
    try:
        # Buscar columnas que son fechas o tienen nombres que sugieren fechas
        posibles_fechas = []
        
        # Columnas que ya son datetime
        columnas_datetime = [col for col in df.columns if pd.api.types.is_datetime64_dtype(df[col])]
        posibles_fechas.extend(columnas_datetime)
        
        # Columnas con nombres que sugieren fechas
        palabras_clave = ['fecha', 'date', 'time', 'hora', 'day', 'mes', 'month', 'año', 'year']
        for col in df.columns:
            if any(palabra in col.lower() for palabra in palabras_clave) and col not in posibles_fechas:
                posibles_fechas.append(col)
        
        # Analizar cada columna potencial de fecha
        for columna in posibles_fechas:
            # Si ya es datetime, verificar zonas horarias inconsistentes
            if pd.api.types.is_datetime64_dtype(df[columna]):
                # Verificar si hay información de zona horaria y si es consistente
                tiene_tz = df[columna].dt.tz is not None
                if not tiene_tz:
                    errores.append({
                        'columna': columna,
                        'mensaje': "Columna de fecha sin zona horaria especificada",
                        'sugerencia': "Considerar agregar información de zona horaria para análisis temporal preciso",
                        'formatos_disponibles': ['UTC', 'Local', 'GMT', 'America/New_York', 'Europe/Madrid']
                    })
            else:
                # Para columnas que no son datetime pero podrían contener fechas
                valores_no_nulos = df[columna].dropna()
                if len(valores_no_nulos) > 0:
                    # Convertir a string para análisis
                    valores_str = valores_no_nulos.astype(str)
                    
                    # Identificar formatos de fecha diferentes
                    formatos_detectados = detectar_formatos_fecha(valores_str)
                    
                    if len(formatos_detectados) > 1:
                        # Hay múltiples formatos en la misma columna
                        ejemplos = valores_str.sample(min(3, len(valores_str))).tolist()
                        errores.append({
                            'columna': columna,
                            'mensaje': f"Formatos de fecha inconsistentes detectados ({len(formatos_detectados)} formatos diferentes)",
                            'ejemplos': ejemplos,
                            'formato_sugerido': 'ISO 8601 (YYYY-MM-DD)',
                            'formatos_disponibles': ['ISO 8601 (YYYY-MM-DD)', 'DD/MM/YYYY', 'MM/DD/YYYY', 'YYYY/MM/DD']
                        })
                    elif len(formatos_detectados) == 1 and not pd.api.types.is_datetime64_dtype(df[columna]):
                        # Un solo formato pero no está como datetime
                        errores.append({
                            'columna': columna,
                            'mensaje': "Columna contiene fechas pero no está en formato datetime",
                            'sugerencia': "Convertir a tipo datetime para análisis temporal",
                            'formato_sugerido': 'ISO 8601 (YYYY-MM-DD)',
                            'formatos_disponibles': ['ISO 8601 (YYYY-MM-DD)', 'DD/MM/YYYY', 'MM/DD/YYYY', 'YYYY/MM/DD']
                        })
        
        # Registrar resultados
        logger.info(f"Validación de fechas completada: {len(errores)} problemas encontrados")
        return errores
        
    except Exception as e:
        logger.error(f"Error al validar fechas: {str(e)}")
        errores.append({
            'columna': 'general',
            'mensaje': f"Error al validar formatos de fecha: {str(e)}",
            'sugerencia': "Revisar el formato de las columnas de fecha"
        })
        return errores


def detectar_formatos_fecha(serie):
    """
    Detecta los diferentes formatos de fecha en una serie.
    
    Args:
        serie (pd.Series): Serie con valores de fecha como strings
        
    Returns:
        list: Lista de formatos detectados
    """
    formatos = set()
    
    # Patrones comunes
    patrones = {
        r'\d{4}-\d{2}-\d{2}': 'YYYY-MM-DD',
        r'\d{2}/\d{2}/\d{4}': 'DD/MM/YYYY o MM/DD/YYYY',
        r'\d{2}-\d{2}-\d{4}': 'DD-MM-YYYY o MM-DD-YYYY',
        r'\d{4}/\d{2}/\d{2}': 'YYYY/MM/DD',
        r'\d{2}\.\d{2}\.\d{4}': 'DD.MM.YYYY',
        r'\d{1,2}\s+[a-zA-Z]{3}\s+\d{4}': 'DD MMM YYYY',
        r'[a-zA-Z]{3}\s+\d{1,2},\s+\d{4}': 'MMM DD, YYYY'
    }
    
    # Muestrear para no procesar toda la serie si es muy grande
    muestra = serie.sample(min(100, len(serie)))
    
    for valor in muestra:
        for patron, formato in patrones.items():
            if re.match(patron, valor):
                formatos.add(formato)
                break
    
    return list(formatos)


def validar_unidades(df):
    """
    Detecta posibles inconsistencias en unidades de medida.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos
        
    Returns:
        list: Lista de diccionarios con errores detectados
    """
    errores = []
    
    try:
        # Definir patrones y palabras clave para unidades comunes
        unidades_temperatura = {
            'celsius': ['°c', 'c', 'celsius', 'centígrados', 'centigrados'],
            'fahrenheit': ['°f', 'f', 'fahrenheit'],
            'kelvin': ['k', 'kelvin']
        }
        
        unidades_peso = {
            'kilogramos': ['kg', 'kgs', 'kilogramos', 'kilos'],
            'gramos': ['g', 'gr', 'grs', 'gramos'],
            'libras': ['lb', 'lbs', 'libras', 'pounds'],
            'onzas': ['oz', 'onzas', 'ounces']
        }
        
        unidades_longitud = {
            'metros': ['m', 'mt', 'mts', 'metros'],
            'centimetros': ['cm', 'cms', 'centímetros', 'centimetros'],
            'pulgadas': ['in', 'inch', 'pulgadas'],
            'pies': ['ft', 'feet', 'pie', 'pies']
        }
        
        unidades_volumen = {
            'litros': ['l', 'lt', 'lts', 'litros'],
            'mililitros': ['ml', 'mls', 'mililitros'],
            'galones': ['gal', 'galones', 'gallons'],
            'onzas_liquidas': ['fl oz', 'fluid ounce', 'onzas líquidas']
        }
        
        # Buscar columnas numéricas que podrían contener unidades
        for columna in df.columns:
            if pd.api.types.is_numeric_dtype(df[columna]):
                # Buscar pistas en el nombre de la columna
                nombre_lower = columna.lower()
                
                # Detectar posibles columnas de temperatura
                if any(keyword in nombre_lower for keyword in ['temp', 'temperatura', 'temperature']):
                    # Analizar el rango de valores para inferir unidades
                    min_val, max_val = df[columna].min(), df[columna].max()
                    
                    if 0 <= min_val <= 50 and 10 <= max_val <= 50:
                        # Probablemente Celsius
                        unidad_inferida = 'celsius'
                    elif 32 <= min_val <= 100 and 50 <= max_val <= 100:
                        # Probablemente Fahrenheit
                        unidad_inferida = 'fahrenheit'
                    elif 273 <= min_val <= 373:
                        # Probablemente Kelvin
                        unidad_inferida = 'kelvin'
                    else:
                        # No se puede determinar
                        continue
                    
                    # Verificar si hay valores atípicos que sugieran mezcla de unidades
                    q1, q3 = df[columna].quantile(0.25), df[columna].quantile(0.75)
                    iqr = q3 - q1
                    limite_inf, limite_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    
                    outliers = df[columna][(df[columna] < limite_inf) | (df[columna] > limite_sup)]
                    if len(outliers) > 0 and len(outliers) / len(df) < 0.1:
                        # Posible mezcla de unidades
                        errores.append({
                            'columna': columna,
                            'mensaje': f"Posible mezcla de unidades de temperatura. Unidad principal inferida: {unidad_inferida}",
                            'sugerencia': f"Estandarizar todas las mediciones a {unidad_inferida}",
                            'unidades_detectadas': [unidad_inferida, 'desconocida'],
                            'unidad_sugerida': unidad_inferida,
                            'unidades_disponibles': ['celsius', 'fahrenheit', 'kelvin']
                        })
                
                # Detectar posibles columnas de peso
                elif any(keyword in nombre_lower for keyword in ['peso', 'weight', 'masa', 'mass']):
                    # Análisis similar al de temperatura pero para pesos
                    # ...
                    pass
                
                # Detectar posibles columnas de longitud
                elif any(keyword in nombre_lower for keyword in ['longitud', 'length', 'altura', 'height', 'ancho', 'width']):
                    # Análisis similar para longitudes
                    # ...
                    pass
                
                # Detectar posibles columnas de volumen
                elif any(keyword in nombre_lower for keyword in ['volumen', 'volume', 'capacidad', 'capacity']):
                    # Análisis similar para volúmenes
                    # ...
                    pass
        
        # Registrar resultados
        logger.info(f"Validación de unidades completada: {len(errores)} problemas encontrados")
        return errores
        
    except Exception as e:
        logger.error(f"Error al validar unidades: {str(e)}")
        errores.append({
            'columna': 'general',
            'mensaje': f"Error al validar unidades de medida: {str(e)}",
            'sugerencia': "Revisar el formato de las columnas con unidades de medida"
        })
        return errores


def detectar_inconsistencias(df):
    """
    Detecta inconsistencias generales en el dataset.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos
        
    Returns:
        list: Lista de diccionarios con errores detectados
    """
    # Combinar los resultados de todas las validaciones
    errores = []
    errores.extend(validar_tipos_datos(df))
    errores.extend(validar_fechas(df))
    errores.extend(validar_unidades(df))
    
    return errores
