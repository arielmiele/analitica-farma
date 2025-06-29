"""
Módulo para validar la estructura de datos y variables objetivo.
"""
import pandas as pd
import re
from datetime import datetime
from src.audit.logger import log_audit

def validar_variable_objetivo(df, variable_objetivo, tipo_problema, id_sesion, usuario):
    """
    Valida que la variable objetivo sea adecuada para el tipo de problema.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos
        variable_objetivo (str): Nombre de la columna objetivo
        tipo_problema (str): 'regresion' o 'clasificacion'
        id_sesion (str): ID de sesión para trazabilidad
        usuario (str): Usuario que ejecuta la acción
    
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
                log_audit(
                    usuario=usuario,
                    accion="ADVERTENCIA_VARIABLE_OBJETIVO_NULOS",
                    entidad="validador",
                    id_entidad=variable_objetivo,
                    detalles=f"La variable objetivo tiene {nulos} valores nulos ({porcentaje:.1f}%)",
                    id_sesion=id_sesion
                )
        
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
        log_audit(
            usuario=usuario,
            accion="ERROR_VALIDAR_VARIABLE_OBJETIVO",
            entidad="validador",
            id_entidad="N/A",
            detalles=f"Error al validar variable objetivo: {str(e)}",
            id_sesion=id_sesion
        )
        return False, f"Error al validar la variable objetivo: {str(e)}"


def validar_estructura(df, configuracion, id_sesion, usuario):
    """
    Valida la estructura completa del dataset para el problema especificado.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos
        configuracion (dict): Diccionario con la configuración del modelo
        id_sesion (str): ID de sesión para trazabilidad
        usuario (str): Usuario que ejecuta la acción
    
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
        objetivo_valida, mensaje = validar_variable_objetivo(df, variable_objetivo, tipo_problema, id_sesion, usuario)
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
        log_audit(
            usuario=usuario,
            accion="ERROR_VALIDAR_ESTRUCTURA",
            entidad="validador",
            id_entidad="N/A",
            detalles=f"Error al validar estructura: {str(e)}",
            id_sesion=id_sesion
        )
        return False, f"Error al validar la estructura: {str(e)}"


def validar_tipos_datos(df, id_sesion, usuario):
    """
    Valida los tipos de datos de cada columna y detecta inconsistencias.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos
        id_sesion (str): ID de sesión para trazabilidad
        usuario (str): Usuario que ejecuta la acción
        
    Returns:
        list: Lista de diccionarios con errores detectados
    """
    errores = []
    
    try:
        # Revisar cada columna del DataFrame
        for columna in df.columns:
            # Obtener el tipo de datos actual
            tipo_actual = df[columna].dtype  # noqa: F841
            
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
                    except Exception:
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
        
        log_audit(
            usuario=usuario,
            accion="INFO_VALIDACION_TIPOS_DATOS",
            entidad="validador",
            id_entidad="N/A",
            detalles=f"Validación de tipos de datos completada: {len(errores)} problemas encontrados",
            id_sesion=id_sesion
        )
        return errores
        
    except Exception as e:
        log_audit(
            usuario=usuario,
            accion="ERROR_VALIDAR_TIPOS_DATOS",
            entidad="validador",
            id_entidad="N/A",
            detalles=f"Error al validar tipos de datos: {str(e)}",
            id_sesion=id_sesion
        )
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
            except Exception:
                continue
                
        return False
    except Exception:
        return False


def validar_fechas(df, id_sesion, usuario):
    """
    Valida columnas de fecha para asegurar que sean útiles en ML, no solo formato.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos
        id_sesion (str): ID de sesión para trazabilidad
        usuario (str): Usuario que ejecuta la acción
    
    Returns:
        list: Lista de advertencias/errores relevantes para ML
    """
    errores = []
    
    try:
        # Detectar columnas candidatas a fecha
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
            serie = df[columna]
            
            # Intentar convertir a datetime
            serie_fecha = pd.to_datetime(serie, errors='coerce')
            
            n_nulos = serie_fecha.isna().sum()
            total = len(serie_fecha)
            n_unicos = serie_fecha.nunique(dropna=True)
            rango = None
            if not serie_fecha.dropna().empty:
                try:
                    rango = (serie_fecha.max() - serie_fecha.min()).days
                except Exception:
                    rango = None
            
            # 1. Demasiados nulos
            if n_nulos > total * 0.2:
                errores.append({
                    'columna': columna,
                    'mensaje': f"Más del 20% de los valores de fecha son nulos o inválidos ({n_nulos} de {total}).",
                    'sugerencia': "Imputar o eliminar filas nulas para usar la fecha en ML."
                })
            
            # 2. No convertible a fecha
            if n_nulos == total:
                errores.append({
                    'columna': columna,
                    'mensaje': "Ningún valor es convertible a fecha.",
                    'sugerencia': "Revisar el formato o el contenido de la columna."
                })
                continue
            
            # 3. Todos los valores únicos (posible identificador)
            if n_unicos == total - n_nulos:
                errores.append({
                    'columna': columna,
                    'mensaje': "Todos los valores de fecha son únicos. Probablemente es un identificador y no aporta valor para ML.",
                    'sugerencia': "Evite usar esta columna como predictor. Considere extraer componentes como año, mes, día."
                })
            
            # 4. Sin variabilidad temporal
            if rango is not None and rango < 2:
                errores.append({
                    'columna': columna,
                    'mensaje': f"La columna de fecha tiene muy poca variabilidad temporal (rango: {rango} días).",
                    'sugerencia': "Verifique si la columna aporta información útil para ML."
                })
            
            # 5. Fechas futuras (opcional, solo si la mayoría son futuras)
            hoy = pd.Timestamp.today()
            n_futuras = (serie_fecha > hoy).sum()
            if n_futuras > total * 0.5:
                errores.append({
                    'columna': columna,
                    'mensaje': "Más del 50% de las fechas están en el futuro.",
                    'sugerencia': "Verifique si esto tiene sentido para el dominio del problema."
                })
            
            # 6. Sugerir extracción de componentes si la columna es válida
            if n_nulos < total * 0.2 and n_unicos < total * 0.9 and rango is not None and rango >= 2:
                errores.append({
                    'columna': columna,
                    'mensaje': "Columna de fecha válida para ML, pero se recomienda extraer variables como año, mes, día, día de la semana, o diferencias temporales.",
                    'sugerencia': "Utilice la función de transformación para crear variables derivadas de la fecha."
                })
        
        log_audit(
            usuario=usuario,
            accion="INFO_VALIDACION_FECHAS",
            entidad="validador",
            id_entidad="N/A",
            detalles=f"Validación avanzada de fechas para ML completada: {len(errores)} advertencias/errores.",
            id_sesion=id_sesion
        )
        return errores
        
    except Exception as e:
        log_audit(
            usuario=usuario,
            accion="ERROR_VALIDAR_FECHAS",
            entidad="validador",
            id_entidad="N/A",
            detalles=f"Error al validar fechas: {str(e)}",
            id_sesion=id_sesion
        )
        errores.append({
            'columna': 'general',
            'mensaje': f"Error al validar fechas: {str(e)}",
            'sugerencia': "Revisar el formato y contenido de las columnas de fecha."
        })
        return errores


# FIN DEL MÓDULO: Se eliminaron las funciones detectar_formatos_fecha, validar_unidades y detectar_inconsistencias por no ser necesarias en la arquitectura y flujo actual.
