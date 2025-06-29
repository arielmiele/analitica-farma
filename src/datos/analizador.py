"""
Módulo para analizar la calidad de los datos cargados.
Proporciona funciones para evaluar nulos, duplicados, outliers y 
generar estadísticas descriptivas.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from src.audit.logger import log_audit

def calcular_metricas_basicas(df: pd.DataFrame, id_sesion: str) -> Dict[str, Any]:
    """
    Calcula métricas básicas de calidad del DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame a analizar
    
    Returns:
        Dict[str, Any]: Diccionario con métricas básicas
    """
    try:
        # Contar filas y columnas
        filas, columnas = df.shape
        
        # Calcular valores nulos totales y porcentaje
        nulos_totales = df.isna().sum().sum()
        porcentaje_nulos = (nulos_totales / (filas * columnas)) * 100
        
        # Verificar duplicados
        duplicados = df.duplicated().sum()
        porcentaje_duplicados = (duplicados / filas) * 100 if filas > 0 else 0
        
        # Valores únicos por columna (promedio)
        valores_unicos_promedio = df.nunique().mean()
        
        # Completitud global (porcentaje de datos no nulos)
        completitud = 100 - porcentaje_nulos
        
        # Resultado en un diccionario
        return {
            "filas": filas,
            "columnas": columnas,
            "nulos_totales": nulos_totales,
            "porcentaje_nulos": porcentaje_nulos,
            "duplicados": duplicados,
            "porcentaje_duplicados": porcentaje_duplicados,
            "valores_unicos_promedio": valores_unicos_promedio,
            "completitud": completitud
        }
    except Exception as e:
        log_audit(
            usuario="sistema",
            accion="ERROR_METRICAS_BASICAS",
            entidad="analizador",
            id_entidad="",
            detalles=f"Error al calcular métricas básicas: {str(e)}",
            id_sesion=id_sesion
        )
        return {"error": str(e)}

def analizar_nulos_por_columna(df: pd.DataFrame, id_sesion: str) -> pd.DataFrame:
    """
    Analiza los valores nulos por columna.
    
    Args:
        df (pd.DataFrame): DataFrame a analizar
    
    Returns:
        pd.DataFrame: DataFrame con estadísticas de nulos por columna
    """
    try:
        # Calcular nulos por columna
        nulos = df.isna().sum()
        porcentaje = (nulos / len(df)) * 100
        
        # Crear DataFrame con los resultados
        resultado = pd.DataFrame({
            'columna': nulos.index,
            'nulos': nulos.values,
            'porcentaje': porcentaje.values,
            'completitud': 100 - np.array(porcentaje.values)
        })
        
        # Ordenar por porcentaje de nulos (descendente)
        resultado = resultado.sort_values('porcentaje', ascending=False)
        
        # Agregar una clasificación según el porcentaje de nulos
        resultado['clasificacion'] = pd.cut(
            resultado['porcentaje'], 
            bins=[0, 5, 20, 50, 100],
            labels=['Excelente', 'Buena', 'Regular', 'Crítica']
        )
        
        return resultado
    except Exception as e:
        log_audit(
            usuario="sistema",
            accion="ERROR_NULOS_POR_COLUMNA",
            entidad="analizador",
            id_entidad="",
            detalles=f"Error al analizar nulos por columna: {str(e)}",
            id_sesion=id_sesion
        )
        return pd.DataFrame()

def detectar_outliers(df: pd.DataFrame, metodo: str = 'iqr', umbral: float = 1.5, id_sesion: str = "sin_sesion") -> Dict[str, Dict[str, Any]]:
    """
    Detecta outliers en columnas numéricas usando diferentes métodos.
    
    Args:
        df (pd.DataFrame): DataFrame a analizar
        metodo (str): Método para detectar outliers ('iqr', 'zscore', 'desviacion')
        umbral (float): Umbral para considerar un valor como outlier
    
    Returns:
        Dict[str, Dict[str, Any]]: Diccionario con resultados por columna
    """
    resultados = {}
    
    try:
        # Solo analizar columnas numéricas
        columnas_numericas = df.select_dtypes(include=['number']).columns
        
        for columna in columnas_numericas:
            serie = df[columna].dropna()
            
            # Si la serie está vacía o tiene un solo valor, saltarla
            if len(serie) <= 1:
                continue
            
            indices_outliers = []
            metrica = {}  # Inicializar metrica por defecto
            
            if metodo == 'iqr':
                # Método del rango intercuartílico (IQR)
                q1 = serie.quantile(0.25)
                q3 = serie.quantile(0.75)
                iqr = q3 - q1
                
                limite_inferior = q1 - umbral * iqr
                limite_superior = q3 + umbral * iqr
                
                indices_outliers = serie[(serie < limite_inferior) | (serie > limite_superior)].index
                metrica = {'q1': q1, 'q3': q3, 'iqr': iqr, 'limite_inferior': limite_inferior, 'limite_superior': limite_superior}
                
            elif metodo == 'zscore':
                # Método del z-score
                media = serie.mean()
                desv_std = serie.std()
                
                z_scores = np.abs((serie - media) / desv_std)
                indices_outliers = serie[z_scores > umbral].index
                metrica = {'media': media, 'desv_std': desv_std, 'umbral_zscore': umbral}
                
            elif metodo == 'desviacion':
                # Método de la desviación estándar
                media = serie.mean()
                desv_std = serie.std()
                
                limite_inferior = media - umbral * desv_std
                limite_superior = media + umbral * desv_std
                
                indices_outliers = serie[(serie < limite_inferior) | (serie > limite_superior)].index
                metrica = {'media': media, 'desv_std': desv_std, 'limite_inferior': limite_inferior, 'limite_superior': limite_superior}
            
            # Calcular porcentaje de outliers
            cantidad_outliers = len(indices_outliers)
            porcentaje_outliers = (cantidad_outliers / len(serie)) * 100 if len(serie) > 0 else 0
            
            # Guardar resultados
            resultados[columna] = {
                'indices': indices_outliers,
                'cantidad': cantidad_outliers,
                'porcentaje': porcentaje_outliers,
                'metodo': metodo,
                'umbral': umbral,
                'metrica': metrica
            }
    
    except Exception as e:
        log_audit(
            usuario="sistema",
            accion="ERROR_OUTLIERS",
            entidad="analizador",
            id_entidad="",
            detalles=f"Error al detectar outliers: {str(e)}",
            id_sesion=id_sesion
        )
    
    return resultados

def analizar_duplicados(df: pd.DataFrame, columnas: Optional[List[str]] = None, id_sesion: str = "sin_sesion") -> Dict[str, Any]:
    """
    Analiza duplicados en el DataFrame, opcionalmente basado en columnas específicas.
    
    Args:
        df (pd.DataFrame): DataFrame a analizar
        columnas (List[str], optional): Lista de columnas para considerar en la detección de duplicados
    
    Returns:
        Dict[str, Any]: Diccionario con resultados del análisis de duplicados
    """
    try:
        # Si no se especifican columnas, usar todas
        if columnas is None:
            columnas = df.columns.tolist()
        
        # Verificar que todas las columnas existan
        columnas_validas = [col for col in columnas if col in df.columns]
        
        # Detectar duplicados
        duplicados = df.duplicated(subset=columnas_validas, keep='first')
        indices_duplicados = df[duplicados].index.tolist()
        
        # Contar duplicados por grupo
        if indices_duplicados:
            # Obtener grupos de duplicados
            grupos = df.groupby(columnas_validas).size().reset_index(name='conteo')
            grupos_duplicados = grupos[grupos['conteo'] > 1]
            
            # Ordenar por conteo descendente
            grupos_duplicados = grupos_duplicados.sort_values('conteo', ascending=False)
        else:
            grupos_duplicados = pd.DataFrame()
        
        return {
            'cantidad': len(indices_duplicados),
            'porcentaje': (len(indices_duplicados) / len(df)) * 100 if len(df) > 0 else 0,
            'indices': indices_duplicados,
            'columnas_analizadas': columnas_validas,
            'grupos_duplicados': grupos_duplicados
        }
    except Exception as e:
        log_audit(
            usuario="sistema",
            accion="ERROR_DUPLICADOS",
            entidad="analizador",
            id_entidad="",
            detalles=f"Error al analizar duplicados: {str(e)}",
            id_sesion=id_sesion
        )
        return {'error': str(e)}

def generar_estadisticas_por_columna(df: pd.DataFrame, id_sesion: str) -> pd.DataFrame:
    """
    Genera estadísticas descriptivas por columna, mostrando solo la información
    relevante para cada tipo de dato y evitando filas y columnas innecesarias.
    
    Args:
        df (pd.DataFrame): DataFrame a analizar
    
    Returns:
        pd.DataFrame: DataFrame con estadísticas por columna
    """
    try:
        # Inicializar DataFrame para resultados
        columnas = df.columns
        resultados = []
        
        # Determinar qué tipos de columnas tenemos para mostrar solo las estadísticas relevantes
        tiene_numericas = not df.select_dtypes(include=['number']).empty
        tiene_texto = not df.select_dtypes(include=['object', 'string']).empty
        tiene_fechas = not df.select_dtypes(include=['datetime']).empty
        
        for col in columnas:
            serie = df[col]
            tipo_dato = serie.dtype
            datos_no_nulos = serie.dropna()
            
            # Estadísticas comunes para todos los tipos
            stats = {
                'columna': col,
                'tipo': str(tipo_dato),
                'nulos': serie.isna().sum(),
                'porcentaje_nulos': (serie.isna().sum() / len(df)) * 100 if len(df) > 0 else 0,
                'valores_unicos': serie.nunique(),
            }
            
            # Estadísticas específicas según el tipo
            if pd.api.types.is_numeric_dtype(serie):
                if not datos_no_nulos.empty:
                    # Para columnas numéricas con datos
                    stats.update({
                        'min': datos_no_nulos.min(),
                        'max': datos_no_nulos.max(),
                        'media': datos_no_nulos.mean(),
                        'mediana': datos_no_nulos.median(),
                        'desv_std': datos_no_nulos.std()
                    })
                else:
                    # Columnas numéricas sin datos
                    stats.update({
                        'min': None,
                        'max': None,
                        'media': None,
                        'mediana': None,
                        'desv_std': None
                    }) if tiene_numericas else None
                    
            elif pd.api.types.is_string_dtype(serie) or pd.api.types.is_object_dtype(serie):
                # Para columnas de texto
                if not datos_no_nulos.empty:
                    valor_counts = datos_no_nulos.value_counts()
                    if not valor_counts.empty:
                        valor_mas_comun = valor_counts.index[0]
                        frecuencia_valor_comun = valor_counts.iloc[0]
                    else:
                        valor_mas_comun = None
                        frecuencia_valor_comun = 0
                    
                    try:
                        # Algunas columnas de objeto pueden no ser strings
                        longitud_promedio = datos_no_nulos.astype(str).str.len().mean()
                    except Exception:
                        longitud_promedio = None
                    
                    stats.update({
                        'valor_mas_comun': valor_mas_comun,
                        'frecuencia_valor_comun': frecuencia_valor_comun,
                        'longitud_promedio': longitud_promedio
                    })
                else:
                    # Columnas de texto sin datos
                    stats.update({
                        'valor_mas_comun': None,
                        'frecuencia_valor_comun': None,
                        'longitud_promedio': None
                    }) if tiene_texto else None
                    
            elif pd.api.types.is_datetime64_dtype(serie):
                # Para columnas de fecha/hora
                if not datos_no_nulos.empty:
                    try:
                        rango_dias = (datos_no_nulos.max() - datos_no_nulos.min()).days
                    except Exception:
                        rango_dias = None
                    
                    stats.update({
                        'min': datos_no_nulos.min(),
                        'max': datos_no_nulos.max(),
                        'rango_dias': rango_dias
                    })
                else:
                    # Columnas de fecha sin datos
                    stats.update({
                        'min': None,
                        'max': None,
                        'rango_dias': None
                    }) if tiene_fechas else None
            
            resultados.append(stats)
        
        # Convertir a DataFrame
        df_resultado = pd.DataFrame(resultados)
        
        # Si el DataFrame está vacío, retornar el DataFrame vacío
        if df_resultado.empty:
            return df_resultado
            
        # Determinar qué columnas mostrar según el contenido real
        # Primero, las columnas comunes que siempre se muestran
        cols_seleccionadas = ['columna', 'tipo', 'nulos', 'porcentaje_nulos', 'valores_unicos']
        
        # Agregar columnas específicas solo si tienen al menos un valor no nulo
        columnas_numericas = ['min', 'max', 'media', 'mediana', 'desv_std']
        columnas_texto = ['valor_mas_comun', 'frecuencia_valor_comun', 'longitud_promedio']
        columnas_fecha = ['min', 'max', 'rango_dias']
        
        # Verificar si hay valores no nulos en cada grupo de columnas
        for grupo in [columnas_numericas, columnas_texto, columnas_fecha]:
            # Filtrar solo las columnas que existen en el DataFrame
            grupo_existente = [col for col in grupo if col in df_resultado.columns]
            if grupo_existente:
                # Verificar si hay al menos una columna con valores no nulos
                tiene_valores = df_resultado[grupo_existente].notna().any().any()
                if tiene_valores:
                    # Agregar solo las columnas que no están ya en la lista
                    cols_seleccionadas.extend([col for col in grupo_existente if col not in cols_seleccionadas])
        
        # Filtrar columnas existentes
        cols_existentes = [col for col in cols_seleccionadas if col in df_resultado.columns]
        
        # Devolver el DataFrame con las columnas seleccionadas
        return df_resultado[cols_existentes]
        
    except Exception as e:
        log_audit(
            usuario="sistema",
            accion="ERROR_ESTADISTICAS_COLUMNA",
            entidad="analizador",
            id_entidad="",
            detalles=f"Error al generar estadísticas por columna: {str(e)}",
            id_sesion=id_sesion
        )
        return pd.DataFrame()

def evaluar_calidad_global(df: pd.DataFrame, id_sesion: str) -> Dict[str, Any]:
    """
    Evalúa la calidad global del dataset y asigna una calificación.
    
    Args:
        df (pd.DataFrame): DataFrame a analizar
    
    Returns:
        Dict[str, Any]: Diccionario con evaluación global
    """
    try:
        # Obtener métricas básicas
        metricas = calcular_metricas_basicas(df, id_sesion)
        
        # Calcular puntaje de calidad (0-100)
        puntaje = 0
        
        # Componente de completitud (0-40 puntos)
        completitud = metricas.get('completitud', 0)
        puntaje_completitud = min(40, completitud * 0.4)
        
        # Componente de duplicados (0-30 puntos)
        porcentaje_duplicados = metricas.get('porcentaje_duplicados', 0)
        puntaje_duplicados = min(30, 30 * (1 - porcentaje_duplicados / 100))
        
        # Componente de outliers (0-30 puntos)
        outliers = detectar_outliers(df, id_sesion=id_sesion)
        porcentaje_outliers_promedio = np.mean([info.get('porcentaje', 0) for info in outliers.values()]) if outliers else 0
        puntaje_outliers = min(30.0, 30 * (1 - float(porcentaje_outliers_promedio) / 100))
        
        # Puntaje total
        puntaje = puntaje_completitud + puntaje_duplicados + puntaje_outliers
        
        # Clasificar calidad
        if puntaje >= 90:
            calificacion = "Excelente"
        elif puntaje >= 75:
            calificacion = "Buena"
        elif puntaje >= 50:
            calificacion = "Regular"
        else:
            calificacion = "Deficiente"
        
        return {
            'puntaje': puntaje,
            'calificacion': calificacion,
            'puntaje_completitud': puntaje_completitud,
            'puntaje_duplicados': puntaje_duplicados,
            'puntaje_outliers': puntaje_outliers,
            'metricas': metricas
        }
    except Exception as e:
        log_audit(
            usuario="sistema",
            accion="ERROR_CALIDAD_GLOBAL",
            entidad="analizador",
            id_entidad="",
            detalles=f"Error al evaluar calidad global: {str(e)}",
            id_sesion=id_sesion
        )
        return {'error': str(e)}

def obtener_recomendaciones(df: pd.DataFrame, id_sesion: str) -> List[Dict[str, str]]:
    """
    Genera recomendaciones automáticas basadas en el análisis de calidad.
    
    Args:
        df (pd.DataFrame): DataFrame a analizar
    
    Returns:
        List[Dict[str, str]]: Lista de recomendaciones con tipo y mensaje
    """
    recomendaciones = []
    
    try:
        # Analizar nulos
        nulos_por_columna = analizar_nulos_por_columna(df, id_sesion)
        
        # Recomendar acciones para columnas con muchos nulos
        columnas_criticas = nulos_por_columna[nulos_por_columna['porcentaje'] > 50]
        if not columnas_criticas.empty:
            recomendaciones.append({
                'tipo': 'advertencia',
                'mensaje': f"Se encontraron {len(columnas_criticas)} columnas con más del 50% de valores nulos. "
                          f"Considere eliminar estas columnas o utilizar métodos avanzados de imputación."
            })
        
        columnas_problema = nulos_por_columna[(nulos_por_columna['porcentaje'] > 20) & (nulos_por_columna['porcentaje'] <= 50)]
        if not columnas_problema.empty:
            recomendaciones.append({
                'tipo': 'advertencia',
                'mensaje': f"Se encontraron {len(columnas_problema)} columnas con 20-50% de valores nulos. "
                          f"Se recomienda imputar estos valores usando métodos como media, mediana o KNN."
            })
        
        # Analizar duplicados
        duplicados = analizar_duplicados(df, id_sesion=id_sesion)
        if duplicados.get('porcentaje', 0) > 5:
            recomendaciones.append({
                'tipo': 'advertencia',
                'mensaje': f"El dataset contiene {duplicados.get('porcentaje', 0):.1f}% de filas duplicadas. "
                          f"Considere eliminar duplicados para mejorar la calidad del modelo."
            })
        
        # Analizar outliers
        outliers = detectar_outliers(df, id_sesion=id_sesion)
        columnas_con_outliers = [col for col, info in outliers.items() if info.get('porcentaje', 0) > 5]
        
        if columnas_con_outliers:
            recomendaciones.append({
                'tipo': 'informacion',
                'mensaje': f"Se detectaron outliers significativos en {len(columnas_con_outliers)} columnas. "
                          f"Considere revisar y tratar estos valores atípicos antes del modelado."
            })
        
        # Recomendación general basada en calidad
        evaluacion = evaluar_calidad_global(df, id_sesion)
        calificacion = evaluacion.get('calificacion', 'Desconocida')
        
        if calificacion == 'Deficiente':
            recomendaciones.append({
                'tipo': 'error',
                'mensaje': "La calidad general de los datos es deficiente. Se recomienda un proceso exhaustivo "
                          "de limpieza y preparación antes de continuar con el modelado."
            })
        elif calificacion == 'Regular':
            recomendaciones.append({
                'tipo': 'advertencia',
                'mensaje': "La calidad general de los datos es regular. Se recomienda aplicar las correcciones "
                          "sugeridas para mejorar la calidad antes del modelado."
            })
        elif calificacion == 'Buena':
            recomendaciones.append({
                'tipo': 'informacion',
                'mensaje': "La calidad general de los datos es buena. Puede continuar con el modelado, "
                          "aunque considere aplicar algunas mejoras menores."
            })
        elif calificacion == 'Excelente':
            recomendaciones.append({
                'tipo': 'exito',
                'mensaje': "La calidad general de los datos es excelente. Puede proceder directamente "
                          "con el modelado sin preocupaciones significativas."
            })
        
        # Si no hay problemas graves, añadir recomendación positiva
        if len(recomendaciones) <= 1:
            recomendaciones.append({
                'tipo': 'exito',
                'mensaje': "No se detectaron problemas significativos en los datos. "
                          "Puede continuar con confianza al siguiente paso."
            })
        
    except Exception as e:
        log_audit(
            usuario="sistema",
            accion="ERROR_RECOMENDACIONES",
            entidad="analizador",
            id_entidad="",
            detalles=f"Error al generar recomendaciones: {str(e)}",
            id_sesion=id_sesion
        )
        recomendaciones.append({
            'tipo': 'error',
            'mensaje': f"Error al analizar los datos: {str(e)}"
        })
    
    return recomendaciones
