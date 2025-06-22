"""
Módulo para validar la estructura de datos y variables objetivo.
"""
import pandas as pd
import logging

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
        predictores_num = [var for var in variables_predictoras 
                          if pd.api.types.is_numeric_dtype(df[var])]
        
        if len(predictores_num) >= 2:
            # Calcular matriz de correlación
            corr_matrix = df[predictores_num].corr().abs()
            
            # Identificar pares con alta correlación (>0.95)
            alta_corr = []
            for i in range(len(predictores_num)):
                for j in range(i+1, len(predictores_num)):
                    if corr_matrix.iloc[i, j] > 0.95:
                        alta_corr.append((predictores_num[i], predictores_num[j], corr_matrix.iloc[i, j]))
            
            if alta_corr:
                mensaje = "Se detectaron variables predictoras con alta correlación entre sí:\n"
                for var1, var2, corr in alta_corr:
                    mensaje += f"- {var1} y {var2}: {corr:.2f}\n"
                mensaje += "Considera eliminar una de cada par para mejorar el modelo."
                logger.warning(mensaje)
        
        # Validar separabilidad (para clasificación)
        if tipo_problema == "clasificacion" and len(predictores_num) > 0:
            # Análisis simplificado de separabilidad para cada predictor numérico
            baja_separabilidad = []
            
            for var in predictores_num:
                # Calcular ratio de varianza entre clases / varianza total
                if df[var].std() == 0:  # Evitar división por cero
                    continue
                    
                # Agrupar por clase y calcular medias
                group_means = df.groupby(variable_objetivo)[var].mean()
                
                # Si todas las medias son muy similares, hay baja separabilidad
                if group_means.std() / df[var].std() < 0.1:
                    baja_separabilidad.append(var)
            
            if baja_separabilidad and len(baja_separabilidad) > len(predictores_num) // 2:
                mensaje = "Muchas variables predictoras muestran baja capacidad de separación entre clases."
                logger.warning(mensaje)
        
        return True, "La estructura de datos es válida para el análisis"
        
    except Exception as e:
        logger.error(f"Error al validar estructura: {str(e)}")
        return False, f"Error al validar la estructura de datos: {str(e)}"
