"""
Módulo para manejar la serialización y deserialización de modelos de machine learning.
Proporciona funciones para convertir modelos de scikit-learn a formatos serializables
y viceversa, permitiendo su almacenamiento en JSON y bases de datos.
"""
import joblib
import base64
import io
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("modelo_serializer")

def serializar_modelo(modelo: Any) -> Optional[str]:
    """
    Serializa un modelo de scikit-learn a una cadena base64 que se puede almacenar en JSON.
    
    Args:
        modelo: El objeto modelo de scikit-learn a serializar
        
    Returns:
        str: Representación base64 del modelo serializado, o None si falla
    """
    try:
        # Usar un buffer en memoria para guardar el modelo
        buffer = io.BytesIO()
        # Serializar el modelo usando joblib (más eficiente que pickle para modelos de ML)
        joblib.dump(modelo, buffer, compress=3)
        # Convertir a base64 para almacenamiento seguro en JSON
        serialized_model = base64.b64encode(buffer.getvalue()).decode('utf-8')
        logger.info(f"Modelo serializado correctamente ({len(serialized_model)} bytes)")
        return serialized_model
    except Exception as e:
        logger.error(f"Error al serializar modelo: {str(e)}")
        return None

def deserializar_modelo(serialized_model: str) -> Optional[Any]:
    """
    Deserializa un modelo desde su representación base64.
    
    Args:
        serialized_model (str): Representación base64 del modelo serializado
        
    Returns:
        Any: El modelo deserializado, o None si falla
    """
    try:
        # Decodificar la cadena base64
        buffer = io.BytesIO(base64.b64decode(serialized_model))
        # Cargar el modelo usando joblib
        modelo = joblib.load(buffer)
        logger.info("Modelo deserializado correctamente")
        return modelo
    except Exception as e:
        logger.error(f"Error al deserializar modelo: {str(e)}")
        return None

def serializar_modelos_benchmarking(resultados: Dict) -> Dict:
    """
    Prepara los resultados del benchmarking para serialización JSON,
    convirtiendo modelos a formatos serializables.
    
    Args:
        resultados (Dict): Resultados del benchmarking con objetos modelo
        
    Returns:
        Dict: Resultados preparados para serialización JSON
    """
    # Crear una copia para no modificar el original
    resultados_serializables = resultados.copy()
    
    # Procesar modelos exitosos
    for modelo in resultados_serializables.get('modelos_exitosos', []):
        if 'modelo_objeto' in modelo:
            # Serializar el objeto modelo
            modelo['modelo_serializado'] = serializar_modelo(modelo['modelo_objeto'])
            # Marcar que tenía un objeto modelo
            modelo['tiene_modelo_objeto'] = True
            # Eliminar el objeto que no es serializable
            del modelo['modelo_objeto']
    
    # Procesar el mejor modelo
    if (resultados_serializables.get('mejor_modelo') and 
        'modelo_objeto' in resultados_serializables['mejor_modelo']):
        mejor_modelo = resultados_serializables['mejor_modelo']
        mejor_modelo['modelo_serializado'] = serializar_modelo(mejor_modelo['modelo_objeto'])
        mejor_modelo['tiene_modelo_objeto'] = True
        del mejor_modelo['modelo_objeto']
    
    return resultados_serializables

def deserializar_modelos_benchmarking(resultados: Dict) -> Dict:
    """
    Reconstruye los objetos modelo a partir de sus representaciones serializadas
    en los resultados del benchmarking.
    
    Args:
        resultados (Dict): Resultados del benchmarking con modelos serializados
        
    Returns:
        Dict: Resultados con objetos modelo reconstruidos
    """
    # Crear una copia para no modificar el original
    resultados_reconstruidos = resultados.copy()
    
    # Reconstruir modelos exitosos
    for modelo in resultados_reconstruidos.get('modelos_exitosos', []):
        if 'modelo_serializado' in modelo and modelo.get('tiene_modelo_objeto', False):
            # Deserializar el objeto modelo
            modelo_objeto = deserializar_modelo(modelo['modelo_serializado'])
            if modelo_objeto:
                modelo['modelo_objeto'] = modelo_objeto
            # Eliminar la representación serializada para ahorrar memoria
            del modelo['modelo_serializado']
    
    # Reconstruir el mejor modelo
    mejor_modelo = resultados_reconstruidos.get('mejor_modelo', {})
    if 'modelo_serializado' in mejor_modelo and mejor_modelo.get('tiene_modelo_objeto', False):
        modelo_objeto = deserializar_modelo(mejor_modelo['modelo_serializado'])
        if modelo_objeto:
            mejor_modelo['modelo_objeto'] = modelo_objeto
        del mejor_modelo['modelo_serializado']
    
    return resultados_reconstruidos
