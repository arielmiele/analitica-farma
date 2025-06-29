"""
Módulo para manejar la serialización y deserialización de modelos de machine learning.
Proporciona funciones para convertir modelos de scikit-learn a formatos serializables
y viceversa, permitiendo su almacenamiento en JSON y bases de datos.
"""
import joblib
import base64
import io
from typing import Any, Dict, Optional
from src.audit.logger import log_audit


def serializar_modelo(modelo: Any, id_sesion: str, usuario: str) -> Optional[str]:
    """
    Serializa un modelo de scikit-learn a una cadena base64 que se puede almacenar en JSON.
    
    Args:
        modelo: El objeto modelo de scikit-learn a serializar
        id_sesion (str): ID de la sesión actual para trazabilidad
        usuario (str): Usuario que ejecuta la operación
    Returns:
        str: Representación base64 del modelo serializado, o None si falla
    """
    try:
        buffer = io.BytesIO()
        joblib.dump(modelo, buffer, compress=3)
        serialized_model = base64.b64encode(buffer.getvalue()).decode('utf-8')
        log_audit(
            id_sesion,
            usuario,
            "SERIALIZACION_MODELO_OK",
            "modelo_serializer",
            f"Modelo serializado correctamente ({len(serialized_model)} bytes)"
        )
        return serialized_model
    except Exception as e:
        log_audit(
            id_sesion,
            usuario,
            "ERROR_SERIALIZAR_MODELO",
            "modelo_serializer",
            f"Error al serializar modelo: {str(e)}"
        )
        return None


def deserializar_modelo(serialized_model: str, id_sesion: str, usuario: str) -> Optional[Any]:
    """
    Deserializa un modelo desde su representación base64.
    
    Args:
        serialized_model (str): Representación base64 del modelo serializado
        id_sesion (str): ID de la sesión actual para trazabilidad
        usuario (str): Usuario que ejecuta la operación
    Returns:
        Any: El modelo deserializado, o None si falla
    """
    try:
        buffer = io.BytesIO(base64.b64decode(serialized_model))
        modelo = joblib.load(buffer)
        log_audit(
            id_sesion,
            usuario,
            "DESERIALIZACION_MODELO_OK",
            "modelo_serializer",
            "Modelo deserializado correctamente"
        )
        return modelo
    except Exception as e:
        log_audit(
            id_sesion,
            usuario,
            "ERROR_DESERIALIZAR_MODELO",
            "modelo_serializer",
            f"Error al deserializar modelo: {str(e)}"
        )
        return None


def serializar_modelos_benchmarking(resultados: Dict, id_sesion: str, usuario: str) -> Dict:
    """
    Prepara los resultados del benchmarking para serialización JSON,
    convirtiendo modelos a formatos serializables.
    
    Args:
        resultados (Dict): Resultados del benchmarking con objetos modelo
        id_sesion (str): ID de la sesión actual para trazabilidad
        usuario (str): Usuario que ejecuta la operación
    Returns:
        Dict: Resultados preparados para serialización JSON
    """
    resultados_serializables = resultados.copy()
    for modelo in resultados_serializables.get('modelos_exitosos', []):
        if 'modelo_objeto' in modelo:
            modelo['modelo_serializado'] = serializar_modelo(modelo['modelo_objeto'], id_sesion, usuario)
            modelo['tiene_modelo_objeto'] = True
            del modelo['modelo_objeto']
    if (resultados_serializables.get('mejor_modelo') and 
        'modelo_objeto' in resultados_serializables['mejor_modelo']):
        mejor_modelo = resultados_serializables['mejor_modelo']
        mejor_modelo['modelo_serializado'] = serializar_modelo(mejor_modelo['modelo_objeto'], id_sesion, usuario)
        mejor_modelo['tiene_modelo_objeto'] = True
        del mejor_modelo['modelo_objeto']
    return resultados_serializables


def deserializar_modelos_benchmarking(resultados: Dict, id_sesion: str, usuario: str) -> Dict:
    """
    Reconstruye los objetos modelo a partir de sus representaciones serializadas
    en los resultados del benchmarking.
    
    Args:
        resultados (Dict): Resultados del benchmarking con modelos serializados
        id_sesion (str): ID de la sesión actual para trazabilidad
        usuario (str): Usuario que ejecuta la operación
    Returns:
        Dict: Resultados con objetos modelo reconstruidos
    """
    resultados_reconstruidos = resultados.copy()
    for modelo in resultados_reconstruidos.get('modelos_exitosos', []):
        if 'modelo_serializado' in modelo and modelo.get('tiene_modelo_objeto', False):
            modelo_objeto = deserializar_modelo(modelo['modelo_serializado'], id_sesion, usuario)
            if modelo_objeto:
                modelo['modelo_objeto'] = modelo_objeto
            del modelo['modelo_serializado']
    mejor_modelo = resultados_reconstruidos.get('mejor_modelo', {})
    if 'modelo_serializado' in mejor_modelo and mejor_modelo.get('tiene_modelo_objeto', False):
        modelo_objeto = deserializar_modelo(mejor_modelo['modelo_serializado'], id_sesion, usuario)
        if modelo_objeto:
            mejor_modelo['modelo_objeto'] = modelo_objeto
        del mejor_modelo['modelo_serializado']
    return resultados_reconstruidos
