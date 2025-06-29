"""
Módulo de explicación de modelos usando SHAP.
Todas las funciones aquí son independientes de la UI y cumplen con el patrón Model-View.
"""
import shap
import numpy as np
import pandas as pd
from src.audit.logger import log_audit

def obtener_importancias_shap(modelo, X, id_sesion, usuario, max_display=15):
    """
    Calcula los valores SHAP y la importancia de cada feature para el modelo dado.
    Args:
        modelo: Modelo entrenado (scikit-learn compatible)
        X: DataFrame de features
        id_sesion (str): ID de sesión para trazabilidad
        usuario (str): Usuario que ejecuta la acción
        max_display: Máximo de features a mostrar
    Returns:
        importancias: DataFrame con importancia media de cada feature
        shap_values: Valores SHAP calculados
    """
    try:
        log_audit(
            usuario=usuario,
            accion="INICIO_EXPLICACION_SHAP",
            entidad="explicador",
            id_entidad="N/A",
            detalles=f"Cálculo de importancias SHAP para modelo {type(modelo).__name__}",
            id_sesion=id_sesion
        )
        # Selección de explainer según tipo de modelo
        if hasattr(modelo, 'predict_proba'):
            explainer = shap.Explainer(modelo, X)
        else:
            explainer = shap.Explainer(modelo.predict, X)

        shap_values = explainer(X)
        # Importancia media absoluta
        importancias = pd.DataFrame({
            'feature': X.columns,
            'importance': np.abs(shap_values.values).mean(axis=0)
        }).sort_values('importance', ascending=False).head(max_display)
        
        log_audit(
            usuario=usuario,
            accion="EXPLICACION_SHAP_EXITOSA",
            entidad="explicador",
            id_entidad="N/A",
            detalles=f"Importancias SHAP calculadas correctamente para modelo {type(modelo).__name__}",
            id_sesion=id_sesion
        )
        return importancias, shap_values
    except Exception as e:
        log_audit(
            usuario=usuario,
            accion="ERROR_EXPLICACION_SHAP",
            entidad="explicador",
            id_entidad="N/A",
            detalles=f"Error al calcular importancias SHAP: {str(e)}",
            id_sesion=id_sesion
        )
        raise
