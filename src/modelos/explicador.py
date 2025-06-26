"""
Módulo de explicación de modelos usando SHAP.
Todas las funciones aquí son independientes de la UI y cumplen con el patrón Model-View.
"""
import shap
import numpy as np
import pandas as pd

def obtener_importancias_shap(modelo, X, max_display=15):
    """
    Calcula los valores SHAP y la importancia de cada feature para el modelo dado.
    Args:
        modelo: Modelo entrenado (scikit-learn compatible)
        X: DataFrame de features
        max_display: Máximo de features a mostrar
    Returns:
        importancias: DataFrame con importancia media de cada feature
        shap_values: Valores SHAP calculados
    """
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
    return importancias, shap_values
