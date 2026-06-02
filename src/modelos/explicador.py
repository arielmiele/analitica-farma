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
    MAX_ROWS_SHAP = 2000
    SAMPLE_SIZE = 1000

    try:
        log_audit(
            usuario=usuario,
            accion="INICIO_EXPLICACION_SHAP",
            entidad="explicador",
            id_entidad="N/A",
            detalles=f"Cálculo de importancias SHAP para modelo {type(modelo).__name__}",
            id_sesion=id_sesion
        )

        # Validar que X no tenga NaN
        if X.isnull().any().any():
            raise ValueError(
                "El dataset contiene valores NaN. Limpiá o imputá los valores faltantes antes de generar la explicación SHAP."
            )

        # Validar que X no tenga columnas no numéricas
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            raise ValueError(
                f"El dataset contiene columnas no numéricas: {non_numeric_cols}. "
                "Aplicá las transformaciones necesarias antes de explicar el modelo."
            )

        # Sampling para datasets grandes (SHAP es O(n²) en tiempo)
        X_shap = X
        sampled = False
        if len(X) > MAX_ROWS_SHAP:
            X_shap = X.sample(n=SAMPLE_SIZE, random_state=42)
            sampled = True
            log_audit(
                usuario=usuario,
                accion="SHAP_SAMPLING_APLICADO",
                entidad="explicador",
                id_entidad="N/A",
                detalles=f"Dataset reducido de {len(X)} a {SAMPLE_SIZE} filas para cálculo SHAP",
                id_sesion=id_sesion
            )

        # Selección de explainer según tipo de modelo
        if hasattr(modelo, 'predict_proba'):
            explainer = shap.Explainer(modelo, X_shap)
        else:
            explainer = shap.Explainer(modelo.predict, X_shap)

        shap_values = explainer(X_shap)
        # Importancia media absoluta
        importancias = pd.DataFrame({
            'feature': X_shap.columns,
            'importance': np.abs(shap_values.values).mean(axis=0)
        }).sort_values('importance', ascending=False).head(max_display)

        log_audit(
            usuario=usuario,
            accion="EXPLICACION_SHAP_EXITOSA",
            entidad="explicador",
            id_entidad="N/A",
            detalles=f"Importancias SHAP calculadas correctamente para modelo {type(modelo).__name__}"
                     + (f" (muestra de {SAMPLE_SIZE} filas)" if sampled else ""),
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
