"""
Módulo de Validación Cruzada - Lógica de Negocio
Contiene toda la lógica de cálculos, análisis y procesamiento de validación cruzada.
Separado de la UI para mantener la arquitectura Model-View.
"""

import numpy as np
from datetime import datetime
from sklearn.model_selection import learning_curve, cross_val_score, StratifiedKFold, KFold
from src.audit.logger import log_audit

def verificar_datos_para_validacion(resultados_benchmarking: dict, id_sesion: str, usuario: str) -> dict:
    """
    Verifica si los datos necesarios están disponibles para la validación cruzada.
    
    Args:
        resultados_benchmarking: Diccionario con resultados del benchmarking
        id_sesion (str): ID de la sesión actual para trazabilidad
        usuario (str): Usuario que ejecuta la operación
    Returns:
        dict: Información sobre disponibilidad de datos
    """
    campos_requeridos = ['tipo_problema', 'variable_objetivo', 'total_filas']
    campos_faltantes = [campo for campo in campos_requeridos if campo not in resultados_benchmarking]
    if campos_faltantes:
        log_audit(
            id_sesion,
            usuario,
            "DATOS_INCOMPLETOS_VALIDACION",
            "validacion_cruzada",
            f"Faltan campos: {', '.join(campos_faltantes)}"
        )
        return {
            'datos_ok': False,
            'mensaje': f"Información incompleta del benchmarking: faltan {', '.join(campos_faltantes)}",
            'solucion': "Ejecute un nuevo benchmarking para obtener toda la información necesaria"
        }
    if 'X_test' not in resultados_benchmarking or 'y_test' not in resultados_benchmarking:
        log_audit(
            id_sesion,
            usuario,
            "DATOS_TEST_NO_DISPONIBLES",
            "validacion_cruzada",
            "Faltan datos de prueba para validación cruzada"
        )
        return {
            'datos_ok': False,
            'mensaje': "Datos de prueba no disponibles",
            'solucion': "Los datos de prueba son necesarios para las visualizaciones. Ejecute un nuevo benchmarking"
        }
    if not resultados_benchmarking.get('modelos_exitosos'):
        log_audit(
            id_sesion,
            usuario,
            "NO_MODELOS_EXITOSOS",
            "validacion_cruzada",
            "No hay modelos exitosos disponibles para validación cruzada"
        )
        return {
            'datos_ok': False,
            'mensaje': "No hay modelos exitosos disponibles",
            'solucion': "Ejecute un benchmarking que produzca al menos un modelo válido"
        }
    log_audit(
        id_sesion,
        usuario,
        "DATOS_VALIDACION_OK",
        "validacion_cruzada",
        "Todos los datos necesarios están disponibles para validación cruzada"
    )
    return {
        'datos_ok': True,
        'mensaje': "Todos los datos necesarios están disponibles",
        'solucion': ""
    }

def ejecutar_validacion_cruzada_completa(modelo, X, y, tipo_problema, id_sesion: str, usuario: str, cv_folds=5) -> dict:
    """
    Ejecuta validación cruzada completa con estadísticas detalladas.
    
    Args:
        modelo: Objeto modelo de scikit-learn
        X: Features
        y: Variable objetivo
        tipo_problema: 'clasificacion' o 'regresion'
        id_sesion (str): ID de la sesión actual para trazabilidad
        usuario (str): Usuario que ejecuta la operación
        cv_folds: Número de folds
    Returns:
        dict: Resultados completos de validación cruzada
    """
    try:
        if tipo_problema == 'clasificacion':
            cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scoring = 'accuracy'
        else:
            cv_strategy = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scoring = 'r2'
        cv_scores = cross_val_score(
            modelo, X, y,
            cv=cv_strategy,
            scoring=scoring,
            n_jobs=-1
        )
        log_audit(
            id_sesion,
            usuario,
            "VALIDACION_CRUZADA_OK",
            "validacion_cruzada",
            f"Validación cruzada ejecutada correctamente. Media: {np.mean(cv_scores):.3f}"
        )
        return {
            'cv_scores': cv_scores.tolist(),
            'mean_score': float(np.mean(cv_scores)),
            'std_score': float(np.std(cv_scores)),
            'min_score': float(np.min(cv_scores)),
            'max_score': float(np.max(cv_scores)),
            'variance': float(np.var(cv_scores)),
            'cv_folds': cv_folds,
            'scoring_metric': scoring
        }
    except Exception as e:
        log_audit(
            id_sesion,
            usuario,
            "ERROR_VALIDACION_CRUZADA",
            "validacion_cruzada",
            f"Error en validación cruzada: {str(e)}"
        )
        return {
            'error': f'Error en validación cruzada: {str(e)}',
            'solucion': 'Verifique que el modelo sea compatible con los datos proporcionados'
        }

def generar_curvas_aprendizaje_reales(modelo, X, y, tipo_problema, id_sesion: str, usuario: str) -> dict:
    """
    Genera curvas de aprendizaje reales usando learning_curve de scikit-learn.
    
    Args:
        modelo: Objeto modelo de scikit-learn
        X: Features
        y: Variable objetivo
        tipo_problema: 'clasificacion' o 'regresion'
        id_sesion (str): ID de la sesión actual para trazabilidad
        usuario (str): Usuario que ejecuta la operación
    Returns:
        dict: Resultados de curvas de aprendizaje
    """
    try:
        scoring = 'accuracy' if tipo_problema == 'clasificacion' else 'r2'
        train_sizes = np.linspace(0.1, 1.0, 10)
        try:
            result = learning_curve(
                modelo, X, y,
                train_sizes=train_sizes,
                cv=5,
                scoring=scoring,
                n_jobs=-1,
                random_state=42
            )
            train_sizes_abs = result[0]
            train_scores = result[1] 
            validation_scores = result[2]
        except Exception as learning_error:
            log_audit(
                id_sesion,
                usuario,
                "ERROR_CURVAS_APRENDIZAJE",
                "validacion_cruzada",
                f"Error en curvas de aprendizaje: {str(learning_error)}"
            )
            return {
                'error': f'Error en curvas de aprendizaje: {str(learning_error)}',
                'solucion': 'Se usará validación cruzada simple en su lugar'
            }
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        validation_scores_mean = np.mean(validation_scores, axis=1)
        validation_scores_std = np.std(validation_scores, axis=1)
        overfitting_gap = train_scores_mean - validation_scores_mean
        log_audit(
            id_sesion,
            usuario,
            "CURVAS_APRENDIZAJE_OK",
            "validacion_cruzada",
            f"Curvas de aprendizaje generadas correctamente. Gap final: {float(overfitting_gap[-1]):.3f}"
        )
        return {
            'train_sizes': train_sizes_abs.tolist(),
            'train_scores_mean': train_scores_mean.tolist(),
            'train_scores_std': train_scores_std.tolist(),
            'validation_scores_mean': validation_scores_mean.tolist(),
            'validation_scores_std': validation_scores_std.tolist(),
            'overfitting_gap': overfitting_gap.tolist(),
            'final_gap': float(overfitting_gap[-1]),
            'max_gap': float(np.max(overfitting_gap)),
            'gap_trend': 'creciente' if overfitting_gap[-1] > overfitting_gap[0] else 'decreciente',
            'scoring_metric': scoring
        }
    except Exception as e:
        log_audit(
            id_sesion,
            usuario,
            "ERROR_GENERAR_CURVAS_APRENDIZAJE",
            "validacion_cruzada",
            f"Error generando curvas de aprendizaje: {str(e)}"
        )
        return {
            'error': f'Error generando curvas de aprendizaje: {str(e)}',
            'solucion': 'Verifique que haya suficientes datos para generar curvas de aprendizaje'
        }

def generar_diagnostico_avanzado(cv_results, learning_results, metricas_originales, tipo_problema, id_sesion: str, usuario: str) -> dict:
    """
    Genera diagnóstico avanzado basado en validación cruzada y curvas de aprendizaje.
    
    Args:
        cv_results: Resultados de validación cruzada
        learning_results: Resultados de curvas de aprendizaje
        metricas_originales: Métricas del benchmarking original
        tipo_problema: Tipo de problema
        id_sesion (str): ID de la sesión actual para trazabilidad
        usuario (str): Usuario que ejecuta la operación
    Returns:
        dict: Diagnóstico completo
    """
    diagnostico = {
        'overfitting': 'desconocido',
        'underfitting': 'desconocido',
        'varianza_cv': cv_results.get('std_score', 0),
        'mensaje': '',
        'nivel_confianza': 'alto',
        'detalles': {}
    }
    cv_std = cv_results.get('std_score', 0)
    cv_mean = cv_results.get('mean_score', 0)
    final_gap = learning_results.get('final_gap', 0)
    max_gap = learning_results.get('max_gap', 0)
    gap_trend = learning_results.get('gap_trend', 'estable')
    if final_gap > 0.1 or max_gap > 0.15:
        diagnostico['overfitting'] = 'posible'
        diagnostico['mensaje'] = f'Brecha significativa entre entrenamiento y validación (gap final: {final_gap:.3f})'
    elif final_gap < 0.03:
        diagnostico['overfitting'] = 'improbable'
        diagnostico['mensaje'] = f'Brecha mínima entre entrenamiento y validación (gap final: {final_gap:.3f})'
    else:
        diagnostico['overfitting'] = 'normal'
        diagnostico['mensaje'] = f'Brecha moderada entre entrenamiento y validación (gap final: {final_gap:.3f})'
    threshold = 0.7 if tipo_problema == 'clasificacion' else 0.5
    if cv_mean < threshold:
        diagnostico['underfitting'] = 'posible'
        diagnostico['mensaje'] += f' Rendimiento bajo (μ={cv_mean:.3f})'
    else:
        diagnostico['underfitting'] = 'improbable'
    if cv_std > 0.1:
        diagnostico['mensaje'] += f' Alta varianza en CV (σ={cv_std:.3f})'
    diagnostico['detalles'] = {
        'gap_final': final_gap,
        'gap_maximo': max_gap,
        'tendencia_gap': gap_trend,
        'cv_media': cv_mean,
        'cv_std': cv_std,
        'scoring_metric': cv_results.get('scoring_metric', 'N/A')
    }
    log_audit(
        id_sesion,
        usuario,
        "DIAGNOSTICO_AVANZADO_OK",
        "validacion_cruzada",
        f"Diagnóstico avanzado generado. Overfitting: {diagnostico['overfitting']}, Underfitting: {diagnostico['underfitting']}"
    )
    return diagnostico

def generar_recomendaciones_avanzadas(diagnostico, cv_results, learning_results, id_sesion: str, usuario: str) -> list:
    """
    Genera recomendaciones específicas basadas en diagnóstico y resultados.
    
    Args:
        diagnostico: Diagnóstico del modelo
        cv_results: Resultados de validación cruzada
        learning_results: Resultados de curvas de aprendizaje
        id_sesion (str): ID de la sesión actual para trazabilidad
        usuario (str): Usuario que ejecuta la operación
    Returns:
        list: Lista de recomendaciones específicas
    """
    recomendaciones = []
    overfitting = diagnostico.get('overfitting', 'desconocido')
    final_gap = learning_results.get('final_gap', 0)
    if overfitting == 'posible':
        if final_gap > 0.15:
            recomendaciones.extend([
                "Aplicar regularización L1 o L2 más agresiva",
                "Reducir la complejidad del modelo (menos capas/parámetros)",
                "Implementar early stopping con paciencia reducida",
                "Aumentar significativamente el tamaño del dataset"
            ])
        else:
            recomendaciones.extend([
                "Aplicar regularización moderada (L2, dropout)",
                "Usar validación cruzada con más folds",
                "Considerar ensemble methods para estabilizar"
            ])
    underfitting = diagnostico.get('underfitting', 'desconocido')
    if underfitting == 'posible':
        recomendaciones.extend([
            "Aumentar la complejidad del modelo",
            "Añadir más features o realizar feature engineering",
            "Reducir la regularización si está aplicada",
            "Entrenar por más épocas/iteraciones"
        ])
    cv_std = cv_results.get('std_score', 0)
    if cv_std > 0.1:
        recomendaciones.extend([
            "El modelo es inconsistente - considerar ensemble methods",
            "Aumentar el número de folds en validación cruzada",
            "Verificar calidad y balance del dataset"
        ])
    if overfitting == 'improbable' and underfitting == 'improbable' and cv_std < 0.05:
        recomendaciones.extend([
            "Modelo con buen balance - considerar optimización de hiperparámetros",
            "Evaluar en conjunto de datos completamente independiente",
            "Preparar para deployment con monitoreo de drift"
        ])
    log_audit(
        id_sesion,
        usuario,
        "RECOMENDACIONES_AVANZADAS_OK",
        "validacion_cruzada",
        f"Recomendaciones avanzadas generadas. Total: {len(recomendaciones)}"
    )
    return recomendaciones

def generar_analisis_completo_validacion_cruzada(modelo, resultados_benchmarking, id_sesion: str, usuario: str) -> dict:
    """
    Función principal que genera el análisis completo de validación cruzada.
    
    Args:
        modelo: Información del modelo incluyendo objeto serializado
        resultados_benchmarking: Resultados del benchmarking
        id_sesion (str): ID de la sesión actual para trazabilidad
        usuario (str): Usuario que ejecuta la operación
    Returns:
        dict: Análisis completo con curvas, diagnóstico y recomendaciones
    """
    try:
        modelo_objeto = modelo.get('modelo_objeto')
        if modelo_objeto is None:
            log_audit(
                id_sesion,
                usuario,
                "ERROR_OBJETO_MODELO",
                "validacion_cruzada",
                "Objeto del modelo no disponible para validación cruzada"
            )
            return {
                'error': 'Objeto del modelo no disponible para validación cruzada',
                'solucion': 'Ejecute un nuevo benchmarking que incluya modelos serializados'
            }
        X_test = np.array(resultados_benchmarking['X_test'])
        y_test = np.array(resultados_benchmarking['y_test'])
        X_total = X_test
        y_total = y_test
        tipo_problema = resultados_benchmarking.get('tipo_problema', 'clasificacion')
        cv_results = ejecutar_validacion_cruzada_completa(
            modelo_objeto, X_total, y_total, tipo_problema, id_sesion, usuario
        )
        if 'error' in cv_results:
            return cv_results
        learning_results = generar_curvas_aprendizaje_reales(
            modelo_objeto, X_total, y_total, tipo_problema, id_sesion, usuario
        )
        if 'error' in learning_results:
            return learning_results
        metricas_originales = modelo.get('metricas', {})
        diagnostico = generar_diagnostico_avanzado(
            cv_results, learning_results, metricas_originales, tipo_problema, id_sesion, usuario
        )
        recomendaciones = generar_recomendaciones_avanzadas(
            diagnostico, cv_results, learning_results, id_sesion, usuario
        )
        resultados = {
            'modelo': modelo['nombre'],
            'tipo_problema': tipo_problema,
            'metricas_principales': metricas_originales,
            'cv_scores': cv_results['cv_scores'],
            'cv_results_completos': cv_results,
            'learning_curves': learning_results,
            'diagnostico': diagnostico,
            'recomendaciones': recomendaciones,
            'datos_disponibles': {
                'X_shape': str(X_total.shape),
                'y_shape': str(y_total.shape),
                'total_filas': resultados_benchmarking.get('total_filas', X_total.shape[0]),
                'porcentaje_test': resultados_benchmarking.get('porcentaje_test', 'N/A'),
                'features': X_total.shape[1] if len(X_total.shape) > 1 else 1
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        log_audit(
            id_sesion,
            usuario,
            "ANALISIS_VALIDACION_CRUZADA_OK",
            "validacion_cruzada",
            f"Análisis completo de validación cruzada generado para modelo: {modelo['nombre']}"
        )
        return resultados
    except Exception as e:
        log_audit(
            id_sesion,
            usuario,
            "ERROR_VALIDACION_CRUZADA",
            "validacion_cruzada",
            f"Error en análisis de validación cruzada: {str(e)}"
        )
        return {
            'error': f'Error generando análisis de validación cruzada: {str(e)}',
            'solucion': 'Verifique que el modelo y los datos estén disponibles correctamente'
        }
