"""
Módulo para la recomendación del mejor modelo de machine learning.
Incluye funciones para generar recomendaciones específicas basadas en diagnósticos.
"""
from datetime import datetime
from typing import Dict, List
from src.audit.logger import log_audit

def recomendar_mejor_modelo(
    benchmarking: Dict,
    criterio: str,
    id_sesion: str,
    usuario: str
) -> Dict:
    """
    Recomienda el mejor modelo basado en los resultados del benchmarking.
    
    Args:
        benchmarking (Dict): Resultados del benchmarking con modelos exitosos
        criterio (str): Criterio para seleccionar el mejor modelo ('accuracy', 'f1', 'r2', 'rmse', 'auto')
        id_sesion (str): ID de la sesión actual para trazabilidad
        usuario (str): Usuario que ejecuta la operación
    Returns:
        Dict: Información del modelo recomendado
    """
    try:
        if not benchmarking['modelos_exitosos']:
            return {"error": "No hay modelos exitosos en el benchmarking."}
        # Si el criterio es 'auto', usar el criterio adecuado según el tipo de problema
        if criterio == 'auto':
            if benchmarking['tipo_problema'] == 'clasificacion':
                criterio = 'accuracy'
            else:
                criterio = 'r2'
        # Ordenar modelos según criterio seleccionado
        if criterio in ['accuracy', 'precision', 'recall', 'f1', 'r2']:
            modelos_ordenados = sorted(
                benchmarking['modelos_exitosos'],
                key=lambda x: x['metricas'].get(criterio, 0),
                reverse=True
            )
        else:  # 'mse', 'rmse', 'mae'
            modelos_ordenados = sorted(
                benchmarking['modelos_exitosos'],
                key=lambda x: x['metricas'].get(criterio, float('inf'))
            )
        modelo_recomendado = modelos_ordenados[0]
        log_audit(
            id_sesion,
            usuario,
            "RECOMENDACION_MODELO",
            "recomendador",
            f"Recomendación de modelo con criterio: {criterio}"
        )
        return {
            "modelo_recomendado": modelo_recomendado,
            "criterio_usado": criterio,
            "tipo_problema": benchmarking['tipo_problema'],
            "variable_objetivo": benchmarking['variable_objetivo'],
            "total_modelos_evaluados": len(benchmarking['modelos_exitosos']),
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    except Exception as e:
        log_audit(
            id_sesion,
            usuario,
            "ERROR_RECOMENDAR_MODELO",
            "recomendador",
            f"Error al recomendar modelo: {str(e)}"
        )
        return {"error": str(e)}

def guardar_modelo_seleccionado(
    nombre_modelo: str,
    benchmarking: Dict,
    comentarios: str,
    id_sesion: str,
    usuario: str
) -> Dict:
    """
    Guarda el modelo seleccionado por el usuario (solo auditoría, no base de datos).
    
    Args:
        nombre_modelo (str): Nombre del modelo seleccionado
        benchmarking (Dict): Resultados del benchmarking
        comentarios (str): Comentarios sobre la selección
        id_sesion (str): ID de la sesión actual para trazabilidad
        usuario (str): Usuario que ejecuta la operación
    Returns:
        Dict: Resultado de la operación
    """
    try:
        log_audit(
            id_sesion,
            usuario,
            "SELECCION_MODELO",
            "recomendador",
            f"Modelo seleccionado: {nombre_modelo}. Comentarios: {comentarios}"
        )
        return {
            "exito": True,
            "mensaje": f"Modelo '{nombre_modelo}' seleccionado correctamente.",
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    except Exception as e:
        log_audit(
            id_sesion,
            usuario,
            "ERROR_GUARDAR_MODELO",
            "recomendador",
            f"Error al guardar modelo seleccionado: {str(e)}"
        )
        return {
            "exito": False,
            "error": str(e)
        }

def generar_recomendaciones_diagnostico(tipo_diag: str) -> List[str]:
    """
    Genera recomendaciones genéricas basadas en el tipo de diagnóstico.
    
    Args:
        tipo_diag: Tipo de diagnóstico ('overfitting', 'underfitting', 'balanceado')
        
    Returns:
        List[str]: Lista de recomendaciones
    """
    if tipo_diag == 'overfitting':
        return [
            "🔄 Considere usar regularización (L1/L2)",
            "📊 Aumente el tamaño del dataset de entrenamiento", 
            "🌳 Reduzca la complejidad del modelo",
            "✂️ Aplique técnicas de feature selection",
            "📈 Implemente early stopping durante el entrenamiento",
            "🔀 Use técnicas de data augmentation si es apropiado",
            "👥 Considere ensemble methods para reducir varianza"
        ]
    elif tipo_diag == 'underfitting':
        return [
            "🔧 Aumente la complejidad del modelo",
            "🎯 Agregue más características relevantes",
            "⚙️ Ajuste los hiperparámetros", 
            "🔍 Verifique la calidad de los datos",
            "📊 Considere feature engineering más sofisticado",
            "🧮 Pruebe modelos más complejos (ensemble, neural networks)",
            "📈 Aumente el número de iteraciones de entrenamiento"
        ]
    else:
        return [
            "✅ El modelo muestra un comportamiento balanceado",
            "🔍 Considere realizar ajuste fino de hiperparámetros",
            "📈 Monitoree el rendimiento en producción",
            "🎯 Evalúe la adición de características adicionales",
            "🔧 Implemente validación A/B en producción",
            "📊 Configure alertas para model drift",
            "🎓 Prepare documentación para transferencia a producción"
        ]

def generar_recomendaciones_industria(tipo_problema: str) -> Dict[str, List[str]]:
    """
    Genera recomendaciones específicas para la industria farmacéutica.
    
    Args:
        tipo_problema: Tipo de problema/diagnóstico
        
    Returns:
        Dict: Recomendaciones categorizadas para la industria
    """
    return {
        "overfitting": [
            "📋 Documentación: Registre todas las correcciones en el batch record",
            "🔄 Validación cruzada: Implemente validación con datos de múltiples lotes",
            "📊 Monitoreo continuo: Establezca alertas para drift del modelo en producción",
            "👥 Revisión por pares: Involucre a QA en la validación del modelo",
            "🔬 Compliance: Asegúrese de cumplir con regulaciones FDA/EMA",
            "📈 Trazabilidad: Mantenga histórico completo de cambios de modelo"
        ],
        "underfitting": [
            "🔬 Revisión de variables: Inclua más CPPs (Critical Process Parameters)",
            "📈 Aumento de datos: Considere datos históricos de sitios similares",
            "🧪 Experimentos dirigidos: Planifique experimentos para llenar gaps de datos",
            "🎯 Refinamiento de objetivos: Revise si los KPIs están bien definidos",
            "📊 Análisis de riesgo: Evalúe impacto de predicciones incorrectas",
            "🔧 Calibración: Implemente procedimientos de calibración de equipos"
        ],
        "balanceado": [
            "✅ Validación final: Proceda con validación en lotes piloto",
            "📝 Documentación GMP: Prepare documentación para transferencia",
            "🔍 Monitoreo de performance: Implemente sistema de seguimiento continuo",
            "🎓 Entrenamiento: Capacite al personal en el uso del modelo",
            "📋 SOPs: Desarrolle procedimientos estándar de operación",
            "🔄 Mantenimiento: Establezca rutinas de mantenimiento del modelo"
        ],
        "general": [
            "🏭 Validación de proceso: Asegure que el modelo respalde controles de proceso",
            "📊 Reportes regulatorios: Configure generación automática de reportes",
            "🔒 Seguridad de datos: Implemente controles de acceso robustos",
            "🎯 ROI: Monitoree retorno de inversión del modelo implementado"
        ]
    }

def generar_recomendaciones_completas(diagnostico: Dict, modelo: Dict, tipo_problema: str, id_sesion: str, usuario: str) -> Dict:
    """
    Genera un conjunto completo de recomendaciones basadas en el diagnóstico del modelo.
    
    Args:
        diagnostico: Diccionario con información del diagnóstico
        modelo: Información del modelo
        tipo_problema: Tipo de problema ('clasificacion', 'regresion')
        id_sesion (str): ID de la sesión actual para trazabilidad
        usuario (str): Usuario que ejecuta la operación
    Returns:
        Dict: Recomendaciones completas organizadas por categoría
    """
    try:
        overfitting = diagnostico.get('overfitting', 'desconocido')
        underfitting = diagnostico.get('underfitting', 'desconocido')
        if overfitting == 'posible':
            tipo_diag = 'overfitting'
        elif underfitting == 'posible':
            tipo_diag = 'underfitting'
        else:
            tipo_diag = 'balanceado'
        recomendaciones_del_analisis = modelo.get('recomendaciones', [])
        recomendaciones_diagnostico = diagnostico.get('recomendaciones', [])
        recomendaciones_genericas = generar_recomendaciones_diagnostico(tipo_diag)
        recomendaciones_industria = generar_recomendaciones_industria(tipo_diag)
        log_audit(
            id_sesion,
            usuario,
            "GENERAR_RECOMENDACIONES",
            "recomendador",
            f"Recomendaciones generadas para tipo_diagnostico: {tipo_diag}"
        )
        return {
            "tipo_diagnostico": tipo_diag,
            "recomendaciones_especificas": recomendaciones_del_analisis + recomendaciones_diagnostico,
            "recomendaciones_genericas": recomendaciones_genericas,
            "recomendaciones_industria": recomendaciones_industria,
            "contexto": {
                "nombre_modelo": modelo.get('nombre', 'Modelo'),
                "tipo_problema": tipo_problema,
                "overfitting": overfitting,
                "underfitting": underfitting
            }
        }
    except Exception as e:
        log_audit(
            id_sesion,
            usuario,
            "ERROR_GENERAR_RECOMENDACIONES",
            "recomendador",
            f"Error al generar recomendaciones: {str(e)}"
        )
        return {
            "error": f"Error al generar recomendaciones: {str(e)}",
            "tipo_diagnostico": "error",
            "recomendaciones_especificas": [],
            "recomendaciones_genericas": [],
            "recomendaciones_industria": {},
            "contexto": {}
        }
