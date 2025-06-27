"""
Módulo para la recomendación del mejor modelo de machine learning.
Incluye funciones para generar recomendaciones específicas basadas en diagnósticos.
"""
import json
import os
import sqlite3
import logging
from datetime import datetime
from typing import Dict, Optional, List

# Configuración del logger
logger = logging.getLogger("recomendador")

def recomendar_mejor_modelo(
    id_benchmarking: Optional[int] = None,
    criterio: str = "auto",
    id_usuario: int = 1,
    db_path: Optional[str] = None
) -> Dict:
    """
    Recomienda el mejor modelo basado en los resultados del benchmarking.
    
    Args:
        id_benchmarking: ID del benchmarking (si es None, se usa el último)
        criterio: Criterio para seleccionar el mejor modelo ('accuracy', 'f1', 'r2', 'rmse', 'auto')
        id_usuario: ID del usuario
        db_path: Ruta a la base de datos
        
    Returns:
        Dict: Información del modelo recomendado
    """
    conn = None
    try:
        # Determinar la ruta de la base de datos
        if db_path is None:
            db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'analitica_farma.db')
        
        # Conectar a la base de datos
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Obtener el benchmarking
        if id_benchmarking:
            query = "SELECT * FROM benchmarking_modelos WHERE id = ?"
            params = (id_benchmarking,)
        else:
            query = "SELECT * FROM benchmarking_modelos WHERE id_usuario = ? ORDER BY fecha_ejecucion DESC LIMIT 1"
            params = (id_usuario,)
        
        cursor.execute(query, params)
        resultado = cursor.fetchone()
        
        if not resultado:
            return {"error": "No se encontró ningún benchmarking."}
        
        # Cargar resultados
        benchmarking = json.loads(resultado['resultados_completos'])
        
        if not benchmarking['modelos_exitosos']:
            return {"error": "No hay modelos exitosos en el benchmarking."}
        
        # Registrar en la tabla de auditoría
        cursor.execute("""
            INSERT INTO auditoria (
                id_usuario, accion, descripcion, fecha
            ) VALUES (?, ?, ?, ?)
        """, (
            id_usuario,
            "RECOMENDACION_MODELO",
            f"Recomendación de modelo con criterio: {criterio}",
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ))
        
        # Confirmar cambios
        conn.commit()
        
        # Si el criterio es 'auto', usar el criterio adecuado según el tipo de problema
        if criterio == 'auto':
            if benchmarking['tipo_problema'] == 'clasificacion':
                criterio = 'accuracy'
            else:
                criterio = 'r2'
        
        # Ordenar modelos según criterio seleccionado
        if criterio in ['accuracy', 'precision', 'recall', 'f1', 'r2']:
            # Para estos criterios, mayor es mejor
            modelos_ordenados = sorted(
                benchmarking['modelos_exitosos'],
                key=lambda x: x['metricas'].get(criterio, 0),
                reverse=True
            )
        else:  # 'mse', 'rmse', 'mae'
            # Para estos criterios, menor es mejor
            modelos_ordenados = sorted(
                benchmarking['modelos_exitosos'],
                key=lambda x: x['metricas'].get(criterio, float('inf'))
            )
        
        # Modelo recomendado es el primero después de ordenar
        modelo_recomendado = modelos_ordenados[0]
        
        return {
            "modelo_recomendado": modelo_recomendado,
            "criterio_usado": criterio,
            "tipo_problema": benchmarking['tipo_problema'],
            "variable_objetivo": benchmarking['variable_objetivo'],
            "total_modelos_evaluados": len(benchmarking['modelos_exitosos']),
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    except Exception as e:
        logger.error(f"Error al recomendar modelo: {str(e)}")
        if conn:
            conn.rollback()
        return {"error": str(e)}
    finally:
        if conn:
            conn.close()

def guardar_modelo_seleccionado(
    nombre_modelo: str,
    id_benchmarking: Optional[int] = None,
    comentarios: str = "",
    id_usuario: int = 1,
    db_path: Optional[str] = None
) -> Dict:
    """
    Guarda el modelo seleccionado por el usuario.
    
    Args:
        nombre_modelo: Nombre del modelo seleccionado
        id_benchmarking: ID del benchmarking
        comentarios: Comentarios sobre la selección
        id_usuario: ID del usuario
        db_path: Ruta a la base de datos
        
    Returns:
        Dict: Resultado de la operación
    """
    conn = None
    try:
        # Determinar la ruta de la base de datos
        if db_path is None:
            db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'analitica_farma.db')
        
        # Conectar a la base de datos
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Verificar si existe la tabla
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS modelos_seleccionados (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                id_usuario INTEGER NOT NULL,
                id_benchmarking INTEGER,
                nombre_modelo TEXT NOT NULL,
                comentarios TEXT,
                fecha_seleccion TIMESTAMP NOT NULL
            )
        """)
        
        # Obtener el ID del benchmarking si no se proporcionó
        if not id_benchmarking:
            cursor.execute(
                "SELECT id FROM benchmarking_modelos WHERE id_usuario = ? ORDER BY fecha_ejecucion DESC LIMIT 1",
                (id_usuario,)
            )
            resultado = cursor.fetchone()
            if resultado:
                id_benchmarking = resultado[0]
        
        # Insertar el modelo seleccionado
        cursor.execute("""
            INSERT INTO modelos_seleccionados (
                id_usuario, id_benchmarking, nombre_modelo, comentarios, fecha_seleccion
            ) VALUES (?, ?, ?, ?, ?)
        """, (
            id_usuario,
            id_benchmarking,
            nombre_modelo,
            comentarios,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ))
        
        # Registrar en la tabla de auditoría
        cursor.execute("""
            INSERT INTO auditoria (
                id_usuario, accion, descripcion, fecha
            ) VALUES (?, ?, ?, ?)
        """, (
            id_usuario,
            "SELECCION_MODELO",
            f"Modelo seleccionado: {nombre_modelo}",
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ))
        
        # Confirmar cambios
        conn.commit()
        
        return {
            "exito": True,
            "mensaje": f"Modelo '{nombre_modelo}' seleccionado correctamente.",
            "id_benchmarking": id_benchmarking,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    except Exception as e:
        logger.error(f"Error al guardar modelo seleccionado: {str(e)}")
        if conn:
            conn.rollback()
        return {
            "exito": False,
            "error": str(e)
        }
    finally:
        if conn:
            conn.close()

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
            "🔬 Revisión de variables: Incluya más CPPs (Critical Process Parameters)",
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

def generar_recomendaciones_completas(diagnostico: Dict, modelo: Dict, tipo_problema: str) -> Dict:
    """
    Genera un conjunto completo de recomendaciones basadas en el diagnóstico del modelo.
    
    Args:
        diagnostico: Diccionario con información del diagnóstico
        modelo: Información del modelo
        tipo_problema: Tipo de problema ('clasificacion', 'regresion')
        
    Returns:
        Dict: Recomendaciones completas organizadas por categoría
    """
    try:
        # Determinar tipo de diagnóstico
        overfitting = diagnostico.get('overfitting', 'desconocido')
        underfitting = diagnostico.get('underfitting', 'desconocido')
        
        if overfitting == 'posible':
            tipo_diag = 'overfitting'
        elif underfitting == 'posible':
            tipo_diag = 'underfitting'
        else:
            tipo_diag = 'balanceado'
        
        # Obtener recomendaciones del análisis si están disponibles
        recomendaciones_del_analisis = modelo.get('recomendaciones', [])
        recomendaciones_diagnostico = diagnostico.get('recomendaciones', [])
        
        # Generar recomendaciones específicas
        recomendaciones_genericas = generar_recomendaciones_diagnostico(tipo_diag)
        recomendaciones_industria = generar_recomendaciones_industria(tipo_diag)
        
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
        logger.error(f"Error generando recomendaciones: {str(e)}")
        return {
            "error": f"Error al generar recomendaciones: {str(e)}",
            "tipo_diagnostico": "error",
            "recomendaciones_especificas": [],
            "recomendaciones_genericas": [],
            "recomendaciones_industria": {},
            "contexto": {}
        }
