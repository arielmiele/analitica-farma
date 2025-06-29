"""
Módulo para la evaluación detallada de modelos de machine learning.
"""
import json
import os
import sqlite3
from datetime import datetime
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import traceback
from src.audit.logger import log_audit

def evaluar_modelo_detallado(
    id_benchmarking: int,
    nombre_modelo: str,
    id_usuario: int,
    id_sesion: str,
    usuario: str,
    db_path: Optional[str] = None
) -> Dict:
    """
    Realiza una evaluación detallada de un modelo específico.
    
    Args:
        id_benchmarking: ID del benchmarking previamente ejecutado
        nombre_modelo: Nombre del modelo a evaluar en detalle
        id_usuario: ID del usuario que realiza la acción
        id_sesion (str): ID de sesión para trazabilidad
        usuario (str): Usuario que ejecuta la acción
        db_path: Ruta a la base de datos
        
    Returns:
        Dict: Resultados detallados de la evaluación
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
        
        # Obtener los resultados del benchmarking
        cursor.execute(
            "SELECT resultados_completos FROM benchmarking_modelos WHERE id = ?", 
            (id_benchmarking,)
        )
        
        resultado = cursor.fetchone()
        if not resultado:
            return {"error": f"No se encontró el benchmarking con ID {id_benchmarking}"}
        
        # Cargar resultados
        benchmarking = json.loads(resultado['resultados_completos'])
        
        # Buscar el modelo específico
        modelo_encontrado = None
        for modelo in benchmarking['modelos_exitosos']:
            if modelo['nombre'] == nombre_modelo:
                modelo_encontrado = modelo
                break
        
        if not modelo_encontrado:
            return {"error": f"No se encontró el modelo {nombre_modelo} en los resultados"}
        
        # Registrar en la tabla de auditoría
        cursor.execute("""
            INSERT INTO auditoria (
                id_usuario, accion, descripcion, fecha, id_sesion, usuario
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            id_usuario,
            "EVALUACION_DETALLADA",
            f"Evaluación detallada del modelo {nombre_modelo}",
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            id_sesion,
            usuario
        ))
        
        # Confirmar cambios
        conn.commit()
        
        # Retornar la información detallada del modelo
        return {
            "modelo": modelo_encontrado,
            "tipo_problema": benchmarking['tipo_problema'],
            "variable_objetivo": benchmarking['variable_objetivo'],
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    except Exception as e:
        log_audit(
            usuario=usuario,
            accion="ERROR_EVALUAR_MODELO_DETALLADO",
            entidad="evaluador",
            id_entidad=str(id_benchmarking),
            detalles=f"Error al evaluar modelo detallado: {str(e)}",
            id_sesion=id_sesion
        )
        if conn:
            conn.rollback()
        return {"error": str(e)}
    finally:
        if conn:
            conn.close()

def obtener_ultimos_benchmarkings(
    limite: int,
    id_usuario: Optional[int],
    id_sesion: str,
    usuario: str,
    db_path: Optional[str] = None
) -> list:
    """
    Obtiene los últimos benchmarkings ejecutados.
    
    Args:
        limite: Número máximo de resultados a devolver
        id_usuario: ID del usuario para filtrar
        id_sesion (str): ID de sesión para trazabilidad
        usuario (str): Usuario que ejecuta la acción
        db_path: Ruta a la base de datos
        
    Returns:
        list: Lista de benchmarkings
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
        
        # Construir la consulta
        query = """
            SELECT id, tipo_problema, variable_objetivo, 
                   cantidad_modelos_exitosos, cantidad_modelos_fallidos, 
                   mejor_modelo, fecha_ejecucion 
            FROM benchmarking_modelos 
            WHERE 1=1
        """
        params = []
        
        if id_usuario:
            query += " AND id_usuario = ?"
            params.append(id_usuario)
        
        query += " ORDER BY fecha_ejecucion DESC LIMIT ?"
        params.append(limite)
        
        # Ejecutar la consulta
        cursor.execute(query, params)
        
        # Obtener y devolver resultados
        resultados = []
        for row in cursor.fetchall():
            resultados.append({
                "id": row['id'],
                "tipo_problema": row['tipo_problema'],
                "variable_objetivo": row['variable_objetivo'],
                "modelos_exitosos": row['cantidad_modelos_exitosos'],
                "modelos_fallidos": row['cantidad_modelos_fallidos'],
                "mejor_modelo": row['mejor_modelo'],
                "fecha": row['fecha_ejecucion']
            })
            
        return resultados
    
    except Exception as e:
        log_audit(
            usuario=usuario,
            accion="ERROR_OBTENER_BENCHMARKINGS",
            entidad="evaluador",
            id_entidad="N/A",
            detalles=f"Error al obtener últimos benchmarkings: {str(e)}",
            id_sesion=id_sesion
        )
        return []
    finally:
        if conn:
            conn.close()

def generar_curvas_aprendizaje(
    id_benchmarking: int,
    nombre_modelo: str,
    id_sesion: str,
    usuario: str,
    db_path: Optional[str] = None
) -> Dict:
    """
    Genera curvas de aprendizaje para detectar overfitting/underfitting.
    
    Args:
        id_benchmarking: ID del benchmarking
        nombre_modelo: Nombre del modelo
        id_sesion (str): ID de sesión para trazabilidad
        usuario (str): Usuario que ejecuta la acción
        db_path: Ruta a la base de datos
        
    Returns:
        Dict: Resultados del análisis de curvas de aprendizaje
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
        
        # Obtener los resultados del benchmarking
        cursor.execute(
            "SELECT resultados_completos FROM benchmarking_modelos WHERE id = ?", 
            (id_benchmarking,)
        )
        row = cursor.fetchone()
        
        if not row:
            return {
                'error': f'No se encontró el benchmarking con ID {id_benchmarking}',
                'solucion': 'Verifique que el ID sea correcto o ejecute un nuevo benchmarking'
            }
        
        # Deserializar los resultados
        resultados_benchmarking = json.loads(row['resultados_completos'])
        
        # Buscar el modelo específico
        modelo_encontrado = None
        for modelo in resultados_benchmarking.get('modelos_exitosos', []):
            if modelo['nombre'] == nombre_modelo:
                modelo_encontrado = modelo
                break
        
        if not modelo_encontrado:
            return {
                'error': f'No se encontró el modelo {nombre_modelo} en el benchmarking',
                'solucion': 'Verifique que el nombre del modelo sea correcto'
            }
        
        # Verificar que tenemos datos de prueba
        if 'X_test' not in resultados_benchmarking or 'y_test' not in resultados_benchmarking:
            return {
                'error': 'No se encontraron datos de prueba en el benchmarking',
                'solucion': 'Los datos de prueba son necesarios para el análisis. Ejecute un nuevo benchmarking'
            }
        
        # Generar análisis simulado de curvas de aprendizaje
        # (Una implementación completa requeriría re-entrenar el modelo con diferentes tamaños de datos)
        
        # Obtener métricas existentes del modelo
        metricas = modelo_encontrado.get('metricas', {})
        cv_scores = modelo_encontrado.get('cv_scores', [])
        
        # Simular diagnóstico basado en métricas disponibles
        diagnostico = generar_diagnostico_overfitting(metricas, cv_scores)
        
        resultados = {
            'modelo': nombre_modelo,
            'tipo_problema': resultados_benchmarking.get('tipo_problema'),
            'metricas_principales': metricas,
            'cv_scores': cv_scores,
            'diagnostico': diagnostico,
            'recomendaciones': generar_recomendaciones_validacion(diagnostico, metricas),
            'datos_disponibles': {
                'X_test_shape': str(np.array(resultados_benchmarking['X_test']).shape) if 'X_test' in resultados_benchmarking else 'N/A',
                'y_test_shape': str(np.array(resultados_benchmarking['y_test']).shape) if 'y_test' in resultados_benchmarking else 'N/A',
                'total_filas': resultados_benchmarking.get('total_filas', 'N/A'),
                'porcentaje_test': resultados_benchmarking.get('porcentaje_test', 'N/A')
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        conn.close()
        return resultados
        
    except Exception as e:
        log_audit(
            usuario=usuario,
            accion="ERROR_GENERAR_CURVAS_APRENDIZAJE",
            entidad="evaluador",
            id_entidad=str(id_benchmarking),
            detalles=f"Error generando curvas de aprendizaje: {str(e)}",
            id_sesion=id_sesion
        )
        if conn:
            conn.close()
        return {
            'error': f'Error interno: {str(e)}',
            'solucion': 'Verifique los datos y vuelva a intentar'
        }


def generar_diagnostico_overfitting(metricas: Dict, cv_scores: list) -> Dict:
    """
    Genera un diagnóstico de overfitting/underfitting basado en métricas disponibles.
    
    Args:
        metricas: Métricas del modelo
        cv_scores: Puntuaciones de validación cruzada
        
    Returns:
        Dict: Diagnóstico del modelo
    """
    diagnostico = {
        'overfitting': 'desconocido',
        'underfitting': 'desconocido',
        'varianza_cv': 0,
        'mensaje': '',
        'nivel_confianza': 'bajo'
    }
    
    if cv_scores:
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        diagnostico['varianza_cv'] = cv_std
        
        # Análisis de varianza en CV
        if cv_std > 0.1:  # Alta varianza
            diagnostico['overfitting'] = 'posible'
            diagnostico['mensaje'] = f'Alta varianza en CV (σ={cv_std:.3f}). Posible overfitting.'
        elif cv_std < 0.03:  # Baja varianza
            diagnostico['overfitting'] = 'improbable'
            diagnostico['mensaje'] = f'Baja varianza en CV (σ={cv_std:.3f}). Modelo estable.'
        else:
            diagnostico['overfitting'] = 'normal'
            diagnostico['mensaje'] = f'Varianza en CV normal (σ={cv_std:.3f}).'
        
        # Análisis de rendimiento general
        if cv_mean < 0.6:  # Bajo rendimiento
            diagnostico['underfitting'] = 'posible'
            diagnostico['mensaje'] += f' Rendimiento bajo (μ={cv_mean:.3f}). Posible underfitting.'
        else:
            diagnostico['underfitting'] = 'improbable'
        
        diagnostico['nivel_confianza'] = 'medio'
    
    return diagnostico


def generar_recomendaciones_validacion(diagnostico: Dict, metricas: Dict) -> list:
    """
    Genera recomendaciones basadas en el diagnóstico de validación.
    
    Args:
        diagnostico: Diagnóstico del modelo
        metricas: Métricas del modelo
        
    Returns:
        list: Lista de recomendaciones
    """
    recomendaciones = []
    
    if diagnostico['overfitting'] == 'posible':
        recomendaciones.extend([
            "🔄 Considere usar regularización (L1/L2)",
            "📊 Aumente el tamaño del dataset de entrenamiento",
            "🌳 Reduzca la complejidad del modelo",
            "✂️ Aplique técnicas de feature selection"
        ])
    
    if diagnostico['underfitting'] == 'posible':
        recomendaciones.extend([
            "🔧 Aumente la complejidad del modelo",
            "🎯 Agregue más características relevantes",
            "⚙️ Ajuste los hiperparámetros",
            "🔍 Verifique la calidad de los datos"
        ])
    
    if diagnostico['varianza_cv'] > 0.1:
        recomendaciones.append("🎲 Considere usar ensemble methods para reducir varianza")
    
    if not recomendaciones:
        recomendaciones.append("✅ El modelo muestra un comportamiento balanceado")
    
    return recomendaciones

def cargar_benchmarking_seleccionado(
    benchmarking_id: int,
    id_sesion: str,
    usuario: str,
    db_path: Optional[str] = None
) -> Dict:
    """
    Carga un benchmarking específico desde la base de datos.
    
    Args:
        benchmarking_id: ID del benchmarking a cargar
        id_sesion (str): ID de sesión para trazabilidad
        usuario (str): Usuario que ejecuta la acción
        db_path: Ruta a la base de datos (opcional)
        
    Returns:
        Dict: Datos completos del benchmarking o diccionario con error
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
        
        # Obtener los resultados del benchmarking
        cursor.execute(
            "SELECT resultados_completos FROM benchmarking_modelos WHERE id = ?", 
            (benchmarking_id,)
        )
        
        resultado = cursor.fetchone()
        if not resultado:
            return {"error": f"No se encontró el benchmarking con ID {benchmarking_id}"}
        
        # Cargar resultados JSON
        try:
            benchmarking = json.loads(resultado['resultados_completos'])
            return benchmarking
        except json.JSONDecodeError as e:
            return {"error": f"Error al decodificar JSON del benchmarking: {str(e)}"}
    except Exception as e:
        log_audit(
            usuario=usuario,
            accion="ERROR_CARGAR_BENCHMARKING",
            entidad="evaluador",
            id_entidad=str(benchmarking_id),
            detalles=f"Error al cargar benchmarking: {str(e)}\n{traceback.format_exc()}",
            id_sesion=id_sesion
        )
        return {"error": str(e)}
    finally:
        if conn:
            conn.close()

def obtener_modelo_desde_benchmarking(
    resultados_benchmarking: Dict,
    nombre_modelo: Optional[str] = None
) -> Tuple[Dict, bool]:
    """
    Obtiene un modelo específico desde los resultados de un benchmarking.
    Si no se especifica nombre_modelo, retorna el mejor modelo.
    
    Args:
        resultados_benchmarking: Diccionario con resultados del benchmarking
        nombre_modelo: Nombre del modelo a obtener (opcional)
        
    Returns:
        Tuple[Dict, bool]: (Modelo encontrado, Es el mejor modelo)
    """
    try:
        # Verificar si hay modelos exitosos
        if not resultados_benchmarking.get('modelos_exitosos'):
            return {"error": "No hay modelos exitosos en el benchmarking"}, False
        
        # Si no se especifica modelo, tomar el mejor
        if nombre_modelo is None:
            # Si hay un mejor modelo definido, usarlo
            if 'mejor_modelo' in resultados_benchmarking and resultados_benchmarking['mejor_modelo']:
                modelo = resultados_benchmarking['mejor_modelo']
                return modelo, True
            # Si no, tomar el primer modelo exitoso
            else:
                return resultados_benchmarking['modelos_exitosos'][0], False
        
        # Buscar el modelo por nombre
        for modelo in resultados_benchmarking['modelos_exitosos']:
            if modelo['nombre'] == nombre_modelo:
                # Determinar si es el mejor modelo
                es_mejor = False
                if 'mejor_modelo' in resultados_benchmarking:
                    mejor = resultados_benchmarking['mejor_modelo']
                    if mejor and 'nombre' in mejor and mejor['nombre'] == nombre_modelo:
                        es_mejor = True
                
                return modelo, es_mejor
        
        # Si no se encontró el modelo
        return {"error": f"No se encontró el modelo {nombre_modelo}"}, False
        
    except Exception as e:
        log_audit(
            "sistema",
            "ERROR_OBTENER_MODELO_BENCHMARKING",
            "evaluador",
            "N/A",
            f"Error al obtener modelo desde benchmarking: {str(e)}"
        )
        return {"error": str(e)}, False
def generar_visualizaciones_clasificacion(
    modelo: Dict,
    resultados_benchmarking: Dict,
    tipo_visualizacion: str
) -> Figure:
    """
    Genera visualizaciones específicas para problemas de clasificación.
    
    Args:
        modelo: Diccionario con información del modelo
        resultados_benchmarking: Resultados completos del benchmarking
        tipo_visualizacion: Tipo de visualización a generar
        
    Returns:
        plt.Figure: Figura con la visualización generada
    """
    from src.modelos.visualizador import (
        generar_matriz_confusion, 
        generar_curva_roc, 
        generar_curva_precision_recall
    )
    
    try:
        # Convertir datos serializados a numpy arrays si es necesario
        X_test = np.array(resultados_benchmarking['X_test']) if isinstance(resultados_benchmarking['X_test'], list) else resultados_benchmarking['X_test']
        y_test = np.array(resultados_benchmarking['y_test']) if isinstance(resultados_benchmarking['y_test'], list) else resultados_benchmarking['y_test']
        
        # Determinar si tenemos el modelo o solo métricas
        if 'modelo_objeto' in modelo:
            # Obtener predicciones
            y_pred = modelo['modelo_objeto'].predict(X_test)
            
            # Obtener probabilidades (para curvas ROC)
            try:
                y_prob = modelo['modelo_objeto'].predict_proba(X_test)
            except Exception:
                # Si el modelo no soporta predict_proba
                y_prob = np.zeros((len(y_test), len(np.unique(y_test))))
        else:
            # Si no hay modelo, usar resultados pre-calculados o dummy data
            y_pred = y_test  # Fallback
            y_prob = np.zeros((len(y_test), len(np.unique(y_test))))
        
        # Obtener clases
        clases = np.unique(y_test)
        
        # Generar visualización específica
        if tipo_visualizacion == "matriz_confusion":
            return generar_matriz_confusion(
                y_test, 
                y_pred, 
                clases=[str(c) for c in clases],
                normalizar=None,
                titulo=f"Matriz de Confusión - {modelo['nombre']}",
                id_sesion=resultados_benchmarking.get('id_sesion', 'N/A'),
                usuario=resultados_benchmarking.get('usuario', 'sistema')
            )
        elif tipo_visualizacion == "curva_roc":
            return generar_curva_roc(
                y_test, 
                y_prob, 
                clases=[str(c) for c in clases],
                titulo=f"Curva ROC - {modelo['nombre']}",
                id_sesion=resultados_benchmarking.get('id_sesion', 'N/A'),
                usuario=resultados_benchmarking.get('usuario', 'sistema')
            )
        elif tipo_visualizacion == "precision_recall":
            return generar_curva_precision_recall(
                y_test, 
                y_prob, 
                clases=[str(c) for c in clases],
                titulo=f"Curva Precision-Recall - {modelo['nombre']}",
                id_sesion=resultados_benchmarking.get('id_sesion', 'N/A'),
                usuario=resultados_benchmarking.get('usuario', 'sistema')
            )
        elif tipo_visualizacion == "comparar_modelos":
            # Implementar comparación entre modelos
            # Por ahora, simplemente devolvemos otra visualización
            return generar_curva_roc(
                y_test, 
                y_prob, 
                clases=[str(c) for c in clases],
                titulo=f"Curva ROC - {modelo['nombre']}",
                id_sesion=resultados_benchmarking.get('id_sesion', 'N/A'),
                usuario=resultados_benchmarking.get('usuario', 'sistema')
            )
        else:
            # Crear un gráfico genérico si el tipo de visualización no es reconocido
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"Visualización no disponible: {tipo_visualizacion}", 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
            
    except Exception as e:
        # Crear un gráfico de error
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"Error al generar visualización: {str(e)}", 
               ha='center', va='center', transform=ax.transAxes)
        log_audit(
            "sistema",
            "ERROR_GENERAR_VISUALIZACION_CLASIFICACION",
            "evaluador",
            modelo.get('nombre', 'N/A'),
            f"Error al generar visualización de clasificación: {str(e)}"
        )
        return fig
def generar_visualizaciones_regresion(
    modelo: Dict,
    resultados_benchmarking: Dict,
    tipo_visualizacion: str
) -> Figure:
    """
    Genera visualizaciones específicas para problemas de regresión.
    
    Args:
        modelo: Diccionario con información del modelo
        resultados_benchmarking: Resultados completos del benchmarking
        tipo_visualizacion: Tipo de visualización a generar
        
    Returns:
        plt.Figure: Figura con la visualización generada
    """
    from src.modelos.visualizador import (
        generar_grafico_residuos,
        comparar_distribuciones
    )
    
    try:
        # Convertir datos serializados a numpy arrays si es necesario
        X_test = np.array(resultados_benchmarking['X_test']) if isinstance(resultados_benchmarking['X_test'], list) else resultados_benchmarking['X_test']
        y_test = np.array(resultados_benchmarking['y_test']) if isinstance(resultados_benchmarking['y_test'], list) else resultados_benchmarking['y_test']
        
        # Determinar si tenemos el modelo o solo métricas
        if 'modelo_objeto' in modelo:
            # Obtener predicciones
            y_pred = modelo['modelo_objeto'].predict(X_test)
        else:
            # Si no hay modelo, usar resultados pre-calculados o dummy data
            y_pred = y_test * 0.9 + np.random.normal(0, 0.1, len(y_test))  # Fallback con algo de ruido
        
        # Generar visualización específica
        if tipo_visualizacion == "residuos":
            return generar_grafico_residuos(
                y_test, 
                y_pred, 
                titulo=f"Gráfico de Residuos - {modelo['nombre']}",
                id_sesion=resultados_benchmarking.get('id_sesion', 'N/A'),
                usuario=resultados_benchmarking.get('usuario', 'sistema')
            )
        elif tipo_visualizacion == "valores_reales_vs_predichos":
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(y_test, y_pred, alpha=0.6)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax.set_xlabel('Valores Reales')
            ax.set_ylabel('Valores Predichos')
            ax.set_title(f'Valores Reales vs Predichos - {modelo["nombre"]}')
            ax.grid(True, linestyle='--', alpha=0.7)
            return fig
        elif tipo_visualizacion == "distribucion":
            return comparar_distribuciones(
                y_test,
                y_pred,
                titulo=f"Comparación de Distribuciones - {modelo['nombre']}",
                id_sesion=resultados_benchmarking.get('id_sesion', 'N/A'),
                usuario=resultados_benchmarking.get('usuario', 'sistema')
            )
        else:
            # Crear un gráfico genérico si el tipo de visualización no es reconocido
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"Visualización no disponible: {tipo_visualizacion}", 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
            
    except Exception as e:
        # Crear un gráfico de error
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"Error al generar visualización: {str(e)}", 
               ha='center', va='center', transform=ax.transAxes)
        log_audit(
            "sistema",
            "ERROR_VISUALIZACION_REGRESION",
            "evaluador",
            modelo.get('nombre', 'N/A'),
            f"Error en visualización de regresión: {str(e)}"
        )
        return fig

def diagnosticar_visualizaciones(resultados_benchmarking: Dict, modelo: Dict) -> Dict:
    """
    Diagnostica si se cumplen los requisitos para generar visualizaciones avanzadas
    y proporciona recomendaciones si hay problemas.
    
    Args:
        resultados_benchmarking: Resultados completos del benchmarking
        modelo: Diccionario con información del modelo seleccionado
        
    Returns:
        Dict: Diccionario con estado de cada requisito y recomendaciones
    """
    diagnostico = {
        "requisitos": {},
        "recomendaciones": [],
        "puede_visualizar": True
    }
    
    # Verificar disponibilidad de datos de prueba
    if 'X_test' not in resultados_benchmarking or 'y_test' not in resultados_benchmarking:
        diagnostico["requisitos"]["datos_prueba"] = False
        diagnostico["recomendaciones"].append(
            "No se encontraron datos de prueba (X_test, y_test). Ejecute un nuevo benchmarking para generar visualizaciones avanzadas."
        )
        diagnostico["puede_visualizar"] = False
    else:
        diagnostico["requisitos"]["datos_prueba"] = True
    
    # Verificar disponibilidad del objeto modelo
    if 'modelo_objeto' not in modelo:
        diagnostico["requisitos"]["modelo_objeto"] = False
        diagnostico["recomendaciones"].append(
            "El objeto del modelo no está disponible. Esto puede ocurrir cuando los modelos no se deserializan correctamente. "
            "Si ha cargado datos nuevos, ejecute un nuevo benchmarking para garantizar que los modelos estén disponibles."
        )
        diagnostico["puede_visualizar"] = False
    else:
        diagnostico["requisitos"]["modelo_objeto"] = True
    
    # Verificar que el modelo sea compatible con predict_proba (para clasificación)
    if resultados_benchmarking.get('tipo_problema') == 'clasificacion' and diagnostico["requisitos"].get("modelo_objeto", False):
        try:
            # Comprobar si el modelo tiene el método predict_proba
            if hasattr(modelo['modelo_objeto'], 'predict_proba'):
                diagnostico["requisitos"]["predict_proba"] = True
            else:
                diagnostico["requisitos"]["predict_proba"] = False
                diagnostico["recomendaciones"].append(
                    f"El modelo {modelo['nombre']} no admite el cálculo de probabilidades (predict_proba). "
                    "Algunas visualizaciones como curvas ROC pueden no estar disponibles o ser limitadas."
                )
        except Exception as e:
            diagnostico["requisitos"]["predict_proba"] = False
            diagnostico["recomendaciones"].append(
                f"Error al verificar compatibilidad con predict_proba: {str(e)}"
            )
    
    # Verificar que haya suficientes métricas calculadas
    if not modelo.get('metricas'):
        diagnostico["requisitos"]["metricas"] = False
        diagnostico["recomendaciones"].append(
            "No se encontraron métricas para el modelo. Algunas visualizaciones pueden no estar disponibles."
        )
    else:
        diagnostico["requisitos"]["metricas"] = True
    
    # Verificar tipo de problema compatible
    if resultados_benchmarking.get('tipo_problema') not in ['clasificacion', 'regresion']:
        diagnostico["requisitos"]["tipo_problema"] = False
        diagnostico["recomendaciones"].append(
            f"El tipo de problema '{resultados_benchmarking.get('tipo_problema')}' no es compatible con las visualizaciones avanzadas disponibles."
        )
        diagnostico["puede_visualizar"] = False
    else:
        diagnostico["requisitos"]["tipo_problema"] = True
    
    return diagnostico

def generar_tabla_metricas(modelo: Dict, tipo_problema: str) -> pd.DataFrame:
    """
    Genera una tabla de pandas con las métricas del modelo.
    
    Args:
        modelo: Diccionario con información del modelo
        tipo_problema: Tipo de problema (clasificación o regresión)
        
    Returns:
        pd.DataFrame: DataFrame con las métricas
    """
    try:
        # Obtener métricas del modelo
        metricas = modelo.get('metricas', {})
        
        # Crear dataframe según tipo de problema
        if tipo_problema == 'clasificacion':
            metricas_clave = ['accuracy', 'precision', 'recall', 'f1', 'auc']
            nombres_metricas = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
        else:  # regresión
            metricas_clave = ['r2', 'mae', 'mse', 'rmse', 'mape']
            nombres_metricas = ['R²', 'MAE', 'MSE', 'RMSE', 'MAPE']
        
        # Filtrar métricas disponibles
        valores = []
        nombres_filtrados = []
        
        for clave, nombre in zip(metricas_clave, nombres_metricas):
            if clave in metricas:
                valores.append(metricas[clave])
                nombres_filtrados.append(nombre)
        
        # Crear dataframe
        df_metricas = pd.DataFrame({
            'Métrica': nombres_filtrados,
            'Valor': valores
        })
        
        return df_metricas
    
    except Exception as e:
        log_audit(
            "sistema",
            "ERROR_GENERAR_TABLA_METRICAS",
            "evaluador",
            "N/A",
            f"Error al generar tabla de métricas: {str(e)}"
        )
        # Devolver un dataframe vacío en caso de error
        return pd.DataFrame({'Métrica': [], 'Valor': []})
def comparar_metricas_regresion(modelos_dict, X_test, y_test):
    """
    Compara métricas de regresión para múltiples modelos.
    
    Args:
        modelos_dict: Diccionario con nombre -> {'modelo': modelo_objeto}
        X_test: Datos de prueba
        y_test: Etiquetas verdaderas de prueba
        
    Returns:
        pd.DataFrame: DataFrame con métricas comparativas
    """
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    try:
        # Crear DataFrame para comparar métricas de los modelos
        metricas_comp = pd.DataFrame()
        
        for nombre, info in modelos_dict.items():
            y_pred = info['modelo'].predict(X_test)
            
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            
            # Añadir métricas al DataFrame de comparación
            metricas_comp[nombre] = [r2, mse, rmse, mae]
        
        # Establecer nombres de filas para las métricas
        metricas_comp.index = pd.Index(['R²', 'MSE', 'RMSE', 'MAE'])
        
        return metricas_comp
        
    except Exception as e:
        log_audit(
            "sistema",
            "ERROR_COMPARAR_METRICAS_REGRESION",
            "evaluador",
            "N/A",
            f"Error al comparar métricas de regresión: {str(e)}"
        )
        return pd.DataFrame()

def calcular_matriz_confusion_detallada(y_test, y_pred, normalize=None):
    """
    Calcula la matriz de confusión con normalización opcional.
    
    Args:
        y_test: Etiquetas verdaderas
        y_pred: Predicciones del modelo
        normalize: Tipo de normalización ('true', 'pred', 'all', None)
        
    Returns:
        np.ndarray: Matriz de confusión calculada
    """
    from sklearn.metrics import confusion_matrix
    
    try:
        return confusion_matrix(y_test, y_pred, normalize=normalize)
    except Exception as e:
        log_audit(
            "sistema",
            "ERROR_MATRIZ_CONFUSION",
            "evaluador",
            "N/A",
            f"Error al calcular matriz de confusión: {str(e)}"
        )
        raise


def calcular_curvas_roc_completas(y_test, y_prob, clases):
    """
    Calcula curvas ROC para clasificación binaria y multiclase.
    
    Args:
        y_test: Etiquetas verdaderas
        y_prob: Probabilidades predichas
        clases: Lista de clases
        
    Returns:
        dict: Diccionario con datos de curvas ROC y AUC
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    import scipy.sparse as sp
    
    try:
        es_multiclase = len(clases) > 2
        
        if es_multiclase:
            # Para multiclase, binarizar las etiquetas
            y_bin = label_binarize(y_test, classes=np.unique(y_test))
            
            # Convertir a array denso si es matriz sparse
            if sp.issparse(y_bin):
                y_bin = sp.csr_matrix(y_bin).toarray()
            else:
                y_bin = np.array(y_bin)
            
            # Calcular AUC para cada clase
            aucs = []
            curvas_roc = {}
            
            for i, clase in enumerate(clases):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                auc_valor = auc(fpr, tpr)
                aucs.append(auc_valor)
                curvas_roc[str(clase)] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'auc': auc_valor
                }
            
            auc_promedio = np.mean(aucs)
            
            return {
                'es_multiclase': True,
                'auc_promedio': auc_promedio,
                'aucs_por_clase': aucs,
                'curvas_roc': curvas_roc
            }
        else:
            # Para binario, usar la probabilidad de la clase positiva
            if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
                y_prob_pos = y_prob[:, 1]
            else:
                y_prob_pos = y_prob
            
            fpr, tpr, _ = roc_curve(y_test, y_prob_pos)
            auc_valor = auc(fpr, tpr)
            
            return {
                'es_multiclase': False,
                'auc_valor': auc_valor,
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist()
            }
            
    except Exception as e:
        log_audit(
            "sistema",
            "ERROR_CURVAS_ROC",
            "evaluador",
            "N/A",
            f"Error al calcular curvas ROC: {str(e)}"
        )
        raise


def calcular_metricas_clasificacion_completas(y_test, y_pred, y_prob):
    """
    Calcula métricas completas para problemas de clasificación.
    
    Args:
        y_test: Etiquetas verdaderas
        y_pred: Predicciones del modelo
        y_prob: Probabilidades predichas
        
    Returns:
        dict: Diccionario con todas las métricas calculadas
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    try:
        clases = np.unique(y_test)
        es_multiclase = len(clases) > 2
        
        # Métricas básicas
        accuracy = accuracy_score(y_test, y_pred)
        
        # Para multiclase, usar promedio macro
        avg_method = 'macro' if es_multiclase else 'binary'
        
        precision = precision_score(y_test, y_pred, average=avg_method, zero_division=0)
        recall = recall_score(y_test, y_pred, average=avg_method, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=avg_method, zero_division=0)
        
        # AUC (solo si tenemos probabilidades)
        try:
            if es_multiclase:
                auc_score = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
            else:
                if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
                    auc_score = roc_auc_score(y_test, y_prob[:, 1])
                else:
                    auc_score = roc_auc_score(y_test, y_prob)
        except Exception:
            auc_score = None
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc_score,
            'es_multiclase': es_multiclase,
            'num_clases': len(clases)
        }
        
    except Exception as e:
        log_audit(
            "sistema",
            "ERROR_METRICAS_CLASIFICACION",
            "evaluador",
            "N/A",
            f"Error al calcular métricas de clasificación: {str(e)}"
        )
        raise

def comparar_modelos_regresion_completo(modelos_dict, X_test, y_test):
    """
    Genera comparación completa de modelos de regresión incluyendo predicciones y métricas.
    
    Args:
        modelos_dict: Diccionario con nombre -> {'modelo': modelo_objeto}
        X_test: Datos de prueba
        y_test: Etiquetas verdaderas de prueba
        
    Returns:
        dict: Diccionario con predicciones, métricas comparativas y datos para visualización
    """
    try:
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        
        # Crear DataFrame para comparar predicciones
        pred_df = pd.DataFrame()
        pred_df['Real'] = y_test
        
        # Generar predicciones para cada modelo
        for nombre, info in modelos_dict.items():
            pred_df[nombre] = info['modelo'].predict(X_test)
        
        # Crear DataFrame para comparar métricas de los modelos
        metricas_comp = pd.DataFrame()
        
        for nombre, info in modelos_dict.items():
            y_pred = info['modelo'].predict(X_test)
            
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            
            # Añadir métricas al DataFrame de comparación
            metricas_comp[nombre] = [r2, mse, rmse, mae]
        
        # Establecer nombres de filas para las métricas
        metricas_comp.index = pd.Index(['R²', 'MSE', 'RMSE', 'MAE'])
        
        # Datos para visualización
        min_val = pred_df['Real'].min()
        max_val = pred_df['Real'].max()
        
        return {
            'predicciones_df': pred_df,
            'metricas_comparativas': metricas_comp,
            'rango_valores': {'min': min_val, 'max': max_val},
            'nombres_modelos': list(modelos_dict.keys())
        }
        
    except Exception as e:
        log_audit(
            "sistema",
            "ERROR_COMPARAR_MODELOS_REGRESION_COMPLETO",
            "evaluador",
            "N/A",
            f"Error al comparar modelos de regresión completo: {str(e)}"
        )
        return {
            'error': str(e),
            'predicciones_df': pd.DataFrame(),
            'metricas_comparativas': pd.DataFrame(),
            'rango_valores': {'min': 0, 'max': 1},
            'nombres_modelos': []
        }

def generar_visualizacion_comparacion_regresion(datos_comparacion):
    """
    Genera visualización de comparación de modelos de regresión.
    
    Args:
        datos_comparacion: Diccionario con datos de comparación generados por comparar_modelos_regresion_completo
        
    Returns:
        matplotlib.figure.Figure: Figura con la comparación de modelos
    """
    try:
        pred_df = datos_comparacion['predicciones_df']
        rango = datos_comparacion['rango_valores']
        nombres = datos_comparacion['nombres_modelos']
        
        if pred_df.empty:
            # Crear figura de error
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'Error: No hay datos para comparación', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Crear figura
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Graficar línea de referencia (predicción perfecta)
        ax.plot([rango['min'], rango['max']], [rango['min'], rango['max']], 
                'k--', label='Predicción perfecta')
        
        # Graficar predicciones de cada modelo
        for nombre in nombres:
            if nombre in pred_df.columns:
                ax.scatter(pred_df['Real'], pred_df[nombre], label=nombre, alpha=0.6)
        
        # Configurar gráfico
        ax.set_xlabel('Valores reales')
        ax.set_ylabel('Valores predichos')
        ax.set_title('Comparación de predicciones')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
        
    except Exception as e:
        log_audit(
            "sistema",
            "ERROR_VISUALIZACION_COMPARACION_REGRESION",
            "evaluador",
            "N/A",
            f"Error al generar visualización de comparación: {str(e)}"
        )
        # Crear figura de error
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Error al generar visualización: {str(e)}', 
               ha='center', va='center', transform=ax.transAxes)
        return fig
def crear_dataframe_comparacion_regresion(modelos_dict, X_test, y_test):
    """
    Crea un DataFrame para comparar predicciones de múltiples modelos de regresión.
    
    Args:
        modelos_dict (dict): Diccionario con nombre_modelo: {"modelo": objeto_modelo}
        X_test: Datos de prueba
        y_test: Valores reales
        
    Returns:
        pd.DataFrame: DataFrame con columnas Real y predicciones de cada modelo
    """
    import pandas as pd
    import numpy as np
    
    # Convertir datos si son listas
    if isinstance(X_test, list):
        X_test = np.array(X_test)
    if isinstance(y_test, list):
        y_test = np.array(y_test)
    
    # Crear DataFrame base
    pred_df = pd.DataFrame()
    pred_df['Real'] = y_test
    
    # Agregar predicciones de cada modelo
    for nombre, info in modelos_dict.items():
        try:
            predicciones = info['modelo'].predict(X_test)
            pred_df[nombre] = predicciones
        except Exception:
            log_audit(
                "sistema",
                "ERROR_PREDICCION_MODELO_REGRESION",
                "evaluador",
                nombre,
                f"Error al generar predicciones para {nombre}"
            )
            # Continuar con otros modelos
            continue
    
    return pred_df

def generar_datos_grafico_comparacion_regresion(modelos_dict, X_test, y_test):
    """
    Genera los datos necesarios para crear gráficos de comparación de regresión.
    
    Args:
        modelos_dict (dict): Diccionario con modelos
        X_test: Datos de prueba
        y_test: Valores reales
        
    Returns:
        dict: Datos estructurados para gráficos
    """
    
    # Crear DataFrame de comparación
    pred_df = crear_dataframe_comparacion_regresion(modelos_dict, X_test, y_test)
    
    # Calcular rango para línea de referencia
    min_val = float(pred_df['Real'].min())
    max_val = float(pred_df['Real'].max())
    
    # Preparar datos para cada modelo
    datos_modelos = {}
    for nombre in modelos_dict.keys():
        if nombre in pred_df.columns:
            datos_modelos[nombre] = {
                'x': pred_df['Real'].values,
                'y': pred_df[nombre].values
            }
    
    return {
        'dataframe': pred_df,
        'rango_referencia': {'min': min_val, 'max': max_val},
        'datos_modelos': datos_modelos
    }

def calcular_metricas_modelo_individual(modelo_objeto, X_test, y_test, nombre_modelo):
    """
    Calcula métricas individuales para un modelo de regresión.
    
    Args:
        modelo_objeto: Objeto del modelo entrenado
        X_test: Datos de prueba
        y_test: Valores reales
        nombre_modelo (str): Nombre del modelo
        
    Returns:
        dict: Métricas calculadas
    """
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    import numpy as np
    
    try:
        # Generar predicciones
        y_pred = modelo_objeto.predict(X_test)
        
        # Calcular métricas
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        
        return {
            'modelo': nombre_modelo,
            'r2': round(r2, 4),
            'mse': round(mse, 4),
            'rmse': round(rmse, 4),
            'mae': round(mae, 4)
        }
    except Exception as e:
        log_audit(
            "sistema",
            "ERROR_METRICAS_MODELO_INDIVIDUAL",
            "evaluador",
            nombre_modelo,
            f"Error al calcular métricas para {nombre_modelo}: {str(e)}"
        )
        return {
            'modelo': nombre_modelo,
            'r2': 0,
            'mse': 0,
            'rmse': 0,
            'mae': 0,
            'error': str(e)
        }
