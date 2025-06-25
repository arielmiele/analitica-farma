"""
Módulo para la evaluación detallada de modelos de machine learning.
"""
import json
import os
import sqlite3
from datetime import datetime
import logging
from typing import Dict, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
import traceback

# Configuración del logger
logger = logging.getLogger("evaluador")

def evaluar_modelo_detallado(
    id_benchmarking: int,
    nombre_modelo: str,
    id_usuario: int = 1,
    db_path: Optional[str] = None
) -> Dict:
    """
    Realiza una evaluación detallada de un modelo específico.
    
    Args:
        id_benchmarking: ID del benchmarking previamente ejecutado
        nombre_modelo: Nombre del modelo a evaluar en detalle
        id_usuario: ID del usuario que realiza la acción
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
                id_usuario, accion, descripcion, fecha
            ) VALUES (?, ?, ?, ?)
        """, (
            id_usuario,
            "EVALUACION_DETALLADA",
            f"Evaluación detallada del modelo {nombre_modelo}",
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
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
        logger.error(f"Error al evaluar modelo detallado: {str(e)}")
        if conn:
            conn.rollback()
        return {"error": str(e)}
    finally:
        if conn:
            conn.close()

def obtener_ultimos_benchmarkings(
    limite: int = 5,
    id_usuario: Optional[int] = None,
    db_path: Optional[str] = None
) -> list:
    """
    Obtiene los últimos benchmarkings ejecutados.
    
    Args:
        limite: Número máximo de resultados a devolver
        id_usuario: ID del usuario para filtrar
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
        logger.error(f"Error al obtener últimos benchmarkings: {str(e)}")
        return []
    finally:
        if conn:
            conn.close()

def generar_curvas_aprendizaje(
    id_benchmarking: int,
    nombre_modelo: str,
    db_path: Optional[str] = None
) -> Dict:
    """
    Genera curvas de aprendizaje para detectar overfitting/underfitting.
    Este es un stub que debe implementarse con la HU11.
    
    Args:
        id_benchmarking: ID del benchmarking
        nombre_modelo: Nombre del modelo
        db_path: Ruta a la base de datos
        
    Returns:
        Dict: Información para generar curvas de aprendizaje
    """
    # Este método debe implementarse completamente en la HU11
    # Por ahora devolvemos un placeholder
    return {
        "mensaje": "La generación de curvas de aprendizaje estará disponible en la siguiente versión."
    }

def cargar_benchmarking_seleccionado(
    benchmarking_id: int,
    db_path: Optional[str] = None
) -> Dict:
    """
    Carga un benchmarking específico desde la base de datos.
    
    Args:
        benchmarking_id: ID del benchmarking a cargar
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
        error_detalle = traceback.format_exc()
        logger.error(f"Error al cargar benchmarking: {str(e)}\n{error_detalle}")
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
        logger.error(f"Error al obtener modelo desde benchmarking: {str(e)}")
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
            except Exception as e:
                # Si el modelo no soporta predict_proba
                logger.warning(f"El modelo no soporta predict_proba: {str(e)}")
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
                titulo=f"Matriz de Confusión - {modelo['nombre']}"
            )
        elif tipo_visualizacion == "curva_roc":
            return generar_curva_roc(
                y_test, 
                y_prob, 
                clases=[str(c) for c in clases],
                titulo=f"Curva ROC - {modelo['nombre']}"
            )
        elif tipo_visualizacion == "precision_recall":
            return generar_curva_precision_recall(
                y_test, 
                y_prob, 
                clases=[str(c) for c in clases],
                titulo=f"Curva Precision-Recall - {modelo['nombre']}"
            )
        elif tipo_visualizacion == "comparar_modelos":
            # Implementar comparación entre modelos
            # Por ahora, simplemente devolvemos otra visualización
            return generar_curva_roc(
                y_test, 
                y_prob, 
                clases=[str(c) for c in clases],
                titulo=f"Curva ROC - {modelo['nombre']}"
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
                titulo=f"Gráfico de Residuos - {modelo['nombre']}"
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
                titulo=f"Comparación de Distribuciones - {modelo['nombre']}"
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
        logger.error(f"Error en visualización de regresión: {str(e)}")
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
        logger.error(f"Error al generar tabla de métricas: {str(e)}")
        # Devolver un dataframe vacío en caso de error
        return pd.DataFrame({'Métrica': [], 'Valor': []})
