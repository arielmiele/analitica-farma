"""
Módulo para la recomendación del mejor modelo de machine learning.
"""
import json
import os
import sqlite3
import logging
from datetime import datetime
from typing import Dict, Optional

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
