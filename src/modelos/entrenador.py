"""
Módulo para entrenar modelos de machine learning.
Implementa la funcionalidad de benchmarking de múltiples modelos.
"""
import pandas as pd
import numpy as np
import time
import logging
import json
import sqlite3
import os
import scipy.sparse as sp
from datetime import datetime
from typing import Dict, Tuple, Optional
from .modelo_serializer import serializar_modelos_benchmarking, deserializar_modelos_benchmarking
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    r2_score, mean_squared_error, mean_absolute_error
)

# Importar modelos de clasificación
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Importar modelos de regresión
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, 
    AdaBoostRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Obtener el logger
logger = logging.getLogger("entrenador")

# Definición de modelos para clasificación y regresión
CLASSIFICATION_MODELS = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'SVC': SVC(probability=True, random_state=42),
    'KNeighbors': KNeighborsClassifier(),
    'GaussianNB': GaussianNB(),
    'AdaBoost': AdaBoostClassifier(random_state=42)
}

REGRESSION_MODELS = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(random_state=42),
    'Lasso': Lasso(random_state=42),
    'ElasticNet': ElasticNet(random_state=42),
    'DecisionTree': DecisionTreeRegressor(random_state=42),
    'RandomForest': RandomForestRegressor(random_state=42),
    'GradientBoosting': GradientBoostingRegressor(random_state=42),
    'SVR': SVR(),
    'KNeighbors': KNeighborsRegressor(),
    'AdaBoost': AdaBoostRegressor(random_state=42)
}

def detectar_tipo_problema(y: pd.Series) -> str:
    """
    Detecta automáticamente si se trata de un problema de clasificación o regresión
    
    Args:
        y (pd.Series): Variable objetivo
        
    Returns:
        str: 'clasificacion' o 'regresion'
    """
    # Si la variable objetivo es categórica o tiene pocos valores únicos, asumimos clasificación
    if y.dtype == 'object' or y.dtype == 'category' or y.dtype == 'bool':
        return 'clasificacion'
    
    # Si tiene pocos valores únicos en relación al total de muestras, probablemente es clasificación
    unique_ratio = len(y.unique()) / len(y)
    if unique_ratio < 0.05 or len(y.unique()) < 10:
        return 'clasificacion'
    
    # En caso contrario, asumimos regresión
    return 'regresion'

def preparar_datos(
    X: pd.DataFrame, 
    y: pd.Series, 
    tipo_problema: str,
    test_size: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[LabelEncoder]]:
    """
    Prepara los datos para entrenar modelos, realizando la división en train/test
    y aplicando las transformaciones necesarias.
    
    Args:
        X (pd.DataFrame): Variables predictoras
        y (pd.Series): Variable objetivo
        tipo_problema (str): 'clasificacion' o 'regresion'
        test_size (float): Tamaño del conjunto de prueba (por defecto 0.2)
        
    Returns:
        Tuple: X_train, X_test, y_train, y_test, le (LabelEncoder si es clasificación)
    """
    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Escalar características
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Para clasificación, codificar etiquetas si no son numéricas
    le = None
    if tipo_problema == 'clasificacion' and (y.dtype == 'object' or y.dtype == 'category'):
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)
    
    # Asegurar que sean arrays de numpy sin importar el caso previo
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    
    return X_train, X_test, y_train, y_test, le

def entrenar_modelo(
    nombre_modelo: str, 
    modelo,
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_test: np.ndarray, 
    y_test: np.ndarray,
    tipo_problema: str
) -> Dict:
    """
    Entrena un modelo específico y calcula sus métricas.
    
    Args:
        nombre_modelo (str): Nombre del modelo
        modelo: Instancia del modelo a entrenar
        X_train, y_train: Datos de entrenamiento
        X_test, y_test: Datos de prueba
        tipo_problema (str): 'clasificacion' o 'regresion'
        
    Returns:
        Dict: Diccionario con resultados del entrenamiento
    """
    resultado = {
        'nombre': nombre_modelo,
        'entrenado': False,
        'error': None,
        'tiempo_entrenamiento': 0,
        'metricas': {},
        'modelo_objeto': None  # Almacenaremos el modelo entrenado aquí
    }
    
    try:
        # Registrar tiempo de inicio
        tiempo_inicio = time.time()
        
        # Entrenar modelo
        modelo.fit(X_train, y_train)
        
        # Guardar el modelo entrenado
        resultado['modelo_objeto'] = modelo
        
        # Calcular tiempo de entrenamiento
        tiempo_fin = time.time()
        resultado['tiempo_entrenamiento'] = tiempo_fin - tiempo_inicio
        
        # Hacer predicciones
        y_pred = modelo.predict(X_test)
        
        # Calcular métricas según el tipo de problema
        if tipo_problema == 'clasificacion':
            # Clasificación binaria o multiclase
            resultado['metricas'] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
            
            # Calcular cross-validation
            cv_scores = cross_val_score(modelo, X_train, y_train, cv=5, scoring='accuracy')
            resultado['metricas']['cv_score_media'] = cv_scores.mean()
            resultado['metricas']['cv_score_std'] = cv_scores.std()
            
        else:  # regresión
            resultado['metricas'] = {
                'r2': r2_score(y_test, y_pred),
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred)
            }
            
            # Calcular cross-validation
            cv_scores = cross_val_score(modelo, X_train, y_train, cv=5, scoring='r2')
            resultado['metricas']['cv_score_media'] = cv_scores.mean()
            resultado['metricas']['cv_score_std'] = cv_scores.std()
        
        resultado['entrenado'] = True
        
    except Exception as e:
        resultado['error'] = str(e)
        logger.error(f"Error al entrenar modelo {nombre_modelo}: {str(e)}")
    
    return resultado

def ejecutar_benchmarking(
    X: pd.DataFrame, 
    y: pd.Series, 
    tipo_problema: Optional[str] = None,
    test_size: float = 0.2,
    id_usuario: int = 1,
    db_path: Optional[str] = None
) -> Dict:
    """
    Ejecuta un benchmarking de múltiples modelos de ML.
    
    Args:
        X (pd.DataFrame): Variables predictoras
        y (pd.Series): Variable objetivo
        tipo_problema (str, opcional): Si no se especifica, se detecta automáticamente
        test_size (float): Tamaño del conjunto de prueba
        id_usuario (int): ID del usuario que realiza la acción
        db_path (str, opcional): Ruta a la base de datos
        
    Returns:
        Dict: Resultados del benchmarking
    """
    # Detectar tipo de problema si no se especificó
    if tipo_problema is None:
        tipo_problema = detectar_tipo_problema(y)
    
    logger.info(f"Iniciando benchmarking para problema de {tipo_problema}")
    
    # Preprocesar los datos para que sean compatibles con ML
    logger.info("Preprocesando datos para ML")
    X_preprocesado = preparar_datos_para_ml(X)
    
    # Preparar datos
    X_train, X_test, y_train, y_test, le = preparar_datos(
        X_preprocesado, y, tipo_problema, test_size
    )
    
    # Seleccionar los modelos según el tipo de problema
    modelos = CLASSIFICATION_MODELS if tipo_problema == 'clasificacion' else REGRESSION_MODELS
    
    # Resultados del benchmarking
    resultados = {
        'tipo_problema': tipo_problema,
        'modelos_exitosos': [],
        'modelos_fallidos': [],
        'mejor_modelo': None,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_modelos': len(modelos),
        'variable_objetivo': y.name,
        'total_filas': len(X),
        'porcentaje_test': test_size * 100,
        # Guardar datos de prueba para visualizaciones avanzadas
        'X_test': X_test,
        'y_test': y_test,
        'columnas_originales': X.columns.tolist(),
        'tiene_label_encoder': le is not None
    }
    
    # Guardar label encoder si existe
    if le is not None:
        resultados['clases'] = le.classes_.tolist()
    
    # Entrenar cada modelo y capturar errores individualmente
    for nombre, modelo in modelos.items():
        logger.info(f"Entrenando modelo: {nombre}")
        
        try:
            resultado = entrenar_modelo(
                nombre, modelo, X_train, y_train, X_test, y_test, tipo_problema
            )
            
            # Agregar a lista de exitosos o fallidos
            if resultado['entrenado']:
                resultados['modelos_exitosos'].append(resultado)
            else:
                resultados['modelos_fallidos'].append(resultado)
                logger.warning(f"El modelo {nombre} no se pudo entrenar: {resultado['error']}")
        except Exception as e:
            # Capturar cualquier error no manejado durante el entrenamiento
            error_msg = f"Error no manejado en modelo {nombre}: {str(e)}"
            logger.error(error_msg)
            resultados['modelos_fallidos'].append({
                'nombre': nombre,
                'entrenado': False,
                'error': error_msg,
                'tiempo_entrenamiento': 0,
                'metricas': {}
            })
    
    # Ordenar modelos según métrica principal
    if resultados['modelos_exitosos']:
        if tipo_problema == 'clasificacion':
            # Ordenar por accuracy descendente
            resultados['modelos_exitosos'] = sorted(
                resultados['modelos_exitosos'], 
                key=lambda x: x['metricas']['accuracy'], 
                reverse=True
            )
        else:
            # Ordenar por R² descendente
            resultados['modelos_exitosos'] = sorted(
                resultados['modelos_exitosos'], 
                key=lambda x: x['metricas']['r2'], 
                reverse=True
            )
        
        # Guardar el mejor modelo
        resultados['mejor_modelo'] = resultados['modelos_exitosos'][0]
    
    # Guardar resultados en la base de datos
    guardar_resultados_benchmarking(resultados, id_usuario, db_path)
    
    return resultados

def guardar_resultados_benchmarking(
    resultados: Dict, 
    id_usuario: int = 1,
    db_path: Optional[str] = None
) -> int:
    """
    Guarda los resultados del benchmarking en la base de datos.
    
    Args:
        resultados (Dict): Resultados del benchmarking
        id_usuario (int): ID del usuario
        db_path (str, opcional): Ruta a la base de datos
        
    Returns:
        int: ID del benchmarking guardado
    """
    conn = None
    benchmarking_id = 0  # Valor por defecto
    
    # Crear una copia del diccionario para no modificar el original
    resultados_serializables = resultados.copy()
    
    # Convertir elementos no serializables a formatos serializables
    if 'X_test' in resultados_serializables:
        # Convertir numpy array a lista para serialización
        X_test = resultados_serializables['X_test']
        if sp.issparse(X_test):
            # Si es una matriz dispersa, convertirla a formato de diccionario
            X_test = X_test.toarray()
        resultados_serializables['X_test'] = X_test.tolist() if hasattr(X_test, 'tolist') else None
    
    if 'y_test' in resultados_serializables:
        # Convertir numpy array a lista para serialización
        y_test = resultados_serializables['y_test']
        resultados_serializables['y_test'] = y_test.tolist() if hasattr(y_test, 'tolist') else None
    
    # Usar el serializador de modelos para manejar los objetos modelo
    resultados_serializables = serializar_modelos_benchmarking(resultados_serializables)
    
    try:
        # Determinar la ruta de la base de datos
        if db_path is None:
            db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'analitica_farma.db')
        
        # Conectar a la base de datos
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Verificar si existe la tabla de benchmarking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS benchmarking_modelos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                id_usuario INTEGER NOT NULL,
                tipo_problema TEXT NOT NULL,
                variable_objetivo TEXT NOT NULL,
                cantidad_modelos_exitosos INTEGER NOT NULL,
                cantidad_modelos_fallidos INTEGER NOT NULL,
                mejor_modelo TEXT NOT NULL,
                resultados_completos TEXT NOT NULL,
                fecha_ejecucion TIMESTAMP NOT NULL
            )        """)
        
        # Preparar datos para inserción
        resultados_json = json.dumps(resultados_serializables)
        
        # Insertar los resultados
        cursor.execute("""
            INSERT INTO benchmarking_modelos (
                id_usuario, tipo_problema, variable_objetivo, 
                cantidad_modelos_exitosos, cantidad_modelos_fallidos,
                mejor_modelo, resultados_completos, fecha_ejecucion
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)        """, (
            id_usuario,
            resultados_serializables['tipo_problema'],
            resultados_serializables['variable_objetivo'],
            len(resultados_serializables['modelos_exitosos']),
            len(resultados_serializables['modelos_fallidos']),
            resultados_serializables['mejor_modelo']['nombre'] if resultados_serializables['mejor_modelo'] else '',
            resultados_json,
            resultados_serializables['timestamp']        ))
        
        # Obtener el ID del benchmarking insertado
        benchmarking_id = cursor.lastrowid or 0
        
        # Registrar en la tabla de auditoría
        cursor.execute("""
            INSERT INTO auditoria (
                id_usuario, accion, descripcion, fecha
            ) VALUES (?, ?, ?, ?)
        """, (
            id_usuario,
            "BENCHMARKING_MODELOS",
            f"Benchmarking de modelos completado: {resultados['tipo_problema']} - {resultados['variable_objetivo']}",
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')        ))
        
        # Confirmar cambios
        conn.commit()
        logger.info(f"Resultados de benchmarking guardados con ID {benchmarking_id}")
        
        return benchmarking_id
    
    except Exception as e:
        logger.error(f"Error al guardar resultados de benchmarking: {str(e)}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

def obtener_ultimo_benchmarking(
    id_usuario: Optional[int] = None,
    db_path: Optional[str] = None
) -> Optional[Dict]:
    """
    Obtiene los resultados del último benchmarking realizado.
    
    Args:
        id_usuario (int, opcional): ID del usuario para filtrar
        db_path (str, opcional): Ruta a la base de datos
        
    Returns:
        Dict: Resultados del benchmarking o None si no hay
    """
    conn = None
    try:
        # Determinar la ruta de la base de datos
        if db_path is None:
            db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'analitica_farma.db')
        
        # Conectar a la base de datos
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # Para acceder a columnas por nombre
        cursor = conn.cursor()
        
        # Construir la consulta según los parámetros
        query = "SELECT * FROM benchmarking_modelos WHERE 1=1"
        params = []
        
        if id_usuario:
            query += " AND id_usuario = ?"
            params.append(id_usuario)
        
        query += " ORDER BY fecha_ejecucion DESC LIMIT 1"
        
        # Ejecutar la consulta
        cursor.execute(query, params)
        resultado = cursor.fetchone()
        
        if resultado:
            # Convertir a diccionario
            return json.loads(resultado['resultados_completos'])
        
        return None
    
    except Exception as e:
        logger.error(f"Error al obtener benchmarking: {str(e)}")
        return None
    finally:
        if conn:
            conn.close()

def obtener_benchmarking_por_id(
    benchmarking_id: int,
    db_path: Optional[str] = None,
    deserializar: bool = True
) -> Optional[Dict]:
    """
    Obtiene los resultados de un benchmarking específico por su ID.
    
    Args:
        benchmarking_id (int): ID del benchmarking a obtener
        db_path (str, opcional): Ruta a la base de datos
        deserializar (bool): Si True, deserializa los objetos modelo
        
    Returns:
        Dict: Resultados del benchmarking o None si no existe
    """
    conn = None
    try:
        # Determinar la ruta de la base de datos
        if db_path is None:
            db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'analitica_farma.db')
        
        # Conectar a la base de datos
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # Para acceder a columnas por nombre
        cursor = conn.cursor()
        
        # Ejecutar la consulta
        cursor.execute(
            "SELECT resultados_completos FROM benchmarking_modelos WHERE id = ?", 
            (benchmarking_id,)
        )
        
        resultado = cursor.fetchone()
        
        if resultado:
            # Convertir a diccionario
            resultados = json.loads(resultado['resultados_completos'])
            
            # Deserializar los modelos si está activada la opción
            if deserializar:
                resultados = deserializar_modelos_benchmarking(resultados)
                logger.info(f"Modelos deserializados para benchmarking ID {benchmarking_id}")
            
            return resultados
        
        return None
    
    except Exception as e:
        logger.error(f"Error al obtener benchmarking por ID {benchmarking_id}: {str(e)}")
        return None
    finally:
        if conn:
            conn.close()

def preparar_datos_para_ml(X: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesa el DataFrame para hacerlo compatible con los algoritmos de ML y SHAP,
    detectando y transformando columnas de fecha, texto y categóricas.
    Asegura que el resultado final sea completamente numérico.
    
    Args:
        X (pd.DataFrame): DataFrame con variables predictoras
        
    Returns:
        pd.DataFrame: DataFrame preprocesado listo para ML y SHAP
    """
    X_preprocesado = X.copy()

    # Identificar columnas de fecha y convertirlas a características numéricas
    for columna in X_preprocesado.columns:
        # Detectar columnas de tipo datetime
        if pd.api.types.is_datetime64_any_dtype(X_preprocesado[columna]):
            logger.info(f"Procesando columna datetime: {columna}")
            
            # Extraer componentes útiles de la fecha
            X_preprocesado[f"{columna}_año"] = X_preprocesado[columna].dt.year
            X_preprocesado[f"{columna}_mes"] = X_preprocesado[columna].dt.month
            X_preprocesado[f"{columna}_dia"] = X_preprocesado[columna].dt.day
            
            # Eliminar columna original de datetime
            X_preprocesado = X_preprocesado.drop(columns=[columna])
            
        # Intentar convertir columnas de texto con formato de fecha
        elif X_preprocesado[columna].dtype == 'object':
            # Comprobar si parece una fecha
            try:
                # Intentar parsear como fecha una muestra de valores
                muestra = X_preprocesado[columna].dropna().head(5)
                fechas_validas = True
                
                for valor in muestra:
                    try:
                        pd.to_datetime(valor)
                    except Exception:
                        fechas_validas = False
                        break
                
                if fechas_validas:
                    logger.info(f"Convirtiendo columna con formato de fecha: {columna}")
                    # Convertir a datetime y extraer componentes
                    fechas = pd.to_datetime(X_preprocesado[columna], errors='coerce')
                    
                    # Solo procesar si se pudieron convertir la mayoría de los valores
                    if fechas.isna().mean() < 0.3:  # Si menos del 30% son NaN
                        X_preprocesado[f"{columna}_año"] = fechas.dt.year
                        X_preprocesado[f"{columna}_mes"] = fechas.dt.month
                        X_preprocesado[f"{columna}_dia"] = fechas.dt.day
                        
                        # Eliminar columna original
                        X_preprocesado = X_preprocesado.drop(columns=[columna])
            except Exception as e:
                logger.warning(f"No se pudo convertir la columna {columna} a fecha: {str(e)}")
    
    # Convertir columnas categóricas (object/category) a numéricas usando one-hot encoding
    columnas_objeto = X_preprocesado.select_dtypes(include=['object', 'category']).columns
    
    for columna in columnas_objeto:
        logger.info(f"Aplicando one-hot encoding a columna categórica: {columna}")
        # Limitar cardinalidad a 10 categorías más frecuentes
        if X_preprocesado[columna].nunique() > 10:
            top_categorias = X_preprocesado[columna].value_counts().nlargest(10).index
            X_preprocesado[columna] = X_preprocesado[columna].apply(
                lambda x: x if x in top_categorias else 'OTROS'
            )
        
        # Aplicar one-hot encoding
        dummies = pd.get_dummies(
            X_preprocesado[columna], 
            prefix=columna, 
            drop_first=True,  # Evitar multicolinealidad
            dummy_na=True  # Considerar NaN como categoría
        )
        
        # Unir los dummies al DataFrame
        X_preprocesado = pd.concat([X_preprocesado, dummies], axis=1)
        
        # Eliminar columna original
        X_preprocesado = X_preprocesado.drop(columns=[columna])
    
    # Eliminar columnas que no sean numéricas (por seguridad)
    columnas_no_numericas = X_preprocesado.select_dtypes(exclude=[np.number]).columns
    if len(columnas_no_numericas) > 0:
        logger.warning(f"Eliminando columnas no numéricas no convertidas: {list(columnas_no_numericas)}")
        X_preprocesado = X_preprocesado.drop(columns=columnas_no_numericas)

    # Manejar valores faltantes (NaN)
    X_preprocesado = X_preprocesado.fillna(X_preprocesado.mean(numeric_only=True))
    X_preprocesado = X_preprocesado.fillna(0)

    return X_preprocesado

def cargar_modelo_entrenado(modelo_id: Optional[str] = None, listar: bool = False, id_usuario: Optional[int] = None, db_path: Optional[str] = None):
    """
    Lista modelos entrenados disponibles o carga un modelo específico por nombre.
    Args:
        modelo_id (str): Nombre del modelo a cargar (opcional)
        listar (bool): Si True, retorna un diccionario de modelos disponibles
        id_usuario (int): Filtra por usuario (opcional)
        db_path (str): Ruta a la base de datos (opcional)
    Returns:
        Si listar: dict {nombre: {'nombre': str, ...}}
        Si modelo_id: modelo deserializado
    """
    try:
        # Obtener el último benchmarking (puede mejorarse para listar todos)
        resultados = obtener_ultimo_benchmarking(id_usuario=id_usuario, db_path=db_path)
        if not resultados:
            logger.warning("No se encontraron resultados de benchmarking.")
            return {} if listar else None
        resultados = deserializar_modelos_benchmarking(resultados)
        modelos = resultados.get('modelos_exitosos', [])
        modelos_dict = {m['nombre']: {'nombre': m['nombre'], 'score': m.get('score', None)} for m in modelos if 'nombre' in m}
        if listar:
            return modelos_dict
        if modelo_id is not None:
            modelo = next((m for m in modelos if m.get('nombre') == modelo_id), None)
            if modelo and 'modelo_objeto' in modelo:
                return modelo['modelo_objeto']
            else:
                logger.error(f"No se encontró el modelo con nombre {modelo_id}.")
                return None
        return None
    except Exception as e:
        logger.error(f"Error en cargar_modelo_entrenado: {str(e)}")
        return {} if listar else None
