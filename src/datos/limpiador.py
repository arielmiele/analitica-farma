import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Literal
from src.audit.logger import log_audit

def detectar_duplicados(df: pd.DataFrame, id_sesion: str, usuario: str, columnas: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Detecta registros duplicados en un DataFrame basado en columnas específicas.
    
    Args:
        df: DataFrame a analizar
        id_sesion: ID de sesión para trazabilidad
        usuario: Usuario que ejecuta la acción
        columnas: Lista de columnas para buscar duplicados. Si es None, usa todas las columnas.
    
    Returns:
        Diccionario con información sobre duplicados encontrados:
        - cantidad: Número de filas duplicadas
        - porcentaje: Porcentaje de filas duplicadas respecto al total
        - grupos_duplicados: DataFrame con información sobre los grupos de duplicados
        - indices: Índices de las filas duplicadas (excepto la primera ocurrencia)
    """
    try:
        # Si no se especifican columnas, usar todas
        if columnas is None or len(columnas) == 0:
            columnas = df.columns.tolist()
        
        # Verificar que las columnas existan en el DataFrame
        columnas_existentes = [col for col in columnas if col in df.columns]
        if len(columnas_existentes) == 0:
            log_audit(
                accion="error_columnas_duplicados",
                entidad="limpiador",
                id_entidad=None,
                detalles="No se encontraron las columnas especificadas en el DataFrame",
                id_sesion=id_sesion,
                usuario=usuario
            )
            return {
                'error': 'No se encontraron las columnas especificadas en el DataFrame',
                'cantidad': 0,
                'porcentaje': 0,
                'grupos_duplicados': pd.DataFrame(),
                'indices': []
            }
        
        # Identificar duplicados
        duplicados = df.duplicated(subset=columnas_existentes, keep='first')
        cantidad_duplicados = duplicados.sum()
        porcentaje_duplicados = (cantidad_duplicados / len(df)) * 100
        
        # Obtener índices de duplicados (excluyendo la primera ocurrencia)
        indices_duplicados = df[duplicados].index.tolist()
        
        # Crear DataFrame con información de grupos de duplicados
        grupos = pd.DataFrame()
        
        if cantidad_duplicados > 0:
            # Agrupar por las columnas seleccionadas y contar ocurrencias
            conteo_grupo = df.groupby(columnas_existentes).size().reset_index(name='conteo')
            # Filtrar solo los grupos con más de una ocurrencia
            grupos = conteo_grupo[conteo_grupo['conteo'] > 1].sort_values('conteo', ascending=False)
            log_audit(
                accion="duplicados_detectados",
                entidad="limpiador",
                id_entidad=None,
                detalles=f"{cantidad_duplicados} duplicados detectados en columnas {columnas_existentes}",
                id_sesion=id_sesion,
                usuario=usuario
            )
        else:
            log_audit(
                accion="sin_duplicados",
                entidad="limpiador",
                id_entidad=None,
                detalles=f"No se detectaron duplicados en columnas {columnas_existentes}",
                id_sesion=id_sesion,
                usuario=usuario
            )
        
        return {
            'cantidad': cantidad_duplicados,
            'porcentaje': porcentaje_duplicados,
            'grupos_duplicados': grupos,
            'indices': indices_duplicados
        }
    except Exception as e:
        log_audit(
            accion="error_detectar_duplicados",
            entidad="limpiador",
            id_entidad=None,
            detalles=f"Error al detectar duplicados: {str(e)}",
            id_sesion=id_sesion,
            usuario=usuario
        )
        return {
            'error': str(e),
            'cantidad': 0,
            'porcentaje': 0,
            'grupos_duplicados': pd.DataFrame(),
            'indices': []
        }

def eliminar_duplicados(df: pd.DataFrame, id_sesion: str, usuario: str, columnas: Optional[List[str]] = None, keep: Literal['first', 'last', False] = 'first') -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Elimina registros duplicados del DataFrame basados en columnas específicas.
    
    Args:
        df: DataFrame a limpiar
        id_sesion: ID de sesión para trazabilidad
        usuario: Usuario que ejecuta la acción
        columnas: Lista de columnas para buscar duplicados. Si es None, usa todas las columnas.
        keep: Qué duplicados conservar ('first', 'last', False). Si es False, elimina todas las ocurrencias.
    
    Returns:
        Tuple con:
        - DataFrame limpio
        - Diccionario con información sobre la operación:
          - filas_antes: Número de filas antes de la eliminación
          - filas_despues: Número de filas después de la eliminación
          - filas_eliminadas: Número de filas eliminadas
          - porcentaje_reduccion: Porcentaje de reducción
    """
    try:
        # Si no se especifican columnas, usar todas
        if columnas is None or len(columnas) == 0:
            columnas = df.columns.tolist()
        
        # Verificar que las columnas existan en el DataFrame
        columnas_existentes = [col for col in columnas if col in df.columns]
        if len(columnas_existentes) == 0:
            log_audit(
                accion="error_columnas_eliminar_duplicados",
                entidad="limpiador",
                id_entidad=None,
                detalles="No se encontraron las columnas especificadas en el DataFrame",
                id_sesion=id_sesion,
                usuario=usuario
            )
            return df.copy(), {
                'error': 'No se encontraron las columnas especificadas en el DataFrame',
                'filas_antes': len(df),
                'filas_despues': len(df),
                'filas_eliminadas': 0,
                'porcentaje_reduccion': 0.0
            }
        
        # Guardar cantidad de filas antes de la eliminación
        filas_antes = len(df)
        
        # Eliminar duplicados
        df_limpio = df.drop_duplicates(subset=columnas_existentes, keep=keep)
        
        # Calcular métricas
        filas_despues = len(df_limpio)
        filas_eliminadas = filas_antes - filas_despues
        porcentaje_reduccion = (filas_eliminadas / filas_antes) * 100 if filas_antes > 0 else 0
        
        log_audit(
            accion="eliminar_duplicados",
            entidad="limpiador",
            id_entidad=None,
            detalles=f"Eliminadas {filas_eliminadas} filas duplicadas en columnas {columnas_existentes} (estrategia: {keep})",
            id_sesion=id_sesion,
            usuario=usuario
        )
        
        return df_limpio, {
            'filas_antes': filas_antes,
            'filas_despues': filas_despues,
            'filas_eliminadas': filas_eliminadas,
            'porcentaje_reduccion': porcentaje_reduccion,
            'columnas_usadas': columnas_existentes,
            'estrategia': keep
        }
    except Exception as e:
        log_audit(
            accion="error_eliminar_duplicados",
            entidad="limpiador",
            id_entidad=None,
            detalles=f"Error al eliminar duplicados: {str(e)}",
            id_sesion=id_sesion,
            usuario=usuario
        )
        return df.copy(), {
            'error': str(e),
            'filas_antes': len(df),
            'filas_despues': len(df),
            'filas_eliminadas': 0,
            'porcentaje_reduccion': 0.0
        }

def fusionar_duplicados(df: pd.DataFrame, id_sesion: str, usuario: str, columnas: List[str], metodo: Optional[Dict[str, str]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fusiona registros duplicados aplicando diferentes métodos de agregación por columna.
    
    Args:
        df: DataFrame a procesar
        id_sesion: ID de sesión para trazabilidad
        usuario: Usuario que ejecuta la acción
        columnas: Lista de columnas que definen los grupos de duplicados
        metodo: Diccionario donde las claves son nombres de columnas y los valores son métodos de agregación
    
    Returns:
        Tuple con:
        - DataFrame con duplicados fusionados
        - Diccionario con información sobre la operación
    """
    try:
        # Si no se especifican columnas, usar todas
        if columnas is None or len(columnas) == 0:
            log_audit(
                accion="error_columnas_fusionar_duplicados",
                entidad="limpiador",
                id_entidad=None,
                detalles="No se especificaron columnas para agrupar",
                id_sesion=id_sesion,
                usuario=usuario
            )
            return df.copy(), {
                'error': 'No se especificaron columnas para agrupar',
                'filas_antes': len(df),
                'filas_despues': len(df),
                'grupos_fusionados': 0
            }
        
        # Verificar que las columnas existan en el DataFrame
        columnas_existentes = [col for col in columnas if col in df.columns]
        if len(columnas_existentes) == 0:
            log_audit(
                accion="error_columnas_fusionar_duplicados",
                entidad="limpiador",
                id_entidad=None,
                detalles="No se encontraron las columnas especificadas en el DataFrame",
                id_sesion=id_sesion,
                usuario=usuario
            )
            return df.copy(), {
                'error': 'No se encontraron las columnas especificadas en el DataFrame',
                'filas_antes': len(df),
                'filas_despues': len(df),
                'grupos_fusionados': 0
            }
        
        # Guardar cantidad de filas antes de la fusión
        filas_antes = len(df)
        
        # Si no se especifica método, usar 'first' para todas las columnas
        if metodo is None:
            metodo = {}
        
        # Construir diccionario de agregación
        agg_dict = {}
        for col in df.columns:
            if col in columnas_existentes:
                continue  # Las columnas de agrupación no necesitan método
            
            # Usar el método especificado o 'first' por defecto
            agg_dict[col] = metodo.get(col, 'first')
        
        # Agrupar y agregar
        df_fusionado = df.groupby(columnas_existentes, as_index=False).agg(agg_dict)
        
        # Calcular métricas
        filas_despues = len(df_fusionado)
        grupos_fusionados = filas_antes - filas_despues
        
        log_audit(
            accion="fusionar_duplicados",
            entidad="limpiador",
            id_entidad=None,
            detalles=f"Fusionados {grupos_fusionados} grupos de duplicados en columnas {columnas_existentes}",
            id_sesion=id_sesion,
            usuario=usuario
        )
        
        return df_fusionado, {
            'filas_antes': filas_antes,
            'filas_despues': filas_despues,
            'grupos_fusionados': grupos_fusionados,
            'columnas_agrupacion': columnas_existentes,
            'metodos_fusion': metodo
        }
    except Exception as e:
        log_audit(
            accion="error_fusionar_duplicados",
            entidad="limpiador",
            id_entidad=None,
            detalles=f"Error al fusionar duplicados: {str(e)}",
            id_sesion=id_sesion,
            usuario=usuario
        )
        return df.copy(), {
            'error': str(e),
            'filas_antes': len(df),
            'filas_despues': len(df),
            'grupos_fusionados': 0
        }
