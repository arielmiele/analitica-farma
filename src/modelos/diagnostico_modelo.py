"""
Función auxiliar para verificar la disponibilidad de objetos modelo
en los resultados de benchmarking.
"""
from src.audit.logger import log_audit

def diagnosticar_objetos_modelo(resultados_benchmarking, id_sesion, usuario):
    """
    Verifica si los objetos modelo están disponibles en los resultados del benchmarking.
    
    Args:
        resultados_benchmarking: Resultados completos del benchmarking
        id_sesion (str): ID de sesión para trazabilidad
        usuario (str): Usuario que ejecuta la acción
        
    Returns:
        dict: {'disponible': bool, 'modelos_con_objeto': int, 'modelos_totales': int, 'mensaje': str}
    """
    if not resultados_benchmarking or not resultados_benchmarking.get('modelos_exitosos'):
        log_audit(
            usuario=usuario,
            accion="DIAGNOSTICO_MODELO_SIN_RESULTADOS",
            entidad="diagnostico_modelo",
            id_entidad="N/A",
            detalles="No hay resultados de benchmarking o no hay modelos exitosos.",
            id_sesion=id_sesion
        )
        return {'disponible': False, 'modelos_con_objeto': 0, 'modelos_totales': 0, 'mensaje': "No hay modelos exitosos."}
    
    modelos_con_objeto = 0
    modelos_totales = len(resultados_benchmarking['modelos_exitosos'])
    
    for modelo in resultados_benchmarking['modelos_exitosos']:
        if 'modelo_objeto' in modelo:
            modelos_con_objeto += 1
    
    mensaje = f"Diagnóstico de objetos modelo: {modelos_con_objeto}/{modelos_totales} modelos tienen objeto disponible."
    
    log_audit(
        usuario=usuario,
        accion="DIAGNOSTICO_MODELO",
        entidad="diagnostico_modelo",
        id_entidad="N/A",
        detalles=mensaje,
        id_sesion=id_sesion
    )
    
    return {
        'disponible': modelos_con_objeto > 0,
        'modelos_con_objeto': modelos_con_objeto,
        'modelos_totales': modelos_totales,
        'mensaje': mensaje
    }
