"""
Función auxiliar para verificar la disponibilidad de objetos modelo
en los resultados de benchmarking.
"""
import streamlit as st

def diagnosticar_objetos_modelo(resultados_benchmarking):
    """
    Verifica si los objetos modelo están disponibles en los resultados del benchmarking.
    
    Args:
        resultados_benchmarking: Resultados completos del benchmarking
        
    Returns:
        bool: True si al menos un modelo tiene su objeto, False en caso contrario
    """
    if not resultados_benchmarking or not resultados_benchmarking.get('modelos_exitosos'):
        return False
    
    modelos_con_objeto = 0
    modelos_totales = len(resultados_benchmarking['modelos_exitosos'])
    
    for modelo in resultados_benchmarking['modelos_exitosos']:
        if 'modelo_objeto' in modelo:
            modelos_con_objeto += 1
    
    # Mostrar información sobre los objetos modelo
    st.info(f"Diagnóstico de objetos modelo: {modelos_con_objeto}/{modelos_totales} modelos tienen objeto disponible.")
    
    return modelos_con_objeto > 0
