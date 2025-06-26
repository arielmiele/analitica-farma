"""
Módulo de UI para Validación Cruzada - Analítica Farma
Contiene funciones modulares para la interfaz de validación cruzada y diagnóstico de modelos.
"""

from .educativo import mostrar_introduccion, mostrar_importancia_validacion
from .configuracion import seleccionar_modelo, configurar_validacion
from .analisis import ejecutar_analisis_completo, realizar_analisis_validacion
from .visualizacion import (
    mostrar_resultados_analisis,
    mostrar_diagnostico_principal,
    mostrar_curvas_aprendizaje_interactivas,
    mostrar_interpretacion_detallada
)
from .recomendaciones import (
    mostrar_recomendaciones_mejora,
    mostrar_recomendaciones_industria_ui
)

__all__ = [
    # Módulo educativo
    "mostrar_introduccion",
    "mostrar_importancia_validacion",
    
    # Módulo configuración
    "seleccionar_modelo",
    "configurar_validacion",
    
    # Módulo análisis
    "ejecutar_analisis_completo",
    "realizar_analisis_validacion",
    
    # Módulo visualización
    "mostrar_resultados_analisis",
    "mostrar_diagnostico_principal",
    "mostrar_curvas_aprendizaje_interactivas",
    "mostrar_interpretacion_detallada",
    
    # Módulo recomendaciones
    "mostrar_recomendaciones_mejora",
    "mostrar_recomendaciones_industria_ui"
]
