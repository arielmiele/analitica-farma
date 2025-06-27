"""
Módulo educativo para Validación Cruzada - Analítica Farma
Contiene funciones para mostrar información educativa sobre overfitting, underfitting y validación cruzada.
"""

import streamlit as st


def mostrar_introduccion():
    """Muestra la introducción teórica sobre validación cruzada y overfitting."""
    st.markdown("""
    ## 🎯 ¿Qué es el Overfitting y Underfitting?
    
    ### Overfitting (Sobreajuste)
    Ocurre cuando un modelo aprende demasiado bien los datos de entrenamiento, memorizando 
    incluso el ruido y las peculiaridades específicas del conjunto de entrenamiento. Esto 
    resulta en un excelente rendimiento en datos de entrenamiento pero pobre generalización 
    a datos nuevos.
    
    **Características del Overfitting:**
    - ✅ Alta precisión en conjunto de entrenamiento
    - ❌ Baja precisión en conjunto de validación/prueba
    - 📈 Gran diferencia (gap) entre ambos rendimientos
    
    ### Underfitting (Subajuste)
    Ocurre cuando un modelo es demasiado simple para capturar los patrones subyacentes 
    en los datos. El modelo no logra aprender adecuadamente ni de los datos de entrenamiento 
    ni de validación.
    
    **Características del Underfitting:**
    - ❌ Baja precisión en conjunto de entrenamiento
    - ❌ Baja precisión en conjunto de validación/prueba
    - 📊 Rendimiento similar pero insuficiente en ambos conjuntos
    
    ### Modelo Balanceado (Objetivo)
    Un modelo bien balanceado logra un buen rendimiento tanto en entrenamiento como en 
    validación, con una diferencia mínima entre ambos.
    
    **Características del Modelo Balanceado:**
    - ✅ Buena precisión en conjunto de entrenamiento
    - ✅ Buena precisión en conjunto de validación/prueba
    - 🎯 Diferencia mínima entre ambos rendimientos
    """)


def mostrar_importancia_validacion():
    """Explica por qué es importante la validación cruzada."""
    with st.expander("🔬 ¿Por qué es importante este análisis?", expanded=False):
        st.markdown("""
        ### Importancia en la Industria Farmacéutica
        
        En procesos biotecnológicos, la **confiabilidad** y **reproducibilidad** de los modelos 
        predictivos es crítica por varios motivos:
        
        #### 1. **Cumplimiento Regulatorio**
        - Los modelos deben ser **validados** y **documentados** según estándares como GMP y FDA
        - La **trazabilidad** del rendimiento del modelo es esencial para auditorías
        - Se requiere **evidencia estadística** de que el modelo generalizará correctamente
        
        #### 2. **Costos de Producción**
        - Un modelo con **overfitting** puede predecir incorrectamente condiciones óptimas
        - Esto puede resultar en **lotes fallidos** o **baja calidad del producto**
        - Los **costos de re-procesamiento** pueden ser de millones de dólares
        
        #### 3. **Seguridad del Paciente**
        - Los modelos predictivos influyen en la **calidad del producto final**
        - Un modelo mal generalizado puede afectar la **eficacia** o **seguridad** del medicamento
        - La **consistencia entre lotes** es fundamental para la seguridad del paciente
        
        #### 4. **Optimización de Procesos**
        - Un modelo balanceado permite **mejora continua** confiable
        - Facilita la **transferencia de tecnología** entre sitios de producción
        - Permite **escalamiento** seguro de procesos piloto a producción comercial
        
        ### Beneficios de las Curvas de Aprendizaje
        
        - **📊 Visualización clara** del comportamiento del modelo
        - **🔍 Detección temprana** de problemas de generalización
        - **💡 Recomendaciones específicas** para mejorar el rendimiento
        - **📝 Documentación automática** para cumplimiento regulatorio
        """)
