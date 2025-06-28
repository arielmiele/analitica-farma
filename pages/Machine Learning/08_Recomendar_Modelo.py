"""
Página para recomendar y seleccionar el mejor modelo de machine learning.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# Importar módulos propios
from src.modelos.recomendador import recomendar_mejor_modelo, guardar_modelo_seleccionado
from src.state.session_manager import SessionManager
from src.audit.logger import Logger

# Inicializar gestor de sesión y logger
session = SessionManager()
logger = Logger("Recomendar_Modelo")

def main():
    st.title("👑 Recomendación de Modelo")
    
    # Verificar si hay resultados de benchmarking
    resultados_benchmarking = session.obtener_estado("resultados_benchmarking")
    
    if not resultados_benchmarking:
        st.warning("⚠️ No hay resultados de benchmarking disponibles.")
        st.info("👈 Vaya a la sección 'Entrenar Modelos' para ejecutar un benchmarking primero.")
        return
    
    # Sección de explicación
    with st.expander("ℹ️ Acerca de esta página", expanded=False):
        st.markdown("""
        ### Recomendación de Modelo
        
        Esta página analiza los resultados del benchmarking y recomienda el mejor modelo 
        para su conjunto de datos basado en diversas métricas.
        
        **Beneficios:**
        - Recomendación automática basada en múltiples criterios
        - Selección personalizada según sus necesidades específicas
        - Documentación y justificación de la elección
        - Persistencia de la selección para futuros análisis
        
        Puede aceptar la recomendación o seleccionar manualmente otro modelo.
        """)
    
    # Opciones de criterio para la recomendación
    st.subheader("🎯 Criterios de recomendación")
    
    # Determinar opciones según tipo de problema
    if resultados_benchmarking['tipo_problema'] == 'clasificacion':
        criterio_options = {
            'accuracy': 'Accuracy (Precisión global)',
            'f1': 'F1-Score (Balance entre precisión y exhaustividad)',
            'precision': 'Precision (Exactitud de los positivos)',
            'recall': 'Recall (Capacidad de encontrar todos los positivos)'
        }
        criterio_default = 'accuracy'
    else:  # regresión
        criterio_options = {
            'r2': 'R² (Coeficiente de determinación)',
            'rmse': 'RMSE (Error cuadrático medio)',
            'mae': 'MAE (Error absoluto medio)',
            'mse': 'MSE (Error cuadrático medio sin raíz)'
        }
        criterio_default = 'r2'
    
    criterio = st.radio(
        "Seleccione el criterio para recomendar el mejor modelo:",
        options=list(criterio_options.keys()),
        format_func=lambda x: criterio_options[x],
        index=list(criterio_options.keys()).index(criterio_default)
    )
    
    # Botón para ejecutar la recomendación
    if st.button("🔍 Obtener recomendación", type="primary"):
        with st.spinner("Analizando modelos..."):
            # Simular procesamiento
            time.sleep(1)
            
            try:
                # Obtener recomendación
                recomendacion = recomendar_mejor_modelo(criterio=criterio)
                
                # Guardar en la sesión
                session.guardar_estado("modelo_recomendado", recomendacion)
                
                # Registrar en el log
                logger.log_evento(
                    "RECOMENDACION_MODELO",
                    f"Recomendación de modelo con criterio: {criterio}",
                    "Recomendar_Modelo"
                )
                
                st.success("✅ Recomendación completada con éxito.")
                
                # Mostrar resultados
                mostrar_recomendacion(recomendacion, resultados_benchmarking)
                
            except Exception as e:
                st.error(f"❌ Error al obtener recomendación: {str(e)}")
                logger.log_evento(
                    "ERROR_RECOMENDACION",
                    f"Error en recomendación: {str(e)}",
                    "Recomendar_Modelo",
                    tipo="error"
                )
    
    # Si ya hay un modelo recomendado en la sesión, mostrarlo
    elif session.obtener_estado("modelo_recomendado"):
        recomendacion = session.obtener_estado("modelo_recomendado")
        st.info("🔍 Mostrando la última recomendación realizada.")
        mostrar_recomendacion(recomendacion, resultados_benchmarking)
    
    # Si no hay una recomendación específica, mostrar el mejor modelo del benchmarking
    else:
        st.info("ℹ️ Presione el botón para obtener una recomendación personalizada, o vea el mejor modelo del benchmarking a continuación.")
        
        # Mostrar el mejor modelo del benchmarking
        if resultados_benchmarking.get('mejor_modelo'):
            mejor_modelo = resultados_benchmarking['mejor_modelo']
            
            st.subheader("🏆 Mejor modelo del benchmarking")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Modelo:** {mejor_modelo['nombre']}")
                st.write(f"**Tipo de problema:** {resultados_benchmarking['tipo_problema'].capitalize()}")
                st.write(f"**Variable objetivo:** {resultados_benchmarking['variable_objetivo']}")
            
            with col2:
                # Mostrar métrica principal según tipo de problema
                if resultados_benchmarking['tipo_problema'] == 'clasificacion':
                    st.metric("Accuracy", f"{mejor_modelo['metricas']['accuracy']:.4f}")
                    st.metric("F1-Score", f"{mejor_modelo['metricas']['f1']:.4f}")
                else:
                    st.metric("R²", f"{mejor_modelo['metricas']['r2']:.4f}")
                    st.metric("RMSE", f"{mejor_modelo['metricas']['rmse']:.4f}")
            
            # Botón para seleccionar este modelo
            if st.button("✅ Seleccionar este modelo"):
                try:
                    # Guardar selección
                    resultado = guardar_modelo_seleccionado(
                        nombre_modelo=mejor_modelo['nombre'],
                        comentarios="Seleccionado automáticamente como mejor modelo del benchmarking."
                    )
                    
                    if resultado.get('exito'):
                        st.success(f"✅ {resultado['mensaje']}")
                        
                        # Registrar en el log
                        logger.log_evento(
                            "SELECCION_MODELO",
                            f"Modelo seleccionado: {mejor_modelo['nombre']}",
                            "Recomendar_Modelo"
                        )
                    else:
                        st.error(f"❌ Error al seleccionar modelo: {resultado.get('error')}")
                        
                except Exception as e:
                    st.error(f"❌ Error al seleccionar modelo: {str(e)}")
                    logger.log_evento(
                        "ERROR_SELECCION",
                        f"Error al seleccionar modelo: {str(e)}",
                        "Recomendar_Modelo",
                        tipo="error"
                    )

def mostrar_recomendacion(recomendacion, resultados_benchmarking):
    """
    Muestra la recomendación del modelo.
    
    Args:
        recomendacion: Diccionario con la recomendación
        resultados_benchmarking: Resultados completos del benchmarking
    """
    if recomendacion.get('error'):
        st.error(f"❌ Error: {recomendacion['error']}")
        return
    
    st.subheader("🏆 Modelo recomendado")
    
    modelo = recomendacion['modelo_recomendado']
    
    # Información principal
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Modelo:** {modelo['nombre']}")
        st.write(f"**Tipo de problema:** {recomendacion['tipo_problema'].capitalize()}")
        st.write(f"**Variable objetivo:** {recomendacion['variable_objetivo']}")
        st.write(f"**Criterio usado:** {recomendacion['criterio_usado']}")
    
    with col2:
        # Mostrar métrica principal según tipo de problema
        if recomendacion['tipo_problema'] == 'clasificacion':
            st.metric("Accuracy", f"{modelo['metricas']['accuracy']:.4f}")
            st.metric("F1-Score", f"{modelo['metricas']['f1']:.4f}")
        else:
            st.metric("R²", f"{modelo['metricas']['r2']:.4f}")
            st.metric("RMSE", f"{modelo['metricas']['rmse']:.4f}")
    
    # Justificación de la recomendación
    st.subheader("📝 Justificación")
    
    if recomendacion['tipo_problema'] == 'clasificacion':
        if recomendacion['criterio_usado'] == 'accuracy':
            st.write("""
            Este modelo ha sido recomendado porque tiene la mayor precisión global (accuracy) 
            entre todos los modelos evaluados. Esto significa que clasifica correctamente 
            la mayor proporción de instancias.
            """)
        elif recomendacion['criterio_usado'] == 'f1':
            st.write("""
            Este modelo ha sido recomendado porque tiene el mejor equilibrio entre precisión 
            y exhaustividad (F1-Score), lo que lo hace especialmente adecuado para conjuntos 
            de datos desbalanceados.
            """)
        elif recomendacion['criterio_usado'] == 'precision':
            st.write("""
            Este modelo ha sido recomendado porque tiene la mayor precisión en sus predicciones 
            positivas, minimizando los falsos positivos, lo que es importante cuando el costo 
            de un falso positivo es alto.
            """)
        elif recomendacion['criterio_usado'] == 'recall':
            st.write("""
            Este modelo ha sido recomendado porque tiene la mayor capacidad para encontrar 
            todos los casos positivos (recall), minimizando los falsos negativos, lo que es 
            crucial cuando no queremos perder ningún caso positivo.
            """)
    else:  # regresión
        if recomendacion['criterio_usado'] == 'r2':
            st.write("""
            Este modelo ha sido recomendado porque tiene el mayor coeficiente de determinación (R²), 
            lo que indica que explica la mayor proporción de la varianza en los datos.
            """)
        elif recomendacion['criterio_usado'] == 'rmse':
            st.write("""
            Este modelo ha sido recomendado porque tiene el menor error cuadrático medio (RMSE), 
            lo que significa que sus predicciones están más cerca de los valores reales, con 
            una penalización mayor para errores grandes.
            """)
        elif recomendacion['criterio_usado'] == 'mae':
            st.write("""
            Este modelo ha sido recomendado porque tiene el menor error absoluto medio (MAE), 
            lo que significa que sus predicciones están más cerca de los valores reales en 
            términos absolutos, sin penalizar especialmente los errores grandes.
            """)
        elif recomendacion['criterio_usado'] == 'mse':
            st.write("""
            Este modelo ha sido recomendado porque tiene el menor error cuadrático medio (MSE), 
            lo que significa que sus predicciones minimizan la suma de los errores al cuadrado.
            """)
    
    # Comparación con otros modelos
    st.subheader("📊 Comparación con otros modelos")
    
    # Crear DataFrame para comparación
    modelos_df = []
    
    # Añadir todos los modelos del benchmarking
    for m in resultados_benchmarking['modelos_exitosos']:
        # Determinar si es el modelo recomendado
        es_recomendado = (m['nombre'] == modelo['nombre'])
        
        # Preparar fila según tipo de problema
        if recomendacion['tipo_problema'] == 'clasificacion':
            modelos_df.append({
                'Modelo': m['nombre'],
                'Accuracy': m['metricas']['accuracy'],
                'F1-Score': m['metricas']['f1'],
                'Precision': m['metricas']['precision'],
                'Recall': m['metricas']['recall'],
                'Tiempo (s)': m['tiempo_entrenamiento'],
                'Recomendado': es_recomendado
            })
        else:
            modelos_df.append({
                'Modelo': m['nombre'],
                'R²': m['metricas']['r2'],
                'RMSE': m['metricas']['rmse'],
                'MAE': m['metricas']['mae'],
                'MSE': m['metricas']['mse'],
                'Tiempo (s)': m['tiempo_entrenamiento'],
                'Recomendado': es_recomendado
            })
    
    # Convertir a DataFrame
    df_comp = pd.DataFrame(modelos_df)
    
    # Ordenar según criterio seleccionado
    if recomendacion['criterio_usado'] in ['accuracy', 'f1', 'precision', 'recall', 'r2']:
        df_comp = df_comp.sort_values(
            by=recomendacion['criterio_usado'].capitalize() if recomendacion['criterio_usado'] != 'r2' else 'R²',
            ascending=False
        )
    else:  # 'rmse', 'mae', 'mse'
        df_comp = df_comp.sort_values(
            by=recomendacion['criterio_usado'].upper(),
            ascending=True
        )
    
    # Mostrar tabla
    st.dataframe(df_comp)
    
    # Visualización gráfica
    st.subheader("📈 Visualización")
    
    # Preparar datos para gráfico
    modelos = df_comp['Modelo'].tolist()
    
    if recomendacion['tipo_problema'] == 'clasificacion':
        # Para clasificación mostraremos accuracy, precision, recall, f1
        metricas = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        valores = [df_comp[m].tolist() for m in metricas]
        
        # Crear gráfico
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Posición de las barras
        x = np.arange(len(modelos))
        width = 0.2  # Ancho de las barras
        
        # Crear barras para cada métrica
        for i, (metrica, vals) in enumerate(zip(metricas, valores)):
            ax.bar(x + (i - 1.5) * width, vals, width, label=metrica)
        
        # Configurar gráfico
        ax.set_ylim(0, 1.1)  # Métricas de clasificación van de 0 a 1
        ax.set_title('Comparación de métricas por modelo')
        ax.set_xticks(x)
        ax.set_xticklabels(modelos, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        st.pyplot(fig)
        
    else:  # regresión
        # Para regresión mostraremos R², RMSE, MAE
        # Pero usaremos un gráfico diferente para cada métrica ya que tienen escalas diferentes
        
        # Crear figura con tres subplots
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        # R² (mayor es mejor)
        axs[0].bar(modelos, df_comp['R²'], color='green')
        axs[0].set_title('R² (mayor es mejor)')
        axs[0].set_xticklabels(modelos, rotation=45, ha='right')
        axs[0].grid(axis='y', linestyle='--', alpha=0.7)
        
        # RMSE (menor es mejor)
        axs[1].bar(modelos, df_comp['RMSE'], color='red')
        axs[1].set_title('RMSE (menor es mejor)')
        axs[1].set_xticklabels(modelos, rotation=45, ha='right')
        axs[1].grid(axis='y', linestyle='--', alpha=0.7)
        
        # MAE (menor es mejor)
        axs[2].bar(modelos, df_comp['MAE'], color='orange')
        axs[2].set_title('MAE (menor es mejor)')
        axs[2].set_xticklabels(modelos, rotation=45, ha='right')
        axs[2].grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Opciones para el usuario
    st.subheader("⚙️ Acciones")
    
    # Permitir al usuario seleccionar este modelo o elegir otro
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ Seleccionar este modelo recomendado", type="primary"):
            try:
                # Guardar selección
                resultado = guardar_modelo_seleccionado(
                    nombre_modelo=modelo['nombre'],
                    comentarios=f"Seleccionado con criterio: {recomendacion['criterio_usado']}"
                )
                
                if resultado.get('exito'):
                    st.success(f"✅ {resultado['mensaje']}")
                    
                    # Registrar en el log
                    logger.log_evento(
                        "SELECCION_MODELO",
                        f"Modelo seleccionado: {modelo['nombre']}",
                        "Recomendar_Modelo"
                    )
                else:
                    st.error(f"❌ Error al seleccionar modelo: {resultado.get('error')}")
                    
            except Exception as e:
                st.error(f"❌ Error al seleccionar modelo: {str(e)}")
                logger.log_evento(
                    "ERROR_SELECCION",
                    f"Error al seleccionar modelo: {str(e)}",
                    "Recomendar_Modelo",
                    tipo="error"
                )
    
    with col2:
        # Permitir seleccionar otro modelo
        otro_modelo = st.selectbox(
            "O seleccione otro modelo:",
            options=df_comp['Modelo'].tolist(),
            index=0
        )
        
        comentarios = st.text_area("Comentarios sobre su selección:", height=100)
        
        if st.button("Seleccionar modelo alternativo"):
            try:
                # Verificar que tenemos un modelo válido seleccionado
                if otro_modelo is None:
                    st.error("❌ No se ha seleccionado ningún modelo válido.")
                    return
                
                # Guardar selección
                resultado = guardar_modelo_seleccionado(
                    nombre_modelo=otro_modelo,
                    comentarios=comentarios
                )
                
                if resultado.get('exito'):
                    st.success(f"✅ {resultado['mensaje']}")
                    
                    # Registrar en el log
                    logger.log_evento(
                        "SELECCION_MODELO_ALTERNATIVO",
                        f"Modelo alternativo seleccionado: {otro_modelo}",
                        "Recomendar_Modelo"
                    )
                else:
                    st.error(f"❌ Error al seleccionar modelo: {resultado.get('error')}")
                    
            except Exception as e:
                st.error(f"❌ Error al seleccionar modelo: {str(e)}")
                logger.log_evento(
                    "ERROR_SELECCION",
                    f"Error al seleccionar modelo alternativo: {str(e)}",
                    "Recomendar_Modelo",
                    tipo="error"
                )
    
    # Navegación
    st.subheader("⏩ Próximos pasos")
    
    # Botones de navegación
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔙 Volver a Evaluar Modelos", use_container_width=True):
            st.switch_page("pages/Machine Learning/06_Evaluar_Modelos.py")
    
    with col2:
        if st.button("🧠 Explicar Modelo", use_container_width=True):
            st.switch_page("pages/Machine Learning/09_Explicar_Modelo.py")

if __name__ == "__main__":
    main()
