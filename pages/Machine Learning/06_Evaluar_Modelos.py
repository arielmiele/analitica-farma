"""
Página para evaluar y comparar en detalle los modelos entrenados.
Incluye visualizaciones avanzadas como matrices de confusión, curvas ROC, gráficos de residuos
y otras herramientas para análisis detallado del rendimiento de modelos.
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io

# Importar módulos propios
from src.modelos.evaluador import (
    obtener_ultimos_benchmarkings, 
    diagnosticar_visualizaciones,
    comparar_metricas_regresion,
    calcular_matriz_confusion_detallada,
    calcular_curvas_roc_completas,
    generar_datos_grafico_comparacion_regresion
)
from src.modelos.visualizador import (
    generar_matriz_confusion, 
    generar_curva_roc, 
    generar_curva_precision_recall,
    generar_grafico_residuos, 
    comparar_distribuciones,
    comparar_modelos_roc,
    generar_interpretacion_matriz_confusion,
    generar_interpretacion_curva_roc,
    generar_interpretacion_residuos
)
from src.modelos.modelo_serializer import deserializar_modelos_benchmarking
from src.modelos.diagnostico_modelo import diagnosticar_objetos_modelo
from src.state.session_manager import SessionManager
from src.audit.logger import log_audit

# Inicializar gestor de sesión
session = SessionManager()

def main():
    st.title("📊 Evaluación Detallada de Modelos")
    
    # Verificar si hay resultados de benchmarking
    resultados_benchmarking = session.obtener_estado("resultados_benchmarking")
    
    # Si hay resultados, verificar si los modelos necesitan ser deserializados
    if resultados_benchmarking:
        # Comprobar si hay modelos que tienen 'modelo_serializado' pero no 'modelo_objeto'
        necesita_deserializacion = False
        
        if resultados_benchmarking.get('modelos_exitosos'):
            for modelo in resultados_benchmarking['modelos_exitosos']:
                if 'modelo_serializado' in modelo and 'modelo_objeto' not in modelo:
                    necesita_deserializacion = True
                    break
        
        # Si se necesita deserialización, aplicarla
        if necesita_deserializacion:
            resultados_benchmarking = deserializar_modelos_benchmarking(
                resultados_benchmarking,
                usuario=session.obtener_estado("usuario_id", "sistema"),
                id_sesion=session.obtener_estado("id_sesion", "sin_sesion")
            )
            session.guardar_estado("resultados_benchmarking", resultados_benchmarking)
            st.success("✅ Modelos deserializados automáticamente")
        
        # Mostrar diagnóstico de objetos modelo (debugging)
        with st.expander("📋 Diagnóstico de modelos", expanded=False):
            diagnosticar_objetos_modelo(
                resultados_benchmarking,
                usuario=session.obtener_estado("usuario_id", "sistema"),
                id_sesion=session.obtener_estado("id_sesion", "sin_sesion")
            )
            st.info("Si los objetos modelo no están disponibles, intente ejecutar un nuevo benchmarking o cargar uno existente.")
    
    if not resultados_benchmarking:
        st.warning("⚠️ No hay resultados de benchmarking disponibles.")
        st.info("👈 Vaya a la sección 'Entrenar Modelos' para ejecutar un benchmarking primero.")
        
        # Mostrar benchmarkings anteriores si existen
        try:
            benchmarkings_previos = obtener_ultimos_benchmarkings(
                limite=5,
                id_usuario=session.obtener_estado("usuario_id", "sistema"),
                id_sesion=session.obtener_estado("id_sesion", "sin_sesion"),
                usuario=session.obtener_estado("usuario_id", "sistema")
            )
            if benchmarkings_previos:
                st.subheader("📜 Benchmarkings anteriores")
                
                # Crear tabla para mostrar los benchmarkings
                df_benchmarkings = pd.DataFrame(benchmarkings_previos)
                df_benchmarkings.columns = [
                    "ID", "Tipo de problema", "Variable objetivo", 
                    "Modelos exitosos", "Modelos fallidos", 
                    "Mejor modelo", "Fecha"
                ]
                
                st.dataframe(df_benchmarkings)
                  # Permitir cargar un benchmarking anterior
                selected_id = st.selectbox(
                    "Seleccione un benchmarking para cargar:",
                    options=df_benchmarkings["ID"].tolist()
                )
                
                if st.button("Cargar benchmarking seleccionado"):
                    try:
                        # Aquí implementamos la carga del benchmarking seleccionado
                        from src.modelos.entrenador import obtener_benchmarking_por_id
                        
                        # Registrar en el log
                        log_audit(
                            session.obtener_estado("usuario_id", "sistema"),
                            "CARGAR_BENCHMARKING",
                            "Evaluar_Modelos",
                            selected_id,
                            f"Cargando benchmarking ID: {selected_id}",
                            id_sesion=session.obtener_estado("id_sesion", "sin_sesion")
                        )
                        
                        # Verificar que selected_id no sea None antes de convertirlo
                        if selected_id is not None:
                            # Obtener el benchmarking y deserializar los modelos
                            benchmarking = obtener_benchmarking_por_id(int(selected_id))
                            
                            # Deserializar explícitamente para asegurar que los objetos modelo estén disponibles
                            if benchmarking:
                                benchmarking = deserializar_modelos_benchmarking(
                                    benchmarking,
                                    usuario=session.obtener_estado("usuario_id", "sistema"),
                                    id_sesion=session.obtener_estado("id_sesion", "sin_sesion")
                                )
                                st.success("✅ Modelos deserializados correctamente")
                        else:
                            st.error("❌ ID de benchmarking no válido.")
                            benchmarking = None
                        
                        if benchmarking:
                            # Guardar en sesión
                            session.guardar_estado("resultados_benchmarking", benchmarking)
                            st.success(f"✅ Benchmarking ID {selected_id} cargado correctamente.")
                            
                            # Verificar si los objetos modelo están disponibles
                            diagnosticar_objetos_modelo(
                                benchmarking,
                                usuario=session.obtener_estado("usuario_id", "sistema"),
                                id_sesion=session.obtener_estado("id_sesion", "sin_sesion")
                            )
                            
                            st.rerun()
                        else:
                            st.error("❌ No se pudo cargar el benchmarking seleccionado.")
                    except Exception as error:
                        st.error(f"❌ Error al cargar benchmarking: {str(error)}")
                        log_audit(
                            session.obtener_estado("usuario_id", "sistema"),
                            "ERROR_CARGAR_BENCHMARKING",
                            "Evaluar_Modelos",
                            selected_id,
                            f"Error al cargar benchmarking ID {selected_id}: {str(error)}",
                            id_sesion=session.obtener_estado("id_sesion", "sin_sesion")
                        )
            else:
                st.info("No hay benchmarkings previos disponibles.")
        except Exception as e:
            st.error(f"Error al cargar benchmarkings anteriores: {str(e)}")
        
        return
    
    # Sección de explicación
    with st.expander("ℹ️ Acerca de esta página", expanded=False):
        st.markdown("""
        ### Evaluación Detallada de Modelos
        
        Esta página permite analizar en profundidad los modelos entrenados en el benchmarking.
        
        **Funcionalidades:**
        - Selección del modelo a evaluar
        - Visualización detallada de métricas
        - Análisis de validación cruzada
        - Comparación con el modelo de referencia
        - **Visualizaciones avanzadas:**
          - Matrices de confusión
          - Curvas ROC y Precision-Recall
          - Gráficos de residuos y distribuciones
          - Comparativas entre modelos
        
        Seleccione un modelo de la lista para ver su evaluación detallada.
        """)
    
    # Selección de modelo a evaluar
    if resultados_benchmarking.get('modelos_exitosos'):
        # Obtener nombres de modelos exitosos
        nombres_modelos = [modelo['nombre'] for modelo in resultados_benchmarking['modelos_exitosos']]
        
        # Determinar el modelo a mostrar (por defecto, el mejor)
        modelo_seleccionado = st.selectbox(
            "Seleccione un modelo para evaluar:",
            options=nombres_modelos,
            index=0  # Por defecto, el mejor modelo (ya están ordenados)
        )
        
        # Encontrar el modelo seleccionado en los resultados
        modelo = None
        for m in resultados_benchmarking['modelos_exitosos']:
            if m['nombre'] == modelo_seleccionado:
                modelo = m
                break
        
        if modelo:
            # Mostrar información general del modelo
            st.subheader(f"📝 Información general: {modelo['nombre']}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Tipo de problema:** {resultados_benchmarking['tipo_problema'].capitalize()}")
                st.write(f"**Variable objetivo:** {resultados_benchmarking['variable_objetivo']}")
                
                # Determinar si es el mejor modelo
                es_mejor = modelo == resultados_benchmarking['mejor_modelo']
                if es_mejor:
                    st.success("✅ Este es el mejor modelo según la evaluación.")
            
            with col2:
                st.write(f"**Tiempo de entrenamiento:** {modelo['tiempo_entrenamiento']:.4f} segundos")
                st.write(f"**Fecha de evaluación:** {resultados_benchmarking['timestamp']}")
            
            # Mostrar métricas detalladas
            st.subheader("📊 Métricas de rendimiento")
            
            # Diseño según tipo de problema
            if resultados_benchmarking['tipo_problema'] == 'clasificacion':
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Accuracy", f"{modelo['metricas']['accuracy']:.4f}")
                
                with col2:
                    st.metric("Precision", f"{modelo['metricas']['precision']:.4f}")
                
                with col3:
                    st.metric("Recall", f"{modelo['metricas']['recall']:.4f}")
                
                with col4:
                    st.metric("F1-Score", f"{modelo['metricas']['f1']:.4f}")
                
                # Validación cruzada
                st.subheader("🔄 Validación cruzada")
                
                # Obtener métricas desde el objeto modelo
                metricas = modelo.get('metricas', {})
                cv_score = metricas.get('cv_score_media', 0)
                cv_std = metricas.get('cv_score_std', 0)
                
                st.write(f"**Score CV (promedio):** {cv_score:.4f}")
                st.write(f"**Desviación estándar CV:** {cv_std:.4f}")
                
                # Gráfico de barras para métricas de clasificación
                fig, ax = plt.subplots(figsize=(10, 6))
                
                metricas = ['accuracy', 'precision', 'recall', 'f1']
                valores = [modelo['metricas'][m] for m in metricas]
                # Asignar colores según el valor (más alto = mejor)
                colores = ['green', 'lightgreen', 'orange', 'red']
                
                barras = ax.bar(metricas, valores, color=colores)
                
                # Añadir etiquetas con valores
                for i, barra in enumerate(barras):
                    ax.text(
                        barra.get_x() + barra.get_width()/2,
                        barra.get_height() + 0.01,
                        f'{valores[i]:.4f}',
                        ha='center'
                    )
                
                # Configurar gráfico
                plt.ylim(0, 1.1)  # Métricas de clasificación van de 0 a 1
                plt.title('Métricas de clasificación')
                plt.ylabel('Valor')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                
                st.pyplot(fig)                
                
                # VISUALIZACIONES AVANZADAS PARA CLASIFICACIÓN
                st.subheader("📈 Visualizaciones avanzadas")
                
                # Realizar diagnóstico de requisitos para visualizaciones
                diagnostico = diagnosticar_visualizaciones(resultados_benchmarking, modelo)
                
                # Mostrar estado de diagnóstico
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Mostrar estado general
                    if diagnostico["puede_visualizar"]:
                        st.success("✅ Todos los requisitos para visualizaciones avanzadas están disponibles")
                    else:
                        st.warning("⚠️ Hay limitaciones para las visualizaciones avanzadas")
                
                with col2:
                    # Mostrar detalles de los requisitos
                    requisitos = diagnostico["requisitos"]
                    
                    # Datos de prueba
                    if requisitos.get("datos_prueba", False):
                        st.write("✅ Datos de prueba disponibles")
                    else:
                        st.write("❌ Datos de prueba no disponibles")
                    
                    # Objeto del modelo
                    if requisitos.get("modelo_objeto", False):
                        st.write("✅ Objeto del modelo disponible")
                    else:
                        st.write("❌ Objeto del modelo no disponible")
                    
                    # Métricas
                    if requisitos.get("metricas", False):
                        st.write("✅ Métricas completas disponibles")
                    else:
                        st.write("❌ Métricas incompletas")
                
                # Mostrar recomendaciones si hay problemas
                if diagnostico["recomendaciones"]:
                    with st.expander("📋 Recomendaciones para mejorar visualizaciones", expanded=True):
                        for recomendacion in diagnostico["recomendaciones"]:
                            st.info(recomendacion)
                
                # Comprobar si tenemos los datos para las visualizaciones
                if 'X_test' in resultados_benchmarking and 'y_test' in resultados_benchmarking:
                    try:
                        # Convertir datos serializados de vuelta a numpy arrays si es necesario
                        X_test = np.array(resultados_benchmarking['X_test']) if isinstance(resultados_benchmarking['X_test'], list) else resultados_benchmarking['X_test']
                        y_test = np.array(resultados_benchmarking['y_test']) if isinstance(resultados_benchmarking['y_test'], list) else resultados_benchmarking['y_test']
                        
                        # Obtener predicciones y probabilidades si está disponible el modelo
                        if 'modelo_objeto' in modelo:
                            # Obtener predicciones
                            y_pred = modelo['modelo_objeto'].predict(X_test)
                            
                            # Obtener probabilidades (para curvas ROC)
                            try:
                                y_prob = modelo['modelo_objeto'].predict_proba(X_test)
                            except Exception as e:
                                # Si el modelo no soporta predict_proba
                                st.warning(f"El modelo no soporta predict_proba: {str(e)}")
                                y_prob = np.zeros((len(y_test), len(np.unique(y_test))))
                                
                            # Obtener clases
                            clases = np.unique(y_test)
                        else:
                            # Si no hay modelo, usar resultados pre-calculados o mostrar mensaje
                            st.warning("El modelo no está disponible para generar predicciones en tiempo real.")
                            # Usar dummy data para evitar errores
                            y_pred = y_test  # Fallback
                            y_prob = np.zeros((len(y_test), len(np.unique(y_test))))
                            clases = np.unique(y_test)
                        
                        # Mostrar visualizaciones simplificadas basadas en métricas disponibles
                        st.info("Mostrando visualizaciones basadas en las métricas calculadas durante el entrenamiento.")
                        
                        # Ejemplo: Mostrar gráfico de barras de métricas
                        fig, ax = plt.subplots(figsize=(10, 6))
                        metricas = modelo['metricas']
                        nombres = list(metricas.keys())
                        valores = list(metricas.values())
                        
                        # Filtrar solo métricas numéricas relevantes (no std, etc.)
                        metricas_a_mostrar = ['accuracy', 'precision', 'recall', 'f1']
                        indices = [i for i, nombre in enumerate(nombres) if nombre in metricas_a_mostrar]
                        nombres_filtrados = [nombres[i] for i in indices]
                        valores_filtrados = [valores[i] for i in indices]
                        
                        ax.bar(nombres_filtrados, valores_filtrados, color='skyblue')
                        ax.set_ylim(0, 1)
                        ax.set_title('Métricas principales')
                        ax.set_ylabel('Valor')
                        ax.grid(axis='y', linestyle='--', alpha=0.7)
                        
                        # Añadir etiquetas de valor
                        for i, v in enumerate(valores_filtrados):
                            ax.text(i, v + 0.02, f'{v:.3f}', ha='center')
                        
                        st.pyplot(fig)
                        
                        # Indicar que se necesita un nuevo benchmarking para visualizaciones completas
                        st.warning("""
                        Para visualizaciones más avanzadas como matrices de confusión, curvas ROC y gráficos detallados,
                        se recomienda ejecutar un nuevo benchmarking desde la sección de Entrenar Modelos.
                        """)
                        
                    except Exception as e:
                        st.error(f"Error al generar visualizaciones: {str(e)}")
                        log_audit(
                            session.obtener_estado("usuario_id", "sistema"),
                            "ERROR_VISUALIZACION",
                            "Evaluar_Modelos",
                            modelo['nombre'] if modelo else "N/A",
                            f"Error al generar visualizaciones para {modelo['nombre'] if modelo else 'N/A'}: {str(e)}",
                            id_sesion=session.obtener_estado("id_sesion", "sin_sesion")
                        )
                        return
                    
                    # Pestañas para diferentes visualizaciones
                    tabs = st.tabs([
                        "Matriz de Confusión", 
                        "Curva ROC", 
                        "Precision-Recall", 
                        "Comparar Modelos"
                    ])
                    
                    # TAB 1: Matriz de Confusión
                    with tabs[0]:
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                # Generar matriz de confusión
                                opciones_norm = {
                                    "Sin normalizar": None,
                                    "Normalizar por filas": "true",
                                    "Normalizar por columnas": "pred",
                                    "Normalizar por total": "all"
                                }
                                
                                tipo_norm = st.selectbox(
                                    "Tipo de normalización:",
                                    options=list(opciones_norm.keys()),
                                    index=0
                                )
                                
                                # Generar y mostrar matriz de confusión
                                matriz_confusion_fig = generar_matriz_confusion(
                                    y_test, 
                                    y_pred, 
                                    clases=[str(c) for c in clases],  # Convertir a lista de strings
                                    normalizar=opciones_norm[tipo_norm],
                                    titulo=f"Matriz de Confusión - {modelo['nombre']}",
                                    id_sesion=session.obtener_estado("id_sesion", "sin_sesion"),
                                    usuario=session.obtener_estado("usuario_id", "sistema")
                                )
                                
                                st.pyplot(matriz_confusion_fig)
                                
                                # Opción para descargar
                                buf = io.BytesIO()
                                matriz_confusion_fig.savefig(buf, format="png", dpi=120)
                                buf.seek(0)
                                st.download_button(
                                    label="Descargar matriz de confusión",
                                    data=buf,
                                    file_name=f"matriz_confusion_{modelo['nombre']}.png",
                                    mime="image/png"
                                )
                            
                            with col2:
                                # Calcular matriz para interpretación usando función del modelo
                                cm = calcular_matriz_confusion_detallada(y_test, y_pred, normalize=opciones_norm[tipo_norm])
                                interpretacion = generar_interpretacion_matriz_confusion(cm, [str(c) for c in clases])
                                st.markdown(interpretacion)
                        
                    # TAB 2: Curva ROC
                    with tabs[1]:
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # Determinar si es multiclase
                            es_multiclase = len(np.unique(y_test)) > 2
                              # Generar y mostrar curva ROC
                            roc_fig = generar_curva_roc(
                                y_test, 
                                y_prob, 
                                clases=[str(c) for c in clases],  # Convertir a lista de strings
                                multi_clase=es_multiclase,
                                titulo=f"Curva ROC - {modelo['nombre']}",
                                id_sesion=session.obtener_estado("id_sesion", "sin_sesion"),
                                usuario=session.obtener_estado("usuario_id", "sistema")
                            )
                                
                            st.pyplot(roc_fig)
                                
                            # Opción para descargar
                            buf = io.BytesIO()
                            roc_fig.savefig(buf, format="png", dpi=120)
                            buf.seek(0)
                            st.download_button(
                                label="Descargar curva ROC",
                                data=buf,
                                file_name=f"curva_roc_{modelo['nombre']}.png",
                                mime="image/png"
                            )
                            
                        with col2:
                            # Calcular AUC para interpretación usando función del modelo
                            curvas_roc_data = calcular_curvas_roc_completas(y_test, y_prob, clases)
                            
                            if curvas_roc_data['es_multiclase']:
                                auc_valor = curvas_roc_data['auc_promedio']
                            else:
                                auc_valor = curvas_roc_data['auc_valor']
                            
                            interpretacion = generar_interpretacion_curva_roc(float(auc_valor))
                            st.markdown(interpretacion)
                        
                    # TAB 3: Precision-Recall
                    with tabs[2]:
                        # Determinar si es multiclase
                        es_multiclase = len(np.unique(y_test)) > 2
                        
                        # Generar y mostrar curva PR
                        pr_fig = generar_curva_precision_recall(
                            y_test, 
                            y_prob, 
                            clases=[str(c) for c in clases],  # Convertir a lista de strings
                            multi_clase=es_multiclase,
                            titulo=f"Curva Precision-Recall - {modelo['nombre']}",
                            id_sesion=session.obtener_estado("id_sesion", "sin_sesion"),
                            usuario=session.obtener_estado("usuario_id", "sistema")
                        )
                        
                        st.pyplot(pr_fig)
                        
                        # Opción para descargar
                        buf = io.BytesIO()
                        pr_fig.savefig(buf, format="png", dpi=120)
                        buf.seek(0)
                        st.download_button(
                            label="Descargar curva Precision-Recall",
                            data=buf,
                            file_name=f"curva_pr_{modelo['nombre']}.png",
                            mime="image/png"
                        )
                          # TAB 4: Comparación de Modelos
                    with tabs[3]:
                        st.write("Seleccione modelos para comparar:")
                        
                        # Permitir selección múltiple de modelos
                        modelos_comparar = st.multiselect(
                            "Modelos a comparar:",
                            options=nombres_modelos,
                            default=[modelo_seleccionado]
                        )
                        
                        if modelos_comparar:
                            # Preparar diccionario de modelos
                            modelos_dict = {}
                            for nombre in modelos_comparar:
                                for m in resultados_benchmarking['modelos_exitosos']:
                                    if m['nombre'] == nombre and 'modelo_objeto' in m:
                                        modelos_dict[nombre] = {"modelo": m['modelo_objeto']}
                                        break
                            
                            # Generar comparación de curvas ROC
                            if modelos_dict:
                                comp_fig = comparar_modelos_roc(
                                    modelos_dict,
                                    X_test,
                                    y_test,
                                    titulo="Comparación de Modelos - Curva ROC",
                                    id_sesion=session.obtener_estado("id_sesion", "sin_sesion"),
                                    usuario=session.obtener_estado("usuario_id", "sistema")
                                )
                                
                                st.pyplot(comp_fig)
                                
                                # Opción para descargar
                                buf = io.BytesIO()
                                comp_fig.savefig(buf, format="png", dpi=120)
                                buf.seek(0)
                                st.download_button(
                                    label="Descargar comparación",
                                    data=buf,
                                    file_name="comparacion_modelos_roc.png",
                                    mime="image/png"
                                )
                            else:
                                st.warning("No se pudieron cargar los modelos seleccionados para la comparación.")
                        else:
                            st.info("Seleccione al menos un modelo para comparar.")
                                
                else:
                    st.info("""
                    ⚠️ No hay datos suficientes para generar visualizaciones avanzadas. 
                    Esto puede deberse a que el modelo fue cargado desde un benchmarking anterior 
                    que no guardó todos los datos necesarios.
                    
                    Intente ejecutar un nuevo benchmarking para acceder a todas las funcionalidades.
                    """)
                
            else:  # Regresión  
                col1, col2, col3, col4 = st.columns(4)
                
                # Verificar si modelo no es None y si existen las métricas antes de mostrarlas
                if modelo is None:
                    st.error("No se pudo cargar el modelo seleccionado.")
                    return
                
                metricas = modelo.get('metricas', {})
                
                with col1:
                    r2_value = metricas.get('r2', 0)
                    st.metric("R²", f"{r2_value:.4f}")
                
                with col2:
                    mse_value = metricas.get('mse', 0)
                    st.metric("MSE", f"{mse_value:.4f}")
                
                with col3:
                    rmse_value = metricas.get('rmse', 0)
                    st.metric("RMSE", f"{rmse_value:.4f}")
                
                with col4:
                    mae_value = metricas.get('mae', 0)
                    st.metric("MAE", f"{mae_value:.4f}")
                
                # Validación cruzada
                st.subheader("🔄 Validación cruzada")
                
                # Obtener métricas de validación cruzada
                cv_score = metricas.get('cv_score_media', 0)
                cv_std = metricas.get('cv_score_std', 0)
                
                st.write(f"**Score CV (promedio):** {cv_score:.4f}")
                st.write(f"**Desviación estándar CV:** {cv_std:.4f}")
                
                # Gráfico de barras para métricas de regresión
                fig, ax = plt.subplots(figsize=(10, 6))
                
                metricas_keys = ['r2', 'mse', 'rmse', 'mae']
                valores = [metricas.get(m, 0) for m in metricas_keys]
                
                # Gráfico de barras para métricas de regresión
                fig, ax = plt.subplots(figsize=(10, 6))
                
                metricas = ['r2', 'mse', 'rmse', 'mae']
                valores = [modelo['metricas'][m] for m in metricas]
                
                # Para regresión, R² más alto es mejor, pero MSE, RMSE, MAE más bajos son mejores
                # Invertimos los valores para la visualización
                valores_norm = [
                    valores[0],  # R² es mejor si es más alto
                    1 / (valores[1] + 1e-10),  # MSE es mejor si es más bajo
                    1 / (valores[2] + 1e-10),  # RMSE es mejor si es más bajo                    
                    1 / (valores[3] + 1e-10)   # MAE es mejor si es más bajo
                ]
                
                # Normalizar para visualización
                max_val = max(valores_norm)
                valores_norm = [v/max_val for v in valores_norm]
                
                # Asignar colores simples
                colores = ['green', 'red', 'orange', 'blue']
                
                barras = ax.bar(metricas, valores, color=colores)
                
                # Añadir etiquetas con valores originales
                for i, barra in enumerate(barras):
                    ax.text(
                        barra.get_x() + barra.get_width()/2,
                        barra.get_height() + 0.01,
                        f'{valores[i]:.4f}',
                        ha='center'
                    )
                
                # Configurar gráfico
                plt.title('Métricas de regresión')
                plt.ylabel('Valor')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                
                st.pyplot(fig)            
                
                # VISUALIZACIONES AVANZADAS PARA REGRESIÓN
                st.subheader("📈 Visualizaciones avanzadas")
                
                # Realizar diagnóstico de requisitos para visualizaciones
                diagnostico = diagnosticar_visualizaciones(resultados_benchmarking, modelo)
                
                # Mostrar estado de diagnóstico
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Mostrar estado general
                    if diagnostico["puede_visualizar"]:
                        st.success("✅ Todos los requisitos para visualizaciones avanzadas están disponibles")
                    else:
                        st.warning("⚠️ Hay limitaciones para las visualizaciones avanzadas")
                
                with col2:
                    # Mostrar detalles de los requisitos
                    requisitos = diagnostico["requisitos"]
                    
                    # Datos de prueba
                    if requisitos.get("datos_prueba", False):
                        st.write("✅ Datos de prueba disponibles")
                    else:
                        st.write("❌ Datos de prueba no disponibles")
                    
                    # Objeto del modelo
                    if requisitos.get("modelo_objeto", False):
                        st.write("✅ Objeto del modelo disponible")
                    else:
                        st.write("❌ Objeto del modelo no disponible")
                    
                    # Métricas
                    if requisitos.get("metricas", False):
                        st.write("✅ Métricas completas disponibles")
                    else:
                        st.write("❌ Métricas incompletas")
                
                # Mostrar recomendaciones si hay problemas
                if diagnostico["recomendaciones"]:
                    with st.expander("📋 Recomendaciones para mejorar visualizaciones", expanded=True):
                        for recomendacion in diagnostico["recomendaciones"]:
                            st.info(recomendacion)
                
                # Comprobar si tenemos los datos para las visualizaciones
                if 'X_test' in resultados_benchmarking and 'y_test' in resultados_benchmarking:
                    try:
                        # Convertir datos serializados de vuelta a numpy arrays si es necesario
                        X_test = np.array(resultados_benchmarking['X_test']) if isinstance(resultados_benchmarking['X_test'], list) else resultados_benchmarking['X_test']
                        y_test = np.array(resultados_benchmarking['y_test']) if isinstance(resultados_benchmarking['y_test'], list) else resultados_benchmarking['y_test']
                        
                        # Obtener predicciones si está disponible el modelo
                        if 'modelo_objeto' in modelo:
                            # Obtener predicciones
                            y_pred = modelo['modelo_objeto'].predict(X_test)
                        else:
                            # Si no hay modelo, usar resultados pre-calculados o dummy data
                            st.warning("El modelo no está disponible para generar predicciones en tiempo real.")
                            y_pred = y_test * 0.9 + np.random.normal(0, 0.1, len(y_test))  # Fallback con algo de ruido
                    
                        # Mostrar visualizaciones simplificadas basadas en métricas disponibles
                        st.info("Mostrando visualizaciones basadas en las métricas calculadas durante el entrenamiento.")
                        
                        # Gráfico comparativo de métricas de regresión
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        metricas = ['r2', 'mse', 'rmse', 'mae']
                        valores = [modelo['metricas'][m] for m in metricas]
                        
                        # Visualizar valores directamente (sin normalización)
                        ax.bar(metricas, valores, color=['green', 'red', 'orange', 'blue'])
                        for i, v in enumerate(valores):
                            ax.text(i, v, f'{v:.3f}', ha='center', va='bottom')
                        
                        ax.set_title(f'Métricas de regresión para {modelo["nombre"]}')
                        ax.set_ylabel('Valor')
                        ax.grid(axis='y', linestyle='--', alpha=0.7)
                        
                        st.pyplot(fig)
                        
                        # Pestañas para diferentes visualizaciones
                        tabs = st.tabs([
                            "Gráfico de Residuos",
                            "Comparación de Distribuciones", 
                            "Comparar Modelos"
                        ])
                        
                        # TAB 1: Gráfico de Residuos
                        with tabs[0]:
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                if 'modelo_objeto' in modelo:
                                    # Generar gráfico de residuos
                                    residuos_fig = generar_grafico_residuos(
                                        y_test, 
                                        y_pred, 
                                        titulo=f"Análisis de Residuos - {modelo['nombre']}",
                                        id_sesion=session.obtener_estado("id_sesion", "sin_sesion"),
                                        usuario=session.obtener_estado("usuario_id", "sistema")
                                    )
                                    
                                    st.pyplot(residuos_fig)
                                    
                                    # Opción para descargar
                                    buf = io.BytesIO()
                                    residuos_fig.savefig(buf, format="png", dpi=120)
                                    buf.seek(0)
                                    st.download_button(
                                        label="Descargar gráfico de residuos",
                                        data=buf,
                                        file_name=f"residuos_{modelo['nombre']}.png",
                                        mime="image/png"
                                    )
                                else:
                                    st.warning("No hay datos suficientes para generar el gráfico de residuos.")
                            
                            with col2:
                                # Generar interpretación de residuos
                                if 'modelo_objeto' in modelo:
                                    try:
                                        # Calcular residuos
                                        residuos = y_test - y_pred
                                        interpretacion = generar_interpretacion_residuos(residuos)
                                        st.markdown(interpretacion)
                                    except Exception as e:
                                        st.error(f"Error al calcular residuos: {str(e)}")
                                        log_audit(
                                            session.obtener_estado("usuario_id", "sistema"),
                                            "ERROR_CALCULO_RESIDUOS",
                                            "Evaluar_Modelos",
                                            modelo['nombre'] if modelo else "N/A",
                                            f"Error al calcular residuos: {str(e)}",
                                            id_sesion=session.obtener_estado("id_sesion", "sin_sesion")
                                        )
                        
                        # TAB 2: Comparación de Distribuciones
                        with tabs[1]:
                            if 'modelo_objeto' in modelo:
                                try:
                                    # Generar y mostrar comparación de distribuciones
                                    dist_fig = comparar_distribuciones(
                                        y_test, 
                                        y_pred, 
                                        titulo=f"Comparación de Distribuciones - {modelo['nombre']}",
                                        id_sesion=session.obtener_estado("id_sesion", "sin_sesion"),
                                        usuario=session.obtener_estado("usuario_id", "sistema")
                                    )
                                    
                                    st.pyplot(dist_fig)
                                    
                                    # Opción para descargar
                                    buf = io.BytesIO()
                                    dist_fig.savefig(buf, format="png", dpi=120)
                                    buf.seek(0)
                                    st.download_button(
                                        label="Descargar comparación de distribuciones",
                                        data=buf,
                                        file_name=f"distribuciones_{modelo['nombre']}.png",
                                        mime="image/png"
                                    )
                                except Exception as e:
                                    st.error(f"Error al generar visualización de distribuciones: {str(e)}")
                                    log_audit(
                                        session.obtener_estado("usuario_id", "sistema"),
                                        "ERROR_VISUALIZACIONES",
                                        "Evaluar_Modelos",
                                        modelo['nombre'] if modelo else "N/A",
                                        f"Error al generar visualización de distribuciones: {str(e)}",
                                        id_sesion=session.obtener_estado("id_sesion", "sin_sesion")
                                    )
                            else:
                                st.warning("No hay datos suficientes para generar la comparación de distribuciones.")
                                    
                        # TAB 3: Comparar Modelos
                        with tabs[2]:
                            st.write("Seleccione modelos para comparar:")
                            
                            # Permitir selección múltiple de modelos
                            modelos_comparar = st.multiselect(
                                "Modelos a comparar:",
                                options=nombres_modelos,
                                default=[modelo_seleccionado]
                            )
                            
                            if modelos_comparar:
                                # Preparar diccionario de modelos
                                modelos_dict = {}
                                
                                for nombre in modelos_comparar:
                                    for m in resultados_benchmarking['modelos_exitosos']:
                                        if m['nombre'] == nombre and 'modelo_objeto' in m:
                                            modelos_dict[nombre] = {"modelo": m['modelo_objeto']}
                                            break
                                
                                if modelos_dict:
                                    try:
                                        # Usar función del modelo para crear datos de comparación
                                        datos_comparacion = generar_datos_grafico_comparacion_regresion(
                                            modelos_dict, X_test, y_test
                                        )
                                        
                                        # Mostrar gráfico de dispersión
                                        fig, ax = plt.subplots(figsize=(10, 6))
                                        
                                        # Graficar línea de referencia (predicción perfecta)
                                        rango = datos_comparacion['rango_referencia']
                                        ax.plot([rango['min'], rango['max']], [rango['min'], rango['max']], 
                                                'k--', label='Predicción perfecta')
                                        
                                        # Graficar predicciones de cada modelo usando datos del modelo
                                        for nombre, datos in datos_comparacion['datos_modelos'].items():
                                            ax.scatter(datos['x'], datos['y'], label=nombre)
                                        
                                        # Añadir etiquetas y leyenda
                                        ax.set_xlabel('Valores reales')
                                        ax.set_ylabel('Valores predichos')
                                        ax.set_title('Comparación de predicciones')
                                        ax.legend()
                                        
                                        # Mostrar el gráfico
                                        st.pyplot(fig)
                                        
                                        # Mostrar tabla de métricas comparativas
                                        st.subheader("Métricas comparativas")
                                        
                                        buf = io.BytesIO()
                                        fig.savefig(buf, format="png", dpi=120)
                                        buf.seek(0)
                                        st.download_button(
                                            label="Descargar comparación",
                                            data=buf,
                                            file_name="comparacion_modelos_regresion.png",
                                            mime="image/png"
                                        )
                                        
                                        # Usar función del modelo para comparar métricas
                                        metricas_comp = comparar_metricas_regresion(modelos_dict, X_test, y_test)
                                        
                                        # Mostrar la tabla de métricas comparativas
                                        st.dataframe(metricas_comp)
                                        
                                    except Exception as e:
                                        st.error(f"Error al comparar modelos: {str(e)}")
                                        log_audit(
                                            session.obtener_estado("usuario_id", "sistema"),
                                            "ERROR_COMPARACION_MODELOS",
                                            "Evaluar_Modelos",
                                            modelo['nombre'] if modelo else "N/A",
                                            f"Error al comparar modelos: {str(e)}",
                                            id_sesion=session.obtener_estado("id_sesion", "sin_sesion")
                                        )
                                else:
                                    st.warning("No se pudieron cargar los modelos seleccionados para la comparación.")
                            else:
                                st.info("Seleccione al menos un modelo para comparar.")
                    
                    except Exception as e:
                        st.error(f"Error al generar visualizaciones: {str(e)}")
                        log_audit(
                            session.obtener_estado("usuario_id", "sistema"),
                            "ERROR_VISUALIZACION",
                            "Evaluar_Modelos",
                            modelo['nombre'] if modelo else "N/A",
                            f"Error al generar visualizaciones para {modelo['nombre'] if modelo else 'N/A'}: {str(e)}",
                            id_sesion=session.obtener_estado("id_sesion", "sin_sesion")
                        )
                else:
                    st.info("""
                        ⚠️ No hay datos suficientes para generar visualizaciones avanzadas. 
                        Esto puede deberse a que el modelo fue cargado desde un benchmarking anterior 
                        que no guardó todos los datos necesarios.
                        
                        Intente ejecutar un nuevo benchmarking para acceder a todas las funcionalidades.
                        """)
                               
            # Historial de ejecuciones
            st.markdown("---")
            with st.expander("📜 Historial de ejecuciones", expanded=False):
                try:
                    from src.database.modelos_db import obtener_historial_ejecuciones
                    historial = obtener_historial_ejecuciones(
                        id_usuario=session.obtener_estado("usuario_id", None),
                        limit=20
                    )
                    if historial:
                        df_hist = pd.DataFrame(historial)
                        df_hist = df_hist.rename(columns={
                            "id": "ID",
                            "dataset_nombre": "Dataset / Objetivo",
                            "tipo_problema": "Tipo",
                            "variable_objetivo": "Variable objetivo",
                            "modelo_ganador": "Mejor modelo",
                            "metrica_nombre": "Métrica",
                            "metrica_valor": "Valor",
                            "modelos_exitosos": "Exitosos",
                            "total_modelos": "Total",
                            "duracion_segundos": "Duración (s)",
                            "timestamp": "Fecha"
                        })
                        st.dataframe(df_hist, use_container_width=True, hide_index=True)
                    else:
                        st.info("No hay ejecuciones registradas aún.")
                except Exception as e:
                    st.warning(f"No se pudo cargar el historial: {e}")

            # Opciones adicionales
            st.subheader("⏩ Próximos pasos")
            
            # Botones para navegación
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔙 Volver a Entrenar Modelos", use_container_width=True):
                    # Guardar instrucción de navegación en la sesión
                    session.guardar_estado("navegacion", "Entrenar_Modelos")
                    # Redirigir a la página de entrenamiento
                    st.switch_page("pages/Machine Learning/05_Entrenar_Modelos.py")
            
            with col2:
                if st.button("🧠 Validación Cruzada", use_container_width=True):
                    # Guardar instrucción de navegación en la sesión
                    session.guardar_estado("navegacion", "Validacion_Cruzada")
                    # Redirigir a la página de validación cruzada
                    st.switch_page("pages/Machine Learning/07_Validacion_Cruzada.py")
            
            with col3:
                if st.button("👑 Recomendar Modelo", use_container_width=True):
                    # Guardar instrucción de navegación en la sesión
                    session.guardar_estado("navegacion", "Recomendar_Modelo")
                    # Redirigir a la página de recomendación
                    st.switch_page("pages/Machine Learning/08_Recomendar_Modelo.py")
    else:
        st.warning("No hay modelos exitosos para evaluar.")

if __name__ == "__main__":
    main()