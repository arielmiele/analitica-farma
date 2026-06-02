import streamlit as st
import os
import sys
from datetime import datetime

# Agregar el directorio src al path para poder importar los módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Importar módulos de la aplicación
from src.audit.logger import setup_logger, log_audit
from src.datos.validador import validar_estructura, validar_variable_objetivo
from src.modelos.configurador import guardar_configuracion_modelo

# Configurar el logger
usuario_id_raw = st.session_state.get("usuario_id", "sistema")
try:
    usuario_id = int(usuario_id_raw)
except (ValueError, TypeError):
    usuario_id = 0  # Valor por defecto si no es convertible a int
logger = setup_logger("configurar_datos")

# Inicializar session_state para esta página
if 'variable_objetivo' not in st.session_state:
    st.session_state.variable_objetivo = None
if 'variables_predictoras' not in st.session_state:
    st.session_state.variables_predictoras = []
if 'tipo_problema' not in st.session_state:
    st.session_state.tipo_problema = None
if 'configuracion_validada' not in st.session_state:
    st.session_state.configuracion_validada = False
if 'paso_configuracion' not in st.session_state:
    st.session_state.paso_configuracion = 0  # 0: inicio, 1: mapeo, 2: resultado

# Título y descripción de la página
st.title("🎯 Configurar Datos para Modelado")

st.markdown("""
Esta página te permite configurar la estructura de tus datos y definir la variable objetivo 
para el análisis y modelado.
""")

# Verificar si hay datos cargados
if 'df' not in st.session_state or st.session_state.df is None:
    st.warning("⚠️ No hay datos cargados. Por favor, carga un dataset primero en la página 'Cargar Datos'.")
    if st.button("Ir a Cargar Datos", use_container_width=True):
        st.session_state.paso_carga = 0  # Reiniciar el paso de carga
        st.switch_page("pages/datos/01_Cargar_Datos.py")
else:
    # Mostrar información del dataset cargado
    st.write(f"### Dataset cargado: {st.session_state.filename}")
    st.write(f"Dimensiones: {st.session_state.df.shape[0]} filas × {st.session_state.df.shape[1]} columnas")
    
    # Pasos de configuración
    if st.session_state.paso_configuracion == 0:
        # Paso inicial: Explicación y selección de tipo de problema
        st.write("### Paso 1: Seleccionar tipo de problema")
        st.markdown("""
        Antes de definir la variable objetivo, necesitamos determinar qué tipo de problema 
        estamos tratando de resolver:
        
        - **Regresión**: Para predecir valores numéricos continuos (ej. temperatura, rendimiento, concentración)
        - **Clasificación**: Para predecir categorías o clases (ej. cumple/no cumple, tipo de defecto)
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔢 Regresión", use_container_width=True):
                st.session_state.tipo_problema = "regresion"
                st.session_state.paso_configuracion = 1
                log_audit(usuario_id, "SELECCION_TIPO_PROBLEMA", "configurar_datos", st.session_state.filename, "Usuario seleccionó problema de regresión", id_sesion=st.session_state.get("id_sesion", "sin_sesion"))
                st.rerun()
                
        with col2:
            if st.button("🏷️ Clasificación", use_container_width=True):
                st.session_state.tipo_problema = "clasificacion"
                st.session_state.paso_configuracion = 1
                log_audit(usuario_id, "SELECCION_TIPO_PROBLEMA", "configurar_datos", st.session_state.filename, "Usuario seleccionó problema de clasificación", id_sesion=st.session_state.get("id_sesion", "sin_sesion"))
                st.rerun()
                
    elif st.session_state.paso_configuracion == 1:        
        # Paso de mapeo: Selección de variable objetivo y variables predictoras
        tipo_problema_cap = st.session_state.tipo_problema.capitalize() if st.session_state.tipo_problema else "Problema"
        st.write(f"### Paso 2: Definir variables para {tipo_problema_cap}")
        
        # Información específica según el tipo de problema
        if st.session_state.tipo_problema == "regresion":
            st.info("ℹ️ Selecciona una variable numérica continua como objetivo para tu modelo de regresión.")
            # Filtrar solo columnas numéricas para regresión
            columnas_candidatas = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
            if not columnas_candidatas:
                st.error("❌ No se encontraron columnas numéricas en el dataset. Una regresión requiere una variable objetivo numérica.")
                if st.button("⬅️ Volver atrás", use_container_width=True):
                    st.session_state.paso_configuracion = 0
                    st.rerun()
        else:  # clasificación
            st.info("ℹ️ Selecciona una variable categórica como objetivo para tu modelo de clasificación.")
            # Para clasificación permitimos cualquier tipo de columna
            columnas_candidatas = st.session_state.df.columns.tolist()
        
        # Selección de variable objetivo (solo una)
        st.write("#### Variable objetivo")
        variable_objetivo = st.selectbox(
            "Selecciona la variable que deseas predecir:",
            options=columnas_candidatas,
            index=None,
            placeholder="Selecciona una variable objetivo..."
        )
        
        # Si se seleccionó una variable objetivo
        if variable_objetivo:
            # Guardar en session_state
            st.session_state.variable_objetivo = variable_objetivo
            
            # Mostrar variables predictoras disponibles (excluyendo la objetivo)
            st.write("#### Variables predictoras")
            variables_disponibles = [col for col in st.session_state.df.columns if col != variable_objetivo]
            
            # Multiselect para variables predictoras
            variables_seleccionadas = st.multiselect(
                "Selecciona las variables que usarás para predecir:",
                options=variables_disponibles,
                default=variables_disponibles,  # Por defecto todas seleccionadas
                placeholder="Selecciona al menos una variable predictora..."
            )
            
            # Guardar en session_state
            st.session_state.variables_predictoras = variables_seleccionadas
            
            # Botones de navegación
            col1, col2 = st.columns([1, 2])
            with col1:
                if st.button("⬅️ Cambiar tipo de problema", use_container_width=True):
                    st.session_state.paso_configuracion = 0
                    st.session_state.tipo_problema = None
                    st.rerun()
            
            with col2:
                # Validación antes de continuar
                if len(variables_seleccionadas) == 0:
                    st.error("❌ Debes seleccionar al menos una variable predictora.")
                else:
                    # Validar configuración
                    es_valida, mensaje = validar_variable_objetivo(
                        st.session_state.df, 
                        variable_objetivo, 
                        st.session_state.tipo_problema,
                        usuario=usuario_id,
                        id_sesion=st.session_state.get("id_sesion", "sin_sesion")
                    )
                    
                    if not es_valida:
                        st.error(f"❌ {mensaje}")
                    else:
                        if st.button("✅ Confirmar selección", use_container_width=True):
                            # Guardar configuración
                            configuracion = {
                                "tipo_problema": st.session_state.tipo_problema,
                                "variable_objetivo": st.session_state.variable_objetivo,
                                "variables_predictoras": st.session_state.variables_predictoras,
                                "fecha_configuracion": datetime.now().isoformat()
                            }
                            
                            # Validar estructura completa
                            estructura_valida, mensaje_estructura = validar_estructura(
                                st.session_state.df,
                                configuracion,
                                usuario=usuario_id,
                                id_sesion=st.session_state.get("id_sesion", "sin_sesion")
                            )
                            
                            if estructura_valida:
                                # Si ya hay benchmarking entrenado con diferente selección, invalidarlo
                                benchmarking_previo = st.session_state.get("resultados_benchmarking")
                                if benchmarking_previo:
                                    cols_previas = set(benchmarking_previo.get("columnas_originales", []))
                                    cols_nuevas = set(variables_seleccionadas)
                                    if cols_previas and cols_previas != cols_nuevas:
                                        st.session_state.resultados_benchmarking = None
                                        st.session_state.modelo_recomendado = None
                                        st.session_state.interpretabilidad = None
                                        log_audit(
                                            usuario_id,
                                            "INVALIDAR_BENCHMARKING",
                                            "configurar_datos",
                                            st.session_state.filename,
                                            f"Benchmarking invalidado: predictores cambiaron de {len(cols_previas)} a {len(cols_nuevas)} variables",
                                            id_sesion=st.session_state.get("id_sesion", "sin_sesion")
                                        )

                                # Guardar configuración en la base de datos
                                guardar_configuracion_modelo(configuracion, usuario=usuario_id, id_usuario=usuario_id, id_sesion=st.session_state.get("id_sesion", "sin_sesion"))
                                
                                # Actualizar estado
                                st.session_state.configuracion_validada = True
                                st.session_state.paso_configuracion = 2
                                
                                # Registrar acción
                                log_audit(
                                    usuario_id, 
                                    "CONFIGURACION_VARIABLES", 
                                    "configurar_datos", 
                                    st.session_state.filename,
                                    f"Variable objetivo: {variable_objetivo}, Predictoras: {variables_seleccionadas}",
                                    id_sesion=st.session_state.get("id_sesion", "sin_sesion")
                                )
                                
                                st.rerun()
                            else:
                                st.error(f"❌ {mensaje_estructura}")
        else:
            st.warning("⚠️ Debes seleccionar una variable objetivo para continuar.")
            
    elif st.session_state.paso_configuracion == 2:
        # Resumen de configuración y próximos pasos
        st.write("### ✅ Configuración completada correctamente")
          # Mostrar resumen de configuración
        st.write("#### Resumen de configuración:")
        
        col1, col2 = st.columns(2)
        with col1:
            tipo_problema_cap = st.session_state.tipo_problema.capitalize() if st.session_state.tipo_problema else "Problema"
            st.write(f"**Tipo de problema:** {tipo_problema_cap}")
            st.write(f"**Variable objetivo:** {st.session_state.variable_objetivo}")
        
        with col2:
            # Mostrar estadísticas de la variable objetivo
            if st.session_state.tipo_problema == "regresion":
                st.write("**Estadísticas de la variable objetivo:**")
                stats = st.session_state.df[st.session_state.variable_objetivo].describe()
                st.write(f"- Min: {stats['min']:.2f}")
                st.write(f"- Max: {stats['max']:.2f}")
                st.write(f"- Media: {stats['mean']:.2f}")
                st.write(f"- Desv. Estándar: {stats['std']:.2f}")
            else:  # clasificación
                st.write("**Distribución de clases:**")
                clase_counts = st.session_state.df[st.session_state.variable_objetivo].value_counts()
                for clase, count in clase_counts.items():
                    st.write(f"- {clase}: {count} ({count/len(st.session_state.df)*100:.1f}%)")
          # Mostrar variables predictoras
        st.write(f"**Variables predictoras seleccionadas ({len(st.session_state.variables_predictoras)}):**")
        if len(st.session_state.variables_predictoras) > 10:
            st.write(", ".join(st.session_state.variables_predictoras[:10]) + f" y {len(st.session_state.variables_predictoras)-10} más...")
        else:
            st.write(", ".join(st.session_state.variables_predictoras))
          
        # Botones de navegación
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Modificar configuración", use_container_width=True):
                st.session_state.paso_configuracion = 1
                st.session_state.configuracion_validada = False
                st.rerun()
        
        with col2:
            if st.button("➡️ Entrenar Modelos", use_container_width=True):
                # Registrar acción
                log_audit(
                    usuario_id, 
                    "NAVEGACION", 
                    "entrenar_modelos", 
                    st.session_state.filename,
                    f"Continuar con entrenamiento de modelos para {st.session_state.filename}",
                    id_sesion=st.session_state.get("id_sesion", "sin_sesion")
                )
                st.switch_page("pages/Machine Learning/05_Entrenar_Modelos.py")
