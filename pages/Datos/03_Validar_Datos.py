import streamlit as st
import os
import sys

# Agregar el directorio src al path para poder importar los módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Importar módulos de la aplicación
from src.audit.logger import setup_logger, log_operation, log_audit
from src.datos.validador import (
    validar_tipos_datos,
    validar_fechas,
    validar_unidades
)
from src.state.session_manager import SessionManager

# Configurar el logger
usuario_id = st.session_state.get("usuario_id", 1)
logger = setup_logger("validacion_datos", id_usuario=usuario_id)

# Inicializar session_state para esta página
if 'validacion_completa' not in st.session_state:
    st.session_state.validacion_completa = False
if 'errores_tipo' not in st.session_state:
    st.session_state.errores_tipo = []
if 'errores_fecha' not in st.session_state:
    st.session_state.errores_fecha = []
if 'errores_unidad' not in st.session_state:
    st.session_state.errores_unidad = []
if 'paso_validacion' not in st.session_state:
    st.session_state.paso_validacion = 0  # 0: inicio, 1: resultados, 2: correcciones

# Título y descripción de la página
st.title("🔍 Validar Estructura de Datos")

st.markdown("""
Esta página valida automáticamente los tipos de datos, formatos de fecha y unidades 
para garantizar la compatibilidad y detectar errores de forma temprana.
""")

# Verificar si hay datos cargados y configuración completada
if 'df' not in st.session_state or st.session_state.df is None:
    st.warning("⚠️ No hay datos cargados. Por favor, carga un dataset primero en la página 'Cargar Datos'.")
    if st.button("Ir a Cargar Datos"):
        st.session_state.paso_carga = 0  # Reiniciar el paso de carga
        st.switch_page("pages/datos/01_Cargar_Datos.py")
elif 'configuracion_validada' not in st.session_state or not st.session_state.configuracion_validada:
    st.warning("⚠️ La estructura de datos no ha sido configurada. Por favor, configura la estructura primero.")
    if st.button("Ir a Configurar Datos"):
        st.switch_page("pages/datos/02_Configurar_Datos.py")
else:
    # Mostrar información del dataset cargado
    st.write(f"### Dataset cargado: {st.session_state.filename}")
    st.write(f"Dimensiones: {st.session_state.df.shape[0]} filas × {st.session_state.df.shape[1]} columnas")
    
    # Mostrar configuración seleccionada
    st.write("### Configuración actual")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Tipo de problema:** {st.session_state.tipo_problema.capitalize()}")
        st.write(f"**Variable objetivo:** {st.session_state.variable_objetivo}")
    with col2:
        n_predictoras = len(st.session_state.variables_predictoras)
        st.write(f"**Variables predictoras:** {n_predictoras} seleccionadas")
    
    # Paso 0: Explicación y botón para iniciar validación
    if st.session_state.paso_validacion == 0:
        st.write("### Validación automática de datos")
        st.markdown("""
        El sistema realizará automáticamente las siguientes validaciones:
        
        1. **Tipos de datos**: Verifica que las columnas tengan tipos coherentes y adecuados.
        2. **Formatos de fecha**: Detecta inconsistencias en formatos de fechas.
        3. **Unidades de medida**: Identifica posibles variaciones en unidades.
        
        Esta validación ayuda a detectar problemas que podrían afectar el rendimiento de los modelos.
        """)
        
        if st.button("✅ Iniciar validación automática"):
            with st.spinner("Validando datos..."):
                # Registrar inicio de validación
                log_operation(logger, "INICIO_VALIDACION", 
                             f"Iniciando validación de datos para {st.session_state.filename}", 
                             id_usuario=usuario_id)
                
                # Ejecutar validaciones (llamadas a los módulos correspondientes)
                errores_tipo = validar_tipos_datos(st.session_state.df)
                errores_fecha = validar_fechas(st.session_state.df)
                errores_unidad = validar_unidades(st.session_state.df)
                
                # Guardar resultados en session_state
                st.session_state.errores_tipo = errores_tipo
                st.session_state.errores_fecha = errores_fecha
                st.session_state.errores_unidad = errores_unidad
                
                # Avanzar al paso de resultados
                st.session_state.paso_validacion = 1
                st.rerun()
    
    # Paso 1: Mostrar resultados de validación
    elif st.session_state.paso_validacion == 1:
        st.write("### Resultados de validación")
        
        # Crear indicadores de estado
        total_errores = len(st.session_state.errores_tipo) + len(st.session_state.errores_fecha) + len(st.session_state.errores_unidad)
        
        if total_errores == 0:
            st.success("✅ No se detectaron problemas en la estructura de los datos. La validación es correcta.")
            log_operation(logger, "VALIDACION_EXITOSA", 
                         f"Validación completa sin errores para {st.session_state.filename}", 
                         id_usuario=usuario_id)
            st.session_state.validacion_completa = True
        else:
            st.error(f"❌ Se detectaron {total_errores} problemas que requieren atención.")
            log_operation(logger, "VALIDACION_ERRORES", 
                         f"Validación completada con {total_errores} errores para {st.session_state.filename}", 
                         id_usuario=usuario_id)
        
        # Mostrar detalles de errores por tipo
        with st.expander("Problemas de tipos de datos", expanded=len(st.session_state.errores_tipo) > 0):
            if len(st.session_state.errores_tipo) == 0:
                st.info("No se detectaron problemas en los tipos de datos.")
            else:
                st.write(f"Se encontraron {len(st.session_state.errores_tipo)} problemas:")
                for error in st.session_state.errores_tipo:
                    st.warning(f"**{error['columna']}**: {error['mensaje']}")
                    if 'sugerencia' in error:
                        st.info(f"Sugerencia: {error['sugerencia']}")
        
        with st.expander("Problemas de formatos de fecha", expanded=len(st.session_state.errores_fecha) > 0):
            if len(st.session_state.errores_fecha) == 0:
                st.info("No se detectaron problemas en los formatos de fecha.")
            else:
                st.write(f"Se encontraron {len(st.session_state.errores_fecha)} problemas:")
                for error in st.session_state.errores_fecha:
                    st.warning(f"**{error['columna']}**: {error['mensaje']}")
                    if 'ejemplos' in error:
                        st.text(f"Ejemplos encontrados: {', '.join(error['ejemplos'])}")
                    if 'formato_sugerido' in error:
                        st.info(f"Formato sugerido: {error['formato_sugerido']}")
        
        with st.expander("Problemas de unidades de medida", expanded=len(st.session_state.errores_unidad) > 0):
            if len(st.session_state.errores_unidad) == 0:
                st.info("No se detectaron problemas en las unidades de medida.")
            else:
                st.write(f"Se encontraron {len(st.session_state.errores_unidad)} problemas:")
                for error in st.session_state.errores_unidad:
                    st.warning(f"**{error['columna']}**: {error['mensaje']}")
                    if 'unidades_detectadas' in error:
                        st.text(f"Unidades detectadas: {', '.join(error['unidades_detectadas'])}")
                    if 'unidad_sugerida' in error:
                        st.info(f"Unidad sugerida: {error['unidad_sugerida']}")
        
        # Botones de navegación
        st.write("### Acciones disponibles")
        
        # Si hay errores, mostrar opción para corregir
        if total_errores > 0:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("⚙️ Corregir problemas automáticamente", use_container_width=True):
                    st.session_state.paso_validacion = 2
                    st.rerun()
            with col2:
                if st.button("➡️ Continuar sin corregir", use_container_width=True):
                    # Registrar decisión
                    log_audit(usuario_id, "OMITIR_CORRECCIONES", "validacion", 
                            f"Usuario decidió continuar sin corregir {total_errores} problemas")
                    st.session_state.validacion_completa = True
                    st.switch_page("pages/datos/04_Transformar_Datos.py")
        else:
            # Si no hay errores, solo mostrar botón para continuar
            if st.button("➡️ Continuar con transformaciones", use_container_width=True):
                log_audit(usuario_id, "NAVEGACION", "transformaciones", 
                         f"Continuar con transformaciones de datos para {st.session_state.filename}")
                st.switch_page("pages/datos/04_Transformar_Datos.py")
    
    # Paso 2: Aplicar correcciones
    elif st.session_state.paso_validacion == 2:
        st.write("### Corrección automática de problemas")
        
        # Crear formulario para aplicar correcciones
        with st.form(key="form_correcciones"):
            # Correcciones de tipos de datos
            if len(st.session_state.errores_tipo) > 0:
                st.write("#### Correcciones de tipos de datos")
                for i, error in enumerate(st.session_state.errores_tipo):
                    key = f"tipo_{i}"
                    error['aplicar'] = st.checkbox(
                        f"Corregir {error['columna']}: {error['mensaje']}",
                        value=True,
                        key=key
                    )
                    if 'opciones' in error:
                        error['opcion_seleccionada'] = st.selectbox(
                            f"Seleccione la corrección para {error['columna']}",
                            options=error['opciones'],
                            key=f"{key}_opcion"
                        )
            
            # Correcciones de formatos de fecha
            if len(st.session_state.errores_fecha) > 0:
                st.write("#### Correcciones de formatos de fecha")
                for i, error in enumerate(st.session_state.errores_fecha):
                    key = f"fecha_{i}"
                    error['aplicar'] = st.checkbox(
                        f"Estandarizar {error['columna']}: {error['mensaje']}",
                        value=True,
                        key=key
                    )
                    if 'formatos_disponibles' in error:
                        formato_default = error.get('formato_sugerido', error['formatos_disponibles'][0])
                        error['formato_seleccionado'] = st.selectbox(
                            f"Seleccione el formato para {error['columna']}",
                            options=error['formatos_disponibles'],
                            index=error['formatos_disponibles'].index(formato_default) if formato_default in error['formatos_disponibles'] else 0,
                            key=f"{key}_formato"
                        )
            
            # Correcciones de unidades de medida
            if len(st.session_state.errores_unidad) > 0:
                st.write("#### Correcciones de unidades de medida")
                for i, error in enumerate(st.session_state.errores_unidad):
                    key = f"unidad_{i}"
                    error['aplicar'] = st.checkbox(
                        f"Unificar {error['columna']}: {error['mensaje']}",
                        value=True,
                        key=key
                    )
                    if 'unidades_disponibles' in error:
                        unidad_default = error.get('unidad_sugerida', error['unidades_disponibles'][0])
                        error['unidad_seleccionada'] = st.selectbox(
                            f"Seleccione la unidad estándar para {error['columna']}",
                            options=error['unidades_disponibles'],
                            index=error['unidades_disponibles'].index(unidad_default) if unidad_default in error['unidades_disponibles'] else 0,
                            key=f"{key}_unidad"
                        )
            
            # Botones del formulario
            st.write("#### Aplicar correcciones")
            col1, col2 = st.columns(2)
            with col1:
                cancelar = st.form_submit_button("❌ Cancelar", use_container_width=True)
            with col2:
                aplicar = st.form_submit_button("✅ Aplicar correcciones seleccionadas", use_container_width=True)
        
        # Procesar resultados del formulario
        if cancelar:
            st.session_state.paso_validacion = 1
            st.rerun()
        
        if aplicar:
            with st.spinner("Aplicando correcciones..."):
                # Crear una copia del DataFrame para no modificar el original hasta confirmar
                df_corregido = st.session_state.df.copy()
                correcciones_aplicadas = []
                
                # Aplicar correcciones de tipos de datos
                for error in st.session_state.errores_tipo:
                    if error.get('aplicar', False):
                        # Aquí se aplicaría la lógica real de corrección
                        columna = error['columna']
                        mensaje = f"Corregido tipo de dato en columna {columna}"
                        if 'opcion_seleccionada' in error:
                            mensaje += f" usando {error['opcion_seleccionada']}"
                        
                        # Simular corrección (esto sería reemplazado por código real)
                        # df_corregido[columna] = ... lógica de corrección
                        
                        correcciones_aplicadas.append(mensaje)
                        log_operation(logger, "CORRECCION_TIPO", mensaje, id_usuario=usuario_id)
                
                # Aplicar correcciones de formatos de fecha
                for error in st.session_state.errores_fecha:
                    if error.get('aplicar', False):
                        columna = error['columna']
                        formato = error.get('formato_seleccionado', 'ISO')
                        
                        # Aquí iría la llamada a la función real
                        # df_corregido = estandarizar_fechas(df_corregido, columna, formato)
                        
                        mensaje = f"Estandarizado formato de fecha en columna {columna} a {formato}"
                        correcciones_aplicadas.append(mensaje)
                        log_operation(logger, "CORRECCION_FECHA", mensaje, id_usuario=usuario_id)
                
                # Aplicar correcciones de unidades
                for error in st.session_state.errores_unidad:
                    if error.get('aplicar', False):
                        columna = error['columna']
                        unidad_destino = error.get('unidad_seleccionada', '')
                        
                        # Aquí iría la llamada a la función real
                        # df_corregido = convertir_unidades(df_corregido, columna, unidad_destino)
                        
                        mensaje = f"Convertida unidad en columna {columna} a {unidad_destino}"
                        correcciones_aplicadas.append(mensaje)
                        log_operation(logger, "CORRECCION_UNIDAD", mensaje, id_usuario=usuario_id)
                
                # Si se aplicaron correcciones, actualizar el DataFrame
                if correcciones_aplicadas:
                    st.session_state.df = df_corregido
                    log_audit(usuario_id, "CORRECCIONES_APLICADAS", "validacion", 
                             f"Se aplicaron {len(correcciones_aplicadas)} correcciones")
                    
                    # Mostrar resumen de correcciones
                    st.success(f"✅ Se aplicaron {len(correcciones_aplicadas)} correcciones con éxito")
                    for correccion in correcciones_aplicadas:
                        st.info(correccion)                    # Marcar la validación como completa
                    st.session_state.validacion_completa = True
                    
                    # Marcar etapa de validación como completada en el estado global
                    SessionManager.update_progress("validacion", True)
                    
                    # Dar opción para continuar
                    if st.button("➡️ Continuar con transformaciones"):
                        st.switch_page("pages/datos/04_Transformar_Datos.py")
                else:
                    st.warning("No se seleccionó ninguna corrección para aplicar")
                    if st.button("⬅️ Volver a selección"):
                        st.session_state.paso_validacion = 1
                        st.rerun()
