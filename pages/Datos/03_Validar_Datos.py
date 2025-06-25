import streamlit as st
import os
import sys
import pandas as pd

# Agregar el directorio src al path para poder importar los módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Importar módulos de la aplicación
from src.audit.logger import setup_logger, log_operation, log_audit
from src.datos.validador import (
    validar_tipos_datos,
    validar_fechas,
    validar_unidades
)
from src.datos.transformador import (
    corregir_tipo_datos,
    estandarizar_fechas,
    convertir_unidades
)
from src.datos.limpiador import (
    detectar_duplicados,
    eliminar_duplicados,
    fusionar_duplicados
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
    st.session_state.paso_validacion = 0  # 0: inicio, 1: resultados, 2: correcciones, 3: duplicados
if 'duplicados_info' not in st.session_state:
    st.session_state.duplicados_info = None

# Título y descripción de la página
st.title("🔍 Validar Estructura de Datos")

st.markdown("""
Esta página valida automáticamente los tipos de datos, formatos de fecha y unidades 
para garantizar la compatibilidad y detectar errores de forma temprana.
""")

# Verificar si hay datos cargados y configuración completada
if 'df' not in st.session_state or st.session_state.df is None:
    st.warning("⚠️ No hay datos cargados. Por favor, carga un dataset primero en la página 'Cargar Datos'.")
    if st.button("Ir a Cargar Datos", use_container_width=True):
        st.session_state.paso_carga = 0  # Reiniciar el paso de carga
        st.switch_page("pages/datos/01_Cargar_Datos.py")
elif 'configuracion_validada' not in st.session_state or not st.session_state.configuracion_validada:
    st.warning("⚠️ La estructura de datos no ha sido configurada. Por favor, configura la estructura primero.")
    if st.button("Ir a Configurar Datos", use_container_width=True):
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
        
        if st.button("✅ Iniciar validación automática", use_container_width=True):
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

        if total_errores > 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("⬅️ Volver a configurar datos", use_container_width=True):
                    st.switch_page("pages/datos/02_Configurar_Datos.py")
            with col2:
                if st.button("⚙️ Corregir automáticamente", use_container_width=True):
                    st.session_state.paso_validacion = 2
                    st.rerun()
            with col3:
                if st.button("➡️ Continuar sin corregir", use_container_width=True):
                    log_audit(usuario_id, "OMITIR_CORRECCIONES", "validacion", 
                          f"Usuario decidió continuar sin corregir {total_errores} problemas")
                    st.session_state.validacion_completa = True
                    st.switch_page("pages/datos/04_Analizar_Calidad.py")
        else:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("⬅️ Volver a configurar datos", use_container_width=True):
                    st.switch_page("pages/datos/02_Configurar_Datos.py")
            with col2:
                if st.button("➡️ Continuar con análisis de calidad", use_container_width=True):
                    log_audit(usuario_id, "NAVEGACION", "análisis de calidad de datos", 
                          f"Continuar con el análisis de calidad de datos para {st.session_state.filename}")
                    st.switch_page("pages/datos/04_Analizar_Calidad.py")   
    
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
                        columna = error['columna']
                        tipo_destino = error.get('opcion_seleccionada', 'str')
                        
                        # Aplicar la corrección real
                        try:
                            df_corregido = corregir_tipo_datos(df_corregido, columna, tipo_destino)
                            mensaje = f"Corregido tipo de dato en columna {columna} a {tipo_destino}"
                            correcciones_aplicadas.append(mensaje)
                            log_operation(logger, "CORRECCION_TIPO", mensaje, id_usuario=usuario_id)
                        except Exception as e:
                            st.error(f"Error al corregir tipo de dato en columna {columna}: {str(e)}")
                            log_operation(logger, "ERROR_CORRECCION", 
                                         f"Error al corregir tipo de {columna}: {str(e)}", 
                                         id_usuario=usuario_id)                # Aplicar correcciones de formatos de fecha
                for error in st.session_state.errores_fecha:
                    if error.get('aplicar', False):
                        columna = error['columna']
                        formato_seleccionado = error.get('formato_seleccionado', 'ISO 8601 (YYYY-MM-DD)')
                        
                        # Mapear el formato seleccionado al formato interno
                        formato_mapping = {
                            'ISO 8601 (YYYY-MM-DD)': 'ISO',
                            'DD/MM/YYYY': 'DMY',
                            'MM/DD/YYYY': 'MDY',
                            'YYYY/MM/DD': 'YMD'
                        }
                        formato = formato_mapping.get(formato_seleccionado, 'ISO')
                        
                        # Aplicar la corrección real
                        try:
                            df_corregido = estandarizar_fechas(df_corregido, columna, formato)
                            mensaje = f"Estandarizado formato de fecha en columna {columna} a {formato_seleccionado}"
                            correcciones_aplicadas.append(mensaje)
                            log_operation(logger, "CORRECCION_FECHA", mensaje, id_usuario=usuario_id)
                        except Exception as e:
                            st.error(f"Error al estandarizar fechas en columna {columna}: {str(e)}")
                            log_operation(logger, "ERROR_CORRECCION", 
                                         f"Error al estandarizar fechas en {columna}: {str(e)}", 
                                         id_usuario=usuario_id)
                  # Aplicar correcciones de unidades
                for error in st.session_state.errores_unidad:
                    if error.get('aplicar', False):
                        columna = error['columna']
                        unidad_destino = error.get('unidad_seleccionada', '')
                        
                        # Aplicar la corrección real
                        try:
                            # Si hay unidades de origen en el error, las usamos
                            unidad_origen = error.get('unidad_origen', None)
                            df_corregido = convertir_unidades(df_corregido, columna, unidad_destino, unidad_origen)
                            mensaje = f"Convertida unidad en columna {columna} a {unidad_destino}"
                            correcciones_aplicadas.append(mensaje)
                            log_operation(logger, "CORRECCION_UNIDAD", mensaje, id_usuario=usuario_id)
                        except Exception as e:
                            st.error(f"Error al convertir unidades en columna {columna}: {str(e)}")
                            log_operation(logger, "ERROR_CORRECCION", 
                                         f"Error al convertir unidades en {columna}: {str(e)}", 
                                         id_usuario=usuario_id)                # Si se aplicaron correcciones, actualizar el DataFrame
                if correcciones_aplicadas:
                    st.session_state.df = df_corregido
                    log_audit(usuario_id, "CORRECCIONES_APLICADAS", "validacion", 
                             f"Se aplicaron {len(correcciones_aplicadas)} correcciones")
                    
                    # Mostrar resumen de correcciones
                    st.success(f"✅ Se aplicaron {len(correcciones_aplicadas)} correcciones con éxito")
                    
                    # Agregar historial detallado de correcciones
                    st.write("### Historial de correcciones aplicadas")
                    
                    # Crear una tabla con las correcciones
                    correcciones_tabla = []
                    for i, correccion in enumerate(correcciones_aplicadas, 1):
                        # Identificar el tipo de corrección
                        tipo = ""
                        if "tipo de dato" in correccion:
                            tipo = "Tipo de dato"
                            icono = "🔢"
                        elif "fecha" in correccion:
                            tipo = "Formato de fecha"
                            icono = "📅"
                        elif "unidad" in correccion:
                            tipo = "Unidad de medida"
                            icono = "📏"
                        else:
                            tipo = "Otro"
                            icono = "🔧"
                        
                        # Extraer columna del mensaje
                        import re
                        columna_match = re.search(r"columna\s+(\w+)", correccion)
                        columna = columna_match.group(1) if columna_match else "—"
                        
                        # Agregar a la lista para la tabla
                        correcciones_tabla.append({"#": i, "Tipo": f"{icono} {tipo}", "Columna": columna, "Detalle": correccion})
                    
                    # Mostrar tabla de correcciones
                    if correcciones_tabla:
                        st.table(correcciones_tabla)
                    
                    # Guardar historial de correcciones en session_state para referencia futura
                    if 'historial_correcciones' not in st.session_state:
                        st.session_state.historial_correcciones = []
                    
                    # Agregar fecha y nombre del archivo al historial
                    import datetime
                    registro_correcciones = {
                        "fecha": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "archivo": st.session_state.filename,
                        "correcciones": correcciones_aplicadas
                    }
                    st.session_state.historial_correcciones.append(registro_correcciones)
                      # Información individual (mantener para compatibilidad)
                    for correccion in correcciones_aplicadas:
                        st.info(correccion)
                    
                    # Marcar etapa de validación como completada en el estado global
                    SessionManager.update_progress("validacion", True)
                      # Opciones para continuar
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🔍 Detectar Duplicados", use_container_width=True):
                            st.session_state.paso_validacion = 3
                            st.session_state.duplicados_info = None  # Importante: inicializar a None para el flujo correcto
                            st.rerun()
                    with col2:
                        if st.button("➡️ Continuar con el análisis de calidad", use_container_width=True):
                            st.session_state.validacion_completa = True
                            st.switch_page("pages/datos/04_Analizar_Calidad.py")
                else:
                    st.warning("No se seleccionó ninguna corrección para aplicar")
                    
                    if st.button("⬅️ Volver a selección", use_container_width=True):
                        st.session_state.paso_validacion = 1
                        st.rerun()    # Paso 3: Gestión de duplicados
    elif st.session_state.paso_validacion == 3:
        st.write("### Gestión de Duplicados")
        
        st.markdown("""
        Esta herramienta le permite detectar, analizar y gestionar datos duplicados en su conjunto de datos.
        Puede seleccionar las columnas clave para identificar duplicados y decidir cómo tratarlos.
        """)
        
        # Paso 1: Selección de columnas para detectar duplicados
        if st.session_state.duplicados_info is None:
            st.write("#### Paso 1: Seleccionar columnas clave para identificar duplicados")
            
            st.info("Seleccione las columnas que deben considerarse para identificar registros duplicados. "
                   "Los registros que tengan los mismos valores en todas estas columnas se considerarán duplicados.")
            
            # Opciones para seleccionar todas o un subconjunto de columnas
            opcion_columnas = st.radio(
                "¿Qué columnas desea usar para identificar duplicados?",
                ["Todas las columnas", "Seleccionar columnas específicas"],
                index=1
            )
            
            # Inicializar columnas_seleccionadas para evitar desvinculación
            columnas_seleccionadas = []
            if opcion_columnas == "Todas las columnas":
                columnas_seleccionadas = st.session_state.df.columns.tolist()
            else:
                # Mostrar multiselect con todas las columnas disponibles
                columnas_seleccionadas = st.multiselect(
                    "Seleccione las columnas clave para identificar duplicados:",
                    st.session_state.df.columns.tolist(),
                    default=[st.session_state.variable_objetivo] if st.session_state.variable_objetivo in st.session_state.df.columns else []
                )
            
            # Botón para detectar duplicados
            col1, col2 = st.columns(2)
            with col1:
                if st.button("⬅️ Volver", use_container_width=True):
                    st.session_state.paso_validacion = 2
                    st.rerun()
            with col2:
                if st.button("🔍 Detectar duplicados", use_container_width=True, disabled=len(columnas_seleccionadas) == 0):
                    if len(columnas_seleccionadas) == 0:
                        st.error("Debe seleccionar al menos una columna para detectar duplicados.")
                    else:
                        with st.spinner("Analizando duplicados..."):
                            # Detectar duplicados
                            st.session_state.duplicados_info = detectar_duplicados(st.session_state.df, columnas_seleccionadas)
                            log_operation(logger, "DETECCION_DUPLICADOS", 
                                        f"Detección de duplicados en {len(columnas_seleccionadas)} columnas", 
                                        id_usuario=usuario_id)
                            st.rerun()
        
        # Paso 2: Mostrar resultados y opciones de gestión
        else:
            # Obtener información de duplicados
            info = st.session_state.duplicados_info
            
            # Verificar si hubo un error
            if 'error' in info and info['error']:
                st.error(f"Error al detectar duplicados: {info['error']}")
                if st.button("⬅️ Volver a seleccionar columnas", use_container_width=True):
                    st.session_state.duplicados_info = None
                    st.rerun()
            else:
                # Mostrar resumen de duplicados encontrados
                st.write("#### Resumen de duplicados encontrados")
                
                if info is not None and info.get('cantidad', 0) == 0:
                    st.success("✅ No se detectaron registros duplicados en las columnas seleccionadas.")
                    
                    # Botones de navegación
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("⬅️ Volver a seleccionar columnas", use_container_width=True):
                            st.session_state.duplicados_info = None
                            st.rerun()
                    with col2:
                        if st.button("➡️ Continuar con el análisis de calidad", use_container_width=True):
                            st.session_state.validacion_completa = True
                            st.switch_page("pages/datos/04_Analizar_Calidad.py")
                else:
                    # Mostrar métricas
                    col1, col2 = st.columns(2)
                    with col1:
                        cantidad_duplicados = info['cantidad'] if info is not None and 'cantidad' in info else 0
                        st.metric("Registros duplicados", f"{cantidad_duplicados} filas")
                    with col2:
                        porcentaje_duplicados = info['porcentaje'] if info is not None and 'porcentaje' in info else 0
                        st.metric("Porcentaje del total", f"{porcentaje_duplicados:.2f}%")
                    
                    # Mostrar grupos de duplicados
                    if info is not None and 'grupos_duplicados' in info and info['grupos_duplicados'] is not None and not info['grupos_duplicados'].empty:
                        st.write("#### Grupos de registros duplicados")
                        st.write(f"Se encontraron {len(info['grupos_duplicados'])} grupos con registros duplicados:")
                        st.dataframe(info['grupos_duplicados'])
                    
                    # Mostrar ejemplos de filas duplicadas
                    st.write("#### Ejemplos de registros duplicados")
                    if info is not None and 'indices' in info and info['indices'] is not None and len(info['indices']) > 0:
                        # Obtener los índices de duplicados y sus filas correspondientes
                        muestra_indices = info['indices'][:min(5, len(info['indices']))]
                        
                        # Obtener las filas originales que son duplicadas (para mostrar pares completos)
                        # Mostrar solo una muestra representativa para evitar sobrecargar la interfaz
                        df_duplicados = st.session_state.df.loc[muestra_indices]
                        st.write(f"Mostrando {len(df_duplicados)} ejemplos de registros duplicados:")
                        st.dataframe(df_duplicados)
                    
                    # Opciones de gestión de duplicados
                    st.write("#### Opciones para gestionar duplicados")
                    
                    metodo_gestion = st.radio(
                        "¿Cómo desea gestionar los registros duplicados?",
                        ["Eliminar duplicados", "Fusionar duplicados"],
                        help="Eliminar conservará solo uno de los registros. Fusionar combinará la información de todos los duplicados."
                    )
                    
                    if metodo_gestion == "Eliminar duplicados":
                        # Opciones de eliminación
                        estrategia = st.radio(
                            "Estrategia de eliminación:",
                            ["Conservar primera ocurrencia", "Conservar última ocurrencia", "Eliminar todas las ocurrencias"],
                            help="Seleccione qué ocurrencias desea mantener cuando encuentre duplicados."
                        )
                        
                        estrategia_map = {
                            "Conservar primera ocurrencia": "first",
                            "Conservar última ocurrencia": "last",
                            "Eliminar todas las ocurrencias": False
                        }
                        
                        # Previsualizar resultado
                        if st.button("👁️ Previsualizar resultado", use_container_width=True):
                            with st.spinner("Calculando resultado..."):
                                # Obtener las columnas originales usadas para detectar duplicados
                                columnas_clave = None
                                if info is not None:
                                    columnas_clave = info.get('columnas_usadas', None)
                                if columnas_clave is None and info is not None:
                                    # Si no están en info, reconstruirlas del primer grupo (columnas menos la de conteo)
                                    if 'grupos_duplicados' in info and not info['grupos_duplicados'].empty:
                                        columnas_clave = info['grupos_duplicados'].columns.tolist()
                                        if 'conteo' in columnas_clave:
                                            columnas_clave.remove('conteo')
                                
                                # Aplicar eliminación (sin guardar cambios)
                                df_resultado, resultado_info = eliminar_duplicados(
                                    st.session_state.df,
                                    columnas_clave,
                                    estrategia_map[estrategia]
                                )
                                
                                # Mostrar resultados
                                st.write("##### Resultado de la eliminación")
                                
                                # Métricas de resultado
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Filas originales", resultado_info['filas_antes'])
                                with col2:
                                    st.metric("Filas eliminadas", resultado_info['filas_eliminadas'])
                                with col3:
                                    st.metric("Filas resultantes", resultado_info['filas_despues'])
                                
                                # Mostrar muestra del DataFrame resultante
                                st.write("Muestra del conjunto de datos después de eliminar duplicados:")
                                st.dataframe(df_resultado.head(10))
                                
                                # Guardar resultado temporalmente para aplicación posterior
                                st.session_state.df_resultado_temp = df_resultado
                                st.session_state.resultado_info_temp = resultado_info
                                
                                # Botón para aplicar cambios
                                if st.button("✅ Aplicar eliminación de duplicados", use_container_width=True):
                                    # Guardar el DataFrame modificado
                                    st.session_state.df = df_resultado
                                    
                                    # Registrar la operación en el log
                                    mensaje = f"Eliminados {resultado_info['filas_eliminadas']} registros duplicados "
                                    mensaje += f"({resultado_info['porcentaje_reduccion']:.2f}% de reducción)"
                                    
                                    log_operation(logger, "ELIMINACION_DUPLICADOS", mensaje, id_usuario=usuario_id)
                                    log_audit(usuario_id, "ELIMINACION_DUPLICADOS", "validacion", mensaje)
                                    
                                    # Actualizar estado y mostrar confirmación
                                    st.success(f"✅ {mensaje}")
                                    
                                    # Registrar en historial de correcciones
                                    if 'historial_correcciones' not in st.session_state:
                                        st.session_state.historial_correcciones = []
                                    
                                    import datetime
                                    registro_correccion = {
                                        "fecha": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        "archivo": st.session_state.filename,
                                        "correcciones": [mensaje]
                                    }
                                    st.session_state.historial_correcciones.append(registro_correccion)
                                    
                                    # Marcar validación como completa
                                    st.session_state.validacion_completa = True
                                    
                                    # Reiniciar variables de duplicados
                                    st.session_state.duplicados_info = None
                                    if 'df_resultado_temp' in st.session_state:
                                        del st.session_state.df_resultado_temp
                                    if 'resultado_info_temp' in st.session_state:
                                        del st.session_state.resultado_info_temp
                                    
                                    # Botón para continuar
                                    if st.button("➡️ Continuar con el análisis de calidad", use_container_width=True):
                                        st.switch_page("pages/datos/04_Analizar_Calidad.py")
                    
                    elif metodo_gestion == "Fusionar duplicados":
                        st.write("#### Configurar métodos de fusión por columna")
                        st.info("Seleccione cómo desea combinar los valores de cada columna para los registros duplicados.")
                        
                        # Obtener las columnas originales usadas para detectar duplicados
                        columnas_clave = None
                        if info is not None:
                            columnas_clave = info.get('columnas_usadas', None)
                        if columnas_clave is None and info is not None:
                            # Si no están en info, reconstruirlas del primer grupo (columnas menos la de conteo)
                            if not info['grupos_duplicados'].empty:
                                columnas_clave = info['grupos_duplicados'].columns.tolist()
                                if 'conteo' in columnas_clave:
                                    columnas_clave.remove('conteo')
                        # Fallback: si columnas_clave sigue siendo None, usar todas las columnas
                        if columnas_clave is None:
                            columnas_clave = st.session_state.df.columns.tolist()
                        
                        # Determinar columnas a fusionar (todas menos las clave)
                        columnas_fusion = [col for col in st.session_state.df.columns if col not in columnas_clave]
                        
                        # Crear diccionario para almacenar métodos de fusión
                        metodos_fusion = {}
                        
                        # Para cada columna, mostrar opciones de método de fusión
                        for col in columnas_fusion:
                            # Determinar qué métodos son aplicables según el tipo de datos
                            tipo_col = st.session_state.df[col].dtype
                            
                            if pd.api.types.is_numeric_dtype(tipo_col):
                                # Columnas numéricas
                                opciones = ["first", "last", "mean", "sum", "min", "max", "median"]
                                descripcion = {
                                    "first": "Primer valor", 
                                    "last": "Último valor",
                                    "mean": "Promedio", 
                                    "sum": "Suma", 
                                    "min": "Mínimo", 
                                    "max": "Máximo",
                                    "median": "Mediana"
                                }
                            else:
                                # Columnas no numéricas
                                opciones = ["first", "last", "most_frequent"]
                                descripcion = {
                                    "first": "Primer valor", 
                                    "last": "Último valor",
                                    "most_frequent": "Valor más frecuente"
                                }
                            
                            # Mostrar selector de método
                            metodo = st.selectbox(
                                f"Método para fusionar '{col}':",
                                options=opciones,
                                format_func=lambda x: f"{descripcion[x]} ({x})",
                                key=f"fusion_{col}"
                            )
                            
                            # Guardar método seleccionado
                            metodos_fusion[col] = metodo
                        
                        # Previsualizar resultado
                        if st.button("👁️ Previsualizar resultado", use_container_width=True):
                            with st.spinner("Calculando resultado..."):
                                # Aplicar fusión (sin guardar cambios)
                                df_resultado, resultado_info = fusionar_duplicados(
                                    st.session_state.df,
                                    columnas_clave,
                                    metodos_fusion
                                )
                                
                                # Mostrar resultados
                                st.write("##### Resultado de la fusión")
                                
                                # Métricas de resultado
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Filas originales", resultado_info['filas_antes'])
                                with col2:
                                    st.metric("Grupos fusionados", resultado_info['grupos_fusionados'])
                                with col3:
                                    st.metric("Filas resultantes", resultado_info['filas_despues'])
                                
                                # Mostrar muestra del DataFrame resultante
                                st.write("Muestra del conjunto de datos después de fusionar duplicados:")
                                st.dataframe(df_resultado.head(10))
                                
                                # Guardar resultado temporalmente para aplicación posterior
                                st.session_state.df_resultado_temp = df_resultado
                                st.session_state.resultado_info_temp = resultado_info
                                
                                # Botón para aplicar cambios
                                if st.button("✅ Aplicar fusión de duplicados", use_container_width=True):
                                    # Guardar el DataFrame modificado
                                    st.session_state.df = df_resultado
                                    
                                    # Registrar la operación en el log
                                    mensaje = f"Fusionados {resultado_info['grupos_fusionados']} grupos de duplicados "
                                    mensaje += f"(reducción de {resultado_info['filas_antes'] - resultado_info['filas_despues']} filas)"
                                    
                                    log_operation(logger, "FUSION_DUPLICADOS", mensaje, id_usuario=usuario_id)
                                    log_audit(usuario_id, "FUSION_DUPLICADOS", "validacion", mensaje)
                                    
                                    # Actualizar estado y mostrar confirmación
                                    st.success(f"✅ {mensaje}")
                                    
                                    # Registrar en historial de correcciones
                                    if 'historial_correcciones' not in st.session_state:
                                        st.session_state.historial_correcciones = []
                                    
                                    import datetime
                                    registro_correccion = {
                                        "fecha": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        "archivo": st.session_state.filename,
                                        "correcciones": [mensaje]
                                    }
                                    st.session_state.historial_correcciones.append(registro_correccion)
                                    
                                    # Marcar validación como completa
                                    st.session_state.validacion_completa = True
                                    
                                    # Reiniciar variables de duplicados
                                    st.session_state.duplicados_info = None
                                    if 'df_resultado_temp' in st.session_state:
                                        del st.session_state.df_resultado_temp
                                    if 'resultado_info_temp' in st.session_state:
                                        del st.session_state.resultado_info_temp
                                    
                                    # Botón para continuar
                                    if st.button("➡️ Continuar con el análisis de calidad", use_container_width=True):
                                        st.switch_page("pages/datos/04_Analizar_Calidad.py")
                    
                    # Opciones de navegación
                    st.write("#### Navegación")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("⬅️ Volver a seleccionar columnas", use_container_width=True):
                            st.session_state.duplicados_info = None
                            st.rerun()
                    with col2:
                        if st.button("❌ Omitir gestión de duplicados", use_container_width=True):
                            cantidad_duplicados = info['cantidad'] if info is not None and 'cantidad' in info else 0
                            log_audit(usuario_id, "OMITIR_DUPLICADOS", "validacion", 
                                    f"Usuario decidió omitir la gestión de {cantidad_duplicados} duplicados")
                            st.session_state.duplicados_info = None
                            st.session_state.paso_validacion = 0
                            st.rerun()
                    with col3:
                        if st.button("➡️ Continuar con análisis de calidad", use_container_width=True):
                            st.session_state.validacion_completa = True
                            st.switch_page("pages/datos/04_Analizar_Calidad.py")
