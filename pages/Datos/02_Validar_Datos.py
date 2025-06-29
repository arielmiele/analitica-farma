import streamlit as st
import os
import sys

# Agregar el directorio src al path para poder importar los módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Importar módulos de la aplicación
from src.audit.logger import log_audit
from src.datos.validador import (
    validar_tipos_datos,
    validar_fechas
)
from src.datos.transformador import (
    corregir_tipo_datos,
    extraer_variables_fecha
)
from src.datos.limpiador import (
    detectar_duplicados
)
from src.datos.formateador import persistir_dataframe

# Inicializar session_state para esta página
if 'validacion_completa' not in st.session_state:
    st.session_state.validacion_completa = False
if 'errores_tipo' not in st.session_state:
    st.session_state.errores_tipo = []
if 'errores_fecha' not in st.session_state:
    st.session_state.errores_fecha = []
if 'validacion_realizada' not in st.session_state:
    st.session_state.validacion_realizada = False
if 'duplicados_info' not in st.session_state:
    st.session_state.duplicados_info = None

# Título y descripción de la página
st.title("🔍 2. Validar Estructura de Datos")

st.markdown("""
Esta página valida automáticamente los tipos de datos y formatos de fecha para garantizar la compatibilidad y detectar errores de forma temprana.
""")

# Verificar si hay datos cargados
if 'df' not in st.session_state or st.session_state.df is None:
    st.warning("⚠️ No hay datos cargados. Por favor, carga un dataset primero en la página 'Cargar Datos'.")
    if st.button("Ir a Cargar Datos", use_container_width=True):
        st.switch_page("pages/Datos/01_Cargar_Datos.py")
else:
    usuario = st.session_state.get("usuario_id", "sistema")
    id_sesion = st.session_state.get("id_sesion", "sin_sesion")
    # === SECCIÓN 1: Información del dataset ===
    st.subheader("Información del dataset")
    st.success(f"Archivo: {st.session_state.filename}")
    st.write(f"Dimensiones: {st.session_state.df.shape[0]} filas × {st.session_state.df.shape[1]} columnas")
    st.divider()

    # === SECCIÓN 2: Explicación y validación ===
    st.subheader("Validación automática de datos")
    st.markdown("""
    El sistema realizará automáticamente las siguientes validaciones:
    1. **Tipos de datos**: Verifica que las columnas tengan tipos coherentes y adecuados.
    2. **Formatos de fecha**: Detecta inconsistencias en formatos de fechas.
    Esta validación ayuda a detectar problemas que podrían afectar el rendimiento de los modelos.
    """)
    # Ejecutar validación automática al cargar la página o al cambiar el DataFrame
    if (
        not st.session_state.get('validacion_realizada', False)
        and st.session_state.df is not None
    ):
        with st.spinner("Validando datos automáticamente..."):
            log_audit(usuario=usuario, accion="INICIO_VALIDACION", entidad="validacion_datos", id_entidad=st.session_state.filename, detalles=f"Iniciando validación de datos para {st.session_state.filename}", id_sesion=id_sesion)
            errores_tipo = validar_tipos_datos(st.session_state.df, usuario=usuario, id_sesion=id_sesion)
            errores_fecha = validar_fechas(st.session_state.df, usuario=usuario, id_sesion=id_sesion)
            st.session_state.errores_tipo = errores_tipo
            st.session_state.errores_fecha = errores_fecha
            st.session_state.validacion_realizada = True
            # No st.rerun() aquí para evitar bucles infinitos

    # === SECCIÓN 3: Resultados de validación ===
    if st.session_state.get('validacion_realizada', False):
        st.divider()
        st.subheader("Resultados de validación")
        total_errores = len(st.session_state.errores_tipo) + len(st.session_state.errores_fecha)
        if total_errores == 0:
            st.success("✅ No se detectaron problemas en la estructura de los datos. La validación es correcta.")
            log_audit(usuario=st.session_state.get("usuario_id", "sistema"), accion="VALIDACION_EXITOSA", entidad="validacion_datos", id_entidad=st.session_state.filename, detalles=f"Validación completa sin errores para {st.session_state.filename}", id_sesion=st.session_state.get("id_sesion", "sin_sesion"))
            st.session_state.validacion_completa = True
        else:
            st.error(f"❌ Se detectaron {total_errores} problemas que requieren atención.")
            log_audit(usuario=st.session_state.get("usuario_id", "sistema"), accion="VALIDACION_ERRORES", entidad="validacion_datos", id_entidad=st.session_state.filename, detalles=f"Validación completada con {total_errores} errores para {st.session_state.filename}", id_sesion=st.session_state.get("id_sesion", "sin_sesion"))
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
        with st.expander("Validación de columnas de fecha para ML", expanded=len(st.session_state.errores_fecha) > 0):
            if len(st.session_state.errores_fecha) == 0:
                st.info("No se detectaron problemas en las columnas de fecha.")
            else:
                st.write(f"Se encontraron {len(st.session_state.errores_fecha)} advertencias o recomendaciones:")
                for error in st.session_state.errores_fecha:
                    st.warning(f"**{error['columna']}**: {error['mensaje']}")
                    if 'sugerencia' in error:
                        st.info(f"Sugerencia: {error['sugerencia']}")
                    if 'ejemplos' in error:
                        st.text(f"Ejemplos encontrados: {', '.join(error['ejemplos'])}")
                    if 'formato_sugerido' in error:
                        st.info(f"Formato sugerido: {error['formato_sugerido']}")

        # === SECCIÓN 4: Corrección de errores ===
        if total_errores > 0:
            st.divider()
            st.subheader("Corrección automática de problemas")
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
                # Botones del formulario
                st.write("#### Aplicar correcciones")
                col1, col2 = st.columns(2)
                with col1:
                    cancelar = st.form_submit_button("❌ Cancelar", use_container_width=True)
                with col2:
                    aplicar = st.form_submit_button("✅ Aplicar correcciones seleccionadas", use_container_width=True)
            # Procesar resultados del formulario
            if cancelar:
                st.session_state.validacion_realizada = False
                st.rerun()
            if aplicar:
                with st.spinner("Aplicando correcciones..."):
                    df_corregido = st.session_state.df.copy()
                    correcciones_aplicadas = []
                    for error in st.session_state.errores_tipo:
                        if error.get('aplicar', False):
                            columna = error['columna']
                            tipo_destino = error.get('opcion_seleccionada', 'str')
                            try:
                                df_corregido = corregir_tipo_datos(df_corregido, columna, tipo_destino, usuario=usuario, id_sesion=id_sesion)
                                mensaje = f"Corregido tipo de dato en columna {columna} a {tipo_destino}"
                                correcciones_aplicadas.append(mensaje)
                                log_audit(
                                    usuario=usuario,
                                    accion="CORRECCION_TIPO",
                                    entidad="validacion_datos",
                                    id_entidad=st.session_state.filename,
                                    detalles=mensaje,
                                    id_sesion=id_sesion
                                )
                            except Exception as e:
                                st.error(f"Error al corregir tipo de dato en columna {columna}: {str(e)}")
                                log_audit(
                                    usuario=usuario,
                                    accion="ERROR_CORRECCION",
                                    entidad="validacion_datos",
                                    id_entidad=st.session_state.filename,
                                    detalles=f"Error al corregir tipo de {columna}: {str(e)}",
                                    id_sesion=id_sesion
                                )
                    if correcciones_aplicadas:
                        # Persistir y mostrar mensaje
                        resultado_persistencia = persistir_dataframe(df_corregido, usuario=usuario, id_sesion=id_sesion)
                        if resultado_persistencia['success']:
                            st.success(f"✅ Se aplicaron {len(correcciones_aplicadas)} correcciones con éxito. {resultado_persistencia['message']}")
                        else:
                            st.error(f"❌ Error al actualizar el DataFrame: {resultado_persistencia['message']}")
                        st.session_state.validacion_realizada = False
                        st.rerun()
                    else:
                        st.warning("No se seleccionó ninguna corrección para aplicar")

    # === SECCIÓN 5: Análisis automático de duplicados ===
    st.divider()
    st.subheader("Análisis automático de duplicados")
    st.markdown("""
    El sistema analizará automáticamente si existen registros duplicados considerando todas las columnas del dataset.
    """)
    if 'duplicados_info' not in st.session_state or st.session_state.duplicados_info is None:
        with st.spinner("Analizando duplicados..."):
            columnas_seleccionadas = st.session_state.df.columns.tolist()
            st.session_state.duplicados_info = detectar_duplicados(st.session_state.df, columnas_seleccionadas, usuario=usuario)
            log_audit(
                usuario=usuario,
                accion="DETECCION_DUPLICADOS",
                entidad="validacion_datos",
                id_entidad=st.session_state.filename,
                detalles="Detección de duplicados en todas las columnas",
                id_sesion=id_sesion
            )
    info = st.session_state.duplicados_info
    if info is not None and 'error' in info and info['error']:
        st.error(f"Error al detectar duplicados: {info['error']}")
    else:
        st.write("#### Resumen de duplicados encontrados")
        if info is not None and info.get('cantidad', 0) == 0:
            st.success("✅ No se detectaron registros duplicados en el dataset.")
            if st.button("➡️ Continuar con el análisis de calidad", use_container_width=True):
                st.session_state.validacion_completa = True
                st.switch_page("pages/Datos/03_Analizar_Calidad.py")
        else:
            col1, col2 = st.columns(2)
            with col1:
                cantidad_duplicados = info['cantidad'] if info is not None and 'cantidad' in info else 0
                st.metric("Registros duplicados", f"{cantidad_duplicados} filas")
            with col2:
                porcentaje_duplicados = info['porcentaje'] if info is not None and 'porcentaje' in info else 0
                st.metric("Porcentaje del total", f"{porcentaje_duplicados:.2f}%")
            if info is not None and 'grupos_duplicados' in info and info['grupos_duplicados'] is not None and not info['grupos_duplicados'].empty:
                st.write("#### Grupos de registros duplicados")
                st.write(f"Se encontraron {len(info['grupos_duplicados'])} grupos con registros duplicados:")
                st.dataframe(info['grupos_duplicados'])
            st.write("#### Ejemplos de registros duplicados")
            if info is not None and 'indices' in info and info['indices'] is not None and len(info['indices']) > 0:
                muestra_indices = info['indices'][:min(5, len(info['indices']))]
                df_duplicados = st.session_state.df.loc[muestra_indices]
                st.write(f"Mostrando {len(df_duplicados)} ejemplos de registros duplicados:")
                st.dataframe(df_duplicados)
            # Botón para gestionar duplicados
            if st.button("Gestionar duplicados", use_container_width=True):
                # Ejemplo: eliminar duplicados antes de pasar a la siguiente página
                try:
                    df_sin_duplicados = st.session_state.df.drop_duplicates()
                    resultado_persistencia = persistir_dataframe(df_sin_duplicados, usuario=usuario, id_sesion=id_sesion)
                    if resultado_persistencia['success']:
                        st.success(f"Registros duplicados eliminados. {resultado_persistencia['message']}")
                        log_audit(
                            usuario=usuario,
                            accion="ELIMINACION_DUPLICADOS",
                            entidad="validacion_datos",
                            id_entidad=st.session_state.filename,
                            detalles="Registros duplicados eliminados",
                            id_sesion=id_sesion
                        )
                        st.session_state.df = df_sin_duplicados
                        st.session_state.duplicados_info = None
                        st.rerun()
                    else:
                        st.error(f"❌ Error al actualizar el DataFrame tras eliminar duplicados: {resultado_persistencia['message']}")
                except Exception as e:
                    st.error(f"Error al eliminar duplicados: {str(e)}")
    
    # === SECCIÓN: Extracción de variables derivadas de fechas ===
    # Buscar columnas de fecha válidas para extracción
    columnas_fecha_validas = []
    for error in st.session_state.errores_fecha:
        if (
            'mensaje' in error and
            error['mensaje'].startswith('Columna de fecha válida para ML')
        ):
            columnas_fecha_validas.append(error['columna'])
    if columnas_fecha_validas:
        st.divider()
        st.subheader("Extraer variables derivadas de fechas")
        st.markdown("""
        Puede crear automáticamente variables como año, mes, día, día de la semana, semana, etc. a partir de las columnas de fecha válidas para ML.
        """)
        columna_seleccionada = st.selectbox(
            "Seleccione la columna de fecha para extraer variables:",
            columnas_fecha_validas,
            key="columna_fecha_extraer"
        )
        variables_disponibles = [
            'anio', 'mes', 'dia', 'dia_semana', 'nombre_dia', 'nombre_mes',
            'dia_anio', 'semana', 'es_fin_de_semana', 'fecha_epoch'
        ]
        variables_seleccionadas = st.multiselect(
            "Variables a extraer:",
            variables_disponibles,
            default=['anio', 'mes', 'dia', 'dia_semana']
        )
        if st.button("➕ Extraer variables derivadas", use_container_width=True):
            with st.spinner("Extrayendo variables derivadas de la fecha..."):
                try:
                    df_nuevo = extraer_variables_fecha(st.session_state.df, columna_seleccionada, variables_seleccionadas, usuario=usuario)
                    resultado_persistencia = persistir_dataframe(df_nuevo, usuario=usuario, id_sesion=id_sesion)
                    if resultado_persistencia['success']:
                        st.success(f"Variables derivadas extraídas y agregadas al dataset: {', '.join(variables_seleccionadas)}. {resultado_persistencia['message']}")
                        log_audit(
                            usuario=usuario,
                            accion="EXTRACCION_FECHA_UI",
                            entidad="validacion_datos",
                            id_entidad=st.session_state.filename,
                            detalles=f"Extraídas variables {variables_seleccionadas} de {columna_seleccionada}",
                            id_sesion=id_sesion
                        )
                        st.rerun()
                    else:
                        st.error(f"❌ Error al actualizar el DataFrame: {resultado_persistencia['message']}")
                except Exception as e:
                    st.error(f"Error al extraer variables derivadas: {str(e)}")
