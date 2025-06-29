import streamlit as st
from src.reportes.generador import generar_reporte_completo, guardar_reporte_en_snowflake
from src.state.session_manager import SessionManager
from src.audit.logger import log_audit

st.set_page_config(page_title="Reporte Completo", page_icon="📄", layout="wide")
st.title("Reporte Completo de Análisis")
st.markdown("""
Esta sección permite generar y descargar un reporte PDF con todos los resultados, gráficos y recomendaciones del análisis realizado.
El reporte incluye: calidad de datos, transformaciones aplicadas, modelos evaluados, modelo seleccionado e interpretabilidad.
""")

# Recuperar datos de la sesión
session = SessionManager()
usuario = session.obtener_estado("usuario_id", "sistema")
id_sesion = session.obtener_estado("id_sesion", "sin_sesion")

# Recuperar los datos necesarios desde session_state o SessionManager
calidad_datos = session.obtener_estado("calidad_datos", {})
transformaciones = session.obtener_estado("transformaciones", [])
benchmarking = session.obtener_estado("resultados_benchmarking", {})
modelo_seleccionado = session.obtener_estado("modelo_recomendado", {})
interpretabilidad = session.obtener_estado("interpretabilidad", {})
nombre_dataset = session.obtener_estado("nombre_dataset", "dataset")
imagenes = session.obtener_estado("imagenes_reporte", None)

# Validación flexible: si falta algún dato, se avisa pero se permite generar el reporte
faltantes = []
if not calidad_datos:
    calidad_datos = {"mensaje": "No existen datos de calidad para esta sesión."}
    faltantes.append("Calidad de datos")
if not benchmarking:
    benchmarking = {"mensaje": "No existen resultados de benchmarking para esta sesión."}
    faltantes.append("Benchmarking de modelos")
if not modelo_seleccionado or not isinstance(modelo_seleccionado, dict):
    modelo_seleccionado = {"mensaje": "No existe modelo seleccionado para esta sesión."}
    faltantes.append("Modelo seleccionado")
if not interpretabilidad:
    interpretabilidad = {"mensaje": "No existen resultados de interpretabilidad para esta sesión."}
    faltantes.append("Interpretabilidad")
if not nombre_dataset:
    nombre_dataset = "No se registró nombre de dataset en esta sesión."
    faltantes.append("Nombre de dataset")

if faltantes:
    st.warning(f"El reporte se generará, pero faltan las siguientes secciones: {', '.join(faltantes)}. Se incluirá un mensaje en cada sección ausente.")

# Botón para generar y descargar el reporte
if st.session_state.get("reporte_generado") and st.session_state.get("resultado_reporte"):
    resultado = st.session_state["resultado_reporte"]
    id_reporte = st.session_state.get("id_reporte", None)
    error_snowflake = st.session_state.get("error_snowflake", None)
    if id_reporte:
        st.info(f"Reporte almacenado en Snowflake con ID: {id_reporte}")
    if error_snowflake:
        st.warning(f"No se pudo almacenar el reporte en Snowflake. El PDF se generó correctamente y puedes descargarlo. Detalle: {error_snowflake}")
    st.success("Reporte generado correctamente. Descárgalo a continuación.")
    st.download_button(
        label="Descargar reporte PDF",
        data=resultado['pdf_bytes'],
        file_name=resultado['nombre_archivo'],
        mime="application/pdf",
        use_container_width=True
    )
    st.markdown("---")
    st.info("""
    ¡Gracias por utilizar la aplicación de Analítica Farma!
    Si tienes sugerencias o necesitas soporte, contacta al equipo de datos industriales.
    """)
    if st.button("🔒 Cerrar sesión y salir", use_container_width=True):
        session.logout()
        st.success("Sesión finalizada correctamente. Puedes cerrar la ventana o volver a la pantalla de inicio.")
        st.session_state["reporte_generado"] = False
        st.session_state["resultado_reporte"] = None
        st.session_state["id_reporte"] = None
        st.session_state["error_snowflake"] = None
        st.rerun()
else:
    if st.button("📄 Generar y descargar reporte completo", use_container_width=True):
        with st.spinner("Generando reporte PDF..."):
            try:
                resultado = generar_reporte_completo(
                    calidad_datos=calidad_datos,
                    benchmarking=benchmarking,
                    modelo_seleccionado=modelo_seleccionado,
                    interpretabilidad=interpretabilidad,
                    nombre_dataset=nombre_dataset,
                    usuario=usuario,
                    imagenes=imagenes
                )
                log_audit(
                    usuario=usuario,
                    accion="GENERAR_REPORTE",
                    entidad="reporte_completo",
                    id_entidad=nombre_dataset,
                    detalles=f"Reporte generado y listo para descarga. Sesión: {id_sesion}",
                    id_sesion=id_sesion
                )
                # Guardar el reporte en Snowflake
                id_reporte = None
                error_snowflake = None
                try:
                    resultados_reporte = {
                        'calidad_datos': calidad_datos,
                        'benchmarking': benchmarking,
                        'modelo_seleccionado': modelo_seleccionado,
                        'interpretabilidad': interpretabilidad,
                        'nombre_dataset': nombre_dataset,
                        'usuario': usuario,
                        'fecha_generacion': resultado['nombre_archivo'].replace('Reporte_' + nombre_dataset + '_', '').replace('.pdf', '')
                    }
                    id_reporte = guardar_reporte_en_snowflake(
                        nombre_archivo=resultado['nombre_archivo'],
                        tipo='PDF',
                        usuario=usuario,
                        id_modelo=modelo_seleccionado.get('id_modelo', '') if isinstance(modelo_seleccionado, dict) else '',
                        id_dataset=nombre_dataset,
                        resultados=resultados_reporte,
                        id_sesion=id_sesion
                    )
                except Exception as e:
                    error_snowflake = str(e)
                    log_audit(
                        usuario=usuario,
                        accion="ERROR_SNOWFLAKE_REPORTE",
                        entidad="reporte_completo",
                        id_entidad=nombre_dataset,
                        detalles=f"Error al guardar en Snowflake: {str(e)}",
                        id_sesion=id_sesion
                    )
                st.session_state["reporte_generado"] = True
                st.session_state["resultado_reporte"] = resultado
                st.session_state["id_reporte"] = id_reporte
                st.session_state["error_snowflake"] = error_snowflake
                st.rerun()
            except Exception as e:
                st.error(f"Error al generar el reporte: {e}")
                log_audit(
                    usuario=usuario,
                    accion="ERROR_REPORTE",
                    entidad="reporte_completo",
                    id_entidad=nombre_dataset,
                    detalles=f"Error al generar reporte: {str(e)}",
                    id_sesion=id_sesion
                )

# Botón de navegación
st.markdown("---")
if st.button("🔙 Volver a Explicación de Modelo", use_container_width=True):
    st.switch_page("pages/Machine Learning/09_Explicar_Modelo.py")