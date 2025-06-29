import streamlit as st
from src.reportes.generador import generar_reporte_completo
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
benchmarking = session.obtener_estado("benchmarking", {})
modelo_seleccionado = session.obtener_estado("modelo_recomendado", {})
interpretabilidad = session.obtener_estado("interpretabilidad", {})
nombre_dataset = session.obtener_estado("nombre_dataset", "dataset")
imagenes = session.obtener_estado("imagenes_reporte", None)

# Validación básica
if not calidad_datos or not benchmarking or not modelo_seleccionado:
    st.warning("No se encontraron resultados completos en la sesión. Finaliza el flujo de análisis antes de generar el reporte.")
    st.stop()

# Botón para generar y descargar el reporte
if st.button("📄 Generar y descargar reporte completo", use_container_width=True):
    with st.spinner("Generando reporte PDF..."):
        try:
            resultado = generar_reporte_completo(
                calidad_datos=calidad_datos,
                transformaciones=transformaciones,
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
                st.switch_page("pages/00_Logueo.py")
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