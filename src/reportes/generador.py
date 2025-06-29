"""
Módulo: generador.py
Responsabilidad: Generación de reportes PDF completos con resultados, gráficos y recomendaciones.
Cumple HU13: compila automáticamente resultados, visualizaciones y recomendaciones en un documento con nombre único y secciones diferenciadas.
"""
from datetime import datetime
from fpdf import FPDF

# Sección: Función principal para generar el reporte completo

def generar_reporte_completo(
    calidad_datos: dict,
    transformaciones: list,
    benchmarking: dict,
    modelo_seleccionado: dict,
    interpretabilidad: dict,
    nombre_dataset: str,
    usuario: str,
    imagenes: dict | None = None
) -> dict:
    """
    Genera un reporte PDF con resultados, gráficos y recomendaciones.
    Devuelve un dict con el nombre único y el binario del PDF listo para descarga.
    Args:
        calidad_datos: métricas y gráficos de calidad de datos
        transformaciones: lista de transformaciones aplicadas
        benchmarking: resultados de modelos evaluados
        modelo_seleccionado: info del modelo elegido
        interpretabilidad: info y gráficos de interpretabilidad
        nombre_dataset: nombre del dataset analizado
        usuario: usuario que ejecuta el análisis
        imagenes: dict opcional de imágenes (clave: sección, valor: bytes)
    Returns:
        dict: {'nombre_archivo': str, 'pdf_bytes': bytes}
    """
    fecha_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    nombre_archivo = f"Reporte_{nombre_dataset}_{fecha_str}.pdf"
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    # Portada
    pdf.cell(0, 10, "Reporte de Análisis de Datos Industriales", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Dataset: {nombre_dataset}", ln=True)
    pdf.cell(0, 10, f"Usuario: {usuario}", ln=True)
    pdf.cell(0, 10, f"Fecha: {fecha_str}", ln=True)
    pdf.ln(10)
    # Sección: Calidad de datos
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "1. Calidad de datos", ln=True)
    pdf.set_font("Arial", '', 12)
    for k, v in calidad_datos.items():
        pdf.cell(0, 8, f"{k}: {v}", ln=True)
    if imagenes and 'calidad' in imagenes:
        _agregar_imagen(pdf, imagenes['calidad'], "calidad.png")
    pdf.ln(5)
    # Sección: Transformaciones aplicadas
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "2. Transformaciones aplicadas", ln=True)
    pdf.set_font("Arial", '', 12)
    for t in transformaciones:
        pdf.multi_cell(0, 8, f"- {t}")
    pdf.ln(5)
    # Sección: Modelos evaluados
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "3. Modelos evaluados", ln=True)
    pdf.set_font("Arial", '', 12)
    for modelo, metricas in benchmarking.items():
        pdf.cell(0, 8, f"{modelo}: {metricas}", ln=True)
    if imagenes and 'benchmarking' in imagenes:
        _agregar_imagen(pdf, imagenes['benchmarking'], "benchmarking.png")
    pdf.ln(5)
    # Sección: Modelo seleccionado
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "4. Modelo seleccionado", ln=True)
    pdf.set_font("Arial", '', 12)
    for k, v in modelo_seleccionado.items():
        pdf.cell(0, 8, f"{k}: {v}", ln=True)
    pdf.ln(5)
    # Sección: Interpretabilidad
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "5. Interpretabilidad", ln=True)
    pdf.set_font("Arial", '', 12)
    for k, v in interpretabilidad.items():
        if k != 'imagen' and k != 'imagenes':
            pdf.cell(0, 8, f"{k}: {v}", ln=True)
    if imagenes and 'interpretabilidad' in imagenes:
        _agregar_imagen(pdf, imagenes['interpretabilidad'], "interpretabilidad.png")
    pdf.ln(5)
    # Recomendaciones finales
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "6. Recomendaciones finales", ln=True)
    pdf.set_font("Arial", '', 12)
    if 'recomendaciones' in interpretabilidad:
        pdf.multi_cell(0, 8, interpretabilidad['recomendaciones'])
    else:
        pdf.cell(0, 8, "No se generaron recomendaciones específicas.", ln=True)
    # Guardar PDF en memoria
    pdf_bytes = pdf.output(dest='S')
    return {'nombre_archivo': nombre_archivo, 'pdf_bytes': pdf_bytes}

def _agregar_imagen(pdf, img_bytes, nombre_temp):
    """Agrega una imagen desde bytes al PDF (usa archivo temporal en memoria)."""
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=nombre_temp) as tmp:
        tmp.write(img_bytes)
        tmp.flush()
        pdf.image(tmp.name, w=170)
    import os
    os.unlink(tmp.name)