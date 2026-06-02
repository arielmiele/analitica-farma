"""
Módulo: generador.py
Responsabilidad: Generación de reportes PDF completos con resultados, gráficos y recomendaciones.
Compatible con fpdf2 >= 2.x (API Unicode nativa).
"""
import io
import os
import tempfile
from datetime import datetime
from fpdf import FPDF
from src.database.reportes_db import guardar_reporte


def _s(valor) -> str:
    """Convierte un valor a string limpio, reemplazando caracteres no imprimibles."""
    if valor is None:
        return ""
    texto = str(valor)
    # Reemplazar caracteres problemáticos comunes
    reemplazos = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
                  "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U",
                  "ñ": "n", "Ñ": "N", "ü": "u", "Ü": "U"}
    for orig, rep in reemplazos.items():
        texto = texto.replace(orig, rep)
    # Eliminar caracteres fuera de ASCII básico
    return "".join(c if ord(c) < 256 else "?" for c in texto)


class _PDF(FPDF):
    """FPDF personalizado con encabezado y pie de página."""

    def set_meta(self, usuario: str, fecha: str) -> None:
        self._meta_usuario = usuario
        self._meta_fecha = fecha

    def footer(self):
        usuario = getattr(self, "_meta_usuario", "")
        fecha = getattr(self, "_meta_fecha", "")
        self.set_y(-12)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(150, 150, 150)
        self.cell(
            0, 5,
            f"Analitica Farma | {_s(fecha)} | Usuario: {_s(usuario)} | Pag. {self.page_no()}",
            align="C",
        )
        self.set_text_color(0, 0, 0)


def _seccion(pdf: _PDF, titulo: str) -> None:
    """Encabezado de sección con fondo azul claro."""
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_fill_color(220, 230, 245)
    pdf.set_x(pdf.l_margin)
    pdf.cell(0, 9, _s(titulo), new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.ln(1)


def _fila(pdf: _PDF, etiqueta: str, valor: str,
          w_label: int = 65, w_valor: int = 115) -> None:
    """Fila de dos columnas con borde."""
    pdf.set_x(pdf.l_margin)
    x0 = pdf.get_x()
    y0 = pdf.get_y()

    pdf.set_font("Helvetica", "B", 9)
    pdf.multi_cell(w_label, 6, _s(etiqueta), border=1, new_x="RIGHT", new_y="TOP")
    y1 = pdf.get_y()

    pdf.set_xy(x0 + w_label, y0)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(w_valor, 6, _s(valor), border=1, new_x="LMARGIN", new_y="NEXT")
    y2 = pdf.get_y()

    if y2 < y1:
        pdf.set_y(y1)


def _grafico_barras_modelos(benchmarking: dict):
    """Genera PNG de barras horizontales con la métrica principal de cada modelo."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        modelos_exitosos = benchmarking.get("modelos_exitosos", [])
        if not modelos_exitosos:
            return None

        tipo = benchmarking.get("tipo_problema", "clasificacion")
        mk = "accuracy" if tipo == "clasificacion" else "r2"
        label = "Accuracy" if tipo == "clasificacion" else "R2"

        nombres, valores = [], []
        for m in modelos_exitosos:
            v = m.get("metricas", {}).get(mk)
            if v is not None:
                try:
                    nombres.append(str(m.get("nombre", "?"))[:22])
                    valores.append(float(v))
                except (ValueError, TypeError):
                    pass

        if not nombres:
            return None

        fig, ax = plt.subplots(figsize=(9, max(2.5, len(nombres) * 0.55)))
        bars = ax.barh(nombres, valores, color="#4C8BF5", edgecolor="white")
        ax.set_xlabel(label)
        ax.set_title(f"Comparacion de modelos - {label}")
        ax.set_xlim(0, 1.05)
        for bar, val in zip(bars, valores):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=8)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    except Exception:
        return None


def _grafico_shap(importancias: dict, top_n: int = 20):
    """Genera PNG de barras horizontales con importancias SHAP (top N variables)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def _to_float(x):
            try:
                return abs(float(x))
            except (ValueError, TypeError):
                return 0.0

        items = sorted(importancias.items(), key=lambda x: _to_float(x[1]), reverse=True)[:top_n]
        if not items:
            return None

        nombres = [str(v)[:45] for v, _ in reversed(items)]
        valores = [_to_float(imp) for _, imp in reversed(items)]

        fig, ax = plt.subplots(figsize=(9, max(3, len(nombres) * 0.45)))
        bars = ax.barh(nombres, valores, color="#5B8DB8", edgecolor="white")
        ax.set_xlabel("Importancia SHAP (valor absoluto medio)")
        ax.set_title(f"Top {len(nombres)} variables - Importancia SHAP")
        ax.set_xlim(0, max(valores) * 1.18 if valores else 1)
        for bar, val in zip(bars, valores):
            ax.text(val + max(valores) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=7)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    except Exception:
        return None


def generar_reporte_completo(
    calidad_datos: dict,
    benchmarking: dict,
    modelo_seleccionado: dict,
    interpretabilidad: dict,
    nombre_dataset: str,
    usuario: str,
    imagenes: dict | None = None,
) -> dict:
    """
    Genera un reporte PDF con resultados, gráficos y recomendaciones.

    Args:
        calidad_datos: métricas de calidad (estructura de analizador.py)
        benchmarking: resultados de entrenamiento (resultados_benchmarking)
        modelo_seleccionado: recomendación final (modelo_recomendado)
        interpretabilidad: datos SHAP (de 09_Explicar_Modelo.py)
        nombre_dataset: nombre del dataset analizado
        usuario: usuario que ejecuta el análisis
        imagenes: dict opcional con imágenes extra

    Returns:
        dict: {'nombre_archivo': str, 'pdf_bytes': bytes}
    """
    fecha_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    fecha_legible = datetime.now().strftime("%d/%m/%Y %H:%M")
    nombre_archivo = f"Reporte_{nombre_dataset}_{fecha_str}.pdf"

    pdf = _PDF()
    pdf.set_meta(usuario=usuario, fecha=fecha_legible)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # ── PORTADA ──────────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 17)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 12, "Reporte de Analisis de Datos Industriales", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 11)
    pdf.ln(4)
    pdf.cell(0, 7, f"Dataset: {_s(nombre_dataset)}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"Usuario: {_s(usuario)}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"Fecha: {_s(fecha_legible)}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(100, 120, 180)
    pdf.line(pdf.l_margin, pdf.get_y() + 3, pdf.w - pdf.r_margin, pdf.get_y() + 3)
    pdf.ln(7)

    # ── SECCIÓN 1: CALIDAD DE DATOS ──────────────────────────────────────────
    _seccion(pdf, "1. Calidad de datos")

    if isinstance(calidad_datos, dict) and "mensaje" in calidad_datos:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 7, _s(calidad_datos["mensaje"]))
    else:
        metricas_g = calidad_datos.get("global", {})
        if metricas_g:
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 7, "Metricas generales:", new_x="LMARGIN", new_y="NEXT")
            for etiq, val in [
                ("Filas", metricas_g.get("filas", "N/D")),
                ("Columnas", metricas_g.get("columnas", "N/D")),
                ("Nulos totales", metricas_g.get("nulos_totales", "N/D")),
                ("% Nulos", f"{metricas_g.get('porcentaje_nulos', 0):.2f}%"),
                ("Duplicados", metricas_g.get("duplicados", "N/D")),
                ("% Duplicados", f"{metricas_g.get('porcentaje_duplicados', 0):.2f}%"),
                ("Completitud", f"{metricas_g.get('completitud', 0):.2f}%"),
            ]:
                _fila(pdf, etiq, str(val))

        evaluacion = calidad_datos.get("evaluacion", {})
        if isinstance(evaluacion, dict) and evaluacion:
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 7, "Evaluacion de calidad:", new_x="LMARGIN", new_y="NEXT")
            for k, v in evaluacion.items():
                if not isinstance(v, (dict, list)):
                    _fila(pdf, str(k), str(v))

        nulos_col = calidad_datos.get("nulos_por_columna", [])
        filas_con_nulos = [r for r in nulos_col
                           if isinstance(r, dict) and r.get("nulos", 0) > 0]
        if filas_con_nulos:
            filas_con_nulos.sort(key=lambda r: r.get("porcentaje", 0), reverse=True)
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 7, "Columnas con nulos (top 10):", new_x="LMARGIN", new_y="NEXT")
            pdf.set_x(pdf.l_margin)
            pdf.set_font("Helvetica", "B", 8)
            pdf.cell(85, 6, "Columna", border=1)
            pdf.cell(40, 6, "Nulos", border=1)
            pdf.cell(45, 6, "Porcentaje", border=1, new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 8)
            for fila in filas_con_nulos[:10]:
                pdf.set_x(pdf.l_margin)
                pdf.cell(85, 6, _s(fila.get("columna", "")), border=1)
                pdf.cell(40, 6, _s(fila.get("nulos", "")), border=1)
                pdf.cell(45, 6, f"{fila.get('porcentaje', 0):.2f}%", border=1,
                         new_x="LMARGIN", new_y="NEXT")

    pdf.ln(4)

    # ── SECCIÓN 2: MODELOS EVALUADOS ─────────────────────────────────────────
    _seccion(pdf, "2. Modelos evaluados (benchmarking)")

    if isinstance(benchmarking, dict) and "mensaje" in benchmarking:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 7, _s(benchmarking["mensaje"]))
    else:
        tipo_p = benchmarking.get("tipo_problema", "N/D")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, f"Tipo de problema: {_s(tipo_p)}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        modelos_exitosos = benchmarking.get("modelos_exitosos", [])
        if modelos_exitosos:
            primera_m = modelos_exitosos[0].get("metricas", {})
            mk_keys = [k for k, v in primera_m.items() if not isinstance(v, (dict, list))][:5]
            mejor_nombre = (benchmarking.get("mejor_modelo") or {}).get("nombre", "")

            col_nombre = 55
            col_mk = min(28, max(15, int(125 / max(len(mk_keys), 1))))

            pdf.set_x(pdf.l_margin)
            pdf.set_font("Helvetica", "B", 8)
            pdf.cell(col_nombre, 6, "Modelo", border=1)
            for k in mk_keys:
                pdf.cell(col_mk, 6, _s(k[:13]), border=1)
            pdf.ln()

            pdf.set_font("Helvetica", "", 8)
            for modelo in modelos_exitosos:
                nm = modelo.get("nombre", "")
                metricas_m = modelo.get("metricas", {})
                if nm == mejor_nombre:
                    pdf.set_fill_color(200, 235, 200)
                    fill = True
                else:
                    pdf.set_fill_color(255, 255, 255)
                    fill = False
                pdf.set_x(pdf.l_margin)
                pdf.cell(col_nombre, 6, _s(nm[:24]), border=1, fill=fill)
                for k in mk_keys:
                    v = metricas_m.get(k, "")
                    try:
                        vs = f"{float(v):.4f}"
                    except (ValueError, TypeError):
                        vs = _s(v)
                    pdf.cell(col_mk, 6, vs, border=1, fill=fill)
                pdf.ln()

            if mejor_nombre:
                pdf.ln(1)
                pdf.set_font("Helvetica", "I", 8)
                pdf.set_x(pdf.l_margin)
                pdf.cell(0, 6, f"(Fila resaltada = mejor modelo: {_s(mejor_nombre)})",
                         new_x="LMARGIN", new_y="NEXT")

        # Gráfico de comparación generado internamente
        img_barras = _grafico_barras_modelos(benchmarking)
        if img_barras:
            _agregar_imagen(pdf, img_barras)

    pdf.ln(4)

    # ── SECCIÓN 3: MODELO SELECCIONADO ───────────────────────────────────────
    _seccion(pdf, "3. Modelo seleccionado")

    if isinstance(modelo_seleccionado, dict) and "mensaje" in modelo_seleccionado:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 7, _s(modelo_seleccionado["mensaje"]))
    else:
        modelo_obj = modelo_seleccionado.get("modelo_recomendado", modelo_seleccionado)
        if isinstance(modelo_obj, dict):
            nombre_mod = modelo_obj.get("nombre", modelo_seleccionado.get("nombre", "N/D"))
            criterio = modelo_seleccionado.get("criterio", "N/D")
            justificacion = modelo_seleccionado.get("justificacion", "")
            metricas_mod = modelo_obj.get("metricas", {})

            _fila(pdf, "Modelo", nombre_mod)
            _fila(pdf, "Criterio de seleccion", criterio)

            if justificacion:
                pdf.ln(2)
                pdf.set_font("Helvetica", "B", 10)
                pdf.set_x(pdf.l_margin)
                pdf.cell(0, 7, "Justificacion:", new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", "", 10)
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(0, 7, _s(justificacion))

            if isinstance(metricas_mod, dict) and metricas_mod:
                pdf.ln(2)
                pdf.set_font("Helvetica", "B", 10)
                pdf.set_x(pdf.l_margin)
                pdf.cell(0, 7, "Metricas:", new_x="LMARGIN", new_y="NEXT")
                for k, v in metricas_mod.items():
                    if not isinstance(v, (dict, list)):
                        try:
                            vs = f"{float(v):.4f}"
                        except (ValueError, TypeError):
                            vs = _s(v)
                        _fila(pdf, _s(k), vs)
        else:
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 7, _s(str(modelo_seleccionado)))

    pdf.ln(4)

    # ── SECCIÓN 4: INTERPRETABILIDAD (SHAP) ──────────────────────────────────
    _seccion(pdf, "4. Interpretabilidad (SHAP)")

    if isinstance(interpretabilidad, dict) and "mensaje" in interpretabilidad:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 7, _s(interpretabilidad["mensaje"]))
    elif isinstance(interpretabilidad, dict) and interpretabilidad:
        mid = interpretabilidad.get("modelo_id", "N/D")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_x(pdf.l_margin)
        pdf.cell(0, 7, f"Modelo analizado: {_s(mid)}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        importancias = interpretabilidad.get("importancias", {})
        if isinstance(importancias, dict) and importancias:
            def _to_float(x):
                try:
                    return abs(float(x))
                except (ValueError, TypeError):
                    return 0.0

            items_ord = sorted(importancias.items(), key=lambda x: _to_float(x[1]), reverse=True)

            # --- Gráfico SHAP embebido ---
            img_shap = _grafico_shap(importancias, top_n=20)
            if img_shap:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp.write(img_shap)
                    tmp_path = tmp.name
                try:
                    disponible = pdf.h - pdf.get_y() - pdf.b_margin - 18
                    alto = min(120, max(60, disponible))
                    pdf.image(tmp_path, x=pdf.l_margin, w=pdf.w - pdf.l_margin - pdf.r_margin, h=alto)
                    pdf.ln(3)
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

            # --- Lista de texto compacta (top 20) ---
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_x(pdf.l_margin)
            pdf.cell(0, 6, "Ranking de importancias SHAP:", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 7.5)
            for rank, (var, imp) in enumerate(items_ord[:20], 1):
                try:
                    imp_s = f"{float(imp):.6f}"
                except (ValueError, TypeError):
                    imp_s = _s(imp)
                nombre = _s(str(var))
                linea = f"  {rank:>2d}. {nombre}  [{imp_s}]"
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(0, 5, linea, new_x="LMARGIN", new_y="NEXT")
            if len(items_ord) > 20:
                pdf.set_font("Helvetica", "I", 7)
                pdf.set_x(pdf.l_margin)
                pdf.cell(0, 5, f"(Mostrando top 20 de {len(items_ord)} variables)",
                         new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 7, "No se ejecuto la explicacion del modelo en esta sesion.")

    if imagenes and "interpretabilidad" in imagenes:
        _agregar_imagen(pdf, imagenes["interpretabilidad"])

    pdf.ln(4)

    # ── SECCIÓN 5: CONCLUSIONES ───────────────────────────────────────────────
    _seccion(pdf, "5. Conclusiones y recomendaciones")
    pdf.set_font("Helvetica", "", 10)

    conclusiones = []
    metricas_g2 = calidad_datos.get("global", {}) if isinstance(calidad_datos, dict) else {}
    if metricas_g2:
        nulos_pct = metricas_g2.get("porcentaje_nulos", 0)
        completitud = metricas_g2.get("completitud", 100)
        if nulos_pct > 20:
            conclusiones.append(
                f"- El dataset presento un {nulos_pct:.1f}% de valores nulos. "
                "Se recomienda revisar la fuente de datos o aplicar tecnicas de imputacion avanzada."
            )
        elif nulos_pct > 5:
            conclusiones.append(
                f"- El dataset presento un {nulos_pct:.1f}% de valores nulos. "
                "Se recomienda monitorear la calidad de datos en futuros ciclos."
            )
        else:
            conclusiones.append(
                f"- La completitud del dataset fue de {completitud:.1f}%, lo cual es adecuado para el analisis."
            )

    mejor_m = benchmarking.get("mejor_modelo") if isinstance(benchmarking, dict) else None
    if isinstance(mejor_m, dict):
        nm_m = mejor_m.get("nombre", "")
        met_m = mejor_m.get("metricas", {})
        tipo_p2 = benchmarking.get("tipo_problema", "")
        k_met = "accuracy" if tipo_p2 == "clasificacion" else "r2"
        v_met = met_m.get(k_met)
        if v_met is not None:
            try:
                lab = "Accuracy" if tipo_p2 == "clasificacion" else "R2"
                conclusiones.append(f"- El mejor modelo fue '{nm_m}' con {lab} = {float(v_met):.4f}.")
            except (ValueError, TypeError):
                pass

    if not conclusiones:
        conclusiones.append("- Se completaron todas las etapas del analisis. Revise las secciones anteriores.")

    conclusiones.append(
        "- Se recomienda validar el modelo seleccionado con nuevos datos antes de su implementacion en produccion."
    )

    for c in conclusiones:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 7, _s(c))

    # ── GENERAR BYTES ─────────────────────────────────────────────────────────
    pdf_bytes = bytes(pdf.output())
    return {"nombre_archivo": nombre_archivo, "pdf_bytes": pdf_bytes}


def _agregar_imagen(pdf: _PDF, img_bytes: bytes) -> None:
    """Agrega una imagen PNG/JPG desde bytes al PDF usando archivo temporal."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(img_bytes)
        tmp_path = tmp.name
    try:
        pdf.set_x(pdf.l_margin)
        pdf.image(tmp_path, w=170)
    finally:
        os.unlink(tmp_path)


def guardar_reporte_local(
    nombre_archivo: str,
    tipo: str,
    usuario: str,
    id_modelo: str,
    id_dataset: str,
    resultados: dict,
    id_sesion: str,
) -> str:
    """
    Guarda los metadatos y resultados del reporte en SQLite.
    Mantiene la misma firma para compatibilidad con el código que la llama.
    """
    return guardar_reporte(
        nombre_archivo=nombre_archivo,
        tipo=tipo,
        id_usuario=usuario,
        id_sesion=id_sesion,
        id_benchmarking=None,
        id_dataset=id_dataset,
        resultados=resultados,
    )

