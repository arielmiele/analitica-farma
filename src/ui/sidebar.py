import streamlit as st
from src.state.session_manager import SessionManager

# Definición de los pasos del workflow con las claves de sesión que indican su completitud.
# Pasos obligatorios del flujo principal (8 pasos). Validación Cruzada es opcional/avanzado.
_WORKFLOW_STEPS = [
    {"label": "Login",                     "key": "usuario_id",              "page": "pages/00_Logueo.py",                           "opcional": False},
    {"label": "Cargar datos",              "key": "df",                      "page": "pages/Datos/01_Cargar_Datos.py",               "opcional": False},
    {"label": "Validar y analizar datos",  "key": "calidad_datos",           "page": "pages/Datos/03_Analizar_Calidad.py",           "opcional": False},
    {"label": "Configurar datos",          "key": "variable_objetivo",       "page": "pages/Datos/04_Configurar_Datos.py",           "opcional": False},
    {"label": "Entrenar modelos",          "key": "resultados_benchmarking", "page": "pages/Machine Learning/05_Entrenar_Modelos.py","opcional": False},
    {"label": "Evaluar modelos",           "key": "resultados_benchmarking", "page": "pages/Machine Learning/06_Evaluar_Modelos.py", "opcional": False},
    {"label": "Recomendar modelo",         "key": "modelo_recomendado",      "page": "pages/Machine Learning/08_Recomendar_Modelo.py","opcional": False},
    {"label": "Validación cruzada ⭐",    "key": "resultados_benchmarking", "page": "pages/Machine Learning/07_Validacion_Cruzada.py","opcional": True},
    {"label": "Explicar modelo",           "key": "interpretabilidad",       "page": "pages/Machine Learning/09_Explicar_Modelo.py", "opcional": False},
]

class SidebarComponents:
    """
    Clase para gestionar los componentes de UI del sidebar.
    Contiene métodos para renderizar diferentes partes del sidebar.
    """
    
    @staticmethod
    def render_workflow_stepper() -> None:
        """
        Renderiza un indicador visual de progreso del workflow.
        Los pasos opcionales (⭐) no se cuentan para la barra de progreso principal.
        """
        if not SessionManager.is_logged_in():
            return

        import pandas as pd

        def _es_completado(key: str) -> bool:
            valor = st.session_state.get(key)
            try:
                if isinstance(valor, pd.DataFrame):
                    return not valor.empty
                if isinstance(valor, (list, dict)):
                    return len(valor) > 0
                return valor is not None and valor is not False
            except Exception:
                return False

        # Progreso sobre pasos obligatorios solamente
        pasos_obligatorios = [s for s in _WORKFLOW_STEPS if not s.get("opcional", False)]
        completados_oblig = sum(1 for s in pasos_obligatorios if _es_completado(s["key"]))
        total_oblig = len(pasos_obligatorios)

        # Paso actual = primero obligatorio no completado
        paso_actual_idx = next(
            (i for i, s in enumerate(_WORKFLOW_STEPS)
             if not s.get("opcional", False) and not _es_completado(s["key"])),
            len(_WORKFLOW_STEPS)
        )

        with st.expander("🗺️ Progreso del Workflow", expanded=False):
            for i, step in enumerate(_WORKFLOW_STEPS):
                completado = _es_completado(step["key"])
                opcional = step.get("opcional", False)
                if completado:
                    icono = "✅"
                    estilo = ""
                elif i == paso_actual_idx and not opcional:
                    icono = "⏳"
                    estilo = "**"
                elif opcional:
                    icono = "⭐"
                    estilo = ""
                else:
                    icono = "⬜"
                    estilo = ""
                label = step["label"]
                num = i + 1
                st.markdown(
                    f"{icono} {estilo}{num}. {label}{estilo}",
                    help=f"Ir a: {step['page']}"
                )

            st.progress(
                completados_oblig / total_oblig if total_oblig else 0,
                text=f"{completados_oblig}/{total_oblig} pasos completados"
            )


    @staticmethod
    def render_user_info() -> None:
        """
        Renderiza la información persistente del usuario conectado en el sidebar
        """
        if SessionManager.is_logged_in():
            with st.expander("👤 Usuario Conectado", expanded=True):
                user = SessionManager.get_user_info()
                st.write("**Nombre:**", user["usuario_nombre"])
                st.write("**Rol:**", user["usuario_rol"])
                st.write("**Email:**", user["usuario_email"])
                if st.button("🚪 Cerrar Sesión", key="btn_logout_sidebar"):
                    SessionManager.logout()
                    st.rerun()
    
    @staticmethod
    def render_dataset_info() -> None:
        """
        Renderiza la información del dataset en el sidebar
        """
        with st.expander("📊 Información del Dataset", expanded=True):
            dataset_info = SessionManager.get_dataset_info()
            
            if dataset_info:
                # Mostrar nombre con estilo más compacto
                st.write(f"Nombre: **{dataset_info['nombre']}**")
                
                # Información compacta del dataset en 2 columnas
                col1, col2 = st.columns(2)
                
                with col1:
                    # Mostrar origen
                    st.write(f"**Origen:** {dataset_info['origen']}")
                    
                    # Fecha más compacta
                    if dataset_info.get('fecha_carga'):
                        st.write(f"**Fecha:** {dataset_info['fecha_carga']}")
                
                with col2:
                    # Dimensiones del dataframe
                    st.write(f"**Filas:** {dataset_info['filas']}")
                    st.write(f"**Cols:** {dataset_info['columnas']}")
                
                # Mostrar información de configuración si está disponible en formato compacto
                if dataset_info.get('variable_objetivo'):
                    st.write("---")
                    tipo_problema = dataset_info.get('tipo_problema', '').capitalize()
                    var_obj = dataset_info.get('variable_objetivo')
                    st.write(f"**Problema:** {tipo_problema} → **Objetivo:** `{var_obj}`")
                    
                    # Mostrar predictores en forma compacta
                    if dataset_info.get('num_predictores', 0) > 0:
                        st.write(f"**Predictores:** {dataset_info['num_predictores']} variables")
                        # Mostrar los primeros 3-5 predictores y un expander para ver todos
                        predictores = dataset_info.get('lista_predictores', [])
                        if predictores:
                            max_show = 5
                            primeros = predictores[:max_show]
                            st.write(", ".join([f"`{p}`" for p in primeros]) + (f" ... (+{len(predictores)-max_show} más)" if len(predictores) > max_show else ""))
                            if len(predictores) > max_show:
                                if st.checkbox("Ver todos los predictores", key="chk_ver_predictores"):
                                    st.write(", ".join([f"`{p}`" for p in predictores]))
            else:
                st.info("No hay dataset cargado")
                
                # Si ya tenemos un usuario logueado, mostrar un botón de acceso rápido a carga
                if SessionManager.is_logged_in():
                    if st.button("📊 Ir a Cargar Datos", key="btn_ir_cargar"):
                        st.switch_page("pages/Datos/01_Cargar_Datos.py")
    
    @staticmethod
    def render_model_info() -> None:
        """
        Renderiza la información del modelo de ML seleccionado en el sidebar
        """
        modelo_recomendado = SessionManager.obtener_estado("modelo_recomendado", None)
        if modelo_recomendado and isinstance(modelo_recomendado, dict):
            modelo = modelo_recomendado.get("modelo_recomendado", {})
            if modelo:
                with st.expander("🤖 Modelo Seleccionado", expanded=True):
                    st.write(f"**Nombre:** {modelo.get('nombre', 'N/A')}")
                    st.write(f"**Tipo de problema:** {modelo_recomendado.get('tipo_problema', 'N/A').capitalize()}")
                    st.write(f"**Variable objetivo:** {modelo_recomendado.get('variable_objetivo', 'N/A')}")
                    st.write(f"**Criterio de selección:** {modelo_recomendado.get('criterio_usado', 'N/A').upper()}")
                    st.write(f"**Evaluados:** {modelo_recomendado.get('total_modelos_evaluados', 'N/A')}")
                    st.write(f"**Fecha recomendación:** {modelo_recomendado.get('timestamp', 'N/A')}")
                    # Métricas principales
                    metricas = modelo.get('metricas', {})
                    if metricas:
                        st.write("---")
                        st.write("**Métricas principales:**")
                        for k, v in metricas.items():
                            if isinstance(v, (int, float)):
                                st.write(f"- {k.title()}: {v:.4f}")
                            else:
                                st.write(f"- {k.title()}: {v}")
            else:
                st.info("No hay modelo seleccionado actualmente.")

    @staticmethod
    def render_sidebar() -> None:
        """
        Renderiza el sidebar completo
        """
        with st.sidebar:
            # Renderizar solo los componentes esenciales
            SidebarComponents.render_user_info()         # Información del usuario
            SidebarComponents.render_workflow_stepper()  # Indicador de progreso
            SidebarComponents.render_dataset_info()      # Información del dataset
            SidebarComponents.render_model_info()        # Información del modelo seleccionado
