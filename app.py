import streamlit as st

# Importar módulos del proyecto reorganizado
from src.state.session_manager import SessionManager
from src.ui.sidebar import SidebarComponents

# Configuración global de la app (título e ícono en la pestaña)
st.set_page_config(page_title="Analitica Farma", page_icon=":pill:", layout="wide")

# Inicializar el estado de sesión
SessionManager.init_session_state()

# Función de deslogueo: cambia el estado de sesión y recarga la app
def deslogueo():
    # Verificar si el usuario está logueado antes de intentar acceder a sus datos
    if SessionManager.is_logged_in() and "usuario_nombre" in st.session_state:
        nombre_usuario = st.session_state.usuario_nombre
        st.title(f"🧪 ¡Bienvenido a Analítica Farma, {nombre_usuario}!")
    else:
        st.title("🧪 ¡Bienvenido a Analítica Farma!")
    
    st.markdown("""
    <div style="text-align: justify; font-size: 16px;">
    <p>Esta aplicación te permite analizar datos de procesos farmacéuticos, entrenar modelos de machine learning y generar reportes de manera ágil e intuitiva.</p>
    <h3>📋 Flujo de trabajo recomendado:</h3>
    <ol>
        <li><strong>Cargar Datos</strong>: Importa datos desde CSV o conecta con Snowflake</li>
        <li><strong>Configurar Datos</strong>: Define variable objetivo y tipo de problema</li>
        <li><strong>Validar Datos</strong>: Verifica la calidad y coherencia</li>
        <li><strong>Analizar Calidad</strong>: Examina estadísticas y distribuciones</li>
        <li><strong>Entrenar Modelos</strong>: Ejecuta benchmarking de algoritmos</li>
        <li><strong>Evaluar Modelos</strong>: Compara métricas y visualizaciones</li>
        <li><strong>Validación Cruzada</strong>: Detecta overfitting/underfitting con curvas de aprendizaje</li>
        <li><strong>Recomendar Modelo</strong>: Obtén sugerencia del mejor modelo</li>
        <li><strong>Generar Reportes</strong>: Exporta informes y dashboards</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")

    # Sección de acciones
    st.markdown("### 🚀 Comenzar")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Cargar datos", use_container_width=True):
            st.switch_page("pages/Datos/01_Cargar_Datos.py")
    
    with col2:
        if st.button("🚪 Deslogueo", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()

# Crear el layout con sidebar permanente
if SessionManager.is_logged_in():    # Definición de páginas usando st.Page (cada una puede ser un script o función)
    pagina_deslogueo = st.Page(deslogueo, title="Bienvenida", icon=":material/home:", default=True)
    cargar_datos = st.Page("pages/Datos/01_Cargar_Datos.py", title="Cargar Datos", icon=":material/database_upload:")
    configurar_datos = st.Page("pages/Datos/02_Configurar_Datos.py", title="Configurar Datos", icon=":material/check_circle:")
    validar_datos = st.Page("pages/Datos/03_Validar_Datos.py", title="Validar Datos", icon=":material/wand_stars:")
    analizar_calidad = st.Page("pages/Datos/04_Analizar_Calidad.py", title="Analizar Calidad", icon=":material/analytics:")
    entrenar_modelos = st.Page("pages/Machine Learning/05_Entrenar_Modelos.py", title="Entrenar Modelos", icon=":material/model_training:")
    evaluar_modelos = st.Page("pages/Machine Learning/06_Evaluar_Modelos.py", title="Evaluar Modelos", icon=":material/network_intel_node:")
    validacion_cruzada = st.Page("pages/Machine Learning/07_Validacion_Cruzada.py", title="Validación Cruzada", icon=":material/science:")
    recomendar_modelo = st.Page("pages/Machine Learning/08_Recomendar_Modelo.py", title="Recomendar Modelo", icon=":material/network_intelligence:")
    reporte = st.Page("pages/Reportes/07_Reporte.py", title="Reporte", icon=":material/description:")
    dashboard = st.Page("pages/Reportes/08_Dashboard.py", title="Dashboard", icon=":material/dashboard:")
    explicar_modelo = st.Page("pages/Machine Learning/09_Explicar_Modelo.py", title="Explicar Modelo", icon=":material/psychology:")
    
    # Mostrar información del sidebar usando el componente dedicado
    SidebarComponents.render_sidebar()
      # Navegación multipágina agrupada por secciones
    pg = st.navigation(
        {
            "Inicio": [pagina_deslogueo],
            "Datos": [cargar_datos, configurar_datos, validar_datos, analizar_calidad],
            "Machine Learning": [entrenar_modelos, evaluar_modelos, validacion_cruzada, recomendar_modelo, explicar_modelo],
            "Reportes & Dashboards": [reporte, dashboard]
        }
    )
    
    # Ejecuta la navegación seleccionada
    pg.run()
else:
    # Página de login
    pagina_logueo = st.Page("pages/00_Logueo.py", title="Logueo", icon=":material/login:", default=True)
    pg = st.navigation([pagina_logueo])
    pg.run()