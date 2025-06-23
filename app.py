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
    if st.button("Deslogueo"):
        st.session_state.logged_in = False
        st.rerun()

# Crear el layout con sidebar permanente
if SessionManager.is_logged_in():
    # Definición de páginas usando st.Page (cada una puede ser un script o función)
    pagina_deslogueo = st.Page(deslogueo, title="Deslogueo", icon=":material/logout:")
    cargar_datos = st.Page("pages/Datos/01_Cargar_Datos.py", title="Cargar Datos", icon=":material/database_upload:")
    configurar_datos = st.Page("pages/Datos/02_Configurar_Datos.py", title="Configurar Datos", icon=":material/check_circle:")
    validar_datos = st.Page("pages/Datos/03_Validar_Datos.py", title="Validar Datos", icon=":material/wand_stars:")
    analizar_calidad = st.Page("pages/Datos/04_Analizar_Calidad.py", title="Analizar Calidad", icon=":material/analytics:")
    # No se usa transformaciones directamente, pero se puede agregar si es necesario
    #transformaciones = st.Page("pages/Datos/03_Transformaciones.py", title="Transformaciones", icon=":material/wand_stars:")
    entrenar_modelos = st.Page("pages/Machine Learning/04_Entrenar_Modelos.py", title="Entrenar Modelos", icon=":material/model_training:")
    evaluar_modelos = st.Page("pages/Machine Learning/05_Evaluar_Modelos.py", title="Evaluar Modelos", icon=":material/network_intel_node:")
    recomendar_modelo = st.Page("pages/Machine Learning/06_Recomendar_Modelo.py", title="Recomendar Modelo", icon=":material/network_intelligence:")
    reporte = st.Page("pages/Reportes/07_Reporte.py", title="Reporte", icon=":material/description:")
    dashboard = st.Page("pages/Reportes/08_Dashboard.py", title="Dashboard", icon=":material/dashboard:")
    
    # Mostrar información del sidebar usando el componente dedicado
    SidebarComponents.render_sidebar()
    
    # Navegación multipágina agrupada por secciones
    pg = st.navigation(
        {
            "Cuenta": [pagina_deslogueo],
            "Datos": [cargar_datos, configurar_datos, validar_datos, analizar_calidad],
            "Machine Learning": [entrenar_modelos, evaluar_modelos, recomendar_modelo],
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