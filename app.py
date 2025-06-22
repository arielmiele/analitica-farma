from logging import config
import streamlit as st

# Configuración global de la app (título e ícono en la pestaña)
st.set_page_config(page_title="Analitica Farma", page_icon=":pill:")

# Estado de sesión para control de login/logout
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Función de deslogueo: cambia el estado de sesión y recarga la app
def deslogueo():
    if st.button("Deslogueo"):
        st.session_state.logged_in = False
        st.rerun()

# Definición de páginas usando st.Page (cada una puede ser un script o función)
pagina_logueo = st.Page("pages/00_Logueo.py", title="Logueo", icon=":material/login:", default=True)
pagina_deslogueo = st.Page(deslogueo, title="Deslogueo", icon=":material/logout:")
cargar_datos = st.Page("pages/Datos/01_Cargar_Datos.py", title="Cargar Datos", icon=":material/database_upload:")
configurar_datos = st.Page("pages/Datos/02_Configurar_Datos.py", title="Configurar Datos", icon=":material/check_circle:")
transformaciones = st.Page("pages/Datos/03_Transformaciones.py", title="Transformaciones", icon=":material/wand_stars:")
entrenar_modelos = st.Page("pages/Machine Learning/04_Entrenar_Modelos.py", title="Entrenar Modelos", icon=":material/model_training:")
evaluar_modelos = st.Page("pages/Machine Learning/05_Evaluar_Modelos.py", title="Evaluar Modelos", icon=":material/network_intel_node:")
recomendar_modelo = st.Page("pages/Machine Learning/06_Recomendar_Modelo.py", title="Recomendar Modelo", icon=":material/network_intelligence:")
reporte = st.Page("pages/Reportes/07_Reporte.py", title="Reporte", icon=":material/description:")
dashboard = st.Page("pages/Reportes/08_Dashboard.py", title="Dashboard", icon=":material/dashboard:")

# Navegación multipágina agrupada por secciones y control de acceso según login
if st.session_state.logged_in:
    pg = st.navigation(
        {
            "Cuenta": [pagina_deslogueo],
            "Datos": [cargar_datos, configurar_datos, transformaciones],
            "Machine Learning": [entrenar_modelos, evaluar_modelos, recomendar_modelo],
            "Reportes & Dashboards": [reporte, dashboard]
        }
    )
else:
    pg = st.navigation([pagina_logueo])

# Ejecuta la navegación seleccionada
pg.run()