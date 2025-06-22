import streamlit as st

# Opcional: configuración global de la app
st.set_page_config(page_title="Analitica Farma", page_icon=":pill:")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def logueo():
    if st.button("Logeo"):
        st.session_state.logged_in = True
        st.rerun()

def deslogueo():
    if st.button("Deslogueo"):
        st.session_state.logged_in = False
        st.rerun()

pagina_logueo = st.Page(logueo, title="Logueo", icon=":material/login:")
pagina_deslogueo = st.Page(deslogueo, title="Deslogueo", icon=":material/logout:")
cargar_datos = st.Page("pages/Datos/01_Cargar_Datos.py", title="Cargar Datos", icon=":material/dashboard:", default=True)
validar_datos = st.Page("pages/Datos/02_Validar_Datos.py", title="Validar Datos", icon=":material/bug_report:")
transformaciones = st.Page("pages/Datos/03_Transformaciones.py", title="Transformaciones", icon=":material/notification_important:")
entrenar_modelos = st.Page("pages/Machine Learning/04_Entrenar_Modelos.py", title="Entrenar Modelos", icon=":material/search:")
evaluar_modelos = st.Page("pages/Machine Learning/05_Evaluar_Modelos.py", title="Evaluar Modelos", icon=":material/history:")
recomendar_modelo = st.Page("pages/Machine Learning/06_Recomendar_Modelo.py", title="Recomendar Modelo", icon=":material/thumb_up:")
reporte = st.Page("pages/Reportes/07_Reporte.py", title="Reporte", icon=":material/picture_as_pdf:")
dashboard = st.Page("pages/Reportes/08_Dashboard.py", title="Dashboard", icon=":material/dashboard:")

if st.session_state.logged_in:
    pg = st.navigation(
        {
            "Cuenta": [pagina_deslogueo],
            "Datos": [cargar_datos, validar_datos, transformaciones],
            "Machine Learning": [entrenar_modelos, evaluar_modelos, recomendar_modelo],
            "Reportes & Dashboards": [reporte, dashboard]
        }
    )
else:
    pg = st.navigation([pagina_logueo])


# Crear la navegación
# pg = st.navigation(pages)
pg.run()