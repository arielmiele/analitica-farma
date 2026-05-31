import streamlit as st

# Importar módulos del proyecto reorganizado
from src.database.init_db import init_db
from src.state.session_manager import SessionManager
from src.ui.sidebar import SidebarComponents

# Configuración global de la app (título e ícono en la pestaña)
st.set_page_config(page_title="Analitica Farma", page_icon=":pill:", layout="wide")

# Inicializar base de datos SQLite (idempotente: solo crea si no existe)
init_db()

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
        st.info("Para acceder a la aplicación, por favor inicia sesión.")
        st.warning("Debes loguearte para acceder a las funcionalidades de la plataforma.")
        if st.button("Ir a Logueo", use_container_width=True):
            st.switch_page("pages/00_Logueo.py")
        return

    st.write("Esta aplicación te permite analizar datos de procesos farmacéuticos, entrenar modelos de machine learning y generar reportes de manera ágil e intuitiva.")
    st.subheader("📋 Flujo de trabajo recomendado:")
    st.markdown("""
    1. **Cargar Datos**: Importa datos desde CSV o conecta con Snowflake
    2. **Validar Datos**: Verifica la estructura, tipos y consistencia básica
    3. **Analizar Calidad**: Examina calidad, duplicados, outliers y estadísticas
    4. **Configurar Datos**: Define variable objetivo, predictores y tipo de problema
    5. **Entrenar Modelos**: Ejecuta benchmarking de algoritmos
    6. **Evaluar Modelos**: Compara métricas y visualizaciones
    7. **Validación Cruzada**: Detecta overfitting/underfitting con curvas de aprendizaje
    8. **Recomendar Modelo**: Obtén sugerencia del mejor modelo
    9. **Explicar Modelo**: Analiza interpretabilidad y variables importantes
    10. **Generar Reportes**: Exporta informes y dashboards
    """)
    
    # Sección de acciones
    st.markdown("### 🚀 Comenzar")
    st.info("Para iniciar, haz clic en \"📊 Cargar datos\" y sigue los pasos del flujo recomendado. El primer paso es cargar tus datos desde un archivo CSV o seleccionar un dataset existente.")
    
    col1, col2 = st.columns(2)
       
    with col1:
        if st.button("🚪 Deslogueo", use_container_width=True):
            SessionManager.logout()
            st.rerun()
    with col2:
        if st.button("📊 Cargar datos", use_container_width=True):
            st.switch_page("pages/Datos/01_Cargar_Datos.py")

# Crear el layout con sidebar permanente
if SessionManager.is_logged_in():    
    # Definición de páginas usando st.Page (cada una puede ser un script o función)
    pagina_deslogueo = st.Page(deslogueo, title="Bienvenida", icon=":material/home:", default=True)
    cargar_datos = st.Page("pages/Datos/01_Cargar_Datos.py", title="Cargar Datos", icon=":material/database_upload:")
    validar_datos = st.Page("pages/Datos/02_Validar_Datos.py", title="Validar Datos", icon=":material/wand_stars:")
    analizar_calidad = st.Page("pages/Datos/03_Analizar_Calidad.py", title="Analizar Calidad", icon=":material/analytics:")
    configurar_datos = st.Page("pages/Datos/04_Configurar_Datos.py", title="Configurar Datos", icon=":material/check_circle:")
    entrenar_modelos = st.Page("pages/Machine Learning/05_Entrenar_Modelos.py", title="Entrenar Modelos", icon=":material/model_training:")
    evaluar_modelos = st.Page("pages/Machine Learning/06_Evaluar_Modelos.py", title="Evaluar Modelos", icon=":material/network_intel_node:")
    validacion_cruzada = st.Page("pages/Machine Learning/07_Validacion_Cruzada.py", title="Validación Cruzada", icon=":material/science:")
    recomendar_modelo = st.Page("pages/Machine Learning/08_Recomendar_Modelo.py", title="Recomendar Modelo", icon=":material/network_intelligence:")
    explicar_modelo = st.Page("pages/Machine Learning/09_Explicar_Modelo.py", title="Explicar Modelo", icon=":material/psychology:")
    reporte = st.Page("pages/Reportes/10_Reporte.py", title="Generar Reporte", icon=":material/report:")
    
    # Mostrar información del sidebar usando el componente dedicado
    SidebarComponents.render_sidebar()
    
    # Navegación multipágina agrupada por secciones
    pg = st.navigation(
        {
            "Inicio": [pagina_deslogueo],
            "Datos": [cargar_datos, validar_datos, analizar_calidad, configurar_datos],
            "Machine Learning": [entrenar_modelos, evaluar_modelos, validacion_cruzada, recomendar_modelo, explicar_modelo],
            "Reporte": [reporte]
        }
    )
    
    # Ejecuta la navegación seleccionada
    pg.run()

else:
    # Página de login
    pagina_logueo = st.Page("pages/00_Logueo.py", title="Logueo", icon=":material/login:", default=True)
    pg = st.navigation([pagina_logueo])
    pg.run()