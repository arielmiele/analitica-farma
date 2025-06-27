import streamlit as st
import sys
import os

# Agregar el directorio src al path para poder importar los módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importar módulos de la aplicación
from src.audit.logger import log_audit
from src.state.session_manager import SessionManager

st.title("🔐 0. Inicio de Sesión y Acceso a Analítica Farma")

st.write("""
Bienvenido a la plataforma de análisis industrial farmacéutico.

Esta aplicación te permitirá realizar un análisis exhaustivo de datos farmacéuticos, desde la carga y validación de datos hasta la recomendación de modelos y generación de reportes.

**Flujo recomendado:**
1. Cargar Datos
2. Validar Datos
3. Analizar Calidad
4. Configurar Datos
5. Entrenar y Evaluar Modelos
6. Generar Reportes
""")

st.subheader("Opciones de acceso")

with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🧪 Acceso rápido (demo)", use_container_width=True, help="Acceso de desarrollo/pruebas. Persiste usuario demo."):
            SessionManager.set_user(
                usuario_id=1,
                usuario_nombre="usuario",
                usuario_rol="analista",
                usuario_email="usuario@empresa.com"
            )
            log_audit(1, "LOGIN", "Sistema", "Acceso rápido (demo)")
            st.success("Acceso demo exitoso. Redirigiendo...")
            st.rerun()
    with col2:
        if st.button("🔑 Login con usuario/contraseña (Snowflake)", use_container_width=True, help="Validación contra tabla de usuarios en Snowflake."):
            st.info("[Futuro] Aquí se implementará el login contra Snowflake.\n\nDeberás pedir usuario y contraseña, validar contra la tabla de usuarios en Snowflake y luego usar SessionManager.set_user().")
    with col3:
        if st.button("🔒 Login SSO corporativo (Snowflake)", use_container_width=True, help="SSO corporativo vía Snowflake/AD."):
            st.info("[Futuro] Aquí se integrará el login SSO corporativo.\n\nSe usará el proveedor de identidad de la empresa y luego se obtendrán los datos del usuario para SessionManager.set_user().")

# Redirección automática si ya está logueado
if SessionManager.is_logged_in():
    st.switch_page("pages/Datos/01_Cargar_Datos.py")