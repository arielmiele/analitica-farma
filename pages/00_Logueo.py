import streamlit as st
import sys
import os

# Agregar el directorio src al path para poder importar los módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importar módulos de la aplicación
from src.audit.logger import log_audit
from src.state.session_manager import SessionManager
from src.seguridad.autenticador import validar_usuario

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
            # log_audit(1, "LOGIN", "Sistema", "Acceso rápido (demo)")
            st.success("Acceso demo exitoso. Redirigiendo...")
            st.rerun()
    with col2:
        with st.expander("🔑 Acceso con usuario/contraseña (Snowflake)", expanded=False):
            st.markdown("""
            <span style='color:#888'>Valida contra la tabla de usuarios en Snowflake.<br>Solo usuarios autorizados pueden acceder.</span>
            """, unsafe_allow_html=True)
            with st.form("login_form_snowflake", clear_on_submit=False):
                usuario = st.text_input("Usuario", max_chars=50)
                password = st.text_input("Contraseña", type="password", max_chars=50)
                submitted = st.form_submit_button("Iniciar sesión")
                if submitted:
                    if not usuario or not password:
                        st.warning("Por favor, completa usuario y contraseña.")
                    else:
                        user_data = validar_usuario(usuario, password)
                        if user_data:
                            SessionManager.set_user(
                                usuario_id=user_data["id"],
                                usuario_nombre=user_data["usuario"],
                                usuario_rol=user_data["rol"],
                                usuario_email=user_data["email"]
                            )
                            # log_audit(user_data["id"], "LOGIN", "Sistema", "Login exitoso Snowflake")
                            st.success("Login exitoso. Redirigiendo...")
                            st.rerun()
                        else:
                            # log_audit(0, "LOGIN_FAIL", "Sistema", f"Intento fallido usuario: {usuario}")
                            st.error("Usuario o contraseña incorrectos, o usuario inactivo.")
    with col3:
        if st.button("🔒 Login SSO corporativo (Snowflake)", use_container_width=True, help="SSO corporativo vía Snowflake/AD."):
            st.info("[Futuro] Aquí se integrará el login SSO corporativo.\n\nSe usará el proveedor de identidad de la empresa y luego se obtendrán los datos del usuario para SessionManager.set_user().")

# Redirección automática si ya está logueado
if SessionManager.is_logged_in():
    st.switch_page("pages/Datos/01_Cargar_Datos.py")