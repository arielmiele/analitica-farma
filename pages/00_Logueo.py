import streamlit as st
import sys
import os
import streamlit_authenticator as stauth
from src.audit.logger import setup_logger, log_audit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.state.session_manager import SessionManager
from src.database.usuarios_db import cargar_credentials_para_auth, obtener_usuario_por_email

logger = setup_logger("login")

USUARIO_SISTEMA = "sistema"
USUARIO_ANONIMO = "anonimo"

st.title("🔐 Inicio de Sesión y Acceso a Analítica Farma")
st.markdown("**Bienvenido a Analítica Farma**")
st.info(
    "Esta aplicación está diseñada para ayudar a empresas industriales, especialmente del sector farmacéutico, "
    "a analizar, validar y transformar datos de producción, entrenar y comparar modelos de machine learning, "
    "y generar reportes de manera segura y centralizada.\n\n"
    "Podés cargar datos desde archivos CSV, realizar limpieza y transformación, evaluar modelos, "
    "obtener recomendaciones y exportar resultados, todo con trazabilidad y control de acceso local."
)

# Redirección automática si ya está logueado
if SessionManager.is_logged_in():
    log_audit(
        usuario=USUARIO_SISTEMA,
        accion="LOGIN_REDIRECT",
        entidad="login",
        id_entidad="",
        detalles="Usuario ya logueado, redirigiendo a carga de datos.",
        id_sesion=SessionManager.obtener_estado("id_sesion", None)
    )
    st.switch_page("pages/Datos/01_Cargar_Datos.py")

# ── Configurar streamlit-authenticator ────────────────────────────────────────
credentials = cargar_credentials_para_auth()

authenticator = stauth.Authenticate(
    credentials=credentials,
    cookie_name="analitica_farma_auth",
    cookie_key="analitica_farma_local_secret_key",
    cookie_expiry_days=1,
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.subheader("Acceso con usuario/contraseña", divider="rainbow")
    st.write("Valida contra la base de datos local. Solo usuarios autorizados pueden acceder.")
    authenticator.login(location="main")

auth_status = st.session_state.get("authentication_status")
username_auth = st.session_state.get("username")

if auth_status is True:
    # Login exitoso via streamlit-authenticator — poblar SessionManager
    user_data = obtener_usuario_por_email(username_auth)
    if user_data:
        SessionManager.set_user(
            usuario_id=user_data["id"],
            usuario_nombre=user_data["nombre"],
            usuario_rol=user_data["rol"],
            usuario_email=user_data["email"],
        )
        SessionManager.crear_sesion(str(user_data["id"]))
        log_audit(
            usuario=str(user_data["id"]),
            accion="LOGIN_EXITO",
            entidad="USUARIO",
            id_entidad=str(user_data["id"]),
            detalles="Login exitoso (local).",
            id_sesion=SessionManager.obtener_estado("id_sesion", None)
        )
        st.success("Login exitoso. Redirigiendo...")
        st.rerun()

elif auth_status is False:
    log_audit(
        usuario=USUARIO_ANONIMO,
        accion="LOGIN_FALLIDO",
        entidad="login_form",
        id_entidad="",
        detalles="Credenciales incorrectas.",
        id_sesion=SessionManager.obtener_estado("id_sesion", None)
    )
    st.error("Email o contraseña incorrectos.")

elif auth_status is None:
    log_audit(
        usuario=USUARIO_ANONIMO,
        accion="LOGIN_PENDIENTE",
        entidad="login_form",
        id_entidad="",
        detalles="Formulario de login mostrado.",
        id_sesion=SessionManager.obtener_estado("id_sesion", None)
    )