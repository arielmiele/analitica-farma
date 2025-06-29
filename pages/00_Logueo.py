import streamlit as st
import sys
import os
from src.audit.logger import setup_logger, log_audit

# Agregar el directorio src al path para poder importar los módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importar módulos de la aplicación
from src.state.session_manager import SessionManager
from src.seguridad.autenticador import validar_usuario

# Configurar logger de auditoría para la página de login
logger = setup_logger("login")

st.title("🔐 Inicio de Sesión y Acceso a Analítica Farma")

st.markdown("**Bienvenido a Analítica Farma**")
st.info(
    "Esta aplicación está diseñada para ayudar a empresas industriales, especialmente del sector farmacéutico, "
    "a analizar, validar y transformar datos de producción, entrenar y comparar modelos de machine learning, "
    "y generar reportes de manera segura y centralizada.\n\n"
    "Podrás cargar datos desde archivos CSV o Snowflake, realizar limpieza y transformación, evaluar modelos, "
    "obtener recomendaciones y exportar resultados, todo con trazabilidad y control de acceso empresarial."
)

# Definir un ID de usuario especial para logs de sistema o anónimos
USUARIO_SISTEMA = "sistema"
USUARIO_ANONIMO = "anonimo"

# Centrar el formulario usando columnas vacías a los lados
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.subheader("Acceso con usuario/contraseña (Snowflake)", divider="rainbow")
    st.write("""Valida contra la tabla de usuarios en Snowflake. Solo usuarios autorizados pueden acceder.""")
    with st.form("login_form_snowflake", clear_on_submit=False):
        email = st.text_input("Email", max_chars=100)
        password = st.text_input("Contraseña", type="password", max_chars=50)
        submitted = st.form_submit_button("❄️ Iniciar sesión en Snowflake")
        if submitted:
            if not email or not password:
                log_audit(
                    usuario=USUARIO_ANONIMO,
                    accion="LOGIN_VACIO",
                    entidad="login_form",
                    id_entidad="",
                    detalles="Intento de login con campos vacíos.",
                    id_sesion=SessionManager.obtener_estado("id_sesion", None)
                )
                st.warning("Por favor, completa email y contraseña.")
            else:
                user_data = validar_usuario(email, password)
                if user_data:
                    SessionManager.set_user(
                        usuario_id=user_data["id"],
                        usuario_nombre=user_data["usuario"],
                        usuario_rol=user_data["rol"],
                        usuario_email=user_data["email"]
                    )
                    # Crear y registrar la sesión en Snowflake
                    SessionManager.crear_sesion(str(user_data["id"]))
                    log_audit(
                        usuario=str(user_data["id"]),
                        accion="LOGIN_EXITO",
                        entidad="USUARIO",
                        id_entidad=str(user_data["id"]),
                        detalles="Login exitoso.",
                        id_sesion=SessionManager.obtener_estado("id_sesion", None)
                    )
                    st.success("Login exitoso. Redirigiendo...")
                    st.rerun()
                else:
                    log_audit(
                        usuario=USUARIO_ANONIMO,
                        accion="LOGIN_FALLIDO",
                        entidad="login_form",
                        id_entidad="",
                        detalles="Intento fallido de login. Email o contraseña incorrectos, o usuario inactivo.",
                        id_sesion=SessionManager.obtener_estado("id_sesion", None)
                    )
                    st.error("Email o contraseña incorrectos, o usuario inactivo.")

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