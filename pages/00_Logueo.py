import streamlit as st
import sys
import os
import sqlite3

# Agregar el directorio src al path para poder importar los módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importar módulos de la aplicación
from src.audit.logger import log_audit

# Ruta a la base de datos
db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'analitica_farma.db')

st.title("Bienvenido a Analítica Farma")

st.markdown('<div style="text-align: justify;">Esta aplicación te permitirá realizar un análisis exhaustivo de datos farmacéuticos, desde la carga y validación de datos hasta la recomendación de modelos y generación de reportes.</div>', unsafe_allow_html=True)

st.markdown('<div style="text-align: justify;"></br>Por favor, ingresa tus credenciales para comenzar.</div>', unsafe_allow_html=True)

st.markdown('</br>', unsafe_allow_html=True)

# Formulario de logueo
with st.form(key="login_form"):
    correo = st.text_input("Correo electrónico", value="usuario@empresa.com")
    st.form_submit_button("Iniciar sesión")

# Procesar el formulario cuando se envía
if st.session_state.get("login_form", False):
    try:
        # Verificar si el usuario existe en la base de datos
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id_usuario, nombre, rol FROM usuarios WHERE correo = ?", (correo,))
        usuario = cursor.fetchone()
        
        if usuario:
            id_usuario, nombre_usuario, rol_usuario = usuario
            
            # Guardar información del usuario en la sesión
            st.session_state.logged_in = True
            st.session_state.usuario_id = id_usuario
            st.session_state.usuario_nombre = nombre_usuario
            st.session_state.usuario_rol = rol_usuario
            st.session_state.usuario_correo = correo
            
            # Registrar el login en la auditoría
            log_audit(id_usuario, "LOGIN", "Sistema", f"Login exitoso como {rol_usuario}")
            
            # Mensaje de éxito y redirección
            st.success(f"¡Bienvenido, {nombre_usuario}!")
            st.rerun()
        else:
            st.error("Usuario no encontrado. Utilizando usuario por defecto.")
            
            # Usar usuario por defecto
            st.session_state.logged_in = True
            st.session_state.usuario_id = 1
            st.session_state.usuario_nombre = "usuario"
            st.session_state.usuario_rol = "analista"
            st.session_state.usuario_correo = "usuario@empresa.com"
            
            # Registrar el login en la auditoría
            log_audit(1, "LOGIN", "Sistema", "Login con usuario por defecto")
            
            st.rerun()
            
        conn.close()
        
    except Exception as e:
        st.error(f"Error al iniciar sesión: {str(e)}")
        
        # Usar usuario por defecto en caso de error
        st.session_state.logged_in = True
        st.session_state.usuario_id = 1
        st.session_state.usuario_nombre = "usuario"
        st.session_state.usuario_rol = "analista"
        st.session_state.usuario_correo = "usuario@empresa.com"
        
        st.rerun()

# Botón simple para logueo rápido (desarrollo)
if st.button("Acceso rápido (demo)"):
    st.session_state.logged_in = True
    st.session_state.usuario_id = 1
    st.session_state.usuario_nombre = "usuario"
    st.session_state.usuario_rol = "analista"
    st.session_state.usuario_correo = "usuario@empresa.com"
    
    # Registrar el login en la auditoría
    log_audit(1, "LOGIN", "Sistema", "Acceso rápido (demo)")
    
    st.rerun()