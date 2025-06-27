import streamlit as st
from src.state.session_manager import SessionManager

class SidebarComponents:
    """
    Clase para gestionar los componentes de UI del sidebar.
    Contiene métodos para renderizar diferentes partes del sidebar.
    """
    
    @staticmethod
    def render_user_info() -> None:
        """
        Renderiza la información persistente del usuario conectado en el sidebar
        """
        if SessionManager.is_logged_in():
            with st.expander("👤 Usuario Conectado", expanded=True):
                user = SessionManager.get_user_info()
                st.write("**ID:**", user["usuario_id"])
                st.write("**Nombre:**", user["usuario_nombre"])
                st.write("**Rol:**", user["usuario_rol"])
                st.write("**Email:**", user["usuario_email"])
                if st.button("🚪 Cerrar Sesión", key="btn_logout_sidebar"):
                    SessionManager.logout()
                    st.rerun()
    
    @staticmethod
    def render_dataset_info() -> None:
        """
        Renderiza la información del dataset en el sidebar
        """
        with st.expander("📊 Información del Dataset", expanded=True):
            dataset_info = SessionManager.get_dataset_info()
            
            if dataset_info:
                # Mostrar nombre con estilo más compacto
                st.write(f"Nombre: **{dataset_info['nombre']}**")
                
                # Información compacta del dataset en 2 columnas
                col1, col2 = st.columns(2)
                
                with col1:
                    # Mostrar origen
                    st.write(f"**Origen:** {dataset_info['origen']}")
                    
                    # Fecha más compacta
                    if dataset_info.get('fecha_carga'):
                        st.write(f"**Fecha:** {dataset_info['fecha_carga']}")
                
                with col2:
                    # Dimensiones del dataframe
                    st.write(f"**Filas:** {dataset_info['filas']}")
                    st.write(f"**Cols:** {dataset_info['columnas']}")
                
                # Mostrar información de configuración si está disponible en formato compacto
                if dataset_info.get('variable_objetivo'):
                    st.write("---")
                    # Mostrar en formato más compacto
                    tipo_problema = dataset_info.get('tipo_problema', '').capitalize()
                    var_obj = dataset_info.get('variable_objetivo')
                    
                    st.write(f"**Problema:** {tipo_problema} → **Objetivo:** {var_obj}")
                    
                    # Mostrar predictores en forma compacta
                    if dataset_info.get('num_predictores', 0) > 0:
                        st.write(f"**Predictores:** {dataset_info['num_predictores']} variables")
                        # Mostrar los primeros 3-5 predictores y un expander para ver todos
                        predictores = dataset_info.get('lista_predictores', [])
                        if predictores:
                            max_show = 5
                            primeros = predictores[:max_show]
                            st.write(", ".join([f"`{p}`" for p in primeros]) + (f" ... (+{len(predictores)-max_show} más)" if len(predictores) > max_show else ""))
                            if len(predictores) > max_show:
                                with st.expander("Ver todos los predictores"):
                                    st.write(", ".join([f"`{p}`" for p in predictores]))
            else:
                st.info("No hay dataset cargado")
                
                # Si ya tenemos un usuario logueado, mostrar un botón de acceso rápido a carga
                if SessionManager.is_logged_in():
                    if st.button("📊 Ir a Cargar Datos", key="btn_ir_cargar"):
                        st.switch_page("pages/Datos/01_Cargar_Datos.py")
    
    @staticmethod
    def render_sidebar() -> None:
        """
        Renderiza el sidebar completo
        """
        with st.sidebar:
            # Renderizar solo los componentes esenciales
            SidebarComponents.render_user_info()  # Información del usuario
            SidebarComponents.render_dataset_info()
