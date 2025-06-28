import streamlit as st

def get_snowflake_connection():
    """
    Devuelve una conexión segura a Snowflake usando la configuración de secrets.toml.
    """
    return st.connection("snowflake")

# Ejemplo de uso para consulta rápida (eliminar o comentar en producción):
if __name__ == "__main__":
    conn = get_snowflake_connection()
    df = conn.query("SELECT CURRENT_TIMESTAMP();")
    print(df)
