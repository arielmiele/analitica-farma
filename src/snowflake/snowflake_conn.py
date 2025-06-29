import streamlit as st
from typing import Optional
import snowflake.connector

def get_native_snowflake_connection(schema: Optional[str] = None):
    """
    Devuelve una conexión nativa de snowflake.connector usando los parámetros de st.secrets.
    Permite especificar el esquema destino.
    """
    conn_params = st.secrets["connections"]["snowflake"]
    return snowflake.connector.connect(
        user=conn_params["user"],
        password=conn_params["password"],
        account=conn_params["account"],
        warehouse=conn_params["warehouse"],
        database=conn_params["database"],
        schema=schema if schema is not None else conn_params["schema"]
    )