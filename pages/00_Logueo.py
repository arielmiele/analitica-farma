import streamlit as st

st.title("Bienvenido a Analítica Farma")

st.markdown('<div style="text-align: justify;">Esta aplicación te permitirá realizar un análisis exhaustivo de datos farmacéuticos, desde la carga y validación de datos hasta la recomendación de modelos y generación de reportes.</div>', unsafe_allow_html=True)

st.markdown('<div style="text-align: justify;"></br>Comencemos por conectarnos a la aplicación.</div>', unsafe_allow_html=True)

st.markdown('</br>', unsafe_allow_html=True)

if st.button("Conectarse"):
    st.session_state.logged_in = True
    st.rerun()