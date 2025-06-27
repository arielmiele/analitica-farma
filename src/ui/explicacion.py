"""
Componentes de UI para visualización de explicaciones de modelos (SHAP).
Solo lógica de presentación, sin cálculos de negocio.
"""
import streamlit as st
import matplotlib.pyplot as plt
import shap

def mostrar_grafico_importancias(importancias, shap_values, X):
    """
    Muestra gráficos de importancia de features usando SHAP y matplotlib en Streamlit.
    Args:
        importancias: DataFrame con importancia media de cada feature
        shap_values: Objeto SHAP values
        X: DataFrame de features
    """
    st.subheader("Gráfico de Importancia de Variables (SHAP)")
    st.dataframe(importancias, hide_index=True)

    # Gráfico de barras de importancias
    fig, ax = plt.subplots(figsize=(8, 4))
    importancias.plot.bar(x='feature', y='importance', ax=ax, legend=False)
    ax.set_ylabel('Importancia media (|SHAP|)')
    ax.set_xlabel('Variable')
    ax.set_title('Importancia de variables según SHAP')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

    # Gráfico summary plot de SHAP
    st.subheader("SHAP Summary Plot")
    fig2 = plt.figure(figsize=(8, 4))
    shap.summary_plot(shap_values.values, X, show=False)
    st.pyplot(fig2)
