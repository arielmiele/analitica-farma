"""
Módulo para generar visualizaciones avanzadas de los modelos de machine learning.
Implementa funciones para crear matrices de confusión, curvas ROC, curvas PR,
gráficos de residuos y otras visualizaciones para la evaluación detallada de modelos.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import io
import base64
from sklearn.metrics import (
    confusion_matrix, 
    roc_curve, 
    precision_recall_curve, 
    auc, 
    RocCurveDisplay,
    PrecisionRecallDisplay,
    ConfusionMatrixDisplay
)
from typing import Dict, List, Optional, Any, Literal
from src.audit.logger import log_audit

# Configuración global para visualizaciones
PALETA_COLORES = {
    "principal": "#4e79a7",
    "secundario": "#f28e2c",
    "terciario": "#e15759",
    "cuaternario": "#76b7b2",
    "complementario": "#59a14f",
    "neutral": "#edc949",
    "acento": "#af7aa1",
    "gris": "#9c9c9c"
}

# Estilos para plots
ESTILO_PLOT = {
    "figsize": (10, 6),
    "dpi": 100,
    "fontsize": 10,
    "titulo_fontsize": 14,
    "etiqueta_fontsize": 12,
    "grid": True,
    "style": "whitegrid"
}

def configurar_estilo_plots():
    """
    Configura el estilo global para los plots.
    """
    sns.set(style=ESTILO_PLOT["style"])
    plt.rcParams.update({
        'font.size': ESTILO_PLOT["fontsize"],
        'axes.titlesize': ESTILO_PLOT["titulo_fontsize"],
        'axes.labelsize': ESTILO_PLOT["etiqueta_fontsize"],
        'xtick.labelsize': ESTILO_PLOT["fontsize"],
        'ytick.labelsize': ESTILO_PLOT["fontsize"],
        'legend.fontsize': ESTILO_PLOT["fontsize"],
        'figure.figsize': ESTILO_PLOT["figsize"],
        'figure.dpi': ESTILO_PLOT["dpi"]
    })

def generar_matriz_confusion(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    id_sesion: str,
    usuario: str,
    clases: Optional[List[str]] = None,
    normalizar: Optional[Literal['true', 'pred', 'all']] = None,
    titulo: Optional[str] = None
) -> Figure:
    """
    Genera una matriz de confusión para un modelo de clasificación.
    
    Args:
        y_true: Valores reales de las clases
        y_pred: Valores predichos por el modelo
        id_sesion (str): ID de la sesión actual para trazabilidad
        usuario (str): Usuario que ejecuta la operación
        clases: Lista de nombres de las clases (opcional)
        normalizar: Tipo de normalización ('true', 'pred', 'all' o None)
        titulo: Título del gráfico
        
    Returns:
        plt.Figure: Figura con la matriz de confusión
    """
    try:
        configurar_estilo_plots()
        
        # Crear matriz de confusión
        cm = confusion_matrix(y_true, y_pred, normalize=normalizar)
        
        # Crear figura
        fig, ax = plt.subplots(figsize=ESTILO_PLOT["figsize"])
        
        # Generar visualización
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clases)
        disp.plot(cmap='Blues', ax=ax, colorbar=True)
        
        # Personalizar gráfico
        ax.set_title(titulo or "Matriz de Confusión")
        plt.grid(False)
        
        if normalizar:
            plt.colorbar(disp.im_, ax=ax, label="Proporción")
        else:
            plt.colorbar(disp.im_, ax=ax, label="Conteo")
        
        # Ajustar diseño
        fig.tight_layout()
        
        log_audit(
            id_sesion,
            usuario,
            "MATRIZ_CONFUSION_OK",
            "visualizador",
            "Matriz de confusión generada correctamente"
        )
        
        return fig
    except Exception as e:
        log_audit(
            id_sesion,
            usuario,
            "ERROR_MATRIZ_CONFUSION",
            "visualizador",
            f"Error al generar matriz de confusión: {str(e)}"
        )
        # Crear figura de error
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Error al generar matriz de confusión:\n{str(e)}",
                ha='center', va='center', fontsize=12, color='red')
        ax.axis('off')
        return fig

def generar_curva_roc(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    id_sesion: str,
    usuario: str,
    clases: Optional[List[str]] = None,
    multi_clase: bool = False,
    titulo: Optional[str] = None
) -> Figure:
    """
    Genera una curva ROC para un modelo de clasificación.
    
    Args:
        y_true: Valores reales de las clases
        y_prob: Probabilidades predichas por el modelo
        id_sesion (str): ID de la sesión actual para trazabilidad
        usuario (str): Usuario que ejecuta la operación
        clases: Lista de nombres de las clases (opcional)
        multi_clase: Si es True, genera curvas ROC para cada clase en problemas multiclase
        titulo: Título del gráfico
        
    Returns:
        plt.Figure: Figura con la curva ROC
    """
    try:
        configurar_estilo_plots()
        fig, ax = plt.subplots(figsize=ESTILO_PLOT["figsize"])
        
        # Caso binario
        if not multi_clase:
            # Para problemas binarios, y_prob puede ser una matriz de [n_samples, 2]
            # o un vector de [n_samples] con la probabilidad de la clase positiva
            if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
                y_prob = y_prob[:, 1]
                
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            
            # Graficar curva ROC
            RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot(ax=ax, color=PALETA_COLORES["principal"])
            
            # Graficar línea de referencia (clasificador aleatorio)
            ax.plot([0, 1], [0, 1], color=PALETA_COLORES["gris"], linestyle='--', label='Aleatorio (AUC = 0.5)')
            
        # Caso multiclase
        else:
            # Si no tenemos nombres de clases, usamos índices
            if clases is None:
                if len(y_prob.shape) > 1:
                    n_clases = y_prob.shape[1]
                else:
                    n_clases = len(np.unique(y_true))
                clases = [f"Clase {i}" for i in range(n_clases)]            # Para problemas multiclase, hacemos un enfoque más simple
            # Calcular la curva ROC directamente para cada clase individualmente
            if len(np.unique(y_true)) > 2:
                clases_unicas = np.unique(y_true)
                
                for i, clase in enumerate(clases_unicas):
                    try:
                        # Crear etiquetas binarias para esta clase (1 si es la clase, 0 en caso contrario)
                        y_true_bin = np.where(y_true == clase, 1, 0)
                        
                        # Si y_prob es una matriz de probabilidades para cada clase
                        if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
                            y_prob_clase = y_prob[:, i]
                        else:
                            # Si solo tenemos una columna, no podemos hacer multiclase
                            log_audit(
                                id_sesion,
                                usuario,
                                "WARNING_ROC_MULTICLASE",
                                "visualizador",
                                "No se pueden generar curvas ROC multiclase con una sola columna de probabilidades"
                            )
                            break
                            
                        # Calcular curva ROC
                        fpr, tpr, _ = roc_curve(y_true_bin, y_prob_clase)
                        roc_auc = auc(fpr, tpr)
                        
                        # Nombre de la clase (usar el proporcionado si existe)
                        nombre_clase = clases[i] if clases and i < len(clases) else f"Clase {clase}"
                        ax.plot(fpr, tpr, label=f'{nombre_clase} (AUC = {roc_auc:.3f})')
                    except Exception as e:
                        log_audit(
                            id_sesion,
                            usuario,
                            "WARNING_ROC_CLASE",
                            "visualizador",
                            f"Error al calcular curva ROC para clase {clase}: {str(e)}"
                        )
                        continue
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, label=f'{clase} (AUC = {roc_auc:.3f})')
            
            # Graficar línea de referencia
            ax.plot([0, 1], [0, 1], color=PALETA_COLORES["gris"], linestyle='--', label='Aleatorio (AUC = 0.5)')
        
        # Personalizar gráfico
        ax.set_xlabel('Tasa de Falsos Positivos')
        ax.set_ylabel('Tasa de Verdaderos Positivos')
        ax.set_title(titulo or "Curva ROC")
        ax.legend(loc="lower right")
        ax.grid(ESTILO_PLOT["grid"])
        
        # Ajustar diseño
        fig.tight_layout()
        
        log_audit(
            id_sesion,
            usuario,
            "CURVA_ROC_OK",
            "visualizador",
            "Curva ROC generada correctamente"
        )
        
        return fig
    except Exception as e:
        log_audit(
            id_sesion,
            usuario,
            "ERROR_CURVA_ROC",
            "visualizador",
            f"Error al generar curva ROC: {str(e)}"
        )
        # Crear figura de error
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Error al generar curva ROC:\n{str(e)}",
                ha='center', va='center', fontsize=12, color='red')
        ax.axis('off')
        return fig

def generar_curva_precision_recall(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    id_sesion: str,
    usuario: str,
    clases: Optional[List[str]] = None,
    multi_clase: bool = False,
    titulo: Optional[str] = None
) -> Figure:
    """
    Genera una curva Precision-Recall para un modelo de clasificación.
    
    Args:
        y_true: Valores reales de las clases
        y_prob: Probabilidades predichas por el modelo
        id_sesion (str): ID de la sesión actual para trazabilidad
        usuario (str): Usuario que ejecuta la operación
        clases: Lista de nombres de las clases (opcional)
        multi_clase: Si es True, genera curvas PR para cada clase en problemas multiclase
        titulo: Título del gráfico
        
    Returns:
        plt.Figure: Figura con la curva Precision-Recall
    """
    try:
        configurar_estilo_plots()
        fig, ax = plt.subplots(figsize=ESTILO_PLOT["figsize"])
        
        # Caso binario
        if not multi_clase:
            # Para problemas binarios, y_prob puede ser una matriz de [n_samples, 2]
            # o un vector de [n_samples] con la probabilidad de la clase positiva
            if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
                y_prob = y_prob[:, 1]
                
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = auc(recall, precision)
            
            # Graficar curva PR
            PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=pr_auc).plot(
                ax=ax, color=PALETA_COLORES["principal"]
            )
            
        # Caso multiclase
        else:
            # Si no tenemos nombres de clases, usamos índices
            if clases is None:
                if len(y_prob.shape) > 1:
                    n_clases = y_prob.shape[1]
                else:
                    n_clases = len(np.unique(y_true))
                clases = [f"Clase {i}" for i in range(n_clases)]            # Para problemas multiclase, hacemos un enfoque más simple
            # Calcular la curva PR directamente para cada clase individualmente
            if len(np.unique(y_true)) > 2:
                clases_unicas = np.unique(y_true)
                
                for i, clase in enumerate(clases_unicas):
                    try:
                        # Crear etiquetas binarias para esta clase (1 si es la clase, 0 en caso contrario)
                        y_true_bin = np.where(y_true == clase, 1, 0)
                        
                        # Si y_prob es una matriz de probabilidades para cada clase
                        if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
                            y_prob_clase = y_prob[:, i]
                        else:
                            # Si solo tenemos una columna, no podemos hacer multiclase
                            log_audit(
                                id_sesion,
                                usuario,
                                "WARNING_PR_MULTICLASE",
                                "visualizador",
                                "No se pueden generar curvas PR multiclase con una sola columna de probabilidades"
                            )
                            break
                            
                        # Calcular curva PR
                        precision, recall, _ = precision_recall_curve(y_true_bin, y_prob_clase)
                        pr_auc = auc(recall, precision)
                        
                        # Nombre de la clase (usar el proporcionado si existe)
                        nombre_clase = clases[i] if clases and i < len(clases) else f"Clase {clase}"
                        ax.plot(recall, precision, label=f'{nombre_clase} (AUC = {pr_auc:.3f})')
                    except Exception as e:
                        log_audit(
                            id_sesion,
                            usuario,
                            "WARNING_PR_CLASE",
                            "visualizador",
                            f"Error al calcular curva PR para clase {clase}: {str(e)}"
                        )
                        continue
                    pr_auc = auc(recall, precision)
                    ax.plot(recall, precision, label=f'{clase} (AUC = {pr_auc:.3f})')
            
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.legend(loc="lower left")
        
        # Personalizar gráfico
        ax.set_title(titulo or "Curva Precision-Recall")
        ax.grid(ESTILO_PLOT["grid"])
        
        # Ajustar diseño
        fig.tight_layout()
        
        log_audit(
            id_sesion,
            usuario,
            "CURVA_PR_OK",
            "visualizador",
            "Curva Precision-Recall generada correctamente"
        )
        
        return fig
    except Exception as e:
        log_audit(
            id_sesion,
            usuario,
            "ERROR_CURVA_PR",
            "visualizador",
            f"Error al generar curva Precision-Recall: {str(e)}"
        )
        # Crear figura de error
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Error al generar curva Precision-Recall:\n{str(e)}",
                ha='center', va='center', fontsize=12, color='red')
        ax.axis('off')
        return fig

def generar_grafico_residuos(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    id_sesion: str,
    usuario: str,
    titulo: Optional[str] = None
) -> Figure:
    """
    Genera un gráfico de residuos para un modelo de regresión.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos por el modelo
        id_sesion (str): ID de la sesión actual para trazabilidad
        usuario (str): Usuario que ejecuta la operación
        titulo: Título del gráfico
        
    Returns:
        plt.Figure: Figura con el gráfico de residuos
    """
    try:
        configurar_estilo_plots()
        
        # Calcular residuos
        residuos = y_true - y_pred
        
        # Crear figura con subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Residuos vs. Valores predichos
        axs[0, 0].scatter(y_pred, residuos, alpha=0.6, color=PALETA_COLORES["principal"])
        axs[0, 0].axhline(y=0, color=PALETA_COLORES["terciario"], linestyle="--")
        axs[0, 0].set_xlabel("Valores predichos")
        axs[0, 0].set_ylabel("Residuos")
        axs[0, 0].set_title("Residuos vs. Predicciones")
        axs[0, 0].grid(ESTILO_PLOT["grid"])
        
        # 2. Histograma de residuos
        axs[0, 1].hist(residuos, bins=30, alpha=0.7, color=PALETA_COLORES["principal"])
        axs[0, 1].axvline(x=0, color=PALETA_COLORES["terciario"], linestyle="--")
        axs[0, 1].set_xlabel("Residuos")
        axs[0, 1].set_ylabel("Frecuencia")
        axs[0, 1].set_title("Distribución de Residuos")
        axs[0, 1].grid(ESTILO_PLOT["grid"])
        
        # 3. QQ-plot de residuos
        import scipy.stats as stats
        stats.probplot(residuos, plot=axs[1, 0])
        axs[1, 0].set_title("Q-Q Plot de Residuos")
        axs[1, 0].grid(ESTILO_PLOT["grid"])
        
        # 4. Predicciones vs. Valores reales
        axs[1, 1].scatter(y_true, y_pred, alpha=0.6, color=PALETA_COLORES["secundario"])
        
        # Añadir línea de referencia (predicción perfecta)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axs[1, 1].plot([min_val, max_val], [min_val, max_val], color=PALETA_COLORES["terciario"], linestyle="--")
        
        axs[1, 1].set_xlabel("Valores reales")
        axs[1, 1].set_ylabel("Valores predichos")
        axs[1, 1].set_title("Predicciones vs. Valores reales")
        axs[1, 1].grid(ESTILO_PLOT["grid"])
        
        # Título global
        fig.suptitle(titulo or "Análisis de Residuos", fontsize=16)
          # Ajustar diseño
        fig.tight_layout(rect=(0, 0, 1, 0.96))  # Dejar espacio para el título
        
        log_audit(
            id_sesion,
            usuario,
            "GRAFICO_RESIDUOS_OK",
            "visualizador",
            "Gráfico de residuos generado correctamente"
        )
        
        return fig
    except Exception as e:
        log_audit(
            id_sesion,
            usuario,
            "ERROR_RESIDUOS",
            "visualizador",
            f"Error al generar gráfico de residuos: {str(e)}"
        )
        # Crear figura de error
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Error al generar gráfico de residuos:\n{str(e)}",
                ha='center', va='center', fontsize=12, color='red')
        ax.axis('off')
        return fig

def comparar_distribuciones(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    id_sesion: str,
    usuario: str,
    titulo: Optional[str] = None
) -> Figure:
    """
    Compara las distribuciones de valores reales y predichos para un modelo de regresión.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos por el modelo
        id_sesion (str): ID de la sesión actual para trazabilidad
        usuario (str): Usuario que ejecuta la operación
        titulo: Título del gráfico
        
    Returns:
        plt.Figure: Figura con la comparación de distribuciones
    """
    try:
        configurar_estilo_plots()
        
        # Crear figura
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. Histogramas superpuestos
        axs[0].hist(y_true, bins=30, alpha=0.7, label="Valores reales", color=PALETA_COLORES["principal"])
        axs[0].hist(y_pred, bins=30, alpha=0.7, label="Predicciones", color=PALETA_COLORES["secundario"])
        axs[0].set_xlabel("Valores")
        axs[0].set_ylabel("Frecuencia")
        axs[0].set_title("Histograma de distribuciones")
        axs[0].legend()
        axs[0].grid(ESTILO_PLOT["grid"])
        
        # 2. Densidad KDE
        sns.kdeplot(y_true, ax=axs[1], label="Valores reales", color=PALETA_COLORES["principal"])
        sns.kdeplot(y_pred, ax=axs[1], label="Predicciones", color=PALETA_COLORES["secundario"])
        axs[1].set_xlabel("Valores")
        axs[1].set_ylabel("Densidad")
        axs[1].set_title("Estimación de densidad")
        axs[1].legend()
        axs[1].grid(ESTILO_PLOT["grid"])
        
        # Título global
        fig.suptitle(titulo or "Comparación de Distribuciones", fontsize=16)
          # Ajustar diseño
        fig.tight_layout(rect=(0, 0, 1, 0.94))  # Dejar espacio para el título
        
        log_audit(
            id_sesion,
            usuario,
            "COMPARAR_DISTRIBUCIONES_OK",
            "visualizador",
            "Comparación de distribuciones generada correctamente"
        )
        
        return fig
    except Exception as e:
        log_audit(
            id_sesion,
            usuario,
            "ERROR_DISTRIBUCIONES",
            "visualizador",
            f"Error al comparar distribuciones: {str(e)}"
        )
        # Crear figura de error
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Error al comparar distribuciones:\n{str(e)}",
                ha='center', va='center', fontsize=12, color='red')
        ax.axis('off')
        return fig

def comparar_modelos_roc(
    modelos: Dict[str, Dict[str, Any]],
    X_test: np.ndarray,
    y_test: np.ndarray,
    id_sesion: str,
    usuario: str,
    titulo: Optional[str] = None
) -> Figure:
    """
    Compara múltiples modelos de clasificación mediante curvas ROC.
    
    Args:
        modelos: Diccionario con los modelos a comparar (nombre: modelo entrenado)
        X_test: Datos de prueba
        y_test: Etiquetas reales
        id_sesion (str): ID de la sesión actual para trazabilidad
        usuario (str): Usuario que ejecuta la operación
        titulo: Título del gráfico
        
    Returns:
        plt.Figure: Figura con la comparación de curvas ROC
    """
    try:
        configurar_estilo_plots()
        
        # Crear figura
        fig, ax = plt.subplots(figsize=ESTILO_PLOT["figsize"])
        
        # Graficar línea de referencia (clasificador aleatorio)
        ax.plot([0, 1], [0, 1], color=PALETA_COLORES["gris"], linestyle='--', label='Aleatorio (AUC = 0.5)')
        
        # Colores para los diferentes modelos
        colores = list(PALETA_COLORES.values())
        
        # Graficar curva ROC para cada modelo
        for i, (nombre, modelo_info) in enumerate(modelos.items()):
            modelo = modelo_info.get("modelo")
            if not modelo:
                continue
                
            # Obtener probabilidades predichas
            try:
                y_prob = modelo.predict_proba(X_test)
                
                # Para problemas binarios, tomamos la probabilidad de la clase positiva
                if y_prob.shape[1] == 2:
                    y_prob = y_prob[:, 1]
                    
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                
                # Usar colores cíclicos
                color_idx = i % len(colores)
                color = colores[color_idx]
                
                ax.plot(fpr, tpr, label=f'{nombre} (AUC = {roc_auc:.3f})', color=color)
            except Exception as e:
                log_audit(
                    id_sesion,
                    usuario,
                    "WARNING_ROC_MODELO",
                    "visualizador",
                    f"No se pudo generar curva ROC para el modelo {nombre}: {str(e)}"
                )
        
        # Personalizar gráfico
        ax.set_xlabel('Tasa de Falsos Positivos')
        ax.set_ylabel('Tasa de Verdaderos Positivos')
        ax.set_title(titulo or "Comparación de Modelos - Curva ROC")
        ax.legend(loc="lower right")
        ax.grid(ESTILO_PLOT["grid"])
        
        # Ajustar diseño
        fig.tight_layout()
        
        log_audit(
            id_sesion,
            usuario,
            "COMPARAR_MODELOS_ROC_OK",
            "visualizador",
            "Comparación de modelos ROC generada correctamente"
        )
        
        return fig
    except Exception as e:
        log_audit(
            id_sesion,
            usuario,
            "ERROR_COMPARAR_MODELOS_ROC",
            "visualizador",
            f"Error al comparar modelos con curvas ROC: {str(e)}"
        )
        # Crear figura de error
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Error al comparar modelos con curvas ROC:\n{str(e)}",
                ha='center', va='center', fontsize=12, color='red')
        ax.axis('off')
        return fig

def figura_a_base64(fig: Figure) -> str:
    """
    Convierte una figura de matplotlib a una cadena base64 para mostrar en HTML.
    
    Args:
        fig: Figura de matplotlib
        
    Returns:
        str: Cadena base64 que representa la imagen
    """
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    imagen_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return imagen_base64

def generar_interpretacion_matriz_confusion(
    matriz: np.ndarray, 
    clases: Optional[List[str]] = None
) -> str:
    """
    Genera una interpretación textual de la matriz de confusión.
    
    Args:
        matriz: Matriz de confusión
        clases: Nombres de las clases
        
    Returns:
        str: Interpretación textual de la matriz
    """
    if clases is None:
        clases = [f"Clase {i}" for i in range(matriz.shape[0])]
    
    # Calcular métricas básicas
    n_clases = matriz.shape[0]
    diagonal = np.diag(matriz)
    
    # Iniciar texto
    texto = "### Interpretación de la Matriz de Confusión\n\n"
    
    # Analizar cada clase
    for i in range(n_clases):
        clase = clases[i]
        verdaderos_positivos = diagonal[i]
        falsos_positivos = np.sum(matriz[:, i]) - verdaderos_positivos
        falsos_negativos = np.sum(matriz[i, :]) - verdaderos_positivos
        
        precision = verdaderos_positivos / (verdaderos_positivos + falsos_positivos) if (verdaderos_positivos + falsos_positivos) > 0 else 0
        recall = verdaderos_positivos / (verdaderos_positivos + falsos_negativos) if (verdaderos_positivos + falsos_negativos) > 0 else 0
        
        texto += f"**{clase}**:\n"
        texto += f"- Verdaderos Positivos: {verdaderos_positivos:.0f}\n"
        texto += f"- Falsos Positivos: {falsos_positivos:.0f}\n"
        texto += f"- Falsos Negativos: {falsos_negativos:.0f}\n"
        texto += f"- Precisión: {precision:.2f}\n"
        texto += f"- Recall: {recall:.2f}\n\n"
    
    # Exactitud global
    exactitud = np.sum(diagonal) / np.sum(matriz)
    texto += f"**Exactitud Global**: {exactitud:.2f}\n\n"
    
    # Interpretación general
    texto += "**Análisis General**:\n"
    
    # Identificar clases con mejor y peor desempeño
    precisions = []
    recalls = []
    
    for i in range(n_clases):
        vp = diagonal[i]
        fp = np.sum(matriz[:, i]) - vp
        fn = np.sum(matriz[i, :]) - vp
        
        prec = vp / (vp + fp) if (vp + fp) > 0 else 0
        rec = vp / (vp + fn) if (vp + fn) > 0 else 0
        
        precisions.append(prec)
        recalls.append(rec)
    
    mejor_precision_idx = np.argmax(precisions)
    peor_precision_idx = np.argmin(precisions)
    mejor_recall_idx = np.argmax(recalls)
    peor_recall_idx = np.argmin(recalls)
    
    texto += f"- Mejor precisión: {clases[mejor_precision_idx]} ({precisions[mejor_precision_idx]:.2f})\n"
    texto += f"- Peor precisión: {clases[peor_precision_idx]} ({precisions[peor_precision_idx]:.2f})\n"
    texto += f"- Mejor recall: {clases[mejor_recall_idx]} ({recalls[mejor_recall_idx]:.2f})\n"
    texto += f"- Peor recall: {clases[peor_recall_idx]} ({recalls[peor_recall_idx]:.2f})\n\n"
    
    # Confusiones más comunes
    confusiones = []
    for i in range(n_clases):
        for j in range(n_clases):
            if i != j:
                confusiones.append((i, j, matriz[i, j]))
    
    # Ordenar por valor de confusión (descendente)
    confusiones.sort(key=lambda x: x[2], reverse=True)
    
    if confusiones:
        texto += "**Confusiones más comunes**:\n"
        for i in range(min(3, len(confusiones))):
            clase_real, clase_pred, valor = confusiones[i]
            texto += f"- {clases[clase_real]} confundido con {clases[clase_pred]}: {valor:.0f} veces\n"
    
    return texto

def generar_interpretacion_curva_roc(auc_valor: float) -> str:
    """
    Genera una interpretación textual del valor AUC de la curva ROC.
    
    Args:
        auc_valor: Valor del área bajo la curva ROC
        
    Returns:
        str: Interpretación textual del AUC
    """
    texto = "### Interpretación de la Curva ROC\n\n"
    texto += f"El área bajo la curva ROC (AUC) es **{auc_valor:.3f}**.\n\n"
    
    # Interpretación según el valor de AUC
    if auc_valor >= 0.9:
        texto += "Este valor indica un **excelente** poder discriminativo del modelo. Tiene una alta capacidad para distinguir entre las clases positiva y negativa.\n\n"
    elif auc_valor >= 0.8:
        texto += "Este valor indica un **buen** poder discriminativo del modelo. Distingue bien entre las clases positiva y negativa en la mayoría de los casos.\n\n"
    elif auc_valor >= 0.7:
        texto += "Este valor indica un poder discriminativo **aceptable** del modelo. Puede distinguir entre las clases positiva y negativa, pero comete algunos errores.\n\n"
    elif auc_valor >= 0.6:
        texto += "Este valor indica un poder discriminativo **débil** del modelo. Tiene dificultades para distinguir entre las clases positiva y negativa.\n\n"
    else:
        texto += "Este valor indica un poder discriminativo **deficiente** del modelo. Su capacidad de predicción es apenas mejor que un clasificador aleatorio.\n\n"
    
    texto += "**Recordatorio**: Un AUC de 0.5 equivale a un clasificador aleatorio, mientras que un AUC de 1.0 representa un clasificador perfecto.\n\n"
    
    # Recomendaciones basadas en el AUC
    texto += "**Recomendaciones**:\n"
    
    if auc_valor < 0.7:
        texto += "- Considere utilizar técnicas de selección de características para mejorar el rendimiento.\n"
        texto += "- Pruebe con diferentes algoritmos de clasificación que puedan adaptarse mejor a sus datos.\n"
        texto += "- Evalúe si hay desequilibrio en las clases y aplique técnicas de muestreo si es necesario.\n"
    elif auc_valor < 0.8:
        texto += "- El modelo tiene un rendimiento aceptable, pero podría mejorar con ajuste de hiperparámetros.\n"
        texto += "- Considere técnicas de ensamblado para mejorar el rendimiento.\n"
    else:
        texto += "- El modelo tiene un buen rendimiento, pero verifique que no haya sobreajuste.\n"
        texto += "- Compare con otros modelos para confirmar que es la mejor opción para sus datos.\n"
    
    return texto

def generar_interpretacion_residuos(residuos: np.ndarray) -> str:
    """
    Genera una interpretación textual del análisis de residuos.
    
    Args:
        residuos: Array de residuos (y_true - y_pred)
        
    Returns:
        str: Interpretación textual de los residuos
    """
    # Calcular estadísticas de los residuos
    media = np.mean(residuos)
    mediana = np.median(residuos)
    desv_est = np.std(residuos)
    min_residuo = np.min(residuos)
    max_residuo = np.max(residuos)
    
    # Evaluar normalidad (aproximadamente)
    from scipy import stats
    _, p_valor = stats.normaltest(residuos)
    
    # Detectar heterocedasticidad (método simple)
    # Dividir en 5 secciones y comparar varianzas
    residuos_ordenados = np.sort(residuos)
    n = len(residuos_ordenados)
    secciones = 5
    varianzas = []
    
    for i in range(secciones):
        inicio = i * n // secciones
        fin = (i + 1) * n // secciones
        if fin > inicio:  # Evitar secciones vacías
            varianzas.append(np.var(residuos_ordenados[inicio:fin]))
    
    ratio_varianzas = max(varianzas) / min(varianzas) if min(varianzas) > 0 else float('inf')
    
    # Construir interpretación
    texto = "### Interpretación del Análisis de Residuos\n\n"
    
    # Estadísticas básicas
    texto += "**Estadísticas de los residuos**:\n"
    texto += f"- Media: {media:.4f}\n"
    texto += f"- Mediana: {mediana:.4f}\n"
    texto += f"- Desviación estándar: {desv_est:.4f}\n"
    texto += f"- Rango: [{min_residuo:.4f}, {max_residuo:.4f}]\n\n"
    
    # Interpretación de media y sesgo
    texto += "**Tendencia central**:\n"
    if abs(media) < 0.1 * desv_est:
        texto += "- Los residuos están **bien centrados** alrededor de cero, lo que indica que el modelo no tiene un sesgo sistemático.\n"
    elif media > 0:
        texto += f"- Los residuos tienen una **media positiva** ({media:.4f}), lo que sugiere que el modelo tiende a **subestimar** los valores reales.\n"
    else:
        texto += f"- Los residuos tienen una **media negativa** ({media:.4f}), lo que sugiere que el modelo tiende a **sobrestimar** los valores reales.\n"
    
    # Normalidad
    texto += "\n**Distribución de residuos**:\n"
    if p_valor > 0.05:
        texto += "- Los residuos siguen aproximadamente una **distribución normal** (p-valor > 0.05), lo que es una buena señal para un modelo de regresión.\n"
    else:
        texto += f"- Los residuos **no siguen una distribución normal** (p-valor = {p_valor:.4f}), lo que podría indicar problemas con el modelo o la presencia de valores atípicos.\n"
    
    # Heterocedasticidad
    texto += "\n**Variabilidad de residuos**:\n"
    if ratio_varianzas < 3:
        texto += "- La varianza de los residuos es **relativamente constante** a lo largo del rango de predicciones, lo que sugiere homocedasticidad.\n"
    else:
        texto += f"- Hay **evidencia de heterocedasticidad** (ratio de varianzas = {ratio_varianzas:.2f}), lo que indica que la variabilidad de los errores no es constante. Esto puede afectar la fiabilidad de los intervalos de confianza.\n"
    
    # Recomendaciones
    texto += "\n**Recomendaciones**:\n"
    
    if abs(media) > 0.1 * desv_est:
        texto += "- Considere añadir variables predictoras adicionales para reducir el sesgo sistemático.\n"
    
    if p_valor <= 0.05:
        texto += "- Investigue la presencia de valores atípicos y considere su tratamiento.\n"
        texto += "- Explore transformaciones de la variable objetivo (logarítmica, raíz cuadrada, etc.).\n"
    
    if ratio_varianzas >= 3:
        texto += "- Considere aplicar una transformación a la variable objetivo para estabilizar la varianza.\n"
        texto += "- Explore modelos que manejen heterocedasticidad, como regresión ponderada.\n"
    
    if desv_est > (max_residuo - min_residuo) / 4:
        texto += "- El modelo podría beneficiarse de técnicas de regularización para reducir la varianza.\n"
    
    return texto
