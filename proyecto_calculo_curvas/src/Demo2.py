import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import interpolate

# Configuración de estilos
sns.set_theme(style="whitegrid")
plt.rcParams['font.size'] = 12

def obtener_formula_polinomio(coeficientes):
    """Genera una cadena legible del polinomio"""
    terms = []
    for i, coef in enumerate(coeficientes[::-1]):
        power = len(coeficientes) - i - 1
        if power == 0:
            terms.append(f"{coef:.4f}")
        elif power == 1:
            terms.append(f"{coef:.4f}x")
        else:
            terms.append(f"{coef:.4f}x^{power}")
    return "f(x) = " + " + ".join(terms)

def procesar_imagen(ruta_imagen):
    """Procesa la imagen y devuelve los puntos de la curva"""
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen desde {ruta_imagen}")
    
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gris, (5,5), 0)
    bordes = cv2.Canny(blur, 50, 150)
    
    puntos = np.column_stack(np.where(bordes > 0))[:, ::-1]
    return imagen, bordes, puntos

def visualizacion_deteccion(imagen, bordes, puntos):
    """Primera visualización: cómo se detectó la curva"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Imagen original
    ax1.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    ax1.set_title('Imagen Original')
    ax1.axis('off')
    
    # Bordes detectados
    ax2.imshow(bordes, cmap='gray')
    ax2.set_title('Bordes Detectados')
    ax2.axis('off')
    
    # Puntos detectados
    ax3.scatter(puntos[:, 0], puntos[:, 1], s=5, color='blue')
    ax3.set_title('Puntos Detectados')
    ax3.set_xlabel('X (px)')
    ax3.set_ylabel('Y (px)')
    ax3.invert_yaxis()
    
    plt.tight_layout()
    return fig

def analisis_matematico(puntos):
    """Segunda visualización: análisis matemático"""
    x, y = puntos[:, 0], puntos[:, 1]
    
    # Ajuste global
    coef_polinomio = np.polyfit(x, y, 3)
    polinomio = np.poly1d(coef_polinomio)
    spline = interpolate.UnivariateSpline(x, y, s=len(x))
    
    # Cálculo de longitudes
    def calcular_longitud(f, x_vals):
        y_vals = f(x_vals)
        return np.sum(np.sqrt(np.diff(x_vals)**2 + np.diff(y_vals)**2))
    
    x_vals = np.linspace(min(x), max(x), 1000)
    l_polinomio = calcular_longitud(polinomio, x_vals)
    l_spline = calcular_longitud(spline, x_vals)
    
    # Ajuste por intervalos
    n_intervalos = 5
    intervalos = np.linspace(min(x), max(x), n_intervalos+1)
    modelos = []
    
    for i in range(n_intervalos):
        mask = (x >= intervalos[i]) & (x <= intervalos[i+1])
        x_intervalo = x[mask]
        y_intervalo = y[mask]
        
        if len(x_intervalo) > 3:
            coef = np.polyfit(x_intervalo, y_intervalo, 2)
            modelo = np.poly1d(coef)
            l_intervalo = calcular_longitud(modelo, np.linspace(intervalos[i], intervalos[i+1], 100))
            
            modelos.append({
                'Intervalo': i+1,
                'x_min': f"{intervalos[i]:.1f}",
                'x_max': f"{intervalos[i+1]:.1f}",
                'Modelo': obtener_formula_polinomio(coef),
                'Longitud': f"{l_intervalo:.2f} px"
            })
    
    df_modelos = pd.DataFrame(modelos)
    
    # Crear figura
    fig = plt.figure(figsize=(14, 10))
    
    # Gráfico de ajustes
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    x_plot = np.linspace(min(x), max(x), 500)
    
    ax1.scatter(x, y, s=5, alpha=0.5, label='Puntos')
    ax1.plot(x_plot, polinomio(x_plot), 'r-', label=f'Polinomio (L={l_polinomio:.2f} px)')
    ax1.plot(x_plot, spline(x_plot), 'g-', label=f'Spline (L={l_spline:.2f} px)')
    
    for i in range(1, n_intervalos):
        ax1.axvline(x=intervalos[i], linestyle='--', alpha=0.3)
    
    ax1.set_title('Ajuste de Curva')
    ax1.legend()
    ax1.invert_yaxis()
    
    # Información de modelos
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    ax2.axis('off')
    
    texto = f"""Modelo Global (Polinomio):
{obtener_formula_polinomio(coef_polinomio)}

Longitudes:
- Polinomio: {l_polinomio:.2f} px
- Spline: {l_spline:.2f} px"""
    
    ax2.text(0.1, 0.5, texto, ha='left', va='center', fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Tabla de intervalos
    ax3 = plt.subplot2grid((2, 2), (1, 1))
    ax3.axis('off')
    
    tabla = ax3.table(cellText=df_modelos.values,
                     colLabels=df_modelos.columns,
                     loc='center',
                     cellLoc='center')
    
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.scale(1, 1.5)
    ax3.set_title('Modelos por Intervalos', pad=20)
    
    plt.tight_layout()
    return fig, df_modelos

def main():
    ruta_imagen = 'C:/Users/USUARIO/Desktop/curva_longitud/proyecto_calculo_curvas/data/resultado/ejemplo_curva.png'
    
    try:
        # Procesamiento inicial
        imagen, bordes, puntos = procesar_imagen(ruta_imagen)
        
        # Primera visualización (detección)
        fig1 = visualizacion_deteccion(imagen, bordes, puntos)
        fig1.savefig('detection_result.png', dpi=300)
        
        # Segunda visualización (análisis)
        fig2, df_modelos = analisis_matematico(puntos)
        fig2.savefig('analysis_result.png', dpi=300)
        
        # Mostrar resultados
        print("\n=== RESULTADOS ===")
        print("1. Visualización de detección guardada en: detection_result.png")
        print("2. Análisis matemático guardado en: analysis_result.png")
        print("\nModelos por intervalos:")
        print(df_modelos.to_string(index=False))
        
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()