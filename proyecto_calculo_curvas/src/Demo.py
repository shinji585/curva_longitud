import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Añadimos seaborn para mejorar los gráficos

# Configuramos el estilo de seaborn
sns.set_theme(style="whitegrid")  # Estilo con cuadrícula para mejor visualización
sns.set_context("notebook", font_scale=1.2)  # Tamaño de fuente para mejor visibilidad

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.procesamiento import cargar_imagen, preprocesar_imagen, detectar_bordes, extraer_puntos_curva
from src.ajuste_curva import ajuste_polinomio, ajuste_spline
from src.calculo_longitud import calcular_longitud_curva

def main(ruta_imagen=None):
    """
    Función principal para procesar una imagen y calcular la longitud de una curva.
    
    Args:
        ruta_imagen: ruta a la imagen a procesar. Si es None, se usa una imagen de ejemplo.
    """
    # Si no se especifica una ruta, usamos la ruta predeterminada
    if ruta_imagen is None:
        directorio_actual = os.path.dirname(os.path.abspath(__file__))
        ruta_imagen = os.path.normpath(os.path.join(directorio_actual, '..', 'data', 'resultado', 'ejemplo_curva.png'))
    
    print(f"Cargando imagen desde: {ruta_imagen}")
    
    # Cargar imagen
    imagen = cargar_imagen(ruta_imagen)
    if imagen is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen desde {ruta_imagen}")
    
    # Preprocesar imagen
    imagen_preprocesada = preprocesar_imagen(imagen)
    
    # Detectar bordes
    bordes = detectar_bordes(imagen_preprocesada)
    
    # Extraer puntos de la curva
    puntos = extraer_puntos_curva(bordes)
    
    # Verificar que tenemos suficientes puntos
    if len(puntos) < 4:
        print("Advertencia: No se detectaron suficientes puntos en la curva.")
        return
    
    # Ajustar curva (polinomio y spline)
    funcion_polinomio = ajuste_polinomio(puntos, grado=3)
    
    # El parámetro s=0.1 ayuda a evitar el error "x must be strictly increasing if s=0"
    funcion_spline = ajuste_spline(puntos, s=0.1)
    
    # Límites para cálculo
    x_min = min(puntos[:, 0])
    x_max = max(puntos[:, 0])
    
    # Calcular longitud de la curva
    try:
        longitud_polinomio = calcular_longitud_curva(funcion_polinomio, x_min, x_max)
        print(f"Longitud de la curva (polinomio): {longitud_polinomio:.2f} píxeles")
    except Exception as e:
        print(f"Error al calcular longitud del polinomio: {e}")
        longitud_polinomio = 0
    
    try:
        longitud_spline = calcular_longitud_curva(funcion_spline, x_min, x_max)
        print(f"Longitud de la curva (spline): {longitud_spline:.2f} píxeles")
    except Exception as e:
        print(f"Error al calcular longitud del spline: {e}")
        longitud_spline = 0
    
    # Visualización con seaborn
    x = np.linspace(x_min, x_max, 100)
    y_polinomio = [funcion_polinomio(xi) for xi in x]
    y_spline = funcion_spline(x)
    
    # Crear una figura con el estilo de seaborn
    plt.figure(figsize=(14, 12))
    
    # Imagen original
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    plt.title('Imagen Original')
    plt.axis('off')  # Quitar ejes para mejor visualización
    
    # Bordes detectados
    plt.subplot(2, 2, 2)
    plt.imshow(bordes, cmap='gray')
    plt.title('Bordes Detectados')
    plt.axis('off')
    
    # Puntos y ajustes
    plt.subplot(2, 2, 3)
    # Usamos paleta de colores de seaborn
    colores = sns.color_palette("deep", 3)
    plt.scatter(puntos[:, 0], puntos[:, 1], color=colores[0], s=30, alpha=0.7, label='Puntos detectados')
    plt.plot(x, y_polinomio, color=colores[1], linewidth=2.5, label=f'Polinomio (L={longitud_polinomio:.2f} px)')
    plt.plot(x, y_spline, color=colores[2], linewidth=2.5, label=f'Spline (L={longitud_spline:.2f} px)')
    plt.legend(fontsize=10)
    plt.title('Ajuste de Curva')
    plt.gca().invert_yaxis()  # Invertir eje y para que coincida con la imagen
    
    # Gráfico adicional con seaborn
    plt.subplot(2, 2, 4)
    # Crear un DataFrame para seaborn
    df_puntos = np.column_stack([x, y_polinomio, y_spline])
    df_labels = ['x', 'Polinomio', 'Spline']
    
    # Comparación de métodos con líneas más gruesas y colores de seaborn
    plt.plot(x, y_polinomio, color=colores[1], linewidth=2.5, label='Polinomio')
    plt.plot(x, y_spline, color=colores[2], linewidth=2.5, label='Spline')
    plt.scatter(puntos[:, 0], puntos[:, 1], color=colores[0], s=20, alpha=0.5, label='Puntos')
    plt.legend(fontsize=10)
    plt.title('Comparación de Métodos')
    plt.gca().invert_yaxis()
    
    # Añadir texto informativo
    info_text = f"Longitud polinomio: {longitud_polinomio:.2f} px\nLongitud spline: {longitud_spline:.2f} px"
    plt.annotate(info_text, xy=(0.02, 0.95), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                 fontsize=10)
    
    plt.tight_layout()
    
    # Guardar resultado
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_guardado = os.path.normpath(os.path.join(directorio_actual, '..', 'data', 'resultado', 'demo_resultado.png'))
    plt.savefig(ruta_guardado, dpi=300)  # Mayor DPI para mejor calidad
    plt.show()
    
    print(f"Resultado guardado en: {ruta_guardado}")
    
    # Retornar los resultados por si se necesitan
    return {
        'longitud_polinomio': longitud_polinomio,
        'longitud_spline': longitud_spline,
        'puntos': puntos
    }

if __name__ == "__main__":
    # Si se pasa una ruta como argumento, usarla
    if len(sys.argv) > 1:
        ruta_imagen_arg = sys.argv[1]
        main(ruta_imagen_arg)
    else:
        main()  # Usar la ruta predeterminada