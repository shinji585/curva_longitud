import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from datetime import datetime

# Crear estructura de directorios si no existe
def crear_directorios():
    """Crea los directorios necesarios."""
    directorios = ['data/resultados']
    for directorio in directorios:
        os.makedirs(directorio, exist_ok=True)

# Procesamiento básico de imagen (simplificado)
def procesar_imagen_simple(ruta_imagen):
    """Procesamiento simplificado de imagen para extraer puntos de una curva.
    Args:
        ruta_imagen: Ruta a la imagen
    Returns:
        Array de puntos (x, y)
    """
    print(f"Cargando imagen: {ruta_imagen}")
    # Cargar imagen
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        print(f"Error: No se pudo cargar la imagen desde {ruta_imagen}")
        return None

    # Convertir a escala de grises
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    # Aplicar desenfoque para reducir ruido
    gris = cv2.GaussianBlur(gris, (5, 5), 0)
    # Detectar bordes
    bordes = cv2.Canny(gris, 50, 150)
    # Encontrar contornos
    contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        print("No se encontraron contornos en la imagen")
        return None

    # Tomar el contorno más grande
    contorno_mayor = max(contornos, key=cv2.contourArea)
    # Extraer puntos del contorno
    puntos = contorno_mayor.reshape(-1, 2)
    # Ordenar puntos por coordenada x
    puntos = puntos[np.argsort(puntos[:, 0])]
    print(f"Se extrajeron {len(puntos)} puntos de la curva")
    return puntos.astype(float)

# Ajuste de polinomio
def ajuste_polinomio(puntos, grado=3):
    """Ajusta un polinomio a los puntos dados."""
    x = puntos[:, 0]
    y = puntos[:, 1]
    # Ajustar el polinomio
    coeficientes = np.polyfit(x, y, grado)

    # Crear función que evalúe el polinomio
    def funcion_ajustada(x_val):
        return np.polyval(coeficientes, x_val)

    return funcion_ajustada, coeficientes

# Ajuste de spline simplificado
def ajuste_spline_simple(puntos):
    """Ajuste simplificado de spline usando interpolación cúbica."""
    from scipy import interpolate
    # Ordenar puntos por x
    puntos_ordenados = puntos[np.argsort(puntos[:, 0])]
    x = puntos_ordenados[:, 0]
    y = puntos_ordenados[:, 1]
    # Eliminar duplicados en x
    x_unicos, indices = np.unique(x, return_index=True)
    y_unicos = y[indices]
    if len(x_unicos) < 4:
        print("No hay suficientes puntos únicos para spline")
        return None
    # Crear spline
    spline = interpolate.interp1d(x_unicos, y_unicos, kind='cubic', bounds_error=False, fill_value="extrapolate")
    return spline

# Cálculo de longitud (aproximación lineal)
def calcular_longitud_simple(funcion, x_min, x_max, num_puntos=500):
    """Calcula la longitud de una curva por aproximación lineal."""
    # Crear puntos de evaluación
    x = np.linspace(x_min, x_max, num_puntos)
    # Evaluar la función
    try:
        y = funcion(x)
    except:
        y = np.array([funcion(xi) for xi in x])
    # Calcular diferencias
    dx = np.diff(x)
    dy = np.diff(y)
    # Calcular longitud total
    segmentos = np.sqrt(dx**2 + dy**2)
    longitud = np.sum(segmentos)
    return longitud

# Función principal
def main():
    print("=== DEMO SIMPLIFICADO DE CÁLCULO DE LONGITUD ===\n")

    # Crear directorios
    crear_directorios()

    # Ruta de la imagen
    ruta_imagen = "https://github.com/shinji585/curva_longitud/raw/master/proyecto_calculo_curvas/data/resultado/ejemplo_curva.png"

    # Descargar la imagen si no existe localmente
    import urllib.request
    nombre_local = "data/resultados/ejemplo_curva.png"
    try:
        print("Descargando imagen de ejemplo...")
        urllib.request.urlretrieve(ruta_imagen, nombre_local)
        print(f"Imagen descargada en: {nombre_local}")
        ruta_imagen = nombre_local
    except Exception as e:
        print(f"Error descargando imagen: {e}")
        return

    # 1. Procesar imagen
    puntos = procesar_imagen_simple(ruta_imagen)
    if puntos is None:
        print("No se pudieron extraer puntos de la imagen")
        return

    # Guardar puntos
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df_puntos = pd.DataFrame(puntos, columns=['x', 'y'])
    ruta_puntos = f"data/resultados/puntos_{timestamp}.csv"
    df_puntos.to_csv(ruta_puntos, index=False)
    print(f"Puntos guardados en: {ruta_puntos}")

    # 2. Ajustar diferentes funciones
    x_min = puntos[:, 0].min()
    x_max = puntos[:, 0].max()
    resultados = {}

    # Ajustar polinomios de diferentes grados
    for grado in [2, 3, 4]:
        print(f"\nAjustando polinomio de grado {grado}...")
        try:
            funcion, coef = ajuste_polinomio(puntos, grado)
            longitud = calcular_longitud_simple(funcion, x_min, x_max)
            resultados[f'Polinomio grado {grado}'] = {
                'funcion': funcion,
                'longitud': longitud,
                'tipo': 'polinomio',
                'parametro': grado
            }
            print(f"  Longitud: {longitud:.2f} píxeles")
        except Exception as e:
            print(f"  Error: {e}")

    # Ajustar spline
    print("\nAjustando spline cúbico...")
    try:
        funcion_spline = ajuste_spline_simple(puntos)
        if funcion_spline is not None:
            longitud = calcular_longitud_simple(funcion_spline, x_min, x_max)
            resultados['Spline cúbico'] = {
                'funcion': funcion_spline,
                'longitud': longitud,
                'tipo': 'spline',
                'parametro': 'cúbico'
            }
            print(f"  Longitud: {longitud:.2f} píxeles")
    except Exception as e:
        print(f"  Error: {e}")

    # 3. Crear visualización
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Gráfico 1: Puntos y ajustes
    ax1.scatter(puntos[:, 0], puntos[:, 1], s=10, alpha=0.6, label='Puntos originales')
    x_plot = np.linspace(x_min, x_max, 200)
    colores = ['red', 'green', 'blue', 'orange']
    for i, (nombre, info) in enumerate(resultados.items()):
        try:
            y_plot = info['funcion'](x_plot)
            ax1.plot(x_plot, y_plot, color=colores[i % len(colores)], linewidth=2, label=f"{nombre}")
        except:
            pass
    ax1.set_title('Ajustes de Curva')
    ax1.set_xlabel('X (píxeles)')
    ax1.set_ylabel('Y (píxeles)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()  # Invertir Y para que coincida con coordenadas de imagen

    # Gráfico 2: Comparación de longitudes
    nombres = list(resultados.keys())
    longitudes = [info['longitud'] for info in resultados.values()]
    ax2.bar(range(len(nombres)), longitudes, color=colores[:len(nombres)])
    ax2.set_title('Comparación de Longitudes')
    ax2.set_xlabel('Método de Ajuste')
    ax2.set_ylabel('Longitud (píxeles)')
    ax2.set_xticks(range(len(nombres)))
    ax2.set_xticklabels(nombres, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    # Añadir valores en las barras
    for i, v in enumerate(longitudes):
        ax2.text(i, v + max(longitudes)*0.01, f'{v:.1f}', ha='center', va='bottom')

    plt.tight_layout()

    # Guardar figura
    ruta_figura = f"data/resultados/comparacion_{timestamp}.png"
    plt.savefig(ruta_figura, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nFigura guardada en: {ruta_figura}")

    # 4. Guardar resumen de resultados
    df_resultados = pd.DataFrame({
        'Método': nombres,
        'Longitud_píxeles': longitudes,
        'Tipo': [info['tipo'] for info in resultados.values()],
        'Parámetro': [info['parametro'] for info in resultados.values()]
    })
    ruta_csv = f"data/resultados/resumen_{timestamp}.csv"
    df_resultados.to_csv(ruta_csv, index=False)
    print(f"Resumen guardado en: {ruta_csv}")

    # 5. Mostrar resumen
    print("\n=== RESUMEN DE RESULTADOS ===")
    print(df_resultados.to_string(index=False))
    print(f"\nDemostración completada exitosamente!")
    print(f"Archivos generados en: data/resultados/")

if __name__ == "__main__":
    main()
