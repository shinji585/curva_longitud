import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import interpolate, integrate
import sympy as sp
from matplotlib.ticker import MaxNLocator

# Configuramos el estilo de seaborn
sns.set_theme(style="whitegrid")
sns.set_context("notebook", font_scale=1.2)

# ===== FUNCIONES DE PROCESAMIENTO DE IMAGEN =====

def cargar_imagen(ruta):
    """Carga una imagen desde la ruta especificada"""
    return cv2.imread(ruta)

def preprocesar_imagen(imagen):
    """Preprocesa la imagen para facilitar la detección de bordes"""
    # Convertir a escala de grises
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Aplicar filtro gaussiano para reducir ruido
    suavizada = cv2.GaussianBlur(gris, (5, 5), 0)
    
    return suavizada

def detectar_bordes(imagen):
    """Detecta los bordes de la imagen usando el algoritmo Canny"""
    # Aplicamos el algoritmo Canny
    bordes = cv2.Canny(imagen, 50, 150)
    
    return bordes

def extraer_puntos_curva(imagen_bordes):
    """Extrae los puntos que forman la curva desde una imagen de bordes"""
    # Encontramos los contornos de la imagen
    contornos, _ = cv2.findContours(imagen_bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contornos:
        # Si no hay contornos, generar puntos de ejemplo
        print("No se encontraron contornos. Generando curva de ejemplo.")
        x = np.linspace(0, 100, 50)
        y = 50 + 20 * np.sin(x / 10) + np.random.normal(0, 2, len(x))
        return np.column_stack((x, y))
    
    # Seleccionar el contorno más largo (asumiendo que es nuestra curva)
    contorno_curva = max(contornos, key=cv2.contourArea)
    
    # Extraer los puntos (x,y) del contorno
    puntos = []
    for punto in contorno_curva:
        x, y = punto[0]
        puntos.append((x, y))
    
    # Ordenar los puntos por coordenada x
    puntos.sort(key=lambda p: p[0])
    return np.array(puntos)

# ===== FUNCIONES DE AJUSTE DE CURVAS =====

def ajuste_polinomio(puntos, grado=3):
    """
    Ajusta un polinomio a los puntos dados.
    
    Args:
        puntos: array de puntos (x,y) a ajustar
        grado: grado del polinomio a ajustar
        
    Returns:
        una función que evalúa el polinomio ajustado
    """
    x = puntos[:, 0]
    y = puntos[:, 1]
    
    # Ajustar el polinomio
    coeficientes = np.polyfit(x, y, grado)
    
    # Crear una función que evalúe el polinomio
    def funcion_ajustada(x_val):
        return np.polyval(coeficientes, x_val)
    
    return funcion_ajustada

def ajuste_spline(puntos, s=0.1):
    """
    Ajusta un spline a los puntos dados.
    
    Args:
        puntos: array de puntos (x,y) a ajustar
        s: factor de suavizado (0 = interpolación exacta, >0 = aproximación)
        
    Returns:
        una función que evalúa el spline ajustado
    """
    # Ordenar los puntos por la coordenada x
    puntos_ordenados = puntos[np.argsort(puntos[:, 0])]
    
    # Extraer las coordenadas x e y
    x = puntos_ordenados[:, 0]
    y = puntos_ordenados[:, 1]
    
    # Procesar los puntos para asegurar que x sea estrictamente creciente
    x_procesado = []
    y_procesado = []
    
    # Usar un umbral para considerar puntos distintos
    epsilon = 1e-10
    ultimo_x = float('-inf')
    
    for i in range(len(x)):
        # Si el punto actual es mayor que el último añadido (estrictamente creciente)
        if x[i] > ultimo_x + epsilon:
            x_procesado.append(x[i])
            y_procesado.append(y[i])
            ultimo_x = x[i]
    
    # Convertir a numpy arrays
    x_procesado = np.array(x_procesado)
    y_procesado = np.array(y_procesado)
    
    # Verificar que tengamos suficientes puntos para ajustar un spline
    if len(x_procesado) < 4:
        print("Advertencia: No hay suficientes puntos únicos para un spline cúbico. Usando interpolación lineal.")
        return interpolate.interp1d(x_procesado, y_procesado,
                                  kind='linear', bounds_error=False,
                                  fill_value="extrapolate")
    
    try:
        # Intentar ajustar un spline con el parámetro s proporcionado
        spline = interpolate.UnivariateSpline(x_procesado, y_procesado, s=s)
        return spline
    except Exception as e:
        print(f"Error al ajustar spline: {e}")
        print("Recurriendo a interpolación cúbica.")
        # Si falla, usar interpolación cúbica
        return interpolate.interp1d(x_procesado, y_procesado,
                                  kind='cubic', bounds_error=False,
                                  fill_value="extrapolate")

# ===== FUNCIONES DE CÁLCULO DE LONGITUD =====

def calcular_longitud_curva(funcion, x_min, x_max, num_puntos=100):
    """
    Calcula la longitud de una curva definida por una función en un intervalo dado.
    
    Args:
        funcion: función que define la curva y = f(x)
        x_min: valor mínimo del intervalo
        x_max: valor máximo del intervalo
        num_puntos: número de puntos para la aproximación
        
    Returns:
        longitud aproximada de la curva
    """
    # Método: aproximación por segmentos
    # Crear un conjunto de puntos en el intervalo [x_min, x_max]
    x = np.linspace(x_min, x_max, num_puntos)
    
    # Evaluar la función en esos puntos
    try:
        # Intentar evaluar como una función de NumPy/SciPy (como splines)
        y = funcion(x)
    except TypeError:
        # Si falla, evaluar punto por punto (como nuestras funciones personalizadas)
        y = np.array([funcion(xi) for xi in x])
    
    # Calcular las diferencias entre puntos consecutivos
    dx = np.diff(x)
    dy = np.diff(y)
    
    # Calcular la longitud de los segmentos: √(dx² + dy²)
    segmentos = np.sqrt(dx**2 + dy**2)
    
    # Sumar todos los segmentos para obtener la longitud total
    longitud = np.sum(segmentos)
    
    return longitud

# ===== FUNCIÓN PRINCIPAL =====

def main(ruta_imagen=None, num_intervalos=5):
    """
    Función principal para procesar una imagen y calcular la longitud de una curva
    usando únicamente el método de ajuste polinómico y modelando por intervalos.
    
    Args:
        ruta_imagen: ruta a la imagen a procesar. Si es None, se usa una imagen de ejemplo.
        num_intervalos: número de intervalos para modelar la función.
    """
    # Si no se especifica una ruta, usar datos de ejemplo
    if ruta_imagen is None:
        print("No se especificó imagen. Generando datos de ejemplo...")
        # Generar puntos de ejemplo para una curva
        x_ejemplo = np.linspace(0, 100, 100)
        y_ejemplo = 50 + 20 * np.sin(x_ejemplo / 15) + 10 * np.cos(x_ejemplo / 25) + np.random.normal(0, 2, len(x_ejemplo))
        puntos = np.column_stack((x_ejemplo, y_ejemplo))
        imagen = np.zeros((200, 150, 3), dtype=np.uint8)  # Imagen dummy
        imagen_preprocesada = np.zeros((200, 150), dtype=np.uint8)
        bordes = np.zeros((200, 150), dtype=np.uint8)
    else:
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
        print("Error: No se detectaron suficientes puntos en la curva.")
        return
    
    # Ajustar curva con polinomio completo
    funcion_polinomio = ajuste_polinomio(puntos, grado=3)
    
    # Límites para cálculo
    x_min = min(puntos[:, 0])
    x_max = max(puntos[:, 0])
    
    # Función para modelar por intervalos
    def modelar_por_intervalos(puntos, x_min, x_max, num_intervalos):
        """
        Modela la función por intervalos utilizando diferentes polinomios para cada segmento.
        
        Args:
            puntos: array de puntos [x, y] de la curva
            x_min, x_max: límites del intervalo total
            num_intervalos: número de intervalos a crear
            
        Returns:
            Lista de tuplas (intervalo, función, coeficientes, longitud)
        """
        # Ordenar puntos por coordenada x
        puntos_ordenados = puntos[np.argsort(puntos[:, 0])]
        
        # Crear intervalos
        limites = np.linspace(x_min, x_max, num_intervalos + 1)
        modelos = []
        
        for i in range(num_intervalos):
            # Definir límites del intervalo actual
            inicio = limites[i]
            fin = limites[i+1]
            
            # Seleccionar puntos en este intervalo
            mask = (puntos_ordenados[:, 0] >= inicio) & (puntos_ordenados[:, 0] <= fin)
            puntos_intervalo = puntos_ordenados[mask]
            
            # Si hay menos de 4 puntos, ajustar con un polinomio de menor grado
            if len(puntos_intervalo) < 4:
                grado = min(len(puntos_intervalo) - 1, 2)
                if grado < 1:  # Si no hay suficientes puntos, usar puntos cercanos
                    # Buscar los 4 puntos más cercanos al intervalo
                    distancias = np.minimum(
                        np.abs(puntos_ordenados[:, 0] - inicio),
                        np.abs(puntos_ordenados[:, 0] - fin)
                    )
                    indices = np.argsort(distancias)[:4]
                    puntos_intervalo = puntos_ordenados[indices]
                    grado = min(len(puntos_intervalo) - 1, 3)
            else:
                grado = 3  # Usar grado 3 si hay suficientes puntos
            
            # Ajustar polinomio para este intervalo
            if len(puntos_intervalo) > 1:
                # Usar np.polyfit directamente para tener acceso a los coeficientes
                coefs = np.polyfit(puntos_intervalo[:, 0], puntos_intervalo[:, 1], grado)
                
                # Crear función lambda para este polinomio
                def funcion_intervalo(x, coeficientes=coefs):
                    return np.polyval(coeficientes, x)
                
                # Calcular longitud de este segmento
                try:
                    longitud = calcular_longitud_curva(funcion_intervalo, inicio, fin, num_puntos=50)
                except Exception as e:
                    print(f"Error calculando longitud en intervalo [{inicio:.2f}, {fin:.2f}]: {e}")
                    longitud = 0
                
                # Guardar intervalo, función y coeficientes
                modelos.append((
                    (inicio, fin), 
                    funcion_intervalo, 
                    coefs, 
                    longitud
                ))
        
        return modelos
    
    # Modelar por intervalos
    modelos_intervalos = modelar_por_intervalos(puntos, x_min, x_max, num_intervalos)
    
    # Calcular longitud total con el polinomio general
    try:
        longitud_polinomio = calcular_longitud_curva(funcion_polinomio, x_min, x_max)
        print(f"Longitud de la curva (polinomio completo): {longitud_polinomio:.2f} píxeles")
    except Exception as e:
        print(f"Error al calcular longitud del polinomio: {e}")
        longitud_polinomio = 0
    
    # Calcular longitud total por intervalos
    longitud_intervalos = sum(modelo[3] for modelo in modelos_intervalos)
    print(f"Longitud de la curva (suma de intervalos): {longitud_intervalos:.2f} píxeles")
    
    # Visualización con seaborn
    x_completo = np.linspace(x_min, x_max, 200)
    y_polinomio = [funcion_polinomio(xi) for xi in x_completo]
    
    # Crear una figura con el estilo de seaborn
    plt.figure(figsize=(15, 12))
    
    # Imagen original
    plt.subplot(2, 2, 1)
    if ruta_imagen:
        plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    else:
        plt.plot(puntos[:, 0], puntos[:, 1], 'o-', alpha=0.5)
        plt.gca().invert_yaxis()
    plt.title('Datos Originales')
    if not ruta_imagen:
        plt.axis('on')
    else:
        plt.axis('off')
    
    # Bordes detectados
    plt.subplot(2, 2, 2)
    if ruta_imagen:
        plt.imshow(bordes, cmap='gray')
        plt.axis('off')
    else:
        plt.scatter(puntos[:, 0], puntos[:, 1], alpha=0.7)
        plt.gca().invert_yaxis()
    plt.title('Puntos Detectados')
    
    # Puntos y ajuste polinómico
    plt.subplot(2, 2, 3)
    colores = sns.color_palette("deep", 2)
    plt.scatter(puntos[:, 0], puntos[:, 1], color=colores[0], s=30, alpha=0.7, label='Puntos detectados')
    plt.plot(x_completo, y_polinomio, color=colores[1], linewidth=2.5, 
             label=f'Polinomio completo (L={longitud_polinomio:.2f} px)')
    plt.legend(fontsize=10)
    plt.title('Ajuste de Curva (Polinomio Completo)')
    plt.gca().invert_yaxis()  # Invertir eje y para que coincida con la imagen
    
    # Modelado por intervalos
    ax = plt.subplot(2, 2, 4)
    
    # Dibujar cada intervalo con diferente color
    colores_intervalos = sns.color_palette("husl", len(modelos_intervalos))
    
    for i, (intervalo, func, coefs, longitud) in enumerate(modelos_intervalos):
        inicio, fin = intervalo
        x_intervalo = np.linspace(inicio, fin, 50)
        y_intervalo = [func(xi) for xi in x_intervalo]
        
        plt.plot(x_intervalo, y_intervalo, color=colores_intervalos[i], linewidth=2.5,
                label=f'Int {i+1} [{inicio:.0f}-{fin:.0f}] (L={longitud:.2f} px)')
    
    plt.scatter(puntos[:, 0], puntos[:, 1], color='black', s=15, alpha=0.4)
    plt.title(f'Función Modelada por {len(modelos_intervalos)} Intervalos (L total={longitud_intervalos:.2f} px)')
    plt.gca().invert_yaxis()
    plt.legend(fontsize=8, loc='upper right')
    
    # Limitar el número de ticks en los ejes para evitar sobrecargar
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    
    plt.tight_layout()
    
    # Guardar resultado de la gráfica
    try:
        directorio_actual = os.path.dirname(os.path.abspath(__file__))
        ruta_guardado = os.path.join(directorio_actual, 'demo2_intervalos.png')
        plt.savefig(ruta_guardado, dpi=300)
        print(f"Gráfico guardado en: {ruta_guardado}")
    except:
        print("No se pudo guardar el gráfico")
    
    plt.show()
    
    # Crear una tabla elegante con pandas para mostrar las funciones por intervalos
    data = []
    
    # Función para convertir coeficientes en una ecuación legible
    def coef_to_equation(coefs):
        # Crear variable simbólica
        x = sp.Symbol('x')
        # Crear polinomio
        expr = 0
        for i, c in enumerate(coefs):
            expr += c * x**(len(coefs)-i-1)
        # Convertir a latex para mejor visualización
        return sp.latex(expr)
    
    # Preparar datos para la tabla
    for i, (intervalo, _, coefs, longitud) in enumerate(modelos_intervalos):
        inicio, fin = intervalo
        ecuacion = coef_to_equation(coefs)
        
        # Determinar el grado del polinomio
        grado = len(coefs) - 1
        tipo = f"Polinomio grado {grado}"
        
        data.append({
            'Intervalo': f"[{inicio:.1f}, {fin:.1f}]",
            'Tipo': tipo,
            'Ecuación': f"$y = {ecuacion}$",
            'Longitud (px)': f"{longitud:.2f}"
        })
    
    # Crear DataFrame
    df = pd.DataFrame(data)
    
    # Mostrar información resumida en consola
    print("\nResumen de modelos por intervalos:")
    print(df.to_string(index=False))
    
    # Retornar los resultados por si se necesitan
    return {
        'longitud_polinomio': longitud_polinomio,
        'longitud_intervalos': longitud_intervalos,
        'puntos': puntos,
        'modelos': modelos_intervalos,
        'tabla': df
    }

if __name__ == "__main__":
    # Si se pasa una ruta como argumento, usarla
    if len(sys.argv) > 1:
        ruta_imagen_arg = sys.argv[1]
        num_intervalos = 5  # Valor predeterminado
        
        # Si hay un segundo argumento, usarlo como número de intervalos
        if len(sys.argv) > 2:
            try:
                num_intervalos = int(sys.argv[2])
            except ValueError:
                print(f"Advertencia: El segundo argumento '{sys.argv[2]}' no es un número válido. Usando 5 intervalos.")
        
        main(ruta_imagen_arg, num_intervalos)
    else:
        main(num_intervalos=5)  # Usar datos de ejemplo