import numpy as np
from scipy import integrate
import sys
import os

# Intentar importar los cálculos numéricos personalizados
try:
    sys.path.append(os.path.abspath('.'))
    from src.calculos_numericos import longitud_arco
    USAR_CALCULOS_NUMERICOS = True
except ImportError:
    USAR_CALCULOS_NUMERICOS = False
    print("Nota: No se encontró el módulo de cálculos numéricos. Usando métodos alternativos.")

def calcular_longitud_curva(funcion, x_min, x_max, num_puntos=100):
    """
    Calcula la longitud de una curva definida por una función en un intervalo dado.
    
    Si está disponible, utiliza el módulo calculos_numericos.py con Simpson.
    Si no, utiliza un método de aproximación por segmentos.
    
    Args:
        funcion: función que define la curva y = f(x)
        x_min: valor mínimo del intervalo
        x_max: valor máximo del intervalo
        num_puntos: número de puntos para la aproximación
        
    Returns:
        longitud aproximada de la curva
    """
    if USAR_CALCULOS_NUMERICOS:
        # Usar la implementación de Simpson si está disponible
        try:
            return longitud_arco(funcion, x_min, x_max, n=num_puntos)
        except Exception as e:
            print(f"Error al usar cálculos numéricos: {e}")
            print("Recurriendo a método alternativo...")
    
    # Método alternativo: aproximación por segmentos
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

# Método alternativo para calibrar la longitud
def calcular_longitud_con_calibracion(longitud_pixeles, factor_escala):
    """
    Convierte una longitud en pixeles a unidades reales usando un factor de escala.
    
    Args:
        longitud_pixeles: longitud en pixeles
        factor_escala: factor de conversión (unidades reales / pixel)
       
    Returns:
        longitud en unidades reales
    """
    return longitud_pixeles * factor_escala