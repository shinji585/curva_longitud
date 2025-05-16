import os 
import sys
sys.path.append(os.path.abspath('.'))

from src.calculos_numericos import longitud_arco


# definimos la funcion que calcula la longitud de curva 
def calcular_longitud_curva(funcion_curva,limite_inferior,limite_superior,n=100):
    """
    Calcula la longitud de una curva definida por la funcion 'funcion_curva'
    en el intervalo [limite_inferior, limite_superior] usando el metodo de Simpson 1/3 compuesto.
    
    Args:
        funcion_curva: funcion que define la curva
        limite_inferior: limite inferior del intervalo
        limite_superior: limite superior del intervalo
        n: numero de subintervalos (debe ser par)
        
    Returns:
        Longitud de la curva en el intervalo [limite_inferior, limite_superior]
    """
    return longitud_arco(funcion_curva, limite_inferior, limite_superior, n)

# calcular la longitud de la curva con calibracion 
def calcular_longitud_con_calibracion(longitud_piexels,factori_escala): 
    """
        convierte una longitud en pixeles a unidades reales usando un factor de escala.
    Args:
        longitud_piexels : longitud en pixeles
        factori_escala : factor de conversion (unidades reales / pixel)
        
    Returns:
        longitud en unidades reales
    """
    return longitud_piexels * factori_escala


    