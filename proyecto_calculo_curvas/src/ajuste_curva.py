import numpy as np 
from scipy import interpolate

# definimos la funcion de ajuste de polinomios 
def ajuste_polinomio(puntos,grado = 3):
    """
      Ajusta un polinomio a los puntos dados.
      
        Args:
            puntos: array de puntos (x,y) a ajustar
            grado: grado del polinomio a ajustar
            
        Returns:
            una funcion que evalua el polinomio ajustado
    """
    x = puntos[:,0]
    y = puntos[:,1]
    
    # ajustar el polinomio
    coeficientes = np.polyfit(x, y, grado)
    
    # creamos una funcion que evalue el polinomio 
    def funcion_ajustada(x_val):
        return np.polyval(coeficientes, x_val)
    return funcion_ajustada

   # definimos la funcionde ajuste de spline
def ajuste_spline(puntos):
    """
      Ajusta un spline a los puntos dados.
      
        Args:
            puntos: array de puntos (x,y) a ajustar
            
        Returns:
            una funcion que evalua el spline ajustado
    """
    x = puntos[:,0]
    y = puntos[:,1]
    
    # ajustar el spline
    spline = interpolate.UnivariateSpline(x, y)
    
    return spline
       
    