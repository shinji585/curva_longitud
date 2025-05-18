import numpy as np
from scipy import interpolate

# definimos la funcion de ajuste de polinomios
def ajuste_polinomio(puntos, grado=3):
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

# definimos la función de ajuste de spline usando interpolación
def ajuste_spline(puntos, s=0.1):
    """
      Ajusta un spline a los puntos dados.
     
        Args:
            puntos: array de puntos (x,y) a ajustar
            s: factor de suavizado (0 = interpolación exacta, >0 = aproximación)
           
        Returns:
            una funcion que evalua el spline ajustado
    """
    # ordenamos los puntos por la coordenada x
    puntos_ordenados = puntos[np.argsort(puntos[:,0])]
    
    # extraemos las coordenadas x e y
    x = puntos_ordenados[:,0]  
    y = puntos_ordenados[:,1]
    
    # Procesamos los puntos para asegurar que x sea estrictamente creciente
    x_procesado = []
    y_procesado = []
    
    # Usamos un umbral para considerar puntos distintos
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