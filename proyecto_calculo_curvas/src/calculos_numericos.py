# derivada de una funcion

def derivada_numerica(f,x,h=0.0001):
    """
    Calcula la derivada numerica de una funcion f en el punto x
    usando el metodo de diferencias finitas.
    
    Args:
        f: funcion a derivar
        x: punto en el que se evalua la derivada
        h: paso para la aproximacion (default=0.0001)
        
    Returns:
        Derivada de f en x
    """
    return (f(x + h) - f(x - h)) / (2 * h)

# aplicamos el metodo de simpson 1/3 compuesto 
def simpson_compuesto(f, a, b, n):
    """
    Calcula la integral definida de una funcion f en el intervalo [a, b]
    usando el metodo de Simpson 1/3 compuesto.
    
    Args:
        f: funcion a integrar
        a: limite inferior de la integral
        b: limite superior de la integral
        n: numero de subintervalos (debe ser par)
        
    Returns:
        Aproximacion de la integral definida de f en [a, b]
    """
    if n % 2 != 0:
        n += 1 
        
    # tamaño de cada subintervalo 
    h = (b - a) / n 
    
    # suma de los terminos 
    suma = f(a) + f(b)
    
    # suma de terminos con coeficiente 4 (indices impares)
    for i in range(1,n,2):
        x_i = a+ i * h
        suma += 4 * f(x_i)
        
    # suma de terminos con coeficiente 2 (indice pares)
    for i in range(2,n,2):
        x_i = a + i * h
        suma += 2 * f(x_i)
        
        
    # resultado final
    resultado = (h / 3) * suma
    
    # retornamos el resultado 
    return resultado

# funcion para calcular la integral de una funcion (longitud de arco)
def longitud_arco(f, a, b, n=100, h=0.0001):
    """
    Calcula la longitud de arco de una funcion f en el intervalo [a, b]
    usando el metodo de Simpson 1/3 compuesto.
    
    Args:
        f: funcion a integrar
        a: limite inferior de la integral
        b: limite superior de la integral
        n: numero de subintervalos (debe ser par)
        
    Returns:
        Aproximacion de la longitud de arco de f en [a, b]
    """
    # definir la funcion integrando (g(x)) = sqrt(1 + (f'(x))^2)
    def integrado(x): 
        derivada = derivada_numerica(f, x, h)
        return (1 + derivada**2)**0.5
    
    # calcular la integral usando simson 1/3 compuesto 
    longitud_arco = simpson_compuesto(integrado, a, b, n)
    
    return longitud_arco
        