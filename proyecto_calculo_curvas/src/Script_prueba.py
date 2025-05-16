import os
import sys
sys.path.append(os.path.abspath('.'))

import numpy as np
import matplotlib.pyplot as plt
from calculos_numericos import derivada_numerica, simpson_compuesto, longitud_arco


# Crear carpeta para guardar imagen si no existe
os.makedirs('data/resultado', exist_ok=True)

# Probar la derivada numérica
def funcion_ejemplo(x):
    return x**2

x_valores = np.linspace(-5, 5, 100)
valores_derivada_numerica = [derivada_numerica(funcion_ejemplo, x) for x in x_valores]
derivada_analitica = [2*x for x in x_valores]

plt.figure(figsize=(10, 6))
plt.plot(x_valores, valores_derivada_numerica, 'b-', label='Derivada Numerica')
plt.plot(x_valores, derivada_analitica, 'r--', label='Derivada Analitica')
plt.legend()
plt.title('Comparacion de Derivadas')
plt.grid(True)
plt.savefig('data/resultado/test_derivada.png')

# Probar integración
def funcion_integral(x):
    return x**2

a, b = 0, 3
resultado_numerico = simpson_compuesto(funcion_integral, a, b, 100)
resultado_analitico = (b**3 - a**3) / 3

print(f"Integral de x^2 de {a} a {b}:")
print(f"Resultado numerico: {resultado_numerico}")
print(f"Resultado analitico: {resultado_analitico}")
print(f"Error: {abs(resultado_numerico - resultado_analitico)}")

# Probar longitud de arco
def funcion_curva(x):
    return x**2

a, b = 0, 1
longitud = longitud_arco(funcion_curva, a, b, n=100)
print(f"Longitud de arco de la curva entre {a} y {b}: {longitud}")
