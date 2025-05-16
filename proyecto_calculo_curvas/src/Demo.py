import os
import sys
import cv2
import numpy as np  
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.procesamiento import cargar_imagen, preprocesar_imagen, detectar_bordes, extraer_puntos_curva
from src.ajuste_curva import ajuste_polinomio, ajuste_spline
from src.calculo_longitud import calcular_longitud_curva


# Construir ruta absoluta a la imagen
directorio_actual = os.path.dirname(os.path.abspath(__file__))
ruta_imagen = os.path.normpath(os.path.join(directorio_actual, '..', 'data', 'resultado', 'ejemplo_curva.png'))

print(f"Cargando imagen desde: {ruta_imagen}")

# Cargar imagen (usar la ruta correcta)
imagen = cargar_imagen(ruta_imagen)
if imagen is None: 
    raise FileNotFoundError(f"No se pudo cargar la imagen desde {ruta_imagen}")

# Preprocesar imagen
imagen_preprocesada = preprocesar_imagen(imagen)

# Detectar bordes
border = detectar_bordes(imagen_preprocesada)

# Extraer puntos de la curva
puntos = extraer_puntos_curva(border)

# Ajustar curva (polinomio y spline)
funcion_polinomio = ajuste_polinomio(puntos, grado=3)
funcion_spline = ajuste_spline(puntos)

# Límites para cálculo
x_min = min(puntos[:, 0])
x_max = max(puntos[:, 0])

# Calcular longitud de la curva
longitud_polinomio = calcular_longitud_curva(funcion_polinomio, x_min, x_max)
longitud_spline = calcular_longitud_curva(funcion_spline, x_min, x_max)

# Mostrar resultados
print(f"Longitud de la curva (polinomio): {longitud_polinomio:.2f} pixeles")
print(f"Longitud de la curva (spline): {longitud_spline:.2f} pixeles")

# Visualización
x = np.linspace(x_min, x_max, 100)
y_polinomio = [funcion_polinomio(xi) for xi in x]
y_spline = funcion_spline(x)

plt.figure(figsize=(12, 10))

# Imagen original
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
plt.title('Imagen Original')

# Bordes detectados
plt.subplot(2, 2, 2)
plt.imshow(border, cmap='gray')
plt.title('Bordes Detectados')

# Puntos y ajustes
plt.subplot(2, 2, 3)
plt.scatter(puntos[:, 0], puntos[:, 1], color='blue', label='Puntos detectados')
plt.plot(x, y_polinomio, linewidth=2, label=f'Polinomio (L={longitud_polinomio:.2f})')
plt.plot(x, y_spline, 'g-', linewidth=2, label=f'Spline (L={longitud_spline:.2f})')
plt.legend()
plt.title('Ajuste de Curva')
plt.gca().invert_yaxis()  # Invertir eje y para que coincida con la imagen

plt.tight_layout()

# Guardar resultado con ruta relativa correcta
ruta_guardado = os.path.normpath(os.path.join(directorio_actual, '..', 'data', 'resultado', 'demo_resultado.png'))
plt.savefig(ruta_guardado)

plt.show()
print(f"Resultado guardado en: {ruta_guardado}")