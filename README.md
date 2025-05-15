# Proyecto: Cálculo de Longitud de Curvas en Imágenes

Este proyecto permite calcular la longitud de curvas (como cables, puentes colgantes o trayectorias) a partir de imágenes digitales mediante técnicas de procesamiento de imágenes y cálculo numérico.

## Descripción

El sistema permite cargar imágenes que contienen curvas, detectar automáticamente la trayectoria de la curva, ajustar funciones matemáticas a los puntos detectados y calcular la longitud real mediante integración numérica.

El programa cuenta con una interfaz gráfica que facilita su uso a personas sin conocimientos técnicos avanzados.

## Integrantes

    [Nombre Apellido 1]
    [Nombre Apellido 2]
    [Nombre Apellido 3]
    [Nombre Apellido 4]

## Funcionalidades principales

- Carga y visualización de imágenes
- Procesamiento y mejora de contraste/nitidez
- Detección de bordes y extracción de puntos de interés
- Ajuste matemático de funciones a segmentos de curva
- Cálculo de longitud mediante integración numérica
- Visualización de resultados con gráficas
- Exportación de resultados

## Requisitos

- Python 3.10 o superior
- Dependencias especificadas en el archivo `environment.yml`

## Instalación

1. Clonar el repositorio:
   ```
   git clone https://github.com/usuario/proyecto_calculo_curvas.git
   cd proyecto_calculo_curvas
   ```

2. Crear y activar el entorno conda:
   ```
   conda env create -f environment.yml
   conda activate curva_longitud
   ```

3. Verificar la instalación:
   ```
   python -c "import cv2, numpy, scipy, matplotlib; print('Instalación correcta')"
   ```

## Estructura del proyecto

```
proyecto_calculo_curvas/
│
├── data/
│   ├── imagenes/           # Imágenes de entrada
│   └── resultados/         # Resultados generados
│
├── src/
│   ├── __init__.py
│   ├── procesamiento.py    # Procesamiento de imágenes
│   ├── ajuste_curvas.py    # Ajuste de funciones
│   ├── calculo_longitud.py # Cálculo de la longitud
│   ├── utils.py            # Funciones auxiliares
│   └── gui/                # Interfaz gráfica
│       ├── __init__.py
│       ├── main_window.py  # Ventana principal
│       ├── image_panel.py  # Panel para imágenes
│       └── plot_panel.py   # Panel para gráficas
│
├── notebooks/
│   ├── exploracion.ipynb   # Exploración de datos
│   └── demo.ipynb          # Demostración del flujo
│
├── main.py                 # Script principal
│
├── README.md
│
└── environment.yml
```

## Uso

1. Ejecutar la aplicación:
   ```
   python main.py
   ```

2. Usar la interfaz para:
   - Cargar una imagen que contenga una curva
   - Ajustar parámetros de detección según sea necesario
   - Procesar la imagen y visualizar la curva detectada
   - Obtener el resultado del cálculo de longitud

## Metodología

El cálculo de la longitud se realiza mediante los siguientes pasos:

1. Preprocesamiento de la imagen
   - Conversión a escala de grises
   - Mejora de contraste
   - Reducción de ruido

2. Detección de la curva
   - Aplicación de detector de bordes
   - Umbralización
   - Extracción de puntos de interés

3. Ajuste matemático
   - División en segmentos
   - Ajuste de funciones polinómicas o splines

4. Cálculo de longitud
   - Aplicación de la fórmula de longitud de arco:
     L = ∫ᵃᵇ √(1 + [f'(x)]²) dx
   - Integración numérica

## Limitaciones actuales

- Funciona mejor con imágenes de alto contraste
- Se requiere una calibración para obtener medidas en unidades reales
- Las curvas muy complejas pueden requerir ajuste manual de parámetros

## Referencias

- Gonzalez, R.C., & Woods, R.E. (2018). Digital Image Processing (4th ed.)
- Burger, W., & Burge, M.J. (2016). Digital Image Processing: An Algorithmic Introduction Using Java
- Documentation of OpenCV and SciPy libraries