import cv2
import numpy as np

def cargar_imagen(ruta): 
    """ carga una imagen desde la ruta especificada """
    return cv2.imread(ruta)

# procesamos la imagen 
def preprocesar_imagen(imagen): 
    """preporcesa la imagen para facilitar la deteccion de bordes"""
    # convertir a escala de grises 
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # aplicar filtro gaussiano para reducir ruido
    suvizada = cv2.GaussianBlur(gris, (5, 5), 0)
    
    # retonramos 
    return suvizada

# creamos la funcion que detecta los bordes 
def detectar_bordes(imagen): 
    """detecta los bordes de la imagen usando el algoritmo Canny"""
    # aplicamos el algoritmo Canny
    bordes = cv2.Canny(imagen, 50, 150)
    
    # retornamos la imagen con los bordes detectados
    return bordes

# extramos los puntos de la curva 
def extraer_puntos_curva(imagen_bordes): 
    """extrae los puntos que forman la curva desde una imagen de bordes"""
    # encontramos los contornos de la imagen
    contornos, _ = cv2.findContours(imagen_bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # seleccionar el contorno mas largo (asumiendo que nuestra curva)
    contorno_curva = max(contornos, key=cv2.contourArea)
    
    # extrae los puntos (x,y) del contorno 
    puntos = []
    for punto in contorno_curva:
        x, y = punto[0]
        puntos.append((x, y))
        
    # ordenamos los puntos por coordenada x 
    puntos.sort(key=lambda p: p[0])
    return np.array(puntos)