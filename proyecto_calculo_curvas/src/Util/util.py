"""
Módulo de utilidades para el proyecto de cálculo de longitud de curvas.
Contiene funciones auxiliares para guardar/cargar datos, gestionar resultados
y facilitar la interoperabilidad entre los diferentes módulos del proyecto.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from datetime import datetime

# Importaciones locales
import sys
sys.path.append(os.path.abspath('.'))
try:
    from src.ajuste_curva import ajuste_polinomio, ajuste_spline
    from src.calculos_numericos import longitud_arco, derivada_numerica
except ImportError:
    print("Advertencia: No se pudieron importar algunos módulos locales.")

# ----- Funciones para gestión de datos -----

def guardar_puntos_curva(puntos, nombre_archivo, directorio='../data/resultados'):
    """
    Guarda los puntos de una curva en un archivo CSV.
    
    Args:
        puntos: Array NumPy con los puntos (x,y) de la curva
        nombre_archivo: Nombre del archivo donde se guardarán los puntos
        directorio: Directorio donde se guardará el archivo
    
    Returns:
        Ruta completa del archivo guardado
    """
    # Asegurar que el directorio existe
    os.makedirs(directorio, exist_ok=True)
    
    # Crear DataFrame con los puntos
    df = pd.DataFrame(puntos, columns=['x', 'y'])
    
    # Definir ruta completa
    ruta_completa = os.path.join(directorio, f"{nombre_archivo}.csv")
    
    # Guardar como CSV
    df.to_csv(ruta_completa, index=False)
    
    print(f"Puntos guardados en: {ruta_completa}")
    return ruta_completa

def cargar_puntos_curva(ruta_archivo):
    """
    Carga puntos de una curva desde un archivo CSV.
    
    Args:
        ruta_archivo: Ruta al archivo CSV con los puntos
    
    Returns:
        Array NumPy con los puntos (x,y)
    """
    df = pd.read_csv(ruta_archivo)
    return df[['x', 'y']].values

# ----- Funciones para gestión de modelos (funciones ajustadas) -----

def guardar_modelo_funcion(funcion, tipo_funcion, params, nombre_archivo, directorio='../data/resultados'):
    """
    Guarda información sobre una función ajustada.
    
    Args:
        funcion: Función Python que representa el modelo
        tipo_funcion: String que indica el tipo ('polinomio', 'spline', etc.)
        params: Diccionario con los parámetros del modelo (grado, coeficientes, etc.)
        nombre_archivo: Nombre base para el archivo
        directorio: Directorio donde se guardará el archivo
    
    Returns:
        Ruta al archivo guardado
    """
    # Asegurar que el directorio existe
    os.makedirs(directorio, exist_ok=True)
    
    # Crear diccionario con toda la información
    info_modelo = {
        'tipo_funcion': tipo_funcion,
        'parametros': params,
        'fecha_creacion': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Para polinomios, podemos guardar directamente los coeficientes
    if tipo_funcion == 'polinomio':
        # Extraer coeficientes si están en los parámetros
        if 'coeficientes' in params:
            info_modelo['coeficientes'] = params['coeficientes']
    
    # Guardar información en formato pickle
    ruta_pickle = os.path.join(directorio, f"{nombre_archivo}_modelo.pkl")
    with open(ruta_pickle, 'wb') as f:
        pickle.dump(info_modelo, f)
    
    # También guardar un resumen en formato CSV para mejor interoperabilidad
    df_resumen = pd.DataFrame({
        'tipo_funcion': [tipo_funcion],
        'fecha_creacion': [info_modelo['fecha_creacion']]
    })
    
    # Añadir los parámetros como columnas
    for key, value in params.items():
        if isinstance(value, (int, float, str)):
            df_resumen[key] = [value]
    
    ruta_csv = os.path.join(directorio, f"{nombre_archivo}_modelo_info.csv")
    df_resumen.to_csv(ruta_csv, index=False)
    
    print(f"Modelo guardado en: {ruta_pickle}")
    print(f"Información del modelo guardada en: {ruta_csv}")
    
    return ruta_pickle

def cargar_modelo_funcion(ruta_archivo):
    """
    Carga información sobre un modelo guardado.
    
    Args:
        ruta_archivo: Ruta al archivo pickle con la información del modelo
    
    Returns:
        Diccionario con la información del modelo
    """
    with open(ruta_archivo, 'rb') as f:
        info_modelo = pickle.load(f)
    
    return info_modelo

def reconstruir_funcion(info_modelo):
    """
    Reconstruye una función a partir de la información guardada.
    
    Args:
        info_modelo: Diccionario con la información del modelo
    
    Returns:
        Función reconstruida que puede ser evaluada
    """
    tipo = info_modelo['tipo_funcion']
    params = info_modelo['parametros']
    
    if tipo == 'polinomio':
        # Si tenemos los coeficientes, podemos reconstruir la función
        if 'coeficientes' in info_modelo:
            coefs = info_modelo['coeficientes']
        elif 'coeficientes' in params:
            coefs = params['coeficientes']
        else:
            raise ValueError("No se encontraron coeficientes para reconstruir el polinomio")
        
        def funcion_reconstruida(x):
            return np.polyval(coefs, x)
        
        return funcion_reconstruida
    
    elif tipo == 'spline':
        # Para splines necesitamos los puntos originales y parámetros
        # Si no los tenemos, debemos indicar que hay que reajustar
        print("Los splines no pueden ser reconstruidos directamente desde los parámetros.")
        print("Se recomienda reajustar con los puntos originales.")
        return None
    
    else:
        print(f"Tipo de función no reconocido: {tipo}")
        return None

# ----- Funciones para muestreo y evaluación de funciones -----

def muestrear_funcion(funcion, x_min, x_max, num_puntos=100):
    """
    Muestrea una función en un intervalo dado.
    
    Args:
        funcion: Función a muestrear
        x_min: Límite inferior del intervalo
        x_max: Límite superior del intervalo
        num_puntos: Número de puntos de muestreo
    
    Returns:
        Tupla (x, y) con los valores muestreados
    """
    x = np.linspace(x_min, x_max, num_puntos)
    
    try:
        # Intentar evaluar como función NumPy/SciPy
        y = funcion(x)
    except TypeError:
        # Si falla, evaluar punto por punto
        y = np.array([funcion(xi) for xi in x])
    
    return x, y

def guardar_muestreo_funcion(funcion, x_min, x_max, nombre_archivo, num_puntos=100, directorio='../data/resultados'):
    """
    Muestrea una función y guarda los resultados en un archivo CSV.
    
    Args:
        funcion: Función a muestrear
        x_min: Límite inferior del intervalo
        x_max: Límite superior del intervalo
        nombre_archivo: Nombre base para el archivo
        num_puntos: Número de puntos de muestreo
        directorio: Directorio donde se guardará el archivo
    
    Returns:
        Ruta al archivo guardado
    """
    # Asegurar que el directorio existe
    os.makedirs(directorio, exist_ok=True)
    
    # Muestrear la función
    x, y = muestrear_funcion(funcion, x_min, x_max, num_puntos)
    
    # Calcular la derivada numérica en cada punto
    try:
        derivadas = np.array([derivada_numerica(funcion, xi) for xi in x])
    except:
        print("No se pudo calcular la derivada numérica")
        derivadas = np.zeros_like(x)
    
    # Crear DataFrame
    df = pd.DataFrame({
        'x': x,
        'y': y,
        'derivada': derivadas
    })
    
    # Guardar como CSV
    ruta_completa = os.path.join(directorio, f"{nombre_archivo}_muestreo.csv")
    df.to_csv(ruta_completa, index=False)
    
    print(f"Muestreo guardado en: {ruta_completa}")
    return ruta_completa

# ----- Funciones para cálculo y gestión de resultados -----

def calcular_longitud_por_tramos(funcion, x_min, x_max, num_tramos=10):
    """
    Calcula la longitud de una curva por tramos y guarda los resultados.
    
    Args:
        funcion: Función que define la curva
        x_min: Límite inferior del intervalo
        x_max: Límite superior del intervalo
        num_tramos: Número de tramos en los que dividir el intervalo
    
    Returns:
        DataFrame con las longitudes por tramo y la longitud total
    """
    # Dividir el intervalo en tramos
    puntos_tramos = np.linspace(x_min, x_max, num_tramos + 1)
    
    # Calcular la longitud en cada tramo
    longitudes = []
    for i in range(num_tramos):
        a = puntos_tramos[i]
        b = puntos_tramos[i+1]
        
        try:
            longitud_tramo = longitud_arco(funcion, a, b)
        except Exception as e:
            print(f"Error al calcular longitud en tramo [{a}, {b}]: {e}")
            longitud_tramo = 0
        
        longitudes.append({
            'tramo': i+1,
            'x_min': a,
            'x_max': b,
            'longitud': longitud_tramo
        })
    
    # Crear DataFrame con los resultados
    df_longitudes = pd.DataFrame(longitudes)
    
    # Añadir la longitud total
    longitud_total = df_longitudes['longitud'].sum()
    
    print(f"Longitud total: {longitud_total:.2f} píxeles")
    
    return df_longitudes, longitud_total

def guardar_resultados_longitud(df_longitudes, longitud_total, nombre_archivo, directorio='../data/resultados'):
    """
    Guarda los resultados del cálculo de longitud en un archivo CSV.
    
    Args:
        df_longitudes: DataFrame con las longitudes por tramo
        longitud_total: Longitud total de la curva
        nombre_archivo: Nombre base para el archivo
        directorio: Directorio donde se guardará el archivo
    
    Returns:
        Ruta al archivo guardado
    """
    # Asegurar que el directorio existe
    os.makedirs(directorio, exist_ok=True)
    
    # Guardar el DataFrame de longitudes
    ruta_tramos = os.path.join(directorio, f"{nombre_archivo}_longitudes_tramos.csv")
    df_longitudes.to_csv(ruta_tramos, index=False)
    
    # Crear y guardar un resumen
    df_resumen = pd.DataFrame({
        'fecha_calculo': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'num_tramos': [len(df_longitudes)],
        'x_min_total': [df_longitudes['x_min'].min()],
        'x_max_total': [df_longitudes['x_max'].max()],
        'longitud_total': [longitud_total]
    })
    
    ruta_resumen = os.path.join(directorio, f"{nombre_archivo}_resumen_longitud.csv")
    df_resumen.to_csv(ruta_resumen, index=False)
    
    print(f"Resultados detallados guardados en: {ruta_tramos}")
    print(f"Resumen guardado en: {ruta_resumen}")
    
    return ruta_resumen

# ----- Funciones para visualización -----

def visualizar_resultados(puntos, funcion, x_min, x_max, longitud_total, 
                         df_longitudes=None, nombre_archivo=None, directorio='../data/resultados'):
    """
    Visualiza los resultados del ajuste y cálculo de longitud.
    
    Args:
        puntos: Array NumPy con los puntos (x,y) originales
        funcion: Función ajustada
        x_min: Límite inferior del intervalo
        x_max: Límite superior del intervalo
        longitud_total: Longitud total calculada
        df_longitudes: DataFrame con las longitudes por tramo
        nombre_archivo: Nombre base para guardar la figura
        directorio: Directorio donde se guardará la figura
    
    Returns:
        Ruta a la figura guardada o None si no se guarda
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Configurar estilo
    sns.set_theme(style="whitegrid")
    sns.set_context("notebook", font_scale=1.2)
    
    # Crear figura
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Muestrear la función
    x_muestra, y_muestra = muestrear_funcion(funcion, x_min, x_max, num_puntos=200)
    
    # Graficar puntos y curva ajustada en el primer subplot
    ax1.scatter(puntos[:, 0], puntos[:, 1], s=30, alpha=0.7, label='Puntos originales')
    ax1.plot(x_muestra, y_muestra, 'r-', linewidth=2, label='Función ajustada')
    ax1.set_title(f'Ajuste de Curva (Longitud Total: {longitud_total:.2f} píxeles)')
    ax1.legend()
    ax1.grid(True)
    
    # Si tenemos información de tramos, mostrarla en el segundo subplot
    if df_longitudes is not None:
        # Crear un array x para mostrar las longitudes por tramo
        tramos = df_longitudes['tramo'].values
        longitudes = df_longitudes['longitud'].values
        
        # Graficar las longitudes por tramo como barras
        ax2.bar(tramos, longitudes, alpha=0.7, color='skyblue')
        ax2.plot(tramos, longitudes, 'ro-', alpha=0.7)
        ax2.set_xlabel('Número de Tramo')
        ax2.set_ylabel('Longitud (píxeles)')
        ax2.set_title('Longitud por Tramos')
        ax2.grid(True)
        
        # Añadir etiquetas con los valores
        for i, v in enumerate(longitudes):
            ax2.text(i+1, v+0.1, f'{v:.2f}', ha='center')
    
    else:
        # Si no hay información de tramos, mostrar la derivada
        try:
            derivadas = np.array([derivada_numerica(funcion, xi) for xi in x_muestra])
            ax2.plot(x_muestra, derivadas, 'g-', linewidth=2, label='Derivada')
            ax2.set_title('Derivada de la Función Ajustada')
            ax2.legend()
            ax2.grid(True)
        except Exception as e:
            print(f"No se pudo calcular la derivada: {e}")
    
    plt.tight_layout()
    
    # Guardar figura si se especifica un nombre
    if nombre_archivo:
        os.makedirs(directorio, exist_ok=True)
        ruta_figura = os.path.join(directorio, f"{nombre_archivo}_visualizacion.png")
        plt.savefig(ruta_figura, dpi=300)
        print(f"Visualización guardada en: {ruta_figura}")
        return ruta_figura
    
    plt.show()
    return None

# ----- Funciones para procesamiento por lotes -----

def procesar_multiples_curvas(lista_puntos, grados_polinomio=None, parametros_spline=None, guardar_resultados=True):
    """
    Procesa múltiples curvas aplicando diferentes ajustes y calculando longitudes.
    
    Args:
        lista_puntos: Lista de arrays NumPy con puntos (x,y)
        grados_polinomio: Lista de grados para ajustar polinomios
        parametros_spline: Lista de parámetros s para ajustar splines
        guardar_resultados: Si True, guarda todos los resultados
    
    Returns:
        DataFrame con un resumen de los resultados
    """
    if grados_polinomio is None:
        grados_polinomio = [2, 3, 4]
    
    if parametros_spline is None:
        parametros_spline = [0.1, 0.5, 1.0]
    
    # Lista para almacenar los resultados
    resultados = []
    
    # Procesar cada conjunto de puntos
    for i, puntos in enumerate(lista_puntos):
        print(f"\nProcesando curva {i+1}/{len(lista_puntos)}...")
        
        # Límites para cálculos
        x_min = min(puntos[:, 0])
        x_max = max(puntos[:, 0])
        
        # Guardar puntos si se solicita
        if guardar_resultados:
            guardar_puntos_curva(puntos, f"curva_{i+1}_puntos")
        
        # Ajustar polinomios de diferentes grados
        for grado in grados_polinomio:
            print(f"  Ajustando polinomio de grado {grado}...")
            
            # Ajustar polinomio
            funcion_polinomio = ajuste_polinomio(puntos, grado=grado)
            
            # Calcular longitud
            try:
                df_longitudes, longitud_total = calcular_longitud_por_tramos(funcion_polinomio, x_min, x_max)
                
                # Guardar resultados si se solicita
                if guardar_resultados:
                    nombre_base = f"curva_{i+1}_polinomio_g{grado}"
                    
                    # Guardar modelo
                    info_modelo = {
                        'grado': grado,
                        'x_min': x_min,
                        'x_max': x_max
                    }
                    guardar_modelo_funcion(funcion_polinomio, 'polinomio', info_modelo, nombre_base)
                    
                    # Guardar muestreo
                    guardar_muestreo_funcion(funcion_polinomio, x_min, x_max, nombre_base)
                    
                    # Guardar longitudes
                    guardar_resultados_longitud(df_longitudes, longitud_total, nombre_base)
                    
                    # Visualizar
                    visualizar_resultados(puntos, funcion_polinomio, x_min, x_max, 
                                        longitud_total, df_longitudes, nombre_base)
                
                # Añadir a los resultados generales
                resultados.append({
                    'curva': i+1,
                    'tipo_ajuste': 'polinomio',
                    'parametro': grado,
                    'longitud': longitud_total,
                    'x_min': x_min,
                    'x_max': x_max
                })
                
            except Exception as e:
                print(f"  Error al procesar polinomio de grado {grado}: {e}")
        
        # Ajustar splines con diferentes parámetros
        for s in parametros_spline:
            print(f"  Ajustando spline con parámetro s={s}...")
            
            # Ajustar spline
            try:
                funcion_spline = ajuste_spline(puntos, s=s)
                
                # Calcular longitud
                df_longitudes, longitud_total = calcular_longitud_por_tramos(funcion_spline, x_min, x_max)
                
                # Guardar resultados si se solicita
                if guardar_resultados:
                    nombre_base = f"curva_{i+1}_spline_s{s:.1f}".replace('.', '_')
                    
                    # Guardar modelo
                    info_modelo = {
                        's': s,
                        'x_min': x_min,
                        'x_max': x_max
                    }
                    guardar_modelo_funcion(funcion_spline, 'spline', info_modelo, nombre_base)
                    
                    # Guardar muestreo
                    guardar_muestreo_funcion(funcion_spline, x_min, x_max, nombre_base)
                    
                    # Guardar longitudes
                    guardar_resultados_longitud(df_longitudes, longitud_total, nombre_base)
                    
                    # Visualizar
                    visualizar_resultados(puntos, funcion_spline, x_min, x_max, 
                                        longitud_total, df_longitudes, nombre_base)
                
                # Añadir a los resultados generales
                resultados.append({
                    'curva': i+1,
                    'tipo_ajuste': 'spline',
                    'parametro': s,
                    'longitud': longitud_total,
                    'x_min': x_min,
                    'x_max': x_max
                })
                
            except Exception as e:
                print(f"  Error al procesar spline con s={s}: {e}")
    
    # Crear DataFrame con todos los resultados
    df_resultados = pd.DataFrame(resultados)
    
    # Guardar resultados generales
    if guardar_resultados and len(df_resultados) > 0:
        ruta_resultados = os.path.join('../data/resultados', 'resultados_generales.csv')
        df_resultados.to_csv(ruta_resultados, index=False)
        print(f"\nResumen de resultados guardado en: {ruta_resultados}")
    
    return df_resultados

# ----- Función principal para procesar una imagen completa -----

def procesar_imagen_completa(ruta_imagen, nombre_base, grados_polinomio=None, parametros_spline=None):
    """
    Procesa una imagen completa: carga, detecta curva, ajusta funciones y calcula longitudes.
    
    Args:
        ruta_imagen: Ruta a la imagen a procesar
        nombre_base: Nombre base para los archivos generados
        grados_polinomio: Lista de grados para ajustar polinomios
        parametros_spline: Lista de parámetros s para ajustar splines
    
    Returns:
        DataFrame con un resumen de los resultados
    """
    from src.procesamiento import cargar_imagen, preprocesar_imagen, detectar_bordes, extraer_puntos_curva
    
    print(f"Procesando imagen: {ruta_imagen}")
    
    # Cargar y procesar la imagen
    imagen = cargar_imagen(ruta_imagen)
    imagen_preprocesada = preprocesar_imagen(imagen)
    bordes = detectar_bordes(imagen_preprocesada)
    puntos = extraer_puntos_curva(bordes)
    
    # Verificar que tenemos suficientes puntos
    if len(puntos) < 4:
        print("Error: No se detectaron suficientes puntos en la curva.")
        return None
    
    # Guardar los puntos detectados
    guardar_puntos_curva(puntos, f"{nombre_base}_puntos")
    
    # Procesar con diferentes ajustes
    if grados_polinomio is None:
        grados_polinomio = [3]  # Por defecto usamos grado 3
    
    if parametros_spline is None:
        parametros_spline = [0.1]  # Por defecto usamos s=0.1
    
    # Usar la función de procesamiento por lotes
    df_resultados = procesar_multiples_curvas([puntos], grados_polinomio, parametros_spline, True)
    
    return df_resultados