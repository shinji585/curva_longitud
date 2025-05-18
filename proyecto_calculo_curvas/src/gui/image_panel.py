# image_panel.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
from threading import Thread
import time

class ImagePanel(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        
        # Variables de estado
        self.imagen_cargada = None
        self.ruta_imagen = None
        self.puntos_procesados = None
        self.imagen_tk = None
        
        self.crear_interfaz()
        self.configurar_eventos()
    
    def crear_interfaz(self):
        # Configurar el frame principal
        self.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Sección superior: Controles de carga
        frame_controles = ttk.Frame(self)
        frame_controles.pack(fill=tk.X, pady=(0, 10))
        
        # Botón de carga estilizado
        self.boton_cargar = ttk.Button(
            frame_controles,
            text="📁 Cargar Imagen",
            command=self.cargar_imagen
        )
        self.boton_cargar.pack(side=tk.LEFT, padx=(0, 10))
        
        # Label para mostrar estado
        self.label_estado = ttk.Label(
            frame_controles,
            text="No hay imagen cargada",
            foreground="gray"
        )
        self.label_estado.pack(side=tk.LEFT)
        
        # Sección central: Vista previa de imagen
        frame_imagen = ttk.LabelFrame(self, text="Vista Previa")
        frame_imagen.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Canvas para mostrar imagen con scrollbars
        canvas_frame = ttk.Frame(frame_imagen)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.canvas_imagen = tk.Canvas(
            canvas_frame,
            width=400,
            height=300,
            bg="white",
            relief=tk.SUNKEN,
            borderwidth=2
        )
        
        # Scrollbars
        scroll_y = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas_imagen.yview)
        scroll_x = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas_imagen.xview)
        self.canvas_imagen.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        
        # Organizar canvas y scrollbars
        self.canvas_imagen.grid(row=0, column=0, sticky="nsew")
        scroll_y.grid(row=0, column=1, sticky="ns")
        scroll_x.grid(row=1, column=0, sticky="ew")
        
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        
        # Sección inferior: Información de imagen
        frame_info = ttk.LabelFrame(self, text="Información de la Imagen")
        frame_info.pack(fill=tk.X)
        
        info_frame = ttk.Frame(frame_info)
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Labels informativos
        self.label_dimensiones = ttk.Label(info_frame, text="Dimensiones: -")
        self.label_dimensiones.grid(row=0, column=0, sticky="w", padx=(0, 20))
        
        self.label_tipo = ttk.Label(info_frame, text="Tipo: -")
        self.label_tipo.grid(row=0, column=1, sticky="w", padx=(0, 20))
        
        self.label_tamaño = ttk.Label(info_frame, text="Tamaño: -")
        self.label_tamaño.grid(row=0, column=2, sticky="w")
        
        # Botón de procesamiento
        self.boton_procesar = ttk.Button(
            frame_info,
            text="🔄 Procesar Imagen",
            command=self.procesar_imagen,
            state=tk.DISABLED
        )
        self.boton_procesar.pack(pady=10)
        
        # Barra de progreso (oculta inicialmente)
        self.progreso = ttk.Progressbar(
            frame_info,
            mode='indeterminate'
        )
        
        info_frame.grid_columnconfigure(3, weight=1)
    
    def configurar_eventos(self):
        # Eventos del canvas para zoom y pan
        self.canvas_imagen.bind("<ButtonPress-1>", self.on_canvas_click)
        self.canvas_imagen.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas_imagen.bind("<MouseWheel>", self.on_canvas_zoom)
    
    def cargar_imagen(self):
        # Abrir diálogo de archivo
        filetypes = [
            ("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif"),
            ("JPEG", "*.jpg *.jpeg"),
            ("PNG", "*.png"),
            ("BMP", "*.bmp"),
            ("TIFF", "*.tiff"),
            ("Todos los archivos", "*.*")
        ]
        
        ruta = filedialog.askopenfilename(
            title="Seleccionar Imagen",
            filetypes=filetypes
        )
        
        if ruta:
            try:
                # Cargar imagen con OpenCV
                imagen = cv2.imread(ruta)
                
                if imagen is None:
                    messagebox.showerror("Error", "No se pudo cargar la imagen")
                    return
                
                # Guardar datos
                self.imagen_cargada = imagen
                self.ruta_imagen = ruta
                
                # Mostrar imagen en canvas
                self.mostrar_imagen_en_canvas(imagen)
                
                # Actualizar información
                self.actualizar_info_imagen(imagen)
                
                # Habilitar botón de procesamiento
                self.boton_procesar.config(state=tk.NORMAL)
                
                # Actualizar estado
                nombre_archivo = os.path.basename(ruta)
                self.label_estado.config(
                    text=f"✅ Imagen cargada: {nombre_archivo}",
                    foreground="green"
                )
                
            except Exception as error:
                messagebox.showerror("Error", f"Error al cargar imagen: {str(error)}")
    
    def mostrar_imagen_en_canvas(self, imagen):
        # Convertir de BGR a RGB
        imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        
        # Redimensionar si es muy grande (manteniendo aspecto)
        altura, ancho = imagen_rgb.shape[:2]
        max_width, max_height = 800, 600
        
        if ancho > max_width or altura > max_height:
            factor = min(max_width / ancho, max_height / altura)
            nuevo_ancho = int(ancho * factor)
            nueva_altura = int(altura * factor)
            imagen_rgb = cv2.resize(imagen_rgb, (nuevo_ancho, nueva_altura))
        
        # Convertir a formato PIL y luego a PhotoImage
        imagen_pil = Image.fromarray(imagen_rgb)
        self.imagen_tk = ImageTk.PhotoImage(imagen_pil)
        
        # Limpiar canvas y mostrar imagen
        self.canvas_imagen.delete("all")
        self.canvas_imagen.create_image(0, 0, anchor=tk.NW, image=self.imagen_tk)
        
        # Configurar región de scroll
        self.canvas_imagen.configure(scrollregion=self.canvas_imagen.bbox("all"))
    
    def actualizar_info_imagen(self, imagen):
        altura, ancho, canales = imagen.shape
        tamaño_bytes = os.path.getsize(self.ruta_imagen)
        tamaño_kb = tamaño_bytes / 1024
        
        # Obtener tipo de archivo
        _, extension = os.path.splitext(self.ruta_imagen)
        tipo = extension[1:].upper() if extension else "Desconocido"
        
        # Actualizar labels
        self.label_dimensiones.config(text=f"Dimensiones: {ancho} x {altura} px")
        self.label_tipo.config(text=f"Tipo: {tipo}")
        
        if tamaño_kb < 1024:
            self.label_tamaño.config(text=f"Tamaño: {tamaño_kb:.1f} KB")
        else:
            tamaño_mb = tamaño_kb / 1024
            self.label_tamaño.config(text=f"Tamaño: {tamaño_mb:.1f} MB")
    
    def procesar_imagen(self):
        if self.imagen_cargada is None:
            return
        
        # Mostrar indicador de progreso
        self.mostrar_progreso("Procesando imagen...")
        
        # Ejecutar en hilo separado para no bloquear GUI
        hilo = Thread(target=self._procesar_imagen_async)
        hilo.daemon = True
        hilo.start()
    
    def _procesar_imagen_async(self):
        try:
            # Simular procesamiento (aquí iría la llamada real a Demo2.py)
            # Por ahora, simularemos con procesamiento básico
            
            # Preprocesamiento básico
            imagen_gris = cv2.cvtColor(self.imagen_cargada, cv2.COLOR_BGR2GRAY)
            
            # Simulamos que toma tiempo
            time.sleep(2)
            
            # Detectar bordes básicos (ejemplo simple)
            bordes = cv2.Canny(imagen_gris, 50, 150, apertureSize=3)
            
            # Extraer puntos de muestra (ejemplo simplificado)
            puntos = self._extraer_puntos_ejemplo(bordes)
            
            # Guardar resultados
            self.puntos_procesados = puntos
            
            # Actualizar GUI en hilo principal
            self.after(0, self._procesamiento_completado, puntos)
            
        except Exception as error:
            self.after(0, self._procesamiento_error, error)
    
    def _extraer_puntos_ejemplo(self, bordes):
        # Ejemplo simplificado de extracción de puntos
        # En la implementación real, esto vendría de Demo2.py
        puntos = []
        contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contornos:
            # Tomar el contorno más grande
            contorno_principal = max(contornos, key=cv2.contourArea)
            
            # Simplificar contorno y extraer puntos
            epsilon = 0.02 * cv2.arcLength(contorno_principal, True)
            contorno_aproximado = cv2.approxPolyDP(contorno_principal, epsilon, True)
            
            for punto in contorno_aproximado:
                x, y = punto[0]
                puntos.append([float(x), float(y)])
        
        return np.array(puntos) if puntos else np.array([])
    
    def _procesamiento_completado(self, puntos):
        self.ocultar_progreso()
        
        # Actualizar estado
        num_puntos = len(puntos)
        self.label_estado.config(
            text=f"✅ Procesamiento completo: {num_puntos} puntos detectados",
            foreground="green"
        )
        
        # Notificar a ventana principal si existe el método
        if hasattr(self.parent, 'on_imagen_procesada'):
            self.parent.on_imagen_procesada(puntos)
    
    def _procesamiento_error(self, error):
        self.ocultar_progreso()
        messagebox.showerror("Error", f"Error en procesamiento: {str(error)}")
        
        self.label_estado.config(
            text="❌ Error en el procesamiento",
            foreground="red"
        )
    
    def mostrar_progreso(self, mensaje="Procesando..."):
        self.label_estado.config(text=mensaje, foreground="blue")
        self.progreso.pack(pady=5)
        self.progreso.start(10)
        self.boton_procesar.config(state=tk.DISABLED)
        self.boton_cargar.config(state=tk.DISABLED)
    
    def ocultar_progreso(self):
        self.progreso.stop()
        self.progreso.pack_forget()
        self.boton_procesar.config(state=tk.NORMAL)
        self.boton_cargar.config(state=tk.NORMAL)
    
    def on_canvas_click(self, event):
        # Iniciar pan de la imagen
        self.canvas_imagen.scan_mark(event.x, event.y)
    
    def on_canvas_drag(self, event):
        # Pan de la imagen
        self.canvas_imagen.scan_dragto(event.x, event.y, gain=1)
    
    def on_canvas_zoom(self, event):
        # Zoom básico (se puede mejorar)
        factor = 1.1 if event.delta > 0 else 0.9
        self.canvas_imagen.scale("all", event.x, event.y, factor, factor)
        self.canvas_imagen.configure(scrollregion=self.canvas_imagen.bbox("all"))
    
    def on_focus(self):
        # Método llamado cuando el tab recibe foco
        pass
    
    # Métodos públicos para acceso externo
    def obtener_puntos(self):
        return self.puntos_procesados
    
    def obtener_imagen(self):
        return self.imagen_cargada
    
    def obtener_ruta_imagen(self):
        return self.ruta_imagen