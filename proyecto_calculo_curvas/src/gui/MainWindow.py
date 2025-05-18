# main_window.py
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
from threading import Thread
import time

# Importar el panel de imagen que ya tienes
from image_panel import ImagePanel

# Cliente Ollama real usando requests
import requests
import json

class OllamaClient:
    def __init__(self, base_url="http://localhost:11434", model="mistral"):
        self.base_url = base_url
        self.model = model
        self.api_url = f"{base_url}/api/generate"
        
        # Contexto del sistema para el asistente
        self.system_prompt = """Eres un asistente especializado en análisis de curvas e imágenes. 
        Tu papel es ayudar a los usuarios a entender:
        - Procesamiento de imágenes con OpenCV
        - Extracción de puntos de curvas
        - Cálculo de longitudes de curva
        - Métodos matemáticos de ajuste de funciones
        - Interpretación de resultados
        
        Responde de forma clara, técnica pero accesible, y siempre en español.
        Usa emojis para hacer las respuestas más amigables."""
    
    def enviar_mensaje(self, mensaje):
        try:
            # Construir el prompt completo con contexto
            prompt_completo = f"{self.system_prompt}\n\nUsuario: {mensaje}\nAsistente:"
            
            # Hacer petición a Ollama
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "prompt": prompt_completo,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 500
                    }
                },
                timeout=30  # Timeout de 30 segundos
            )
            
            # Verificar que la respuesta sea exitosa
            response.raise_for_status()
            
            # Extraer respuesta
            data = response.json()
            return data["response"].strip()
            
        except requests.exceptions.ConnectionError:
            return "❌ Error: No se pudo conectar con Ollama. Asegúrate de que esté ejecutándose en localhost:11434"
        except requests.exceptions.Timeout:
            return "⏱️ Error: La respuesta de Ollama tardó demasiado. Intenta con una pregunta más corta."
        except requests.exceptions.RequestException as e:
            return f"❌ Error de conexión: {str(e)}"
        except KeyError:
            return "❌ Error: Respuesta inválida de Ollama"
        except Exception as e:
            return f"❌ Error inesperado: {str(e)}"
    
    def verificar_conexion(self):
        """Verifica si Ollama está disponible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            return True
        except:
            return False

# Panels placeholder (crear estos después si es necesario)
class PlotPanel(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Placeholder para panel de gráficas
        label = ttk.Label(self, text="📊 Panel de Gráficas\n\n(Aquí se mostrarán las gráficas cuando proceses una imagen)", 
                         font=("Arial", 14))
        label.pack(expand=True)
        
        self.puntos = None
    
    def cargar_puntos(self, puntos):
        self.puntos = puntos
        # Aquí iría la lógica para crear gráficas con matplotlib
        print(f"PlotPanel: Recibidos {len(puntos)} puntos para graficar")
    
    def on_focus(self):
        pass

class ChatPanel(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Placeholder para panel de chat
        label = ttk.Label(self, text="🤖 Panel de Chat Avanzado\n\n(Aquí irá el chat especializado en análisis)", 
                         font=("Arial", 14))
        label.pack(expand=True)
    
    def on_focus(self):
        pass

# Colores y configuración
class Config:
    COLOR_PRIMARIO = "#2E86AB"
    COLOR_SECUNDARIO = "#A23B72"
    COLOR_TEXTO = "#333333"
    COLOR_FONDO_CHAT = "#F8F9FA"
    COLOR_ASISTENTE = "#E3F2FD"
    COLOR_USUARIO = "#E8F5E9"

class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        
        # Inicializar cliente Ollama real
        self.chat_ollama = OllamaClient(model="mistral")
        
        # Verificar conexión con Ollama al inicio
        if self.chat_ollama.verificar_conexion():
            print("✅ Conectado a Ollama exitosamente")
        else:
            print("⚠️  Advertencia: No se pudo conectar a Ollama. Verificando en segundo plano...")
        
        # Variables de estado
        self.imagen_actual = None
        self.puntos_actuales = None
        
        # Configurar ventana
        self.configurar_ventana()
        
        # Crear navegación por tabs
        self.crear_navegacion()
        
        # Crear tab de bienvenida
        self.crear_tab_bienvenida()
        
        # Mostrar mensaje de bienvenida después de 1 segundo
        self.after(1000, self.mostrar_bienvenida_ollama)
    
    def configurar_ventana(self):
        # Propiedades básicas
        self.title("🔬 Calculadora de Longitud de Curvas")
        self.geometry("1200x800")
        self.minsize(1000, 700)
        
        # Centrar ventana
        self.centrar_ventana()
        
        # Configuración visual
        self.configure(bg="#F5F5F5")
        
        # Estilo moderno para ttk widgets
        style = ttk.Style()
        style.theme_use('clam')
    
    def centrar_ventana(self):
        self.update_idletasks()
        ancho = self.winfo_width()
        alto = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (ancho // 2)
        y = (self.winfo_screenheight() // 2) - (alto // 2)
        self.geometry(f"{ancho}x{alto}+{x}+{y}")
    
    def crear_navegacion(self):
        # Notebook para tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Bienvenida con Ollama
        self.tab_bienvenida = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_bienvenida, text="🏠 Inicio")
        
        # Tab 2: Carga de Imagen
        self.tab_imagen = ttk.Frame(self.notebook)
        self.panel_imagen = ImagePanel(self.tab_imagen)
        # Conectar callback para cuando se procese una imagen
        self.panel_imagen.parent = self
        self.notebook.add(self.tab_imagen, text="📸 Imagen")
        
        # Tab 3: Gráficas
        self.tab_graficas = ttk.Frame(self.notebook)
        self.panel_graficas = PlotPanel(self.tab_graficas)
        self.notebook.add(self.tab_graficas, text="📊 Gráficas")
        
        # Tab 4: Chat Avanzado
        self.tab_chat = ttk.Frame(self.notebook)
        self.panel_chat = ChatPanel(self.tab_chat)
        self.notebook.add(self.tab_chat, text="🤖 Análisis IA")
        
        # Configurar eventos de cambio de tab
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
    
    def crear_tab_bienvenida(self):
        # Frame principal centrado
        frame_principal = ttk.Frame(self.tab_bienvenida)
        frame_principal.pack(expand=True, fill=tk.BOTH, padx=50, pady=50)
        
        # Título elegante
        titulo = tk.Label(
            frame_principal,
            text="🔬 Calculadora de Longitud de Curvas",
            font=("Arial", 24, "bold"),
            fg=Config.COLOR_PRIMARIO,
            bg="#F5F5F5"
        )
        titulo.pack(pady=30)
        
        # Descripción
        descripcion_texto = """🎯 Esta aplicación utiliza análisis computacional avanzado para:

• 📷 Procesar imágenes de curvas y cables
• 🧮 Extraer puntos de forma automática
• 📐 Ajustar funciones matemáticas por intervalos
• 📏 Calcular longitudes de curva con precisión
• 🤖 Explicar los resultados con IA

👈 Comience cargando una imagen en la pestaña "Imagen\""""
        
        descripcion = tk.Label(
            frame_principal,
            text=descripcion_texto,
            font=("Arial", 12),
            justify=tk.LEFT,
            fg=Config.COLOR_TEXTO,
            bg="#F5F5F5"
        )
        descripcion.pack(pady=20)
        
        # Frame para chat de bienvenida
        frame_chat = ttk.LabelFrame(
            frame_principal,
            text="💬 Chat con Asistente IA",
            padding=20
        )
        frame_chat.pack(fill=tk.BOTH, expand=True, pady=20)
        
        # Área de chat
        chat_frame = ttk.Frame(frame_chat)
        chat_frame.pack(fill=tk.BOTH, expand=True)
        
        # Texto del chat con scrollbar
        texto_frame = ttk.Frame(chat_frame)
        texto_frame.pack(fill=tk.BOTH, expand=True)
        
        self.texto_chat = tk.Text(
            texto_frame,
            height=10,
            width=80,
            state=tk.DISABLED,
            font=("Consolas", 10),
            bg=Config.COLOR_FONDO_CHAT,
            wrap=tk.WORD
        )
        
        scrollbar_chat = ttk.Scrollbar(texto_frame, command=self.texto_chat.yview)
        self.texto_chat.configure(yscrollcommand=scrollbar_chat.set)
        
        self.texto_chat.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_chat.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Frame para input de chat
        frame_input = ttk.Frame(frame_chat)
        frame_input.pack(fill=tk.X, pady=(10, 0))
        
        self.entrada_chat = ttk.Entry(
            frame_input,
            font=("Arial", 10)
        )
        self.entrada_chat.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        boton_enviar = ttk.Button(
            frame_input,
            text="📤 Enviar",
            command=self.enviar_mensaje_chat
        )
        boton_enviar.pack(side=tk.RIGHT)
        
        # Vincular Enter para enviar
        self.entrada_chat.bind("<Return>", lambda e: self.enviar_mensaje_chat())
        
        # Placeholder en el entry
        self.entrada_chat.insert(0, "Escriba su pregunta aquí...")
        self.entrada_chat.bind("<FocusIn>", self.on_entry_focus_in)
        self.entrada_chat.bind("<FocusOut>", self.on_entry_focus_out)
        self.entrada_chat.configure(foreground="gray")
    
    def on_entry_focus_in(self, event):
        if self.entrada_chat.get() == "Escriba su pregunta aquí...":
            self.entrada_chat.delete(0, tk.END)
            self.entrada_chat.configure(foreground="black")
    
    def on_entry_focus_out(self, event):
        if not self.entrada_chat.get():
            self.entrada_chat.insert(0, "Escriba su pregunta aquí...")
            self.entrada_chat.configure(foreground="gray")
    
    def mostrar_bienvenida_ollama(self):
        # Verificar conexión antes de mostrar bienvenida
        if self.chat_ollama.verificar_conexion():
            mensaje_bienvenida = """🤖 ¡Hola! Soy tu asistente de análisis de curvas con Ollama Mistral.

Estoy aquí para ayudarte a entender el funcionamiento de esta aplicación.
Puedes preguntarme sobre:

• Cómo cargar y procesar imágenes
• Interpretación de las funciones matemáticas  
• Métodos de cálculo de longitudes
• Explicación de los resultados
• Algoritmos de visión computacional

¿En qué puedo ayudarte hoy?"""
        else:
            mensaje_bienvenida = """⚠️ Asistente IA temporalmente desconectado.

No se pudo conectar con Ollama. Por favor verifica:
• Que Ollama esté ejecutándose (comando: ollama serve)
• Que el modelo 'mistral' esté disponible (comando: ollama list)
• Que el puerto 11434 esté libre

Puedes intentar hacer preguntas y reconectará automáticamente."""
        
        self.añadir_mensaje_chat("Asistente", mensaje_bienvenida, Config.COLOR_ASISTENTE)
    
    def enviar_mensaje_chat(self):
        mensaje = self.entrada_chat.get().strip()
        
        # Verificar placeholder
        if mensaje == "" or mensaje == "Escriba su pregunta aquí...":
            return
        
        # Mostrar mensaje del usuario
        self.añadir_mensaje_chat("Tú", mensaje, Config.COLOR_USUARIO)
        self.entrada_chat.delete(0, tk.END)
        
        # Mostrar indicador de "escribiendo"
        self.añadir_mensaje_chat("Asistente", "🤔 Pensando...", Config.COLOR_ASISTENTE)
        
        # Enviar a Ollama en hilo separado
        hilo = Thread(target=self._procesar_chat_ollama, args=(mensaje,))
        hilo.daemon = True
        hilo.start()
    
    def _procesar_chat_ollama(self, mensaje):
        try:
            # Conectar con Ollama (o simulador)
            respuesta = self.chat_ollama.enviar_mensaje(mensaje)
            
            # Actualizar GUI en hilo principal
            self.after(0, self._actualizar_respuesta_chat, respuesta)
            
        except Exception as error:
            self.after(0, self._error_chat, str(error))
    
    def _actualizar_respuesta_chat(self, respuesta):
        # Eliminar mensaje "Pensando..."
        self.texto_chat.config(state=tk.NORMAL)
        
        # Encontrar y eliminar la última línea "Pensando..."
        contenido = self.texto_chat.get("1.0", tk.END)
        lines = contenido.split('\n')
        if lines and "🤔 Pensando..." in lines[-2]:
            # Eliminar las últimas líneas del mensaje "Pensando..."
            self.texto_chat.delete("end-3l", "end-1l")
        
        self.texto_chat.config(state=tk.DISABLED)
        
        # Añadir respuesta real
        self.añadir_mensaje_chat("Asistente", respuesta, Config.COLOR_ASISTENTE)
    
    def _error_chat(self, error):
        # Eliminar mensaje "Pensando..."
        self.texto_chat.config(state=tk.NORMAL)
        contenido = self.texto_chat.get("1.0", tk.END)
        lines = contenido.split('\n')
        if lines and "🤔 Pensando..." in lines[-2]:
            self.texto_chat.delete("end-3l", "end-1l")
        self.texto_chat.config(state=tk.DISABLED)
        
        # Mostrar error
        self.añadir_mensaje_chat("Asistente", f"❌ Error: {error}", Config.COLOR_ASISTENTE)
    
    def añadir_mensaje_chat(self, remitente, mensaje, color):
        self.texto_chat.config(state=tk.NORMAL)
        
        # Timestamp
        tiempo = datetime.now().strftime("%H:%M")
        
        # Formatear mensaje
        texto_completo = f"[{tiempo}] {remitente}:\n{mensaje}\n\n"
        
        # Configurar tags para colores
        tag_name = f"{remitente}_{tiempo}"
        self.texto_chat.tag_configure(tag_name, background=color, relief=tk.RAISED, borderwidth=1)
        
        # Insertar texto con formato
        start_index = self.texto_chat.index(tk.END)
        self.texto_chat.insert(tk.END, texto_completo)
        end_index = self.texto_chat.index(tk.END)
        
        # Aplicar tag al mensaje
        self.texto_chat.tag_add(tag_name, start_index, end_index)
        
        # Scroll automático
        self.texto_chat.see(tk.END)
        
        self.texto_chat.config(state=tk.DISABLED)
    
    def on_tab_changed(self, evento):
        tab_actual = self.notebook.select()
        indice = self.notebook.index(tab_actual)
        
        # Acciones específicas por tab
        if indice == 1:  # Tab de imagen
            self.panel_imagen.on_focus()
        elif indice == 2:  # Tab de gráficas
            self.panel_graficas.on_focus()
        elif indice == 3:  # Tab de chat
            self.panel_chat.on_focus()
    
    def on_imagen_procesada(self, puntos):
        """Callback cuando imagen es procesada desde ImagePanel"""
        # Guardar puntos actuales
        self.puntos_actuales = puntos
        
        # Activar tab de gráficas si estaba deshabilitado
        self.notebook.tab(2, state="normal")
        
        # Pasar datos al panel de gráficas
        self.panel_graficas.cargar_puntos(puntos)
        
        # Opcional: cambiar automáticamente al tab de gráficas
        self.notebook.select(2)
        
        # Mostrar mensaje en el chat de bienvenida
        mensaje = f"✅ Imagen procesada exitosamente. Se detectaron {len(puntos)} puntos. Puedes revisar las gráficas en la pestaña correspondiente."
        self.añadir_mensaje_chat("Sistema", mensaje, "#FFF3CD")
    
    def ejecutar(self):
        self.mainloop()

# Función principal de inicio
def main():
    app = MainWindow()
    app.ejecutar()

if __name__ == "__main__":
    main()