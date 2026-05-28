import customtkinter as ctk
from PIL import Image, ImageDraw
import numpy as np
import os
import sys
import io
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# ══════════════════════════════════════════════════════════════════════════
# NUEVO: AÑADIR LA CARPETA 'CLUSTER DGX' AL PATH DE PYTHON
# ══════════════════════════════════════════════════════════════════════════
# Obtenemos la ruta actual donde está este script (interfaz_app.py)
ruta_actual = os.path.dirname(os.path.abspath(__file__))
# Subimos un nivel ("..") y entramos en "CLUSTER DGX"
ruta_cluster = os.path.abspath(os.path.join(ruta_actual, "..", "CLUSTER DGX"))
# Lo añadimos a las rutas del sistema donde Python busca librerías
sys.path.append(ruta_cluster)

# Ahora Python ya sabe dónde encontrar tus scripts SOTA, importamos de ellos:
from generar_galaxia_mi_u_net import (
    load_checkpoint, generate_galaxy, GalaxySpec, tensor_to_pil
)
# ══════════════════════════════════════════════════════════════════════════

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# --- RUTA A TU MODELO ENTRENADO ---
# Asegúrate de que esta ruta sigue siendo correcta (donde está tu mejor_modelo.pt)
RUTA_CHECKPOINT = os.path.expanduser("~/Downloads/modelo_epoca_100.pt")

class GeneradorGalaxiasApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Generador de galaxias (TFG) - Universidad de Alicante")
        self.geometry("1200x750")
        
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.LADO_P = 8
        self.TOTAL_PARCHES_MAX = 64*64
        self.espiral = self.generar_espiral(self.TOTAL_PARCHES_MAX)
        
        self.diccionario_sliders = {}

        self.crear_panel_izquierdo()
        self.crear_panel_derecho()

        # Cargamos la IA justo después de dibujar la ventana gráfica
        self.after(100, self.cargar_modelo)

    def cargar_modelo(self):
        """Carga los pesos EMA en la VRAM una sola vez al arrancar."""
        self.btn_generar.configure(text="Cargando\nmodelo...", state="disabled")
        self.update()
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            (
                self.unet, self.projector, self.noise_scheduler, 
                self.variables, self.norm_stats
            ) = load_checkpoint(RUTA_CHECKPOINT, self.device)
            
            # Configuramos DDIM a 30 pasos para que en la app sea casi instantáneo
            self.pasos_ddim = 30 
            self.noise_scheduler.set_timesteps(self.pasos_ddim)
            
            self.btn_generar.configure(text="Generar\ngalaxia", state="normal")
            print("Modelo cargado y listo para inferencia.")
        except Exception as e:
            self.btn_generar.configure(text="Error al\ncargar", text_color="red")
            print(f"Error cargando el modelo: {e}")

    def generar_espiral(self, n_max):
        """Genera las coordenadas relativas en espiral para el muestreo."""
        coords = []
        gx, gy = 0, 0
        dx, dy = 1, 0
        pasos_tramo, longitud_tramo, giros = 0, 1, 0
        while len(coords) < n_max:
            coords.append((gy, gx))
            gx += dx
            gy += dy
            pasos_tramo += 1
            if pasos_tramo == longitud_tramo:
                pasos_tramo = 0
                dx, dy = -dy, dx
                giros += 1
                if giros % 2 == 0:
                    longitud_tramo += 1
        return coords

    def crear_panel_izquierdo(self):
        self.frame_izq = ctk.CTkFrame(self, width=320, corner_radius=0, fg_color="#ffffff")
        self.frame_izq.grid(row=0, column=0, sticky="nsew")
        self.frame_izq.grid_rowconfigure(13, weight=1)

        # Logo UA (Asegúrate de que logo_ua.jpeg está en la misma carpeta que interfaz_app.py)
        try:
            ruta_logo = os.path.join(os.path.dirname(__file__), "logo_ua.jpeg")
            img_logo = ctk.CTkImage(light_image=Image.open(ruta_logo), size=(190, 190))
            self.label_logo = ctk.CTkLabel(self.frame_izq, image=img_logo, text="")
        except:
            self.label_logo = ctk.CTkLabel(self.frame_izq, text="LOGO UA", text_color="black")
        self.label_logo.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Título
        self.label_titulo = ctk.CTkLabel(self.frame_izq, text="Generador U-Net", 
                                         font=ctk.CTkFont(size=20, weight="bold"), text_color="#00A390")
        self.label_titulo.grid(row=1, column=0, padx=20, pady=(0, 10))

        # Sliders mapeados a rangos físicos reales
        self.crear_slider_dinamico("escala_kpc_px", "Escala (kpc/px)", 0.15, 2.5, 0.8, 2)
        self.crear_slider_dinamico("log_ms", "Masa (log M☉)", 8.0, 12.0, 10.0, 4)
        self.crear_slider_dinamico("ea_gyr", "Edad Estelar (Gyr)", 0.5, 10.0, 3.0, 6)
        self.crear_slider_dinamico("radio_p_arcsec", "Radio Petrosian ('')", 3.0, 30.0, 10.0, 8)
        
        # Slider de CFG para ver cómo influye la guía
        self.crear_slider_dinamico("guidance", "Escala de Guía (CFG)", 1.0, 15.0, 7.5, 10)

        # Botón Generar
        self.btn_generar = ctk.CTkButton(self.frame_izq, text="Esperando\nmodelo...", 
                                         fg_color="#00A390", hover_color="#007A6C",
                                         font=ctk.CTkFont(size=22, weight="bold"), 
                                         width=210, height=80, corner_radius=0,
                                         state="disabled",
                                         command=self.ejecutar_generacion)
        self.btn_generar.grid(row=12, column=0, pady=(20, 30))

    def crear_slider_dinamico(self, clave, texto, v_min, v_max, v_default, row):
        """Crea un slider cuyo label se actualiza en tiempo real."""
        lbl_text = f"{texto}: {v_default:.2f}"
        label = ctk.CTkLabel(self.frame_izq, text=lbl_text, font=ctk.CTkFont(size=14, weight="bold"), text_color="black")
        label.grid(row=row, column=0, padx=35, pady=(5, 0), sticky="w")
        
        def actualizar_label(valor):
            label.configure(text=f"{texto}: {valor:.2f}")

        slider = ctk.CTkSlider(self.frame_izq, from_=v_min, to=v_max, width=220, 
                               button_color="#00A390", command=actualizar_label)
        slider.set(v_default)
        slider.grid(row=row+1, column=0, padx=20, pady=(0, 10))
        
        self.diccionario_sliders[clave] = slider

    def crear_panel_derecho(self):
        self.frame_der = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_der.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.frame_der.grid_columnconfigure((0, 1), weight=1)
        self.frame_der.grid_rowconfigure(1, weight=1)

        self.lbl_img_tit = ctk.CTkLabel(self.frame_der, text="Galaxia sintética", font=ctk.CTkFont(size=22, weight="bold"))
        self.lbl_img_tit.grid(row=0, column=0, pady=10)
        
        self.lbl_rgb_tit = ctk.CTkLabel(self.frame_der, text="Análisis RGB", font=ctk.CTkFont(size=22, weight="bold"))
        self.lbl_rgb_tit.grid(row=0, column=1, pady=10)

        self.caja_galaxia = ctk.CTkLabel(self.frame_der, text="Pulsa generar", fg_color="#121212", 
                                 width=512, height=512, corner_radius=10)
        self.caja_galaxia.grid(row=1, column=0, padx=15, pady=15, sticky="nsew")

        self.caja_grafica = ctk.CTkLabel(self.frame_der, text="Esperando datos", fg_color="#121212", 
                                        width=512, height=512, corner_radius=10)
        self.caja_grafica.grid(row=1, column=1, padx=15, pady=15, sticky="nsew")

    def ejecutar_generacion(self):
        self.btn_generar.configure(text="Generando...", state="disabled")
        self.update() # Forzamos el refresco de la UI para que se vea el "Generando..."
        
        # 1. Recoger valores físicos actuales de los sliders
        spec = GalaxySpec(
            escala_kpc_px=self.diccionario_sliders["escala_kpc_px"].get(),
            log_ms=self.diccionario_sliders["log_ms"].get(),
            ea_gyr=self.diccionario_sliders["ea_gyr"].get(),
            radio_p_arcsec=self.diccionario_sliders["radio_p_arcsec"].get()
        )
        guidance_act = self.diccionario_sliders["guidance"].get()

        # 2. Generar el tensor de la galaxia en memoria usando la U-Net
        try:
            tensor_galaxia = generate_galaxy(
                spec=spec,
                unet=self.unet,
                projector=self.projector,
                noise_scheduler=self.noise_scheduler,
                variables=self.variables,
                norm_stats=self.norm_stats,
                device=self.device,
                guidance_scale=guidance_act,
                seed=None, # Semilla aleatoria para que cada pulsación sea única
                img_size=128
            )

            # 3. Convertir Tensor [-1, 1] a PIL Image RGB [0, 255]
            img_pil = tensor_to_pil(tensor_galaxia).convert("RGB")
            
            # Dibujar borde blanco decorativo
            draw_galaxia = ImageDraw.Draw(img_pil)
            draw_galaxia.rectangle([0, 0, img_pil.width-1, img_pil.height-1], outline="white", width=2)
            
            # Ampliarla visualmente para la interfaz (a 512x512)
            ctk_img = ctk.CTkImage(light_image=img_pil, size=(512, 512))
            self.caja_galaxia.configure(image=ctk_img, text="", fg_color="transparent")

            # 4. Extraer la analítica RGB de los brazos espirales
            self.analizar_y_plotear_rgb(img_pil)

        except Exception as e:
            self.caja_galaxia.configure(text=f"Error en la generación:\n{e}", text_color="red")
            print(e)

        self.btn_generar.configure(text="Generar\n galaxia", state="normal")

    def analizar_y_plotear_rgb(self, img_pil):
        """Extrae el perfil radial a lo largo de los brazos y genera la gráfica."""
        img_array = np.array(img_pil)
        h, w = img_array.shape[:2]
        cy, cx = h // 2, w // 2
        
        # Filtramos un poco el fondo oscuro para que la máscara coja la galaxia
        bg_rgb = img_array[0, 0] 
        mascara_galaxia = ~np.all(np.abs(img_array - bg_rgb) < 10, axis=-1)
        
        r_vals, g_vals, b_vals = [], [], []
        
        for gy_rel, gx_rel in self.espiral:
            py = int(cy - (self.LADO_P//2) + (gy_rel * self.LADO_P))
            px = int(cx - (self.LADO_P//2) + (gx_rel * self.LADO_P))
            
            if 0 <= py <= h - self.LADO_P and 0 <= px <= w - self.LADO_P:
                if np.any(mascara_galaxia[py:py+self.LADO_P, px:px+self.LADO_P]):
                    parche = img_array[py:py+self.LADO_P, px:px+self.LADO_P]
                    r_vals.append(np.mean(parche[:,:,0]))
                    g_vals.append(np.mean(parche[:,:,1]))
                    b_vals.append(np.mean(parche[:,:,2]))

        if len(r_vals) > 0:
            self.mostrar_grafico_rgb(r_vals, g_vals, b_vals)
        else:
            self.caja_grafica.configure(image="", text="Galaxia muy difusa\npara el muestreo.")

    def mostrar_grafico_rgb(self, r, g, b):
        """Dibuja el histograma RGB en memoria y lo pone en la interfaz."""
        fig = Figure(figsize=(5.12, 5.12), dpi=100, facecolor='#121212')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#121212')
        
        fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.12)
        
        x = np.arange(len(r))
        bw = 0.3
        
        ax.bar(x - bw, r, width=bw, color="#EB1224", label='R', alpha=0.9)
        ax.bar(x, g, width=bw, color="#13BB5F", label='G', alpha=0.9)
        ax.bar(x + bw, b, width=bw, color="#1E9AE7", label='B', alpha=0.9)
        
        ax.set_xlabel('Nº Parche desde el centro', color='white')
        ax.set_ylabel('Intensidad fotométrica', color='white')
        ax.tick_params(axis='both', colors='white')
        ax.grid(True, linestyle='--', alpha=0.2)
        
        # Volcar plot a memoria RAM en vez de a disco
        buf = io.BytesIO()
        fig.savefig(buf, format='png', facecolor=fig.get_facecolor())
        buf.seek(0)
        
        plot_img = Image.open(buf)
        
        # Borde blanco a la gráfica
        draw_plot = ImageDraw.Draw(plot_img)
        draw_plot.rectangle([0, 0, plot_img.width-1, plot_img.height-1], outline="white", width=2)
        
        ctk_plot = ctk.CTkImage(light_image=plot_img, size=(512, 512))
        self.caja_grafica.configure(image=ctk_plot, text="", fg_color="transparent")

if __name__ == "__main__":
    app = GeneradorGalaxiasApp()
    app.mainloop()