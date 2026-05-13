import customtkinter as ctk
from PIL import Image, ImageDraw
import numpy as np
import time
import os
import io
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class GeneradorGalaxiasApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Generador de galaxias (TFG) - Universidad de Alicante")
        self.geometry("1200x700")
        
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.LADO_P = 8
        self.TOTAL_PARCHES_MAX = 64*64
        self.espiral = self.generar_espiral(self.TOTAL_PARCHES_MAX)

        self.crear_panel_izquierdo()
        self.crear_panel_derecho()

    def generar_espiral(self, n_max):
        """Genera las coordenadas relativas en espiral para el muestreo[cite: 3]"""
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
        # ... (Tu código actual de Sliders y Logo se mantiene igual)[cite: 2]
        self.frame_izq = ctk.CTkFrame(self, width=320, corner_radius=0, fg_color="#ffffff")
        self.frame_izq.grid(row=0, column=0, sticky="nsew")
        self.frame_izq.grid_rowconfigure(11, weight=1)

        # Logo UA
        try:
            ruta_logo = os.path.join(os.path.dirname(__file__), "logo_ua.jpeg")
            img_logo = ctk.CTkImage(light_image=Image.open(ruta_logo), size=(190, 190))
            self.label_logo = ctk.CTkLabel(self.frame_izq, image=img_logo, text="")
        except:
            self.label_logo = ctk.CTkLabel(self.frame_izq, text="LOGO UA")
        self.label_logo.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Título
        self.label_titulo = ctk.CTkLabel(self.frame_izq, text="Red neuronal U-Net", 
                                         font=ctk.CTkFont(size=20, weight="bold"), text_color="#00A390")
        self.label_titulo.grid(row=1, column=0, padx=20, pady=(0, 20))

        # Definición de Sliders simplificada para el ejemplo
        self.crear_slider("Escala kpc-píxel", 2)
        self.crear_slider("Masa (LOG_MS)", 4)
        self.crear_slider("Formación (SFR)", 6)
        self.crear_slider("Edad (EA)", 8)

        # Botón Generar
        self.btn_generar = ctk.CTkButton(self.frame_izq, text="Generar\n galaxia", 
                                         fg_color="#00A390", hover_color="#007A6C",
                                         font=ctk.CTkFont(size=22, weight="bold"), 
                                         width=210, height=100, corner_radius=0,
                                         command=self.ejecutar_generacion)
        self.btn_generar.grid(row=12, column=0, pady=(20, 30))

    def crear_slider(self, texto, row):
        label = ctk.CTkLabel(self.frame_izq, text=f"{texto}: 0.50", font=ctk.CTkFont(size=14, weight="bold"), text_color="black")
        label.grid(row=row, column=0, padx=35, pady=(10, 0), sticky="w")
        slider = ctk.CTkSlider(self.frame_izq, from_=0.0, to=1.0, width=220, button_color="#00A390")
        slider.grid(row=row+1, column=0, padx=20, pady=(5, 10))
        return slider

    def crear_panel_derecho(self):
        self.frame_der = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_der.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.frame_der.grid_columnconfigure((0, 1), weight=1)
        self.frame_der.grid_rowconfigure(1, weight=1)

        self.lbl_img_tit = ctk.CTkLabel(self.frame_der, text="Galaxia sintética", font=ctk.CTkFont(size=22, weight="bold"))
        self.lbl_img_tit.grid(row=0, column=0, pady=10)
        
        self.lbl_rgb_tit = ctk.CTkLabel(self.frame_der, text="Análisis RGB", font=ctk.CTkFont(size=22, weight="bold"))
        self.lbl_rgb_tit.grid(row=0, column=1, pady=10)

        self.caja_galaxia = ctk.CTkLabel(self.frame_der, text="", fg_color="#121212", 
                                 width=512, height=512, corner_radius=10)

        self.caja_galaxia.grid(row=1, column=0, padx=15, pady=15, sticky="nsew")

        self.caja_grafica = ctk.CTkLabel(self.frame_der, text="", fg_color="#121212", 
                                        width=512, height=512, corner_radius=10)
        self.caja_grafica.grid(row=1, column=1, padx=15, pady=15, sticky="nsew")

    def ejecutar_generacion(self):
        self.btn_generar.configure(text="Generando...", state="disabled")
        self.update()
        
        ruta_test = os.path.join(os.path.dirname(__file__), "1237648704055017848.png")
        
        if os.path.exists(ruta_test):
            img_pil = Image.open(ruta_test).convert("RGB")
            
            # --- NUEVO: DIBUJAR BORDE BLANCO A LA GALAXIA ---
            draw_galaxia = ImageDraw.Draw(img_pil)
            draw_galaxia.rectangle([0, 0, img_pil.width-1, img_pil.height-1], outline="white", width=3)
            # ------------------------------------------------
            
            ctk_img = ctk.CTkImage(light_image=img_pil, size=(512, 512))
            self.caja_galaxia.configure(image=ctk_img, text="", fg_color="transparent")

            # --- RECUPERAMOS EL ANÁLISIS RGB QUE FALTABA ---
            img_array = np.array(img_pil)
            h, w = img_array.shape[:2]
            cy, cx = h // 2, w // 2
            
            # Detectamos el fondo para crear la máscara (píxel superior izquierdo)
            bg_rgb = img_array[0, 0] 
            mascara_galaxia = ~np.all(img_array == bg_rgb, axis=-1)
            
            r_vals, g_vals, b_vals = [], [], []
            
            # Muestreo basado en tu lógica de parches
            for gy_rel, gx_rel in self.espiral:
                py = int(cy - (self.LADO_P//2) + (gy_rel * self.LADO_P))
                px = int(cx - (self.LADO_P//2) + (gx_rel * self.LADO_P))
                
                if 0 <= py <= h - self.LADO_P and 0 <= px <= w - self.LADO_P:
                    if np.any(mascara_galaxia[py:py+self.LADO_P, px:px+self.LADO_P]):
                        parche = img_array[py:py+self.LADO_P, px:px+self.LADO_P]
                        r_vals.append(np.mean(parche[:,:,0]))
                        g_vals.append(np.mean(parche[:,:,1]))
                        b_vals.append(np.mean(parche[:,:,2]))
            # ------------------------------------------------

            # 3. Generar el gráfico Matplotlib (ahora sí tiene datos)
            self.mostrar_grafico_rgb(r_vals, g_vals, b_vals)
        else:
            self.caja_galaxia.configure(text="Archivo 'test_galaxia.png'\n no encontrado", text_color="orange")

        self.btn_generar.configure(text="Generar\n galaxia", state="normal")

    def mostrar_grafico_rgb(self, r, g, b):
        fig = Figure(figsize=(5.12, 5.12), dpi=100, facecolor='#121212')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#121212')
        
        fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.12)
        
        x = np.arange(len(r))
        bw = 0.3
        
        ax.bar(x - bw, r, width=bw, color="#EB1224", label='R', alpha=0.9)
        ax.bar(x, g, width=bw, color="#13BB5F", label='G', alpha=0.9)
        ax.bar(x + bw, b, width=bw, color="#1E9AE7", label='B', alpha=0.9)
        
        ax.set_xlabel('Nº Parche', color='white')
        ax.set_ylabel('Intensidad', color='white')
        ax.tick_params(axis='both', colors='white')
        ax.grid(True, linestyle='--', alpha=0.2)
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', facecolor=fig.get_facecolor())
        buf.seek(0)
        plot_img = Image.open(buf)
        
        # --- NUEVO: DIBUJAR BORDE BLANCO A LA GRÁFICA ---
        draw_plot = ImageDraw.Draw(plot_img)
        draw_plot.rectangle([0, 0, plot_img.width-1, plot_img.height-1], outline="white", width=3)
        # ------------------------------------------------
        
        ctk_plot = ctk.CTkImage(light_image=plot_img, size=(512, 512))
        self.caja_grafica.configure(image=ctk_plot, text="", fg_color="transparent")

if __name__ == "__main__":
    app = GeneradorGalaxiasApp()
    app.mainloop()