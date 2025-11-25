import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from Blackbox import load_model, predict_batch
import tkinter as tk
from tkinter import ttk

class BlackboxAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Análisis Blackbox - Visualización Interactiva")
        self.root.geometry("1400x800")
        self.root.configure(bg="#1e1e2e")
        self.model = load_model()
        self.procesar_datos()
        self.crear_interfaz()
        
    def obtener_frontera(self, x1):
        x2_min = -0.5
        x2_max = 5.0
        samples = int(1500 / (1 + 0.6*x1)) + 400
        xs = np.linspace(x2_min, x2_max, samples)
        x1s = np.full_like(xs, x1)
        vals = np.round(predict_batch(self.model, x1s.tolist(), xs.tolist())).astype(int)
        for i in range(1, len(vals)):
            if vals[i-1] == 1 and vals[i] == 0:
                return xs[i]
        return np.nan
    
    def procesar_datos(self):
        x1_values = np.linspace(0.1, 15, 400)
        x2_values = np.array([self.obtener_frontera(x1) for x1 in x1_values])
        mask = ~np.isnan(x2_values)
        self.x1_clean = x1_values[mask]
        self.x2_clean = x2_values[mask]
        self.spline = CubicSpline(self.x1_clean, self.x2_clean)
        self.x_fine = np.linspace(self.x1_clean.min(), self.x1_clean.max(), 2000)
        self.y_spline = self.spline(self.x_fine)
        self.y_spline_data = self.spline(self.x1_clean)
        
        def modelo_senx_sobre_x(x, A, w):
            z = w * x
            y = np.empty_like(x, dtype=float)
            y[z == 0] = A
            mask = ~(z == 0)
            y[mask] = A * np.sin(z[mask]) / (z[mask])
            return y
        
        p0 = [1.0, 10.0]
        params_sinc, _ = curve_fit(modelo_senx_sobre_x, self.x1_clean, self.x2_clean, p0=p0, maxfev=20000)
        
        self.A_opt, self.w_opt = params_sinc
        self.y_fit_sinc = modelo_senx_sobre_x(self.x_fine, self.A_opt, self.w_opt)
        self.y_pred_sinc = modelo_senx_sobre_x(self.x1_clean, self.A_opt, self.w_opt)
        
        residuales_sinc = self.x2_clean - self.y_pred_sinc
        self.SSR_sinc = np.sum(residuales_sinc**2)
        self.RMSE_sinc = np.sqrt(np.mean(residuales_sinc**2))
        
        def error_relativo(real, approx):
            return np.abs(real - approx) / (np.abs(real) + 1e-12)
        
        self.err_spline = error_relativo(self.x2_clean, self.y_spline_data)
        self.err_sinc = error_relativo(self.x2_clean, self.y_pred_sinc)
    
    def crear_interfaz(self):
        frame_botones = tk.Frame(self.root, bg="#2d2d44", pady=15)
        frame_botones.pack(side=tk.TOP, fill=tk.X)
        titulo = tk.Label(frame_botones, text="Análisis de Blackbox - Interpolación y Ajuste",
                         font=("Arial", 18, "bold"), bg="#2d2d44", fg="#ffffff")
        titulo.pack(pady=(0, 10))
        frame_btn = tk.Frame(frame_botones, bg="#2d2d44")
        frame_btn.pack()
        
        botones = [
            ("📊 Datos Blackbox", self.mostrar_datos_blackbox),
            ("📈 Ajuste sen(x)/x", self.mostrar_ajuste_sinc),
            ("🔄 Comparación", self.mostrar_comparacion),
            ("📉 Error Relativo", self.mostrar_error_relativo)
        ]
        
        for texto, comando in botones:
            btn = tk.Button(frame_btn, text=texto, command=comando,
                          font=("Arial", 11, "bold"), bg="#5865f2", fg="white",
                          padx=20, pady=10, relief=tk.FLAT, cursor="hand2")
            btn.pack(side=tk.LEFT, padx=5)
            btn.bind("<Enter>", lambda e, b=btn: b.config(bg="#4752c4"))
            btn.bind("<Leave>", lambda e, b=btn: b.config(bg="#5865f2"))
        
        self.frame_grafica = tk.Frame(self.root, bg="#1e1e2e")
        self.frame_grafica.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=20, pady=10)
        self.mostrar_datos_blackbox()
    
    def limpiar_grafica(self):
        for widget in self.frame_grafica.winfo_children():
            widget.destroy()
    
    def mostrar_datos_blackbox(self):
        self.limpiar_grafica()
        fig, ax = plt.subplots(figsize=(14, 6))
        fig.patch.set_facecolor('#1e1e2e')
        ax.set_facecolor('#2d2d44')
        ax.scatter(self.x1_clean, self.x2_clean, s=12, color="red", alpha=0.7)
        ax.set_title("Puntos obtenidos de la Blackbox", fontsize=16, color='white', pad=20)
        ax.set_xlabel("x1", fontsize=12, color='white')
        ax.set_ylabel("x2", fontsize=12, color='white')
        ax.grid(True, alpha=0.3, color='gray')
        ax.tick_params(colors='white')
        canvas = FigureCanvasTkAgg(fig, master=self.frame_grafica)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def mostrar_ajuste_sinc(self):
        self.limpiar_grafica()
        fig, ax = plt.subplots(figsize=(14, 6))
        fig.patch.set_facecolor('#1e1e2e')
        ax.set_facecolor('#2d2d44')
        
        ax.scatter(self.x1_clean, self.x2_clean, s=12, color="blue", alpha=0.6)
        ax.plot(self.x_fine, self.y_fit_sinc, color="orange", linewidth=2.5)
        
        for i in range(0, len(self.x1_clean), 5):
            ax.plot([self.x1_clean[i], self.x1_clean[i]],
                   [self.x2_clean[i], self.y_pred_sinc[i]],
                   color="red", alpha=0.4, linewidth=1)
        
        texto = (f"$f(x) = {self.A_opt:.4f} \\cdot \\frac{{\\sin({self.w_opt:.4f}x)}}"
                f"{{{self.w_opt:.4f}x}}$\n"
                f"SSR = {self.SSR_sinc:.4f}\n"
                f"RMSE = {self.RMSE_sinc:.4f}")
        
        ax.text(0.5, 0.8, texto, fontsize=11, color="white",
               bbox=dict(boxstyle='round', facecolor='#2d2d44', alpha=0.8))
        
        ax.set_title("Ajuste tipo sen(x)/x — Mínimos Cuadrados", fontsize=16, color='white', pad=20)
        ax.set_xlabel("x1", fontsize=12, color='white')
        ax.set_ylabel("x2", fontsize=12, color='white')
        ax.grid(True, alpha=0.3, color='gray')
        ax.tick_params(colors='white')
        
        canvas = FigureCanvasTkAgg(fig, master=self.frame_grafica)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def mostrar_comparacion(self):
        self.limpiar_grafica()
        fig, ax = plt.subplots(figsize=(14, 6))
        fig.patch.set_facecolor('#1e1e2e')
        ax.set_facecolor('#2d2d44')
        
        ax.scatter(self.x1_clean, self.x2_clean, s=12, color="black")
        ax.plot(self.x_fine, self.y_spline, linewidth=2, color='#3b82f6')
        ax.plot(self.x_fine, self.y_fit_sinc, linewidth=2.3, color='#fb923c')
        
        ax.set_title("Spline Cúbico vs Ajuste sen(x)/x", fontsize=16, color='white', pad=20)
        ax.set_xlabel("x1", fontsize=12, color='white')
        ax.set_ylabel("x2", fontsize=12, color='white')
        ax.grid(True, alpha=0.3, color='gray')
        ax.tick_params(colors='white')
        
        canvas = FigureCanvasTkAgg(fig, master=self.frame_grafica)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def mostrar_error_relativo(self):
        self.limpiar_grafica()
        fig, ax = plt.subplots(figsize=(14, 6))
        fig.patch.set_facecolor('#1e1e2e')
        ax.set_facecolor('#2d2d44')
        
        ax.plot(self.x1_clean, self.err_spline, linewidth=2, color='#3b82f6')
        ax.plot(self.x1_clean, self.err_sinc, linewidth=2, color='#fb923c')
        ax.set_yscale("log")
        
        texto = (f"Spline Cúbico:\n"
                f"  Promedio: {np.mean(self.err_spline):.6f}\n"
                f"  Máximo: {np.max(self.err_spline):.6f}\n\n"
                f"sen(x)/x:\n"
                f"  Promedio: {np.mean(self.err_sinc):.6f}\n"
                f"  Máximo: {np.max(self.err_sinc):.6f}")
        
        ax.text(0.02, 0.98, texto, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', color='white',
               bbox=dict(boxstyle='round', facecolor='#2d2d44', alpha=0.8))
        
        ax.set_title("Error Relativo — Spline Cúbico vs sen(x)/x", fontsize=16, color='white', pad=20)
        ax.set_xlabel("x1", fontsize=12, color='white')
        ax.set_ylabel("Error relativo (log)", fontsize=12, color='white')
        ax.grid(True, alpha=0.3, color='gray')
        ax.tick_params(colors='white')
        
        canvas = FigureCanvasTkAgg(fig, master=self.frame_grafica)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = BlackboxAnalysisGUI(root)
    root.mainloop()
