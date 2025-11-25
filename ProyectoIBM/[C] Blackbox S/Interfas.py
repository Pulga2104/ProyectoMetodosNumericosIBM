import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.optimize import curve_fit
from scipy import ndimage
from Blackbox import load_model, predict_batch
import tkinter as tk
from tkinter import ttk, messagebox
import threading

model = load_model()

X_global = None
Y_global = None

# GENERACION DE PUNTOS

def generar_puntos(progress_callback):
    x1_min, x1_max = 0.1, 8.0
    x2_min, x2_max = -1.0, 5.0
    n_x1, n_x2 = 1000, 1000
    ancho_banda = 3
    
    progress_callback("Creando malla...", 10)
    
    x1_vals = np.linspace(x1_min, x1_max, n_x1)
    x2_vals = np.linspace(x2_min, x2_max, n_x2)
    X1_grid, X2_grid = np.meshgrid(x1_vals, x2_vals)
    X1_flat = X1_grid.flatten()
    X2_flat = X2_grid.flatten()
    
    progress_callback("Prediciendo clases...", 30)
    
    batch_size = 25000
    predictions = []
    for i in range(0, len(X1_flat), batch_size):
        batch_x1 = X1_flat[i:i+batch_size].tolist()
        batch_x2 = X2_flat[i:i+batch_size].tolist()
        batch_pred = predict_batch(model, batch_x1, batch_x2)
        predictions.extend(batch_pred)
    
    predictions = np.round(np.array(predictions)).astype(int).reshape(X1_grid.shape)
    
    progress_callback("Detectando bordes...", 60)
    
    bordes = np.zeros_like(predictions, dtype=bool)
    for i in range(1, n_x2-1):
        for j in range(1, n_x1-1):
            center = predictions[i, j]
            neighbors = [
                predictions[i-1, j-1], predictions[i-1, j], predictions[i-1, j+1],
                predictions[i, j-1], predictions[i, j+1],
                predictions[i+1, j-1], predictions[i+1, j], predictions[i+1, j+1]
            ]
            if any(n != center for n in neighbors):
                bordes[i, j] = True
    
    progress_callback("Expandiendo banda...", 80)
    
    estructura = ndimage.generate_binary_structure(2, 2)
    banda = ndimage.binary_dilation(bordes, structure=estructura, iterations=ancho_banda)
    
    indices = np.where(banda)
    X = X1_grid[indices]
    Y = X2_grid[indices]
    
    progress_callback(f"Completado: {len(X):,} puntos", 100)
    return X, Y



# AJUSTE DE MODELOS

def ajustar_modelos(X, Y, progress_callback):
    progress_callback("Ajustando sen(x)/x...", 30)
    
    def modelo_sinc(x, A, w, offset):
        z = w * x
        y = np.empty_like(x, dtype=float)
        y[z == 0] = A + offset
        mask = ~(z == 0)
        y[mask] = A * np.sin(z[mask]) / (z[mask]) + offset
        return y
    
    params_sinc, _ = curve_fit(modelo_sinc, X, Y, p0=[1.0, 8.0, 0.0], maxfev=40000)
    A, w, offset = params_sinc
    
    progress_callback("Ajustando polinomio...", 60)
    
    grado = 10
    coef_poly = np.polyfit(X, Y, grado)
    poly_model = np.poly1d(coef_poly)
    
    x_fine = np.linspace(min(X), max(X), 4000)
    y_sinc = modelo_sinc(x_fine, *params_sinc)
    y_poly = poly_model(x_fine)
    
    y_sinc_pred = modelo_sinc(X, *params_sinc)
    y_poly_pred = poly_model(X)
    
    progress_callback("Calculando errores...", 80)
    err_sinc = np.abs(Y - y_sinc_pred) / (np.abs(Y) + 1e-12)
    err_poly = np.abs(Y - y_poly_pred) / (np.abs(Y) + 1e-12)
    
    progress_callback("Completado", 100)
    
    return x_fine, y_sinc, y_poly, grado, params_sinc, coef_poly, err_sinc, err_poly


# INTERFAZ GRAFICA

class Aplicacion:
    def __init__(self, root):
        self.root = root
        self.root.title("Análisis Blackbox")
        self.root.geometry("1400x700")
        self.root.configure(bg='white')
        
        self.crear_controles()
        self.crear_pestanas()
    
    def crear_controles(self):
        frame = tk.Frame(self.root, bg='white')
        frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        tk.Label(frame, text="Análisis de Frontera Blackbox", 
                font=('Arial', 14, 'bold'), bg='white').pack(pady=5)
        
        btn_frame = tk.Frame(frame, bg='white')
        btn_frame.pack(pady=5)
        
        self.btn_generar = tk.Button(btn_frame, text="Generar Puntos", 
                                     command=self.generar, width=20, height=2)
        self.btn_generar.pack(side=tk.LEFT, padx=5)
        
        self.btn_ajustar = tk.Button(btn_frame, text="Ajustar Modelos", 
                                     command=self.ajustar, width=20, height=2, state=tk.DISABLED)
        self.btn_ajustar.pack(side=tk.LEFT, padx=5)
        
        self.progress = ttk.Progressbar(frame, length=400, mode='determinate')
        self.progress.pack(pady=5)
        
        self.status = tk.Label(frame, text="Presiona 'Generar Puntos' para comenzar", 
                              bg='white', font=('Arial', 10))
        self.status.pack(pady=5)
    
    def crear_pestanas(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        frame1 = tk.Frame(self.notebook, bg='white')
        self.notebook.add(frame1, text="Nube de Puntos")
        
        self.fig1 = plt.Figure(figsize=(12, 6), facecolor='white')
        self.ax1 = self.fig1.add_subplot(111)
        self.ax1.set_xlabel('x1')
        self.ax1.set_ylabel('x2')
        self.ax1.set_title('Puntos de Frontera')
        self.ax1.grid(True, alpha=0.3)
        
        canvas1 = FigureCanvasTkAgg(self.fig1, frame1)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(canvas1, frame1)
        
        frame2 = tk.Frame(self.notebook, bg='white')
        self.notebook.add(frame2, text="Comparación de Modelos")
        
        self.fig2 = plt.Figure(figsize=(12, 6), facecolor='white')
        self.ax2 = self.fig2.add_subplot(111)
        self.ax2.set_xlabel('x1')
        self.ax2.set_ylabel('x2')
        self.ax2.set_title('Ajuste de Modelos')
        self.ax2.grid(True, alpha=0.3)
        
        canvas2 = FigureCanvasTkAgg(self.fig2, frame2)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(canvas2, frame2)
        
        frame3 = tk.Frame(self.notebook, bg='white')
        self.notebook.add(frame3, text="Resultados")
        
        self.text_resultados = tk.Text(frame3, font=('Courier', 11), bg='white', 
                                       wrap=tk.WORD, padx=20, pady=20)
        self.text_resultados.pack(fill=tk.BOTH, expand=True)
        self.text_resultados.insert('1.0', "Genera y ajusta los modelos para ver los resultados...")
    
    def actualizar_progreso(self, mensaje, valor):
        self.progress['value'] = valor
        self.status.config(text=mensaje)
        self.root.update_idletasks()
    
    def generar(self):
        self.btn_generar.config(state=tk.DISABLED)
        
        def tarea():
            global X_global, Y_global
            try:
                X_global, Y_global = generar_puntos(self.actualizar_progreso)
                self.root.after(0, self.mostrar_puntos)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                self.root.after(0, lambda: self.btn_generar.config(state=tk.NORMAL))
        
        threading.Thread(target=tarea, daemon=True).start()
    
    def mostrar_puntos(self):
        self.ax1.clear()
        self.ax1.scatter(X_global, Y_global, s=0.8, color='orange', alpha=0.7)
        self.ax1.set_xlabel('x1', fontsize=11)
        self.ax1.set_ylabel('x2', fontsize=11)
        self.ax1.set_title(f'Puntos de Frontera ({len(X_global):,} puntos)', fontsize=12)
        self.ax1.grid(True, alpha=0.3)
        self.fig1.canvas.draw()
        
        self.btn_ajustar.config(state=tk.NORMAL)
        messagebox.showinfo("Listo", f"Se generaron {len(X_global):,} puntos")
    
    def ajustar(self):
        if X_global is None:
            messagebox.showwarning("Error", "Primero genera los puntos")
            return
        
        self.btn_ajustar.config(state=tk.DISABLED)
        
        def tarea():
            try:
                resultados = ajustar_modelos(X_global, Y_global, self.actualizar_progreso)
                self.root.after(0, lambda: self.mostrar_ajuste(*resultados))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                self.root.after(0, lambda: self.btn_ajustar.config(state=tk.NORMAL))
        
        threading.Thread(target=tarea, daemon=True).start()
    
    def mostrar_ajuste(self, x_fine, y_sinc, y_poly, grado, params_sinc, coef_poly, err_sinc, err_poly):
        self.ax2.clear()
        self.ax2.scatter(X_global, Y_global, color='lightgray', s=0.5, alpha=0.4, 
                        label='Datos')
        self.ax2.plot(x_fine, y_sinc, color='orange', linewidth=3, label='sen(x)/x')
        self.ax2.plot(x_fine, y_poly, color='blue', linewidth=3, 
                     label=f'Polinomio grado {grado}')
        self.ax2.set_xlabel('x1', fontsize=11)
        self.ax2.set_ylabel('x2', fontsize=11)
        self.ax2.set_title('Comparación de Modelos', fontsize=12)
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend()
        self.fig2.canvas.draw()
        
        self.mostrar_resultados(params_sinc, coef_poly, grado, err_sinc, err_poly)
        
        messagebox.showinfo("Listo", "Modelos ajustados correctamente")
    
    def mostrar_resultados(self, params_sinc, coef_poly, grado, err_sinc, err_poly):
        A, w, offset = params_sinc
        
        texto = "=" * 80 + "\n"
        texto += "FUNCIONES AJUSTADAS\n"
        texto += "=" * 80 + "\n\n"
        
        texto += "MODELO SEN(X)/X:\n"
        texto += f"  f(x) = A * sin(w*x) / (w*x) + offset\n\n"
        texto += f"  Donde:\n"
        texto += f"    A      = {A:.6f}\n"
        texto += f"    w      = {w:.6f}\n"
        texto += f"    offset = {offset:.6f}\n\n"
        texto += f"  Función final:\n"
        texto += f"    f(x) = {A:.6f} * sin({w:.6f}*x) / ({w:.6f}*x) + {offset:.6f}\n\n"
        
        texto += "-" * 80 + "\n\n"
        
        texto += f"MODELO POLINOMIAL (Grado {grado}):\n"
        texto += f"  f(x) = "
        
        for i, c in enumerate(coef_poly):
            exp = grado - i
            if i == 0:
                texto += f"{c:.6f}*x^{exp}"
            else:
                signo = "+" if c >= 0 else ""
                if exp > 1:
                    texto += f" {signo} {c:.6f}*x^{exp}"
                elif exp == 1:
                    texto += f" {signo} {c:.6f}*x"
                else:
                    texto += f" {signo} {c:.6f}"
        
        texto += "\n\n"
        texto += "=" * 80 + "\n"
        texto += "ANÁLISIS DE ERROR\n"
        texto += "=" * 80 + "\n\n"
        
        texto += f"{'Métrica':<30} {'sen(x)/x':<20} {'Polinomial':<20}\n"
        texto += "-" * 80 + "\n"
        texto += f"{'Error promedio':<30} {np.mean(err_sinc):<20.10f} {np.mean(err_poly):<20.10f}\n"
        texto += f"{'Error máximo':<30} {np.max(err_sinc):<20.10f} {np.max(err_poly):<20.10f}\n"
        texto += f"{'Error mínimo':<30} {np.min(err_sinc):<20.10f} {np.min(err_poly):<20.10f}\n"
        texto += f"{'Error mediano':<30} {np.median(err_sinc):<20.10f} {np.median(err_poly):<20.10f}\n"
        texto += f"{'Desviación estándar':<30} {np.std(err_sinc):<20.10f} {np.std(err_poly):<20.10f}\n"
        
        texto += "\n" + "=" * 80 + "\n"
        
        mejor = "sen(x)/x" if np.mean(err_sinc) < np.mean(err_poly) else "Polinomial"
        texto += f"MEJOR MODELO (por error promedio): {mejor}\n"
        texto += "=" * 80 + "\n"
        
        self.text_resultados.delete('1.0', tk.END)
        self.text_resultados.insert('1.0', texto)


# EJECUTAR

if __name__ == "__main__":
    root = tk.Tk()
    app = Aplicacion(root)
    root.mainloop()