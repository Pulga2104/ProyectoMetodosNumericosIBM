import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from Blackbox import load_model, predict_batch

model = load_model()

def obtener_frontera(model, x1):
    x2_min = -0.5
    x2_max = 5.0
    samples = int(1500 / (1 + 0.6*x1)) + 400
    xs = np.linspace(x2_min, x2_max, samples)
    x1s = np.full_like(xs, x1)
    vals = np.round(predict_batch(model, x1s.tolist(), xs.tolist())).astype(int)
    for i in range(1, len(vals)):
        if vals[i-1] == 1 and vals[i] == 0:
            return xs[i]
    return np.nan

x1_values = np.linspace(0.1, 15, 400)
x2_values = np.array([obtener_frontera(model, x1) for x1 in x1_values])

mask = ~np.isnan(x2_values)
x1_clean = x1_values[mask]
x2_clean = x2_values[mask]

plt.figure(figsize=(18,5))
plt.scatter(x1_clean, x2_clean, s=12, color="red")
plt.title("Puntos obtenidos de la Blackbox")
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid(True, alpha=0.35)
plt.show()

spline = CubicSpline(x1_clean, x2_clean)

x_fine = np.linspace(x1_clean.min(), x1_clean.max(), 2000)
y_spline = spline(x_fine)
y_spline_data = spline(x1_clean)

def modelo_senx_sobre_x(x, A, w):
    z = w * x
    y = np.empty_like(x, dtype=float)
    y[z == 0] = A
    mask = ~(z == 0)
    y[mask] = A * np.sin(z[mask]) / (z[mask])
    return y

p0 = [1.0, 10.0]

params_sinc, cov_sinc = curve_fit(
    modelo_senx_sobre_x,
    x1_clean,
    x2_clean,
    p0=p0,
    maxfev=20000
)

A_opt, w_opt = params_sinc

y_fit_sinc = modelo_senx_sobre_x(x_fine, A_opt, w_opt)
y_pred_sinc = modelo_senx_sobre_x(x1_clean, A_opt, w_opt)

residuales_sinc = x2_clean - y_pred_sinc
SSR_sinc = np.sum(residuales_sinc**2)
RMSE_sinc = np.sqrt(np.mean(residuales_sinc**2))

def error_relativo(real, approx):
    return np.abs(real - approx) / (np.abs(real) + 1e-12)

err_spline = error_relativo(x2_clean, y_spline_data)
err_sinc   = error_relativo(x2_clean, y_pred_sinc)

plt.figure(figsize=(18,6))
plt.scatter(x1_clean, x2_clean, s=12, color="blue")
plt.plot(x_fine, y_fit_sinc, color="orange", linewidth=2.5)

for i in range(len(x1_clean)):
    plt.plot(
        [x1_clean[i], x1_clean[i]],
        [x2_clean[i], y_pred_sinc[i]],
        color="red", alpha=0.5, linewidth=1
    )

plt.text(
    0.5, 0.8,
    r"$f(x) = 1.0448 \cdot \dfrac{\sin(9.5182\,x)}{9.5182\,x}$" "\n"
    f"SSR = {SSR_sinc:.4f}\n"
    f"RMSE = {RMSE_sinc:.4f}",
    fontsize=12,
    color="black"
)

plt.title("Ajuste tipo sen(x)/x — Mínimos Cuadrados")
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid(True, alpha=0.35)
plt.show()

plt.figure(figsize=(18,6))
plt.scatter(x1_clean, x2_clean, s=12, color="black")
plt.plot(x_fine, y_spline, linewidth=2, label="Spline Cúbico")
plt.plot(x_fine, y_fit_sinc, linewidth=2.3, label="Ajuste sen(x)/x")
plt.title("Comparación: Spline Cúbico vs Ajuste sen(x)/x")
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid(True, alpha=0.35)
plt.legend()
plt.show()

plt.figure(figsize=(18,6))
plt.plot(x1_clean, err_spline, label="Error Spline Cúbico")
plt.plot(x1_clean, err_sinc, label="Error Modelo sen(x)/x")
plt.yscale("log")
plt.title("Error Relativo — Spline Cúbico vs Modelo sen(x)/x")
plt.xlabel("x1")
plt.ylabel("Error relativo (log)")
plt.grid(True, alpha=0.35)
plt.legend()
plt.show()

print("\n=========== ERROR PROMEDIO ===========")
print("Spline Cúbico: ", np.mean(err_spline))
print("sen(x)/x:      ", np.mean(err_sinc))

print("\n=========== ERROR MÁXIMO ===========")
print("Spline Cúbico: ", np.max(err_spline))
print("sen(x)/x:      ", np.max(err_sinc))
