1. Instalar Python 3.10 o superior.

2. Instalar dependencias:
   pip install -r requirements.txt

3. Asegúrese de tener los archivos:
   - Blackbox.py
   - modelo entrenado en la carpeta correspondiente

4. Ejecución:
   python Interfas.py

5. El programa generará:
   - Gráfica de puntos originales de la Blackbox
   - Interpolación con spline cúbico
   - Ajuste por mínimos cuadrados usando modelo sen(x)/x
   - Comparación entre métodos
   - Gráfica de error relativo

6. Los resultados numéricos se mostraran en la grafica de error relativo.
