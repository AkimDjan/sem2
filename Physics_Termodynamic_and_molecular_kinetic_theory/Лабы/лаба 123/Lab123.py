import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Данные калибровки
Tc = np.array([23.4, 26.8, 29.8, 32.2, 35.6, 37.9, 39.8, 41.4, 43.5, 45.5])  # °C
Rc = np.array([3.8259, 3.8666, 3.9139, 3.9405, 3.9996, 4.0241, 4.0515, 4.0787, 4.1015, 4.1296])  # Ом

# Погрешности измерений (примерные)
dT = 0.1  # Погрешность температуры, °C
dR = 0.005  # Погрешность сопротивления, Ом

# Линейная регрессия
slope, intercept, _, _, std_err = linregress(Tc, Rc)
R0 = intercept
alpha = slope

# Расчет погрешностей коэффициентов
n = len(Tc)
mean_T = np.mean(Tc)
Sxx = np.sum((Tc - mean_T)**2)
std_err_R0 = std_err * np.sqrt(1/n + mean_T**2/Sxx)
std_err_alpha = std_err / np.sqrt(Sxx)

# Создание фигуры
plt.figure(figsize=(16, 10))  # Размер A5 в см

# Построение экспериментальных точек
plt.errorbar(Tc, Rc, xerr=dT, yerr=dR, fmt='o', color='blue',
             markersize=5, capsize=4, capthick=1, elinewidth=1,
             label='Экспериментальные точки')

# Линия регрессии
plt.plot(Tc, R0 + alpha*Tc, 'r-', linewidth=2, label='Линейная аппроксимация')

# Настройка осей
plt.xlabel('Температура проволочки, T (°C)', fontsize=14)
plt.ylabel('Сопротивление, R (Ом)', fontsize=14)
plt.xticks(np.arange(30, 61, 5))
plt.yticks(np.arange(4.0, 4.6, 0.1))
plt.grid(True, linestyle='--', alpha=0.3)

# Легенда и параметры
plt.legend(loc='lower right', fontsize=14)
plt.title('Калибровочная зависимость сопротивления\nпроволочки от температуры', fontsize=17)

# Текст с параметрами (без LaTeX-нотации)
param_text = (f'R(T) = R₀ + α·T\n'
              f'R₀ = {R0:.3f} ± {std_err_R0:.3f} Ом\n'
              f'α = {alpha:.5f} ± {std_err_alpha:.5f} Ом/°C')
plt.text(0.35, 0.15, param_text, transform=plt.gca().transAxes,
         fontsize=14, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('calibration_plot.png', dpi=300, bbox_inches='tight')
plt.show()

plt.close()

# Вывод погрешностей
print("Погрешности коэффициентов:")
print(f"ΔR₀ = {std_err_R0:.5f} Ом")
print(f"Δα = {std_err_alpha:.5f} Ом/°C")