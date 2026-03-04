import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Экспериментальные данные
data = np.array([
    [273+28.08, 2.76],  [273+28.50, 2.717], [273+29.14, 2.712],  [273+30.16, 2.673], [273+31.10, 2.688],
    [273+40.28, 2.806],  [273+40.68, 2.708], [273+41.19, 2.709 ],  [273+42.10, 2.719], [273+43.10, 2.686],
    [273+50.26, 2.753], [273+50.54, 2.721], [273+50.99, 2.721], [273+51.79, 2.719], [273+52.65, 2.730]
])

T_cp = data[:, 0]  # градусы целься
K_exp = data[:, 1]  # 10^-2 Вт/(м·К)

# Справочные данные
"""T_ref = np.array([270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380])
K_ref = np.array([2.38, 2.46, 2.54, 2.62, 2.69, 2.77, 2.85, 2.92, 3.00, 3.08, 3.15, 3.23])"""
T_ref = np.array([273+28.08, 273+28.50, 273+29.14, 273+30.16, 273+31.10, 273+40.28, 273+40.68, 273+41.19, 273+42.10, 273+43.10, 273+50.26, 273+50.54, 273+50.99, 273+51.79, 273+52.65])
K_ref = np.array([2.625, 2.629, 2.634, 2.642, 2.649, 2.721, 2.724, 2.728, 2.735, 2.743, 2.799, 2.801, 2.805, 2.811, 2.817])

# МНК-аппроксимация
slope, intercept, _, _, std_err = linregress(T_cp, K_exp)
n = len(T_cp)
mean_x = np.mean(T_cp)
Sxx = np.sum((T_cp - mean_x)**2)

# Погрешности коэффициентов
SE_slope = std_err / np.sqrt(Sxx)
SE_intercept = std_err * np.sqrt(np.mean(T_cp**2)/n)

# Построение графика
plt.figure(figsize=(10, 7), dpi=150)  # A5 размер

plt.scatter(T_cp, K_exp, color='blue', s=40,
           label='Экспериментальные данные')
plt.scatter(T_ref, K_ref, color='red', marker='s', s=40,
           label='Справочные данные')

x_fit = np.linspace(min(T_cp), max(T_cp), 100)
plt.plot(x_fit, intercept + slope*x_fit, 'k--',
        label=f'МНК: κ = ({intercept:.2f}±{SE_intercept:.2f}) + ({slope:.4f}±{SE_slope:.4f})·T')

# Оформление
plt.xlabel('Средняя температура, $T_{ср}$ (K)', fontsize=14)
plt.ylabel('Теплопроводность, $κ$ ($10^{-2}$ Вт/(м·К))', fontsize=14)
plt.xticks(np.arange(280, 350, 10))
plt.yticks(np.arange(2.5, 3, 0.1))
plt.grid(True, linestyle='--', alpha=0.3)

plt.legend(loc='upper left', fontsize=10)
plt.title('Зависимость теплопроводности от температуры', fontsize=16)
plt.tight_layout()
plt.savefig('thermal_conductivity_final.png', dpi=300, bbox_inches='tight')
plt.show()