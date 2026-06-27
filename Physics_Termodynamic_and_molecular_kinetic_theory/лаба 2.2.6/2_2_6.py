import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Константы
g = 9.81  # ускорение свободного падения, м/с²

# ВСТАВЬТЕ СВОИ ДАННЫЕ ЗДЕСЬ:

# Температуры измерений (°C)
temperatures = np.array([22.02, 28.00, 36.01, 44.01])  # пример: 4 температуры

# Плотности жидкости для каждой температуры (г/см³)
liquid_densities = np.array([1.243, 1.2403, 1.2344, 1.2282]) * 1000  # переводим в кг/м³

# Радиусы сосудов для каждого измерения (см)


# Плотности шариков (г/см³) - массив 4x5 (4 температуры, 5 шариков)
ball_densities = np.array([
    [2.6, 2.6, 2.6, 2.6, 2.6],  # для 20°C
    [2.6, 2.6, 2.6, 2.6, 2.6],  # для 30°C
    [2.6, 2.6, 2.6, 2.6, 2.6],  # для 40°C
    [2.6, 2.6, 2.6, 2.6, 2.6]   # для 50°C
]) * 1000  # переводим в кг/м³

# Диаметры шариков (мм) - массив 4x5
ball_diameters = np.array([
    [1.29, 1.01, 1.04, 1.10, 1.36],
    [1.00, 1.05, 1.33, 1.13, 1.05],
    [1.08, 1.23, 1.02, 1.15, 1.11],
    [1.25, 1.08, 1.14, 1.04, 1.11]
]) / 1000  # переводим в м (радиусы посчитаем ниже)

# Расстояния падения (см) - массив 4x5
fall_distances = np.array([
    [10, 10, 10, 10, 10],  # для 20°C
    [10, 10, 10, 10, 10],  # для 30°C
    [10, 10, 10, 10, 10],  # для 40°C
    [10, 10, 10, 10, 10]   # для 50°C
]) / 100  # переводим в м

# Времена падения (с) - массив 4x5
fall_times = np.array([
    [105.4, 135.5, 126.0, 113.9, 91.7],  # для 20°C
    [101.6, 78.0, 57.7, 64.4, 78.3],    # для 30°C
    [45.2, 37.2, 46.7, 37.0, 39.4],    # для 40°C
    [23.1, 27.9, 23.2, 26.7, 24.3]     # для 50°C
])

# КОНЕЦ ВВОДА ДАННЫХ

# Рассчитываем радиусы шариков из диаметров
ball_radii = ball_diameters / 2

# Расчет всех величин
num_temperatures = len(temperatures)
num_balls_per_temp = ball_densities.shape[1]

v_ust = np.zeros((num_temperatures, num_balls_per_temp))  # установившаяся скорость
eta = np.zeros((num_temperatures, num_balls_per_temp))  # вязкость
Re = np.zeros((num_temperatures, num_balls_per_temp))  # число Рейнольдса
tau = np.zeros((num_temperatures, num_balls_per_temp))  # время релаксации
S = np.zeros((num_temperatures, num_balls_per_temp))  # путь релаксации

for i in range(num_temperatures):
    for j in range(num_balls_per_temp):
        v_ust[i,j] = fall_distances[i,j] / fall_times[i,j]
        eta[i,j] = (2/9) * g * ball_radii[i,j]**2 * (ball_densities[i,j] - liquid_densities[i]) / v_ust[i,j]
        Re[i,j] = v_ust[i,j] * ball_radii[i,j] * liquid_densities[i] / eta[i,j]
        tau[i,j] = (2/9) * ball_radii[i,j]**2 * ball_densities[i,j] / eta[i,j]
        S[i,j] = v_ust[i,j] * tau[i,j]

# Вывод результатов
print("Результаты расчетов:")
for i in range(num_temperatures):
    print(f"\nТемпература: {temperatures[i]}°C")
    print("Шарик | v_уст (м/с) | η (Па·с)   | Re       | τ (с)     | S (м)")
    print("--------------------------------------------------------------")
    for j in range(num_balls_per_temp):
        print(f"{j+1:4}  | {v_ust[i,j]:.3e} | {eta[i,j]:.3e} | {Re[i,j]:.3e} | {tau[i,j]:.3e} | {S[i,j]:.3e}")

# Построение графика ln(η) от 1/T
# Берем средние значения вязкости для каждого шарика при каждой температуре
mean_eta = np.mean(eta, axis=1)
T_K = temperatures + 273.15  # переводим в Кельвины
inv_T = 1 / T_K
ln_eta = np.log(mean_eta)

# Линейная регрессия
slope, intercept, r_value, p_value, std_err = linregress(inv_T, ln_eta)
res = linregress(inv_T, ln_eta)
W = round(slope * 1.38 / 1000, 2)  # энергия активации в Дж/моль (R = 8.314 Дж/(моль·К))

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(inv_T, ln_eta, 'bo', label='Экспериментальные точки')
plt.plot(inv_T, slope*inv_T + intercept, 'r-',
         label=f'Линейная аппроксимация\ny = {slope:.2f}x + {intercept:.2f}')

plt.xlabel('1/T (1/K)')
plt.ylabel('ln(η)')
plt.title(f'Зависимость ln(η) от 1/T\nЭнергия активации W = {W:.2f} Дж')
plt.legend()
plt.grid(True)



# Добавляем аннотацию с энергией активации
plt.annotate(f'Энергия активации:\nW = {W} * 10 ** (-20) Дж',
             xy=(0.5, 0.2), xycoords='axes fraction',
             bbox=dict(boxstyle="round", fc="w"))
print('sigma_k:', std_err, 'sigma_b:', res.intercept_stderr)
plt.show()
