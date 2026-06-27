import numpy as np

t_1 = [8.40, 7.63, 7.53, 8.89, 15.52, 21.22, 29.10]
h_1 = [12, 11.8, 12.4, 10.0, 7.2, 3.8, 2.7]
v_1 = [0.5 for _ in range(7)]
delta_l_1 = [5, 35, 20, 20, 20]
delta_h_1 = [1, 10.2, 6.1, 5.7, 5.9]
t_av_1 = 8.00
v_av_1 = 0.5

t_2 = [4.55, 4.33, 4.43, 4.60, 5.09, 8.54, 21.6]
h_2 = [7.6, 9.2, 8.2, 7.7, 4.9, 2.7, 1.1]
v_2 = [0.5 for _ in range(7)]
delta_l_2 = [10, 30, 20, 20, 20]
delta_h_2 = [1.3, 4, 2.5, 2.8, 2.5]
t_av_2 = 4.87
v_av_2 = 0.5

t_3 = [7.20, 6.31, 6.41, 6.70, 8.30, 9.80, 15.52]
h_3 = [4.2, 5.5, 5.1, 4.6, 2.2, 1.7, 1.2]
v_3 = [1 for _ in range(7)]
delta_l_3 = [10, 25, 20, 20, 20]
delta_h_3 = [1, 2.4, 2.1, 1.6, 3.5]
t_av_3 = 7.51
v_av_3 = 1

ro = 805  # плотность спирта
Q1 = 0.5e-3 / np.array(t_1)
Q2 = 0.5e-3 / np.array(t_2)
Q3 = 1.e-3 / np.array(t_3)

delta_P_1 = np.array([947.6, 931.8, 979.2, 789.7, 568.6, 300.1, 213.2])
delta_P_2 = np.array([600.2, 726.5, 647.5, 608.1, 386.9, 213.2, 86.9])
delta_P_3 = np.array([331.7, 434.3, 402.7, 363.2, 173.7, 134.2, 94.8])

delta_P_L_1 = np.array([h * 1e-2 * ro * 9.81 for h in delta_h_1])
delta_P_L_2 = np.array([h * 1e-2 * ro * 9.81 for h in delta_h_2])
delta_P_L_3 = np.array([h * 1e-2 * ro * 9.81 for h in delta_h_3])

P_L_1 = np.cumsum(delta_P_L_1)
P_L_2 = np.cumsum(delta_P_L_2)
P_L_3 = np.cumsum(delta_P_L_3)

L_1 = np.cumsum(delta_l_1) / 100
L_2 = np.cumsum(delta_l_2) / 100
L_3 = np.cumsum(delta_l_3) / 100

P_1 = np.cumsum(delta_P_1)
P_2 = np.cumsum(delta_P_2)
P_3 = np.cumsum(delta_P_3)

rv = 1.2
r = 0.002
eta1 = (np.pi * r ** 4 * delta_P_3) / (8 * Q3 * 0.4)
Re_1 = (Q3 * rv) / (np.pi * r * eta1)

et1 = 2.8532503795594584e-05
et2 = 3.258382643771493e-05
et3 = 3.2249763570024235e-05

k: float = 1.5 * 10 ** 7
pi = np.pi
r: float = 0.001
l: float = 0.2

print(k * pi * r ** 4 / 8 / l)

import numpy as np
import matplotlib.pyplot as plt


def format_coefficient(value):
    """Форматирует число в виде умножения на 10 в степени."""
    exponent = int(np.floor(np.log10(abs(value)))) if value != 0 else 0
    coefficient = value / 10 ** exponent
    return f"{coefficient:.2f} × 10^{{{exponent}}}"


def tr_I_delta_PQ():
    # Пример данных измерений
    Q = np.array([5.95 * (10 ** -5), 6.55 * (10 ** -5), 6.64 * (10 ** -5), 5.62 * (10 ** -5), 3.22 * (10 ** -5),
                  2.35 * (10 ** -5), 1.71 * (10 ** -5)])  # Объемный расход (м^3/с)
    delta_P = np.array([947.6, 931.8, 979.2, 789.7, 568.6, 300.1, 213.2])  # Разница давлений (Па)

    # Метод наименьших квадратов для линейной зависимости delta_P = k * Q + b
    A = np.vstack([Q, np.ones(len(Q))]).T
    k, b = np.linalg.lstsq(A, delta_P, rcond=None)[0]

    # Расчет среднеквадратичного отклонения
    residuals = delta_P - (k * Q + b)
    rms_error = np.sqrt(np.mean(residuals ** 2))

    # Вычисление средних значений
    n = len(Q)
    Q_mean = np.mean(Q)
    delta_P_mean = np.mean(delta_P)
    Q2_mean = np.mean(Q ** 2)
    Q_delta_P_mean = np.mean(Q * delta_P)

    # Вычисление дисперсий
    D_xx = Q2_mean - Q_mean ** 2
    D_yy = np.mean(delta_P ** 2) - delta_P_mean ** 2

    # Вычисление погрешностей
    sigma_k = np.sqrt(1 / (n - 2)) * np.sqrt((D_yy / D_xx) - k ** 2)
    sigma_b = sigma_k * np.sqrt(Q2_mean)

    # Вывод коэффициентов, среднеквадратичного отклонения и погрешностей
    print("Тр I: Коэффициенты прямой: k =", k, ", b =", b)
    print("Тр I: Среднеквадратичное отклонение:", rms_error)
    print("Тр I: Погрешность коэффициента k (σk):", sigma_k)
    print("Тр I: Погрешность коэффициента b (σb):", sigma_b)

    # Построение графика
    plt.figure(figsize=(6.9 * 1.4, 4.3 * 1.4))
    plt.plot(Q, delta_P, 'bo', label='Измерения')
    plt.plot(Q, k * Q + b, 'r-', label=f'Аппроксимация: ΔP = {format_coefficient(k)} * Q + {format_coefficient(b)}')

    # Оформление графика
    plt.title('Зависимость ΔP(Q)')
    plt.xlabel('Q (м³/с)')
    plt.ylabel('ΔP (Па)')
    plt.grid(True)
    plt.legend()

    # Показ графика
    plt.tight_layout()
    plt.savefig('tr_I_delta_PQ.png')
    plt.show()


def tr_II_delta_PQ():
    # Пример данных измерений
    Q = np.array(
        [10.9 * (10 ** -5), 11.5 * (10 ** -5), 11.2 * (10 ** -5), 10.8 * (10 ** -5), 9.8 * (10 ** -5), 5.8 * (10 ** -5),
         2.3 * (10 ** -5)])  # Объемный расход (м^3/с)
    delta_P = np.array([600.2, 726.5, 647.5, 608.1, 386.9, 213.2, 86.9])  # Разница давлений (Па)

    # Метод наименьших квадратов для линейной зависимости delta_P = k * Q + b
    A = np.vstack([Q, np.ones(len(Q))]).T
    k, b = np.linalg.lstsq(A, delta_P, rcond=None)[0]

    # Расчет среднеквадратичного отклонения
    residuals = delta_P - (k * Q + b)
    rms_error = np.sqrt(np.mean(residuals ** 2))

    # Вычисление средних значений
    n = len(Q)
    Q_mean = np.mean(Q)
    delta_P_mean = np.mean(delta_P)
    Q2_mean = np.mean(Q ** 2)
    Q_delta_P_mean = np.mean(Q * delta_P)

    # Вычисление дисперсий
    D_xx = Q2_mean - Q_mean ** 2
    D_yy = np.mean(delta_P ** 2) - delta_P_mean ** 2

    # Вычисление погрешностей
    sigma_k = np.sqrt(1 / (n - 2)) * np.sqrt((D_yy / D_xx) - k ** 2)
    sigma_b = sigma_k * np.sqrt(Q2_mean)

    # Вывод коэффициентов, среднеквадратичного отклонения и погрешностей
    print("Тр I: Коэффициенты прямой: k =", k, ", b =", b)
    print("Тр I: Среднеквадратичное отклонение:", rms_error)
    print("Тр I: Погрешность коэффициента k (σk):", sigma_k)
    print("Тр I: Погрешность коэффициента b (σb):", sigma_b)

    # Построение графика
    plt.figure(figsize=(6.9 * 1.4, 4.3 * 1.4))
    plt.plot(Q, delta_P, 'bo', label='Измерения')
    plt.plot(Q, k * Q + b, 'r-', label=f'Аппроксимация: ΔP = {format_coefficient(k)} * Q + {format_coefficient(b)}')

    # Оформление графика
    plt.title('Зависимость ΔP(Q)')
    plt.xlabel('Q (м³/с)')
    plt.ylabel('ΔP (Па)')
    plt.grid(True)
    plt.legend()



    # Показ графика
    plt.tight_layout()
    plt.savefig('tr_II_delta_PQ.png')
    plt.show()


def tr_III_delta_PQ():
    # Пример данных измерений
    Q = np.array([13.8 * (10 ** -5), 15.8 * (10 ** -5), 15.6 * (10 ** -5), 14.9 * (10 ** -5), 12.1 * (10 ** -5),
                  10.2 * (10 ** -5), 6.4 * (10 ** -5)])  # Объемный расход (м^3/с)
    delta_P = np.array([331.7, 434.3, 402.7, 363.2, 173.7, 134.2, 94.8])  # Разница давлений (Па)

    # Метод наименьших квадратов для линейной зависимости delta_P = k * Q + b
    A = np.vstack([Q, np.ones(len(Q))]).T
    k, b = np.linalg.lstsq(A, delta_P, rcond=None)[0]

    # Расчет среднеквадратичного отклонения
    residuals = delta_P - (k * Q + b)
    rms_error = np.sqrt(np.mean(residuals ** 2))

    # Вычисление средних значений
    n = len(Q)
    Q_mean = np.mean(Q)
    delta_P_mean = np.mean(delta_P)
    Q2_mean = np.mean(Q ** 2)
    Q_delta_P_mean = np.mean(Q * delta_P)

    # Вычисление дисперсий
    D_xx = Q2_mean - Q_mean ** 2
    D_yy = np.mean(delta_P ** 2) - delta_P_mean ** 2

    # Вычисление погрешностей
    sigma_k = np.sqrt(1 / (n - 2)) * np.sqrt((D_yy / D_xx) - k ** 2)
    sigma_b = sigma_k * np.sqrt(Q2_mean)

    # Вывод коэффициентов, среднеквадратичного отклонения и погрешностей
    print("Тр I: Коэффициенты прямой: k =", k, ", b =", b)
    print("Тр I: Среднеквадратичное отклонение:", rms_error)
    print("Тр I: Погрешность коэффициента k (σk):", sigma_k)
    print("Тр I: Погрешность коэффициента b (σb):", sigma_b)

    # Построение графика
    plt.figure(figsize=(6.9 * 1.4, 4.3 * 1.4))
    plt.plot(Q, delta_P, 'bo', label='Измерения')
    plt.plot(Q, k * Q + b, 'r-', label=f'Аппроксимация: ΔP = {format_coefficient(k)} * Q + {format_coefficient(b)}')

    # Оформление графика
    plt.title('Зависимость ΔP(Q)')
    plt.xlabel('Q (м³/с)')
    plt.ylabel('ΔP (Па)')
    plt.grid(True)
    plt.legend()

    # Показ графика
    plt.tight_layout()
    plt.savefig('tr_III_delta_PQ.png')
    plt.show()


def tr_I_P_delta_L():
    # Пример данных измерений
    l = np.array([0.05, 0.4, 0.6, 0.8, 1.])  # Длина (м)
    P = np.array([78.9705, 884.4696, 1366.18965, 1816.3215, 2282.24745])  # Давление (Па)

    # Метод наименьших квадратов
    A = np.vstack([l, np.ones(len(l))]).T
    k, b = np.linalg.lstsq(A, P, rcond=None)[0]

    # Расчет погрешностей
    residuals = P - (k * l + b)
    rms_error = np.sqrt(np.mean(residuals ** 2))

    n = len(l)
    l_mean = np.mean(l)
    P_mean = np.mean(P)
    l2_mean = np.mean(l ** 2)
    l_P_mean = np.mean(l * P)

    D_xx = l2_mean - l_mean ** 2
    D_yy = np.mean(P ** 2) - P_mean ** 2

    sigma_k = np.sqrt(1 / (n - 2)) * np.sqrt((D_yy / D_xx) - k ** 2)
    sigma_b = sigma_k * np.sqrt(l2_mean)

    # Вывод результатов
    print(f"Коэффициенты прямой: k = {k:.2f}, b = {b:.2f}")
    print(f"Среднеквадратичное отклонение: {rms_error:.2f}")
    print(f"Погрешность коэффициента k (σk): {sigma_k:.2f}")
    print(f"Погрешность коэффициента b (σb): {sigma_b:.2f}")

    # Построение графика
    plt.figure(figsize=(6.9 * 1.4, 4.3 * 1.4))
    plt.plot(l, P, 'bo', label='Измерения')
    plt.plot(l, k * l + b, 'r-', label=f'Аппроксимация: P = {k:.2f} * l + {b:.2f}')

    plt.title('Зависимость P(L)')
    plt.xlabel('L (м)')
    plt.ylabel('P (Па)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('tr_I_P_delta_L.png')
    plt.show()


def tr_II_P_delta_L():
    # Пример данных измерений
    l = np.array([0.1, 0.4, 0.6, 0.8, 1.])  # Длина (м)
    P = np.array([102.66165, 418.54365, 615.9699, 837.0873, 1034.51355])  # Давление (Па)

    # Метод наименьших квадратов
    A = np.vstack([l, np.ones(len(l))]).T
    k, b = np.linalg.lstsq(A, P, rcond=None)[0]

    # Расчет погрешностей
    residuals = P - (k * l + b)
    rms_error = np.sqrt(np.mean(residuals ** 2))

    n = len(l)
    l_mean = np.mean(l)
    P_mean = np.mean(P)
    l2_mean = np.mean(l ** 2)
    l_P_mean = np.mean(l * P)

    D_xx = l2_mean - l_mean ** 2
    D_yy = np.mean(P ** 2) - P_mean ** 2


    sigma_k = np.sqrt(1 / (n - 2)) * np.sqrt((D_yy / D_xx) - k ** 2)
    sigma_b = sigma_k * np.sqrt(l2_mean)

    # Вывод результатов
    print(f"Коэффициенты прямой: k = {k:.2f}, b = {b:.2f}")
    print(f"Среднеквадратичное отклонение: {rms_error:.2f}")
    print(f"Погрешность коэффициента k (σk): {sigma_k:.2f}")
    print(f"Погрешность коэффициента b (σb): {sigma_b:.2f}")

    # Построение графика
    plt.figure(figsize=(6.9 * 1.4, 4.3 * 1.4))
    plt.plot(l, P, 'bo', label='Измерения')
    plt.plot(l, k * l + b, 'r-', label=f'Аппроксимация: P = {k:.2f} * l + {b:.2f}')

    plt.title('Зависимость P(L)')
    plt.xlabel('L (м)')
    plt.ylabel('P (Па)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('tr_II_P_delta_L.png')
    plt.show()


def tr_III_P_delta_L():
    # Пример данных измерений
    l = np.array([0.1, 0.35, 0.55, 0.75, 0.95])  # Длина (м)
    P = np.array([78.9705, 268.4997, 434.33775, 560.69055, 837.0873])  # Давление (Па)

    # Метод наименьших квадратов
    A = np.vstack([l, np.ones(len(l))]).T
    k, b = np.linalg.lstsq(A, P, rcond=None)[0]

    # Расчет погрешностей
    residuals = P - (k * l + b)
    rms_error = np.sqrt(np.mean(residuals ** 2))

    n = len(l)
    l_mean = np.mean(l)
    P_mean = np.mean(P)
    l2_mean = np.mean(l ** 2)
    l_P_mean = np.mean(l * P)

    D_xx = l2_mean - l_mean ** 2
    D_yy = np.mean(P ** 2) - P_mean ** 2

    sigma_k = np.sqrt(1 / (n - 2)) * np.sqrt((D_yy / D_xx) - k ** 2)
    sigma_b = sigma_k * np.sqrt(l2_mean)

    # Вывод результатов
    print(f"Коэффициенты прямой: k = {k:.2f}, b = {b:.2f}")
    print(f"Среднеквадратичное отклонение: {rms_error:.2f}")
    print(f"Погрешность коэффициента k (σk): {sigma_k:.2f}")
    print(f"Погрешность коэффициента b (σb): {sigma_b:.2f}")

    # Построение графика
    plt.figure(figsize=(6.9 * 1.4, 4.3 * 1.4))
    plt.plot(l, P, 'bo', label='Измерения')
    plt.plot(l, k * l + b, 'r-', label=f'Аппроксимация: P = {k:.2f} * l + {b:.2f}')

    plt.title('Зависимость P(L)')
    plt.xlabel('L (м)')
    plt.ylabel('P (Па)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('tr_III_P_delta_L.png')
    plt.show()


def log():
    # Данные эксперимента
    r = np.array([0.001, 0.0015, 0.002])  # радиусы трубок [м]
    Q = np.array([4.580856331356954e-05, 8.955057213659115e-05, 0.00012708331422830725])  # расходы [м³/с]
    delta_P = np.array([675.7428571428572, 467.04285714285714, 276.37142857142857])  # перепады давлений [Па]
    L = np.array([0.2, 0.3, 0.4])  # длины трубок [м]
    eta = np.array([2.8532503795594584e-05, 3.258382643771493e-05, 3.2249763570024235e-05])  # вязкости [Па·с] (разные для каждого измерения)

    # Вычисление координат для графика
    x = np.log(r)
    y = np.log((8 * eta * L * Q) / (np.pi * delta_P))

    # Метод наименьших квадратов
    A = np.vstack([x, np.ones(len(x))]).T
    k, b = np.linalg.lstsq(A, y, rcond=None)[0]

    # Расчет погрешностей
    residuals = y - (k * x + b)
    rms_error = np.sqrt(np.mean(residuals ** 2))

    n_points = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x2_mean = np.mean(x ** 2)
    xy_mean = np.mean(x * y)

    D_xx = x2_mean - x_mean ** 2
    D_yy = np.mean(y ** 2) - y_mean ** 2

    sigma_k = np.sqrt(1 / (n_points - 2)) * np.sqrt((D_yy / D_xx) - k ** 2)
    sigma_b = sigma_k * np.sqrt(x2_mean)

    # Построение графика
    plt.figure(figsize=(6.9 * 1.4, 4.3 * 1.4))
    plt.plot(x, y, 'bo', label='Экспериментальные точки', markersize=8)
    plt.plot(x, k * x + b, 'r-', linewidth=2,
             label=f'Аппроксимация y = {k:.2f}x + {b:.2f}')

    # Оформление
    plt.title('Зависимость Q(P) в двойном логарифмическом масштабе', fontsize=14)
    plt.xlabel('$\ln \, r$', fontsize=12)
    plt.ylabel('$\ln \left( \\frac{8 \eta L Q}{\pi \Delta P} \\right)$', fontsize=12)
    plt.grid(True, linestyle='-', alpha=0.6)
    plt.legend(loc='best', fontsize=10)

    plt.tight_layout()
    plt.savefig('poiseuille_verification.png', dpi=300)
    plt.show()


    # Вывод результатов
    print("\nРезультаты анализа:")
    print(f"Показатель степени n = {k:.2f} ± {sigma_k:.2f}")
    print(f"Свободный член C = {b:.2f} ± {sigma_b:.2f}")
    print(f"Среднеквадратичное отклонение: {rms_error:.3f}")
    print(f"Теоретическое значение n = 4")

    if abs(k - 4) < 2 * sigma_k:
        print("\nВывод: Результаты согласуются с законом Пуазейля (n ≈ 4)")
    else:
        print("\nВывод: Обнаружено статистически значимое расхождение с теорией!")


tr_I_delta_PQ()
tr_II_delta_PQ()
tr_III_delta_PQ()

tr_I_P_delta_L()
tr_II_P_delta_L()
tr_III_P_delta_L()

log()
