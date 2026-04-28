"""
Alice
Программа на Питон, строящая карту потенциальной функции с несколькими источниками (15-20)

Модифицируй программу так, чтобы влияние функций было более распространено по карте, а не концентрировалось около источников.
Т.е. чтобы карта функции была похожа на карту гравитационного поля

Вот модифицированная программа, которая создаёт более распространённое влияние источников — ближе к реальному гравитационному полю.
Для этого используем сглаживающее ядро (гауссовское размытие) и логарифмическую зависимость потенциала от расстояния вместо 1/r.

Ключевые изменения для распространения влияния
1/ Логарифмическая зависимость потенциала: вместо V∼1/r  используется V∼−ln(r+1), что даёт более плавное убывание потенциала с расстоянием.
2/ Гауссовское сглаживание: функция gaussian_filter из scipy.ndimage распространяет влияние каждого источника на окружающую область.
3/ Увеличенный параметр сглаживания: smoothing_sigma=3.0 обеспечивает более широкое распространение поля.
4/ Цветовая палитра: расширенная палитра от тёмно‑синего до красного лучше отображает плавные переходы потенциала.

Как настроить распространение влияния
1/ Измените параметр smoothing_sigma в вызове create_gravitational_potential_map:
2/ smoothing_sigma=1.0 — более локализованное влияние (ближе к исходному варианту);
3/ smoothing_sigma=3.0–5.0 — умеренное распространение (рекомендуется);
4/ smoothing_sigma=8.0+ — очень широкое распространение, почти однородное поле.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

def create_gravitational_potential_map(n_sources=18, grid_size=250, strength_range=(0.8, 2.5), smoothing_sigma=2.0):
    """
    Создаёт карту потенциальной функции с несколькими источниками,
    имитирующую гравитационное поле с распространённым влиянием.

    Параметры:
    - n_sources: количество источников
    - grid_size: размер сетки
    - strength_range: диапазон силы источников
    - smoothing_sigma: параметр сглаживания (больше → более распространённое поле)
    """

    # Создаём сетку координат
    x = np.linspace(-10, 10, grid_size)
    y = np.linspace(-10, 10, grid_size)
    X, Y = np.meshgrid(x, y)

    # Инициализируем потенциальное поле нулевыми значениями
    potential = np.zeros_like(X)

    # Генерируем случайные позиции и силы источников
    np.random.seed(42)  # для воспроизводимости
    sources = []

    for _ in range(n_sources):
        # Случайные координаты источника
        x_source = np.random.uniform(-8, 8)
        y_source = np.random.uniform(-8, 8)

        # Случайная сила источника
        strength = np.random.uniform(*strength_range)

        sources.append((x_source, y_source, strength))

        # Вычисляем расстояние от каждой точки сетки до источника
        r = np.sqrt((X - x_source)**2 + (Y - y_source)**2)

        # Избегаем деления на ноль
        r[r == 0] = 1e-10

        # Используем логарифмическую зависимость для более плавного убывания потенциала
        # V ~ -ln(r) для больших r, плавно переходит к -1/r вблизи источника
        potential_contribution = -strength * np.log(r + 1)

        # Альтернативный вариант — гауссовское распределение влияния
        # potential_contribution = strength * np.exp(-r**2 / (2 * smoothing_sigma**2))

        potential += potential_contribution

    # Применяем сглаживание для распространения влияния
    potential_smoothed = gaussian_filter(potential, sigma=smoothing_sigma)

    return X, Y, potential_smoothed, sources

def plot_gravitational_map(X, Y, potential, sources, title="Карта гравитационного потенциала"):
    """
    Визуализирует карту гравитационного потенциала и источники.
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Первая панель: цветовая карта потенциала
    cmap = LinearSegmentedColormap.from_list(
        "grav_cmap", ["darkblue", "blue", "cyan", "green", "yellow", "orange", "red"]
    )

    contour = ax1.contourf(X, Y, potential, levels=50, cmap=cmap, alpha=0.9)
    cbar = plt.colorbar(contour, ax=ax1)
    cbar.set_label('Значение потенциала', rotation=270, labelpad=20)

    # Отмечаем источники
    source_x, source_y, _ = zip(*sources)
    ax1.scatter(source_x, source_y, c='white', s=80, edgecolors='black', linewidth=2,
               zorder=6, label=f'{len(sources)} масс-концентраций')

    ax1.set_title('Цветовая карта гравитационного потенциала')
    ax1.set_xlabel('X (условные единицы)')
    ax1.set_ylabel('Y (условные единицы)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Вторая панель: линии уровня (изолинии)
    contour_lines = ax2.contour(X, Y, potential, levels=25, colors='black', linewidths=0.8)
    ax2.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')

    # Отмечаем источники на карте изолиний
    ax2.scatter(source_x, source_y, c='red', s=70, edgecolors='black',
                zorder=6, label=f'{len(sources)} масс-концентраций')

    ax2.set_title('Линии уровня гравитационного потенциала')
    ax2.set_xlabel('X (условные единицы)')
    ax2.set_ylabel('Y (условные единицы)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

    return fig

# Основной блок программы
if __name__ == "__main__":
    print("Создание карты гравитационного потенциала...")

    # Создаём карту с 18 источниками и сглаживанием
    X, Y, potential, sources = create_gravitational_potential_map(
        n_sources=18,
        grid_size=250,
        strength_range=(0.8, 2.5),
        smoothing_sigma=3.0  # увеличенный параметр сглаживания
    )

    # Визуализируем результаты
    fig = plot_gravitational_map(X, Y, potential, sources)

    # Дополнительная информация о источниках
    print(f"\nИнформация о масс-концентрациях:")
    print(f"Количество источников: {len(sources)}")
    strengths = [s[2] for s in sources]
    print(f"Средняя масса: {np.mean(strengths):.2f}")
    print(f"Диапазон масс: {min(strengths):.2f} – {max(strengths):.2f}")

    # Статистика по потенциалу
    print(f"\nСтатистика гравитационного потенциала:")
    print(f"Минимум: {np.min(potential):.2f}")
    print(f"Максимум: {np.max(potential):.2f}")
    print(f"Среднее: {np.mean(potential):.2f}")

