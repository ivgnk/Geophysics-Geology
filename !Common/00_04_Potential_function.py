"""
Alice
Модифицируй программу: убери векторы и сделай, чтобы задавались для расчета не массы,
а аномальные плотности в диапазоне от 0.01 до 0.3 г/куб.см
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

# Константа гравитации в м³·кг⁻¹·с⁻²
G = 6.67430e-11
# Коэффициент перевода в мГал: 1 м/с² = 100 000 мГал
M_TO_MGAL = 1e5
# Перевод г/см³ в кг/м³: 1 г/см³ = 1000 кг/м³
DENSITY_CONVERSION = 1000

def create_gravitational_field_map(n_sources=18, grid_size=250,
                            density_range=(0.01, 0.3),  # г/см³
                            radius_range=(200, 800),  # метры
                            smoothing_sigma=2.0):
    """
    Создаёт карту гравитационного поля (в мГал) от 15–20 сферических аномалий плотности.

    Параметры:
    - n_sources: количество источников
    - grid_size: размер сетки
    - density_range: диапазон аномальных плотностей в г/см³
    - radius_range: диапазон радиусов источников в метрах
    - smoothing_sigma: параметр сглаживания
    """

    # Создаём сетку координат в метрах
    x = np.linspace(-5000, 5000, grid_size)  # -5 км до +5 км
    y = np.linspace(-5000, 5000, grid_size)
    X, Y = np.meshgrid(x, y)

    # Инициализируем гравитационное поле
    g_total = np.zeros_like(X)

    # Генерируем случайные позиции, плотности и радиусы источников
    np.random.seed(42)  # для воспроизводимости
    sources = []

    for _ in range(n_sources):
        # Случайные координаты источника в метрах
        x_source = np.random.uniform(-4000, 4000)
        y_source = np.random.uniform(-4000, 4000)

        # Случайная аномальная плотность в г/см³, переводим в кг/м³
        density_g_cm3 = np.random.uniform(*density_range)
        density_kg_m3 = density_g_cm3 * DENSITY_CONVERSION

        # Случайный радиус источника в метрах
        radius = np.random.uniform(*radius_range)

        # Вычисляем объём сферы: V = (4/3)πr³
        volume = (4/3) * np.pi * radius**3

        # Масса = плотность × объём
        mass = density_kg_m3 * volume

        sources.append((x_source, y_source, density_g_cm3, radius, mass))

        # Вычисляем расстояния от каждой точки сетки до источника
        dx = X - x_source
        dy = Y - y_source
        r = np.sqrt(dx**2 + dy**2)

        # Избегаем деления на ноль
        r[r == 0] = 1

        # Компоненты гравитационного ускорения (в м/с²)
        g_magnitude = G * mass / r**2

        # Суммируем вклады всех источников
        g_total += g_magnitude

    # Переводим из м/с² в миллиГалы (1 м/с² = 100 000 мГал)
    g_mgal = g_total * M_TO_MGAL

    # Применяем сглаживание для реалистичного распространения
    g_smoothed = gaussian_filter(g_mgal, sigma=smoothing_sigma)

    return X, Y, g_smoothed, sources

def plot_gravitational_field(X, Y, field_mgal, sources, title="Гравитационное поле (мГал)"):
    """
    Визуализирует карту гравитационного поля и источники.
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Первая панель: цветовая карта гравитационного поля
    cmap = LinearSegmentedColormap.from_list(
        "grav_cmap", ["darkblue", "blue", "cyan", "green", "yellow", "orange", "red"]
    )

    contour = ax1.contourf(X/1000, Y/1000, field_mgal, levels=50, cmap=cmap, alpha=0.9)
    cbar = plt.colorbar(contour, ax=ax1)
    cbar.set_label('Гравитационное поле, мГал', rotation=270, labelpad=20)

    # Отмечаем источники
    source_x, source_y, _, _, _ = zip(*sources)
    ax1.scatter(
        np.array(source_x)/1000, np.array(source_y)/1000,
        c='white', s=80, edgecolors='black', linewidth=2,
        zorder=6, label=f'{len(sources)} аномалий плотности'
    )

    ax1.set_title('Цветовая карта гравитационного поля')
    ax1.set_xlabel('X, км')
    ax1.set_ylabel('Y, км')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Вторая панель: линии уровня (изолинии)
    contour_lines = ax2.contour(X/1000, Y/1000, field_mgal, levels=25, colors='black', linewidths=0.8)
    ax2.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')

    # Отмечаем источники на карте изолиний
    ax2.scatter(np.array(source_x)/1000, np.array(source_y)/1000,
                c='red', s=70, edgecolors='black', zorder=6,
                label=f'{len(sources)} аномалий плотности')

    ax2.set_title('Линии уровня гравитационного поля')
    ax2.set_xlabel('X, км')
    ax2.set_ylabel('Y, км')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

    return fig

# Основной блок программы
if __name__ == "__main__":
    print("Создание карты гравитационного поля от аномалий плотности...")

    # Создаём карту с 18 источниками
    X, Y, gravitational_field, sources = create_gravitational_field_map(
        n_sources=18,
        grid_size=250,
        density_range=(0.01, 0.3),  # г/см³
        radius_range=(30, 170),      # метры
        smoothing_sigma=3.0
    )

    # Визуализируем результаты
    fig = plot_gravitational_field(X, Y, gravitational_field, sources)

    # Дополнительная информация о источниках
    print(f"\nИнформация об аномалиях плотности:")
    print(f"Количество источников: {len(sources)}")

    densities_g_cm3 = [s[2] for s in sources]
    radii = [s[3] for s in sources]
    masses = [s[4] for s in sources]

    print(f"Средняя аномальная плотность: {np.mean(densities_g_cm3):.3f} г/см³")
    print(f"Диапазон плотностей: {min(densities_g_cm3):.3f} – {max(densities_g_cm3):.3f} г/см³")
    print(f"Средний радиус: {np.mean(radii):.1f} м")
    print(f"Диапазон радиусов: {min(radii):.1f} – {max(radii):.1f} м")