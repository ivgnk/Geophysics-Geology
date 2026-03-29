"""
Alice
Модифицируй предыдущую программу, чтобы для создания двумерного поля вычислялось гравитационное поле 15-20 шаров.
Сделай чтобы поле было в миллиГалах.(мГал)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

# Константа гравитации в м³·кг⁻¹·с⁻²
G = 6.67430e-11
# Коэффициент перевода в мГал: 1 м/с² = 100 000 мГал
M_TO_MGAL = 1e5

def create_gravitational_field_map(n_sources=18, grid_size=250, mass_range=(1e12, 1e13), smoothing_sigma=2.0):
    """
    Создаёт карту гравитационного поля (в мГал) от 15–20 сферических масс.
    """

    # Создаём сетку координат в метрах
    x = np.linspace(-5000, 5000, grid_size)  # -5 км до +5 км
    y = np.linspace(-5000, 5000, grid_size)
    X, Y = np.meshgrid(x, y)

    # Инициализируем компоненты гравитационного поля
    gx = np.zeros_like(X)
    gy = np.zeros_like(Y)

    # Генерируем случайные позиции и массы источников
    np.random.seed(42)  # для воспроизводимости
    sources = []

    for _ in range(n_sources):
        # Случайные координаты источника в метрах
        x_source = np.random.uniform(-4000, 4000)
        y_source = np.random.uniform(-4000, 4000)

        # Случайная масса источника в кг (10¹²–10¹³ кг)
        mass = np.random.uniform(*mass_range)

        sources.append((x_source, y_source, mass))

        # Вычисляем расстояния от каждой точки сетки до источника
        dx = X - x_source
        dy = Y - y_source
        r = np.sqrt(dx**2 + dy**2)

        # Избегаем деления на ноль
        r[r == 0] = 1

        # Компоненты гравитационного ускорения (в м/с²)
        g_magnitude = G * mass / r**2

        # Направления компонентов
        gx_contribution = g_magnitude * (dx / r)
        gy_contribution = g_magnitude * (dy / r)

        # Суммируем вклады всех источников
        gx += gx_contribution
        gy += gy_contribution

    # Переводим из м/с² в миллиГалы (1 м/с² = 100 000 мГал)
    gx_mgal = gx * M_TO_MGAL
    gy_mgal = gy * M_TO_MGAL

    # Вычисляем модуль вектора гравитационного поля
    g_magnitude_mgal = np.sqrt(gx_mgal**2 + gy_mgal**2)

    # Применяем сглаживание для реалистичного распространения
    g_smoothed = gaussian_filter(g_magnitude_mgal, sigma=smoothing_sigma)

    return X, Y, g_smoothed, sources, gx_mgal, gy_mgal, grid_size

def plot_gravitational_field(X, Y, field_mgal, sources, gx_mgal, gy_mgal, grid_size, title="Гравитационное поле (мГал)"):
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
    source_x, source_y, _ = zip(*sources)
    ax1.scatter(
        np.array(source_x)/1000, np.array(source_y)/1000,
        c='white', s=80, edgecolors='black', linewidth=2,
        zorder=6, label=f'{len(sources)} масс-концентраций'
    )

    ax1.set_title('Цветовая карта гравитационного поля')
    ax1.set_xlabel('X, км')
    ax1.set_ylabel('Y, км')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Вторая панель: линии уровня (изолинии) и векторы поля
    contour_lines = ax2.contour(X/1000, Y/1000, field_mgal, levels=25, colors='black', linewidths=0.8)
    ax2.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')

    # Добавляем векторы гравитационного поля (разреженная сетка для наглядности)
    step = grid_size // 20
    quiver_x = X[::step, ::step] / 1000
    quiver_y = Y[::step, ::step] / 1000

    # Для векторов берём компоненты поля с тем же шагом
    gx_sample = gx_mgal[::step, ::step]
    gy_sample = gy_mgal[::step, ::step]

    ax2.quiver(quiver_x, quiver_y, gx_sample, gy_sample,
              scale=500, color='purple', alpha=0.7, width=0.005)

    # Отмечаем источники на карте изолиний
    ax2.scatter(np.array(source_x)/1000, np.array(source_y)/1000,
                c='red', s=70, edgecolors='black', zorder=6,
                label=f'{len(sources)} масс-концентраций')

    ax2.set_title('Линии уровня и векторы гравитационного поля')
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
    print("Создание карты гравитационного поля...")

    # Создаём карту с 18 источниками
    X, Y, gravitational_field, sources, gx_mgal, gy_mgal, grid_size = create_gravitational_field_map(
        n_sources=18,
        grid_size=250,
        mass_range=(5e12, 2e13),  # массы 5×10¹² – 2×10¹³ кг
        smoothing_sigma=3.0
    )

    # Визуализируем результаты
    fig = plot_gravitational_field(X, Y, gravitational_field, sources, gx_mgal, gy_mgal, grid_size)

    # Дополнительная информация о источниках
    print(f"\nИнформация о масс-концентрациях:")
    print(f"Количество источников: {len(sources)}")
    masses = [s[2] for s in sources]
    print(f"Средняя масса: {np.mean(masses):.2e} кг")
    print(f"Диапазон масс: {min(masses):.2e} – {max(masses):.2e}")