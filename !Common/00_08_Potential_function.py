"""
Alice
Модифицируй программу так, чтобы:
1) можно было задавать несколько раз тренды (от 1 до 4)
2) в параметрах функции add_trend_polynomial можно было задавать коэффициент, который бы изменял размах поля тренда
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
                            radius_range=(0.5, 3000),    # метры
                            z_level=500,             # высота измерения над уровнем моря (м)
                            smoothing_sigma=2.0):
    """Создаёт карту гравитационного поля (в мГал) от сферических аномалий плотности."""
    # Создаём сетку координат в метрах
    x = np.linspace(-5000, 5000, grid_size)  # -5 км до +5 км
    y = np.linspace(-5000, 5000, grid_size)
    X, Y = np.meshgrid(x, y)

    # Инициализируем гравитационное поле
    g_total = np.zeros_like(X)

    # Генерируем случайные позиции, плотности, радиусы и глубины источников
    np.random.seed(42)  # для воспроизводимости
    sources = []

    for _ in range(n_sources):
        # Случайные координаты источника в метрах (X, Y)
        x_source = np.random.uniform(-4000, 4000)
        y_source = np.random.uniform(-4000, 4000)

        # Случайный радиус источника в метрах
        radius = np.random.uniform(*radius_range)

        # Глубина центра шара: верхняя кромка должна быть минимум в 10 м от уровня Z
        buffer_distance = np.random.uniform(10, 100)
        z_center = z_level - (radius + buffer_distance)

        # Случайная аномальная плотность в г/см³, переводим в кг/м³
        density_g_cm3 = np.random.uniform(*density_range)
        density_kg_m3 = density_g_cm3 * DENSITY_CONVERSION

        # Вычисляем объём сферы: V = (4/3)πr³
        volume = (4/3) * np.pi * radius**3

        # Масса = плотность × объём
        mass = density_kg_m3 * volume

        sources.append((x_source, y_source, z_center, density_g_cm3, radius, mass, buffer_distance))

        # Вычисляем расстояния от каждой точки сетки до источника в 3D
        dx = X - x_source
        dy = Y - y_source
        dz = z_level - z_center  # разница по высоте
        r = np.sqrt(dx**2 + dy**2 + dz**2)

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

    return X, Y, g_smoothed, sources, z_level

def add_random_noise(field, noise_level=0.1, noise_type='gaussian'):
    """
    Добавляет случайный шум к гравитационному полю.

    Параметры:
    - field: исходное гравитационное поле (2D массив)
    - noise_level: уровень шума (относительный или абсолютный)
    - noise_type: тип шума ('gaussian', 'uniform', 'salt_pepper')
    """
    if noise_type == 'gaussian':
        noise = np.random.normal(0, noise_level, field.shape)
    elif noise_type == 'uniform':
        noise = np.random.uniform(-noise_level, noise_level, field.shape)
    elif noise_type == 'salt_pepper':
        noise = np.zeros(field.shape)
        salt_mask = np.random.random(field.shape) < 0.01  # 1% соли
        pepper_mask = np.random.random(field.shape) < 0.01  # 1% перца
        noise[salt_mask] = noise_level
        noise[pepper_mask] = -noise_level
    else:
        raise ValueError("Неизвестный тип шума")

    noisy_field = field + noise
    return noisy_field

def add_trend_polynomial(field, X, Y, order=2, trend_coefficients=None, amplitude_factor=1.0):
    """
    Добавляет тренд, описываемый 2D полиномом заданного порядка.

    Параметры:
    - field: исходное гравитационное поле (2D массив)
    - X, Y: координаты сетки (2D массивы)
    - order: порядок полинома (1–4)
    - trend_coefficients: коэффициенты полинома. Если None, генерируются случайно
    - amplitude_factor: коэффициент, изменяющий размах поля тренда
    """
    # Нормализуем координаты для лучшей устойчивости полинома
    x_norm = X / 1000  # в км
    y_norm = Y / 1000  # в км

    trend = np.zeros_like(field)

    if trend_coefficients is None:
        # Генерируем случайные коэффициенты для полинома
        n_terms = (order + 1) * (order + 2) // 2  # число членов полинома
        trend_coefficients = np.random.uniform(-0.5, 0.5, n_terms)

    coeff_idx = 0
    for i in range(order + 1):
        for j in range(order - i + 1):
            trend += trend_coefficients[coeff_idx] * (x_norm**i) * (y_norm**j)
            coeff_idx += 1

    # Применяем коэффициент амплитуды
    trend *= amplitude_factor

    trend_field = field + trend
    return trend_field, trend_coefficients, amplitude_factor

def apply_multiple_trends(field, X, Y, trends_config):
    """
    Применяет несколько трендов последовательно.

    Параметры:
    - field: исходное гравитационное поле (2D массив)
    - X, Y: координаты сетки (2D массивы)
    - trends_config: список словарей с параметрами для каждого тренда, например:
      [{'order': 2, 'amplitude_factor': 0.8}, {'order': 3, 'amplitude_factor': 0.3}]
    """
    current_field = field.copy()
    all_coefficients = []
    all_factors = []

    for trend_params in trends_config:
        order = trend_params.get('order', 2)
        amplitude_factor = trend_params.get('amplitude_factor', 1.0)
        coefficients = trend_params.get('trend_coefficients', None)

        current_field, coeffs, factor = add_trend_polynomial(
            current_field, X, Y, order, coefficients, amplitude_factor
        )
        all_coefficients.append(coeffs)
        all_factors.append(factor)

    return current_field, all_coefficients, all_factors


def plot_gravitational_field(X, Y, field_mgal, sources, z_level, title="Гравитационное поле (мГал)"):
    """Визуализирует карту гравитационного поля и источники."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))


    # Первая панель: цветовая карта гравитационного поля
    cmap = LinearSegmentedColormap.from_list(
        "grav_cmap", ["darkblue", "blue", "cyan", "green", "yellow", "orange", "red"]
    )

    contour = ax1.contourf(X/1000, Y/1000, field_mgal, levels=50, cmap=cmap, alpha=0.9)
    cbar = plt.colorbar(contour, ax=ax1)
    cbar.set_label('Гравитационное поле, мГал', rotation=270, labelpad=20)

    # Извлекаем координаты источников без zip
    source_x = []
    source_y = []
    for source in sources:
        x_src, y_src, z_center, density, radius, mass, buffer = source
        source_x.append(x_src)
        source_y.append(y_src)

    # Отмечаем источники
    ax1.scatter(
        np.array(source_x)/1000, np.array(source_y)/1000,
        c='white', s=80, edgecolors='black', linewidth=2,
        zorder=6, label=f'{len(sources)} аномалий плотности'
    )

    ax1.set_title(f'Цветовая карта гравитационного поля\nУровень измерения: {z_level} м над уровнем моря')
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

    ax2.set_title(f'Линии уровня гравитационного поля\nУровень измерения: {z_level} м над уровнем моря')
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

    # Задаём уровень измерения (можно менять от 0 до 1500 м)
    measurement_level = 800  # м над уровнем моря

    # Создаём карту с 18 источниками
    X, Y, gravitational_field, sources, z_level = create_gravitational_field_map(
        n_sources=18,
        grid_size=250,
        density_range=(0.01, 0.3),
        radius_range=(0.5, 3000),
        z_level=measurement_level,
        smoothing_sigma=3.0
    )

    print(f"Исходное поле создано. Диапазон значений: {np.min(gravitational_field):.3f} – {np.max(gravitational_field):.3f} мГал")

    # Добавляем случайный шум
    noisy_field = add_random_noise(
        gravitational_field,
        noise_level=0.05,  # 5% от стандартного отклонения исходного поля
        noise_type='gaussian'
    )
    print(f"Добавлен гауссовский шум. Диапазон значений с шумом: {np.min(noisy_field):.3f} – {np.max(noisy_field):.3f} мГал")

    # Конфигурация нескольких трендов
    trends_config = [
        {'order': 2, 'amplitude_factor': 0.8},
        {'order': 3, 'amplitude_factor': 0.3},
        {'order': 1, 'amplitude_factor': 0.5},
        {'order': 4, 'amplitude_factor': 0.2}
    ]

    # Применяем несколько трендов
    final_field, all_coeffs, all_factors = apply_multiple_trends(
        noisy_field, X, Y, trends_config
    )

    print(f"Добавлено {len(trends_config)} трендов. Диапазон значений с трендами: {np.min(final_field):.3f} – {np.max(final_field):.3f} мГал")
    for i, (coeffs, factor) in enumerate(zip(all_coeffs, all_factors)):
        print(f"Тренд {i+1}: порядок {trends_config[i]['order']}, коэффициент амплитуды {factor:.2f}")

    # Визуализируем результаты
    fig1 = plot_gravitational_field(X, Y, gravitational_field, sources, z_level, "Исходное гравитационное поле")
    fig2 = plot_gravitational_field(X, Y, noisy_field, sources, z_level, "Гравитационное поле с шумом")
    fig3 = plot_gravitational_field(X, Y, final_field, sources, z_level, "Гравитационное поле с шумом и трендами")

    # Дополнительная информация о источниках
    print(f"\nИнформация об аномалиях плотности:")
    print(f"Количество источников: {len(sources)}")
    print(f"Уровень измерения: {z_level} м над уровнем моря")

    densities_g_cm3 = [s[3] for s in sources]
    radii = [s[4] for s in sources]
    masses = [s[5] for s in sources]
    depths = [s[2] for s in sources]  # глубины центров
    buffers = [s[6] for s in sources]  # буферные расстояния

    print(f"Средняя аномальная плотность: {np.mean(densities_g_cm3):.3f} г/см³")
    print(f"Диапазон плотностей: {min(densities_g_cm3):.3f} – {max(densities_g_cm3):.3f} г/см³")

    print(f"Средний радиус: {np.mean(radii):.1f} м")
    print(f"Диапазон радиусов: {min(radii):.1f} – {max(radii):.1f} м")

    print(f"Средняя глубина центра: {np.mean(depths):.1f} м")
    print(f"Диапазон глубин центров: {min(depths):.1f} – {max(depths):.1f} м")

    print(f"Среднее буферное расстояние (запас): {np.mean(buffers):.1f} м")
    print(f"Диапазон буферных расстояний: {min(buffers):.1f} – {max(buffers):.1f} м")

    # Проверяем условие: верхняя кромка шара должна быть минимум в 10 м от уровня Z
    min_distance_to_surface = []
    for source in sources:
        x_src, y_src, z_center, density, radius, mass, buffer = source
        # Верхняя кромка шара: z_top = z_center + radius
        z_top = z_center + radius
        # Расстояние от верхней кромки до уровня измерения
        distance = z_level - z_top
        min_distance_to_surface.append(distance)

    print(f"\nПроверка условия расположения шаров:")
    print(f"Минимальное расстояние от верхней кромки до уровня Z: {min(min_distance_to_surface):.1f} м")
    print(f"Максимальное расстояние от верхней кромки до уровня Z: {max(min_distance_to_surface):.1f} м")