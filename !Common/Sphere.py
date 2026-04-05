"""
Alice
Модифицируй программу так, чтобы:
1) можно было задавать несколько раз тренды (от 1 до 4)
2) в параметрах функции add_trend_polynomial можно было задавать коэффициент, который бы изменял размах поля тренда
3) добавлено минимальное расстояние между шарами (не касаются и не пересекаются)
4) добавлена 3D визуализация взаимного положения шаров
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

# Константа гравитации в м³·кг⁻¹·с⁻²
G = 6.67430e-11
# Коэффициент перевода в мГал: 1 м/с² = 100 000 мГал
M_TO_MGAL = 1e5
# Перевод г/см³ в кг/м³: 1 г/см³ = 1000 кг/м³
DENSITY_CONVERSION = 1000


def check_sphere_collision(x1, y1, z1, r1, x2, y2, z2, r2, min_distance_factor=1.0):
    """
    Проверяет, пересекаются ли две сферы.

    Параметры:
    - min_distance_factor: коэффициент минимального расстояния (1.0 = поверхности касаются)
    Возвращает:
    - True если пересекаются или касаются, False если нет
    """
    center_distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
    min_distance = (r1 + r2) * min_distance_factor
    return center_distance < min_distance, center_distance


def resolve_sphere_collision(s1, s2, strategy='horizontal'):
    """
    Разрешает коллизию между двумя сферами.

    Параметры:
    - s1, s2: словари с параметрами сфер (x, y, z, radius, density)
    - strategy: 'horizontal' - раздвинуть по горизонтали
                'vertical' - раздвинуть по вертикали
                'radius' - уменьшить радиус, увеличить плотность

    Возвращает:
    - модифицированные сферы
    """
    x1, y1, z1, r1, d1 = s1['x'], s1['y'], s1['z'], s1['radius'], s1['density']
    x2, y2, z2, r2, d2 = s2['x'], s2['y'], s2['z'], s2['radius'], s2['density']

    center_dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
    min_dist = r1 + r2
    overlap = min_dist - center_dist

    if overlap <= 0:
        return s1, s2

    if strategy == 'horizontal':
        # Раздвигаем по горизонтали
        if center_dist > 0:
            dx = x2 - x1
            dy = y2 - y1
            dist_xy = np.sqrt(dx ** 2 + dy ** 2)
            if dist_xy > 0:
                shift = (overlap + 1) / 2  # небольшой запас
                s1_new = s1.copy()
                s2_new = s2.copy()
                s1_new['x'] = x1 - (dx / dist_xy) * shift
                s1_new['y'] = y1 - (dy / dist_xy) * shift
                s2_new['x'] = x2 + (dx / dist_xy) * shift
                s2_new['y'] = y2 + (dy / dist_xy) * shift
                return s1_new, s2_new

    elif strategy == 'vertical':
        # Раздвигаем по вертикали
        shift = (overlap + 1) / 2
        s1_new = s1.copy()
        s2_new = s2.copy()
        # Опускаем более глубокую сферу ещё глубже, поднимаем более мелкую
        if z1 < z2:  # z1 глубже (меньше значение, т.к. ось вверх)
            s1_new['z'] = z1 - shift
            s2_new['z'] = z2 + shift
        else:
            s1_new['z'] = z1 + shift
            s2_new['z'] = z2 - shift
        return s1_new, s2_new

    elif strategy == 'radius':
        # Уменьшаем радиусы, увеличиваем плотность (сохраняем массу)
        volume1 = (4 / 3) * np.pi * r1 ** 3
        volume2 = (4 / 3) * np.pi * r2 ** 3
        mass1 = d1 * volume1
        mass2 = d2 * volume2

        # Уменьшаем радиусы пропорционально
        scale_factor = min_dist / (center_dist + 0.1)
        new_r1 = r1 * scale_factor
        new_r2 = r2 * scale_factor

        # Увеличиваем плотность для сохранения массы
        new_d1 = mass1 / ((4 / 3) * np.pi * new_r1 ** 3) if new_r1 > 0 else d1
        new_d2 = mass2 / ((4 / 3) * np.pi * new_r2 ** 3) if new_r2 > 0 else d2

        s1_new = s1.copy()
        s2_new = s2.copy()
        s1_new['radius'] = new_r1
        s1_new['density'] = new_d1
        s2_new['radius'] = new_r2
        s2_new['density'] = new_d2
        return s1_new, s2_new

    return s1, s2


def generate_non_overlapping_sources(n_sources, x_range=(-4000, 4000), y_range=(-4000, 4000),
                                     radius_range=(0.5, 3000), density_range=(0.01, 0.3),
                                     z_level=500, min_distance_factor=1.2, max_attempts=100):
    """
    Генерирует источники-шары, которые не пересекаются друг с другом.

    Параметры:
    - min_distance_factor: минимальное расстояние между центрами в радиусах (1.2 = зазор 20%)
    - max_attempts: максимальное количество попыток разместить шар
    """
    sources = []

    for i in range(n_sources):
        for attempt in range(max_attempts):
            # Генерируем случайные параметры
            x_source = np.random.uniform(*x_range)
            y_source = np.random.uniform(*y_range)
            radius = np.random.uniform(*radius_range)

            # Глубина центра шара
            buffer_distance = np.random.uniform(10, 100)
            z_center = z_level - (radius + buffer_distance)

            # Аномальная плотность
            density_g_cm3 = np.random.uniform(*density_range)
            density_kg_m3 = density_g_cm3 * DENSITY_CONVERSION

            # Проверяем коллизии с существующими шарами
            collision = False
            for existing in sources:
                x_e, y_e, z_e, d_e, r_e, m_e, b_e = existing
                center_dist = np.sqrt((x_source - x_e) ** 2 + (y_source - y_e) ** 2 + (z_center - z_e) ** 2)
                min_dist = (radius + r_e) * min_distance_factor
                if center_dist < min_dist:
                    collision = True
                    break

            if not collision:
                volume = (4 / 3) * np.pi * radius ** 3
                mass = density_kg_m3 * volume
                sources.append((x_source, y_source, z_center, density_g_cm3, radius, mass, buffer_distance))
                break
        else:
            # Если не удалось разместить шар после всех попыток, размещаем с предупреждением
            print(f"Предупреждение: Шар {i + 1} не удалось разместить без пересечений после {max_attempts} попыток")
            volume = (4 / 3) * np.pi * radius ** 3
            mass = density_kg_m3 * volume
            sources.append((x_source, y_source, z_center, density_g_cm3, radius, mass, buffer_distance))

    return sources


def create_gravitational_field_map(n_sources=18, grid_size=250,
                                   density_range=(0.01, 0.3),  # г/см³
                                   radius_range=(0.5, 3000),  # метры
                                   z_level=500,  # высота измерения над уровнем моря (м)
                                   smoothing_sigma=2.0,
                                   min_distance_factor=1.2):
    """Создаёт карту гравитационного поля (в мГал) от сферических аномалий плотности."""
    # Создаём сетку координат в метрах
    x = np.linspace(-5000, 5000, grid_size)  # -5 км до +5 км
    y = np.linspace(-5000, 5000, grid_size)
    X, Y = np.meshgrid(x, y)

    # Инициализируем гравитационное поле
    g_total = np.zeros_like(X)

    # Генерируем непересекающиеся источники
    np.random.seed(42)  # для воспроизводимости
    sources = generate_non_overlapping_sources(
        n_sources=n_sources,
        x_range=(-4000, 4000),
        y_range=(-4000, 4000),
        radius_range=radius_range,
        density_range=density_range,
        z_level=z_level,
        min_distance_factor=min_distance_factor
    )

    for source in sources:
        x_source, y_source, z_center, density_g_cm3, radius, mass, buffer_distance = source
        density_kg_m3 = density_g_cm3 * DENSITY_CONVERSION

        # Вычисляем расстояния от каждой точки сетки до источника в 3D
        dx = X - x_source
        dy = Y - y_source
        dz = z_level - z_center  # разница по высоте
        r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        # Избегаем деления на ноль
        r = np.maximum(r, 1e-6)

        # Вертикальная компонента гравитационного ускорения (в м/с²)
        g_vertical = G * mass * dz / r ** 3

        # Суммируем вклады всех источников
        g_total += g_vertical

    # Переводим из м/с² в миллиГалы (1 м/с² = 100 000 мГал)
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
            trend += trend_coefficients[coeff_idx] * (x_norm ** i) * (y_norm ** j)
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


def plot_3d_spheres(sources, z_level, title="3D визуализация аномалий плотности"):
    """
    Создаёт 3D визуализацию шаров-источников с фиксированным соотношением осей.

    Параметры:
    - sources: список источников (x, y, z_center, density, radius, mass, buffer)
    - z_level: уровень измерения
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Цветовая карта для плотностей
    densities = [s[3] for s in sources]
    norm = plt.Normalize(min(densities), max(densities))
    cmap = plt.cm.viridis

    # Определяем границы для всех осей
    all_x = []
    all_y = []
    all_z = []

    # Отрисовка каждого шара
    for source in sources:
        x, y, z_center, density, radius, mass, buffer = source

        # Сохраняем координаты для определения границ
        all_x.extend([x - radius, x + radius])
        all_y.extend([y - radius, y + radius])
        all_z.extend([z_center - radius, z_center + radius])

        # Создаём сферу с более высоким разрешением
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        sphere_x = x + radius * np.outer(np.cos(u), np.sin(v))
        sphere_y = y + radius * np.outer(np.sin(u), np.sin(v))
        sphere_z = z_center + radius * np.outer(np.ones(np.size(u)), np.cos(v))

        # Цвет в зависимости от плотности
        color = cmap(norm(density))
        ax.plot_surface(sphere_x, sphere_y, sphere_z, color=color, alpha=0.6, edgecolor='none', linewidth=0)

    # Добавляем уровень измерения
    x_range = np.array([-5000, 5000])
    y_range = np.array([-5000, 5000])
    X_mesh, Y_mesh = np.meshgrid(x_range, y_range)
    Z_mesh = np.ones_like(X_mesh) * z_level
    ax.plot_surface(X_mesh, Y_mesh, Z_mesh, alpha=0.3, color='lightblue', edgecolor='none')

    # Определяем границы для всех осей
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    z_min, z_max = min(all_z), max(all_z)

    # Добавляем небольшой отступ (10% от диапазона)
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center_plot = (z_min + z_max) / 2

    # Определяем максимальный диапазон для кубического отображения
    max_range = max(x_range, y_range, z_range) * 0.6

    # Устанавливаем одинаковые границы для всех осей
    ax.set_xlim(x_center - max_range, x_center + max_range)
    ax.set_ylim(y_center - max_range, y_center + max_range)
    ax.set_zlim(z_center_plot - max_range, z_center_plot + max_range)

    # Настройка меток
    ax.set_xlabel('X, м')
    ax.set_ylabel('Y, м')
    ax.set_zlabel('Z, м')
    ax.set_title(title)

    # Добавляем цветовую шкалу
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(densities)
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label('Аномальная плотность, г/см³')

    # Отображаем информацию
    ax.text2D(0.02, 0.98, f"Уровень измерения: {z_level} м\nКоличество источников: {len(sources)}",
              transform=ax.transAxes, fontsize=10, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Настройка угла обзора для лучшей видимости
    ax.view_init(elev=25, azim=-60)

    # Включаем равное соотношение осей (альтернативный метод)
    ax.set_box_aspect([1, 1, 1])

    plt.tight_layout()
    plt.show()

    return fig

def plot_gravitational_field(X, Y, field_mgal, sources, z_level, title="Гравитационное поле (мГал)"):
    """Визуализирует карту гравитационного поля и источники."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Первая панель: цветовая карта гравитационного поля
    cmap = LinearSegmentedColormap.from_list(
        "grav_cmap", ["darkblue", "blue", "cyan", "green", "yellow", "orange", "red"]
    )

    contour = ax1.contourf(X / 1000, Y / 1000, field_mgal, levels=50, cmap=cmap, alpha=0.9)
    cbar = plt.colorbar(contour, ax=ax1)
    cbar.set_label('Гравитационное поле, мГал', rotation=270, labelpad=20)

    # Извлекаем координаты источников
    source_x = []
    source_y = []
    source_radii = []
    for source in sources:
        x_src, y_src, z_center, density, radius, mass, buffer = source
        source_x.append(x_src)
        source_y.append(y_src)
        source_radii.append(radius)

    # Отмечаем источники (размер кружка пропорционален радиусу)
    sizes = np.array(source_radii) / 50  # нормализация для визуализации
    ax1.scatter(
        np.array(source_x) / 1000, np.array(source_y) / 1000,
        c='white', s=sizes, edgecolors='black', linewidth=2,
        zorder=6, label=f'{len(sources)} аномалий плотности'
    )

    ax1.set_title(f'Цветовая карта гравитационного поля\nУровень измерения: {z_level} м над уровнем моря')
    ax1.set_xlabel('X, км')
    ax1.set_ylabel('Y, км')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Вторая панель: линии уровня (изолинии)
    contour_lines = ax2.contour(X / 1000, Y / 1000, field_mgal, levels=25, colors='black', linewidths=0.8)
    ax2.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')

    # Отмечаем источники на карте изолиний
    ax2.scatter(np.array(source_x) / 1000, np.array(source_y) / 1000,
                c='red', s=sizes, edgecolors='black', zorder=6,
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
    print("Шары генерируются без пересечений друг с другом.\n")

    # Задаём уровень измерения (можно менять от 0 до 1500 м)
    measurement_level = 800  # м над уровнем моря

    # Создаём карту с 18 источниками (без пересечений)
    X, Y, gravitational_field, sources, z_level = create_gravitational_field_map(
        n_sources=18,
        grid_size=250,
        density_range=(0.01, 0.3),
        radius_range=(0.5, 3000),
        z_level=measurement_level,
        smoothing_sigma=3.0,
        min_distance_factor=1.2  # зазор 20% между шарами
    )

    print(
        f"Исходное поле создано. Диапазон значений: {np.min(gravitational_field):.3f} – {np.max(gravitational_field):.3f} мГал")

    # 3D визуализация шаров
    print("\nОтображение 3D визуализации шаров-источников...")
    plot_3d_spheres(sources, z_level, "3D визуализация аномалий плотности (шары не пересекаются)")

    # Добавляем случайный шум
    noisy_field = add_random_noise(
        gravitational_field,
        noise_level=0.05,
        noise_type='gaussian'
    )
    print(
        f"Добавлен гауссовский шум. Диапазон значений с шумом: {np.min(noisy_field):.3f} – {np.max(noisy_field):.3f} мГал")

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

    print(
        f"Добавлено {len(trends_config)} трендов. Диапазон значений с трендами: {np.min(final_field):.3f} – {np.max(final_field):.3f} мГал")
    for i, (coeffs, factor) in enumerate(zip(all_coeffs, all_factors)):
        print(f"Тренд {i + 1}: порядок {trends_config[i]['order']}, коэффициент амплитуды {factor:.2f}")

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
    depths = [s[2] for s in sources]
    buffers = [s[6] for s in sources]

    print(f"Средняя аномальная плотность: {np.mean(densities_g_cm3):.3f} г/см³")
    print(f"Диапазон плотностей: {min(densities_g_cm3):.3f} – {max(densities_g_cm3):.3f} г/см³")

    print(f"Средний радиус: {np.mean(radii):.1f} м")
    print(f"Диапазон радиусов: {min(radii):.1f} – {max(radii):.1f} м")

    print(f"Средняя глубина центра: {np.mean(depths):.1f} м")
    print(f"Диапазон глубин центров: {min(depths):.1f} – {max(depths):.1f} м")

    print(f"Среднее буферное расстояние (запас): {np.mean(buffers):.1f} м")
    print(f"Диапазон буферных расстояний: {min(buffers):.1f} – {max(buffers):.1f} м")

    # Проверяем условие расположения шаров
    min_distance_to_surface = []
    for source in sources:
        x_src, y_src, z_center, density, radius, mass, buffer = source
        z_top = z_center + radius
        distance = z_level - z_top
        min_distance_to_surface.append(distance)

    print(f"\nПроверка условия расположения шаров:")
    print(f"Минимальное расстояние от верхней кромки до уровня Z: {min(min_distance_to_surface):.1f} м")
    print(f"Максимальное расстояние от верхней кромки до уровня Z: {max(min_distance_to_surface):.1f} м")

    # Проверяем расстояния между шарами
    print(f"\nПроверка расстояний между шарами:")
    min_center_distance = float('inf')
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            x1, y1, z1, _, r1, _, _ = sources[i]
            x2, y2, z2, _, r2, _, _ = sources[j]
            center_dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
            min_center_distance = min(min_center_distance, center_dist)
            surface_dist = center_dist - (r1 + r2)
            if surface_dist < 0:
                print(f"  ВНИМАНИЕ! Шары {i + 1} и {j + 1} пересекаются! Зазор: {surface_dist:.1f} м")

    if min_center_distance > 0:
        print(f"  Минимальное расстояние между центрами шаров: {min_center_distance:.1f} м")