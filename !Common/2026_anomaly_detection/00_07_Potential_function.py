"""
Alice
Теперь в программу вычисления гравитационного поля добавь случайный шум и случайный тренд,
описываемый 2D полиномом 1-4 порядков.
Добавление шума и тренда сделай в виде отдельных функций, где задаются параметры шума и тренда
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

def add_trend_polynomial(field, X, Y, order=2, trend_coefficients=None):
    """
    Добавляет тренд, описываемый 2D полиномом заданного порядка.

    Параметры:
    - field: исходное гравитационное поле (2D массив)
    - X, Y: координаты сетки (2D массивы)
    - order: порядок полинома (1–4)
    - trend_coefficients: коэффициенты полинома. Если None, генерируются случайно
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

    trend_field = field + trend
    return trend_field, trend_coefficients

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

    # Отмечаем источники
    source_x, source_y, _, _, _, _, _ = zip(*sources)
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
        noise_level=0.45,  # 0.05 - 5% от стандартного отклонения исходного поля
        noise_type='gaussian'
    )
    print(f"Добавлен гауссовский шум. Диапазон значений с шумом: {np.min(noisy_field):.3f} – {np.max(noisy_field):.3f} мГал")

    # Добавляем тренд (полином 2-го порядка)
    final_field, trend_coeffs = add_trend_polynomial(
        noisy_field,
        X, Y,
        order=3, # 2
        trend_coefficients=None  # случайные коэффициенты
    )
    print(f"Добавлен полиномиальный тренд. Диапазон значений с трендом: {np.min(final_field):.3f} – {np.max(final_field):.3f} мГал")
    print(f"Коэффициенты тренда: {trend_coeffs}")

    # Визуализируем результаты
    fig1 = plot_gravitational_field(X, Y, gravitational_field, sources, z_level, "Исходное гравитационное поле")
    fig2 = plot_gravitational_field(X, Y, noisy_field, sources, z_level, "Гравитационное поле с шумом")
    fig3 = plot_gravitational_field(X, Y, final_field, sources, z_level, "Гравитационное поле с шумом и трендом")

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

    if min(min_distance_to_surface) >= 10:
        print("✓ Условие выполнено: все шары расположены минимум в 10 м от уровня измерения")
    else:
        # Находим проблемные источники
        problematic_sources = []
        for i, source in enumerate(sources):
            x_src, y_src, z_center, density, radius, mass, buffer = source
            z_top = z_center + radius
            distance = z_level - z_top
            if distance < 10:
                problematic_sources.append((i, distance, radius, z_center))
        print(f"✗ ВНИМАНИЕ: найдено {len(problematic_sources)} источников, нарушающих условие:")
        for idx, dist, rad, zc in problematic_sources:
            print(f"   Источник {idx+1}: расстояние = {dist:.1f} м, радиус = {rad:.1f} м, глубина центра = {zc:.1f} м")

    # Статистика по гравитационному полю
    print(f"\nСтатистика гравитационного поля (окончательное):")
    print(f"Минимум: {np.min(final_field):.3f} мГал")
    print(f"Максимум: {np.max(final_field):.3f} мГал")
    print(f"Среднее: {np.mean(final_field):.3f} мГал")
    print(f"Стандартное отклонение: {np.std(final_field):.3f} мГал")

    # Дополнительная информация о самых сильных аномалиях
    print(f"\nТоп-3 самых сильных аномалий по массе:")
    sorted_sources = sorted(sources, key=lambda x: x[5], reverse=True)
    for i, source in enumerate(sorted_sources[:3]):
        x_src, y_src, z_center, density, radius, mass, buffer = source
        z_top = z_center + radius  # верхняя кромка
        distance_to_z = z_level - z_top  # расстояние до уровня измерения
        print(f"{i+1}. Масса: {mass:.2e} кг, "
              f"плотность: {density:.3f} г/см³, "
              f"радиус: {radius:.1f} м, "
              f"глубина центра: {z_center:.1f} м, "
              f"расстояние до Z: {distance_to_z:.1f} м")

    # Анализ распределения глубин
    print(f"\nАнализ распределения глубин:")
    shallow_sources = sum(1 for d in depths if d < 0)  # выше уровня моря (отрицательные глубины)
    deep_sources = sum(1 for d in depths if d <= -1000)  # очень глубокие (>1000 м ниже уровня моря)
    medium_sources = len(sources) - shallow_sources - deep_sources
    print(f"Источников выше уровня моря: {shallow_sources}")
    print(f"Источников на глубине >1000 м: {deep_sources}")
    print(f"Источников на средней глубине: {medium_sources}")

    # Дополнительная визуализация: сравнение полей
    fig_compare, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig_compare.suptitle("Сравнение гравитационных полей на разных этапах обработки", fontsize=16)

    cmap = LinearSegmentedColormap.from_list(
        "grav_cmap", ["darkblue", "blue", "cyan", "green", "yellow", "orange", "red"]
    )

    # Исходное поле
    contour1 = axes[0, 0].contourf(X/1000, Y/1000, gravitational_field, levels=50, cmap=cmap, alpha=0.9)
    axes[0, 0].set_title("Исходное поле")
    axes[0, 0].set_xlabel("X, км")
    axes[0, 0].set_ylabel("Y, км")
    plt.colorbar(contour1, ax=axes[0, 0])

    # Поле с шумом
    contour2 = axes[0, 1].contourf(X/1000, Y/1000, noisy_field, levels=50, cmap=cmap, alpha=0.9)
    axes[0, 1].set_title("С шумом")
    axes[0, 1].set_xlabel("X, км")
    axes[0, 1].set_ylabel("Y, км")
    plt.colorbar(contour2, ax=axes[0, 1])

    # Поле с трендом
    contour3 = axes[1, 0].contourf(X/1000, Y/1000, final_field, levels=50, cmap=cmap, alpha=0.9)
    axes[1, 0].set_title("С шумом и трендом")
    axes[1, 0].set_xlabel("X, км")
    axes[1, 0].set_ylabel("Y, км")
    plt.colorbar(contour3, ax=axes[1, 0])

    # Разница между исходным и финальным полем
    difference = final_field - gravitational_field
    contour4 = axes[1, 1].contourf(X/1000, Y/1000, difference, levels=50, cmap='RdBu', alpha=0.9)
    axes[1, 1].set_title("Разница (финальное − исходное)")
    axes[1, 1].set_xlabel("X, км")
    axes[1, 1].set_ylabel("Y, км")
    plt.colorbar(contour4, ax=axes[1, 1])

    plt.tight_layout()
    plt.show()

    # Статистика по добавленным компонентам
    print(f"\nСтатистика добавленных компонентов:")
    trend_only = final_field - noisy_field
    noise_only = noisy_field - gravitational_field

    print(f"Шум: диапазон {np.min(noise_only):.3f} – {np.max(noise_only):.3f} мГал, "
          f"среднее: {np.mean(noise_only):.3f} мГал, "
          f"стандартное отклонение: {np.std(noise_only):.3f} мГал")
    print(f"Тренд: диапазон {np.min(trend_only):.3f} – {np.max(trend_only):.3f} мГал, "
          f"среднее: {np.mean(trend_only):.3f} мГал, "
          f"стандартное отклонение: {np.std(trend_only):.3f} мГал")

    print("\nПрограмма завершена успешно!")
