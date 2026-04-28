"""
Alice
Модифицируй программу расчета чтобы в ней
1) рассчитывалось гравитационное поле на некотором заданном уровне Z (диапазон значений от 0 до 1500 м над уровнем моря)
2) глубины залегания центров изменялись так, чтобы верхняя кромка каждого шара была как минимум в 10 м от заданного уровня Z
3) радиусы шаров от 0.5 м до 3000 м
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
    """
    Создаёт карту гравитационного поля (в мГал) от 15–20 сферических аномалий плотности
    на заданном уровне Z. Шары могут залегать на любой глубине, главное условие:
    верхняя кромка шара должна быть минимум в 10 м от уровня измерения.

    Параметры:
    - n_sources: количество источников
    - grid_size: размер сетки
    - density_range: диапазон аномальных плотностей в г/см³
    - radius_range: диапазон радиусов источников в метрах
    - z_level: высота измерения над уровнем моря (0–1500 м)
    - smoothing_sigma: параметр сглаживания
    """

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
        # z_center = z_level - (radius + запас)
        # Запас: случайное значение от 10 до 100 м
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

def plot_gravitational_field(X, Y, field_mgal, sources, z_level, title="Гравитационное поле (мГал)"):
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
        density_range=(0.01, 0.3),  # г/см³
        radius_range=(0.5, 3000),  # метры
        z_level=measurement_level,  # уровень измерения
        smoothing_sigma=3.0
    )

    # Визуализируем результаты
    fig = plot_gravitational_field(X, Y, gravitational_field, sources, z_level)

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
            print(
                f"   Источник {idx + 1}: расстояние = {dist:.1f} м, радиус = {rad:.1f} м, глубина центра = {zc:.1f} м")

#    else:
#    print("✓ Все источники удовлетворяют условию расположения")

# Статистика по гравитационному полю
print(f"\nСтатистика гравитационного поля:")
print(f"Минимум: {np.min(gravitational_field):.3f} мГал")
print(f"Максимум: {np.max(gravitational_field):.3f} мГал")
print(f"Среднее: {np.mean(gravitational_field):.3f} мГал")
print(f"Стандартное отклонение: {np.std(gravitational_field):.3f} мГал")

# Дополнительная информация о самых сильных аномалиях
print(f"\nТоп-3 самых сильных аномалий по массе:")
sorted_sources = sorted(sources, key=lambda x: x[5], reverse=True)
for i, source in enumerate(sorted_sources[:3]):
    x_src, y_src, z_center, density, radius, mass, buffer = source
    z_top = z_center + radius  # верхняя кромка
    distance_to_z = z_level - z_top  # расстояние до уровня измерения
    print(f"{i + 1}. Масса: {mass:.2e} кг, "
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

print("\nПрограмма завершена успешно!")
