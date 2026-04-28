"""
МЕТОД А: Выделение аномалий по правилу "трёх сигм" (3σ)
Программа для анализа гравитационного поля от сферических аномалий плотности

Использует:
- Модельные данные от 18 сферических источников
- Сравнение результатов на поле без шума/тренда и с шумом/трендом
- Визуализация выделенных аномалий в виде многоугольников (≥12 вершин) и эллипсов
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon, Ellipse
from scipy.ndimage import gaussian_filter, binary_closing, binary_opening, label
from skimage import measure
from typing import List, Tuple, Dict, Any

# ==================== КОНСТАНТЫ ====================
G = 6.67430e-11  # гравитационная постоянная
M_TO_MGAL = 1e5  # перевод м/с² в мГал
DENSITY_CONVERSION = 1000  # перевод г/см³ в кг/м³


# ==================== ГЕНЕРАЦИЯ МОДЕЛЬНЫХ ДАННЫХ ====================

def generate_non_overlapping_sources(n_sources: int, x_range: Tuple = (-4000, 4000),
                                     y_range: Tuple = (-4000, 4000),
                                     radius_range: Tuple = (0.5, 3000),
                                     density_range: Tuple = (0.01, 0.3),
                                     z_level: float = 500,
                                     min_distance_factor: float = 1.2,
                                     max_attempts: int = 100):
    """
    Генерирует источники-шары, которые не пересекаются друг с другом
    """
    sources = []

    for _ in range(n_sources):
        for attempt in range(max_attempts):
            x_source = np.random.uniform(*x_range)
            y_source = np.random.uniform(*y_range)
            radius = np.random.uniform(*radius_range)
            buffer_distance = np.random.uniform(10, 100)
            z_center = z_level - (radius + buffer_distance)
            density_g_cm3 = np.random.uniform(*density_range)

            collision = False
            for existing in sources:
                x_e, y_e, z_e, d_e, r_e, m_e, b_e = existing
                center_dist = np.sqrt((x_source - x_e) ** 2 + (y_source - y_e) ** 2 + (z_center - z_e) ** 2)
                min_dist = (radius + r_e) * min_distance_factor
                if center_dist < min_dist:
                    collision = True
                    break

            if not collision:
                density_kg_m3 = density_g_cm3 * DENSITY_CONVERSION
                volume = (4 / 3) * np.pi * radius ** 3
                mass = density_kg_m3 * volume
                sources.append((x_source, y_source, z_center, density_g_cm3, radius, mass, buffer_distance))
                break

    return sources


def create_gravitational_field_map(n_sources: int = 18, grid_size: int = 250,
                                   density_range: Tuple = (0.01, 0.3),
                                   radius_range: Tuple = (0.5, 3000),
                                   z_level: float = 500,
                                   smoothing_sigma: float = 2.0,
                                   min_distance_factor: float = 1.2) -> Tuple:
    """
    Создаёт карту гравитационного поля от сферических аномалий
    """
    x = np.linspace(-5000, 5000, grid_size)
    y = np.linspace(-5000, 5000, grid_size)
    X, Y = np.meshgrid(x, y)

    np.random.seed(42)
    sources = generate_non_overlapping_sources(n_sources, x_range=(-4000, 4000),
                                               y_range=(-4000, 4000),
                                               radius_range=radius_range,
                                               density_range=density_range,
                                               z_level=z_level,
                                               min_distance_factor=min_distance_factor)

    g_total = np.zeros_like(X)

    for source in sources:
        x_source, y_source, z_center, density_g_cm3, radius, mass, buffer_distance = source
        dx = X - x_source
        dy = Y - y_source
        dz = z_level - z_center
        r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        r = np.maximum(r, 1e-6)
        g_vertical = G * mass * dz / r ** 3
        g_total += g_vertical

    g_mgal = g_total * M_TO_MGAL
    g_smoothed = gaussian_filter(g_mgal, sigma=smoothing_sigma)

    return X, Y, g_smoothed, sources, z_level


def add_random_noise(field: np.ndarray, noise_level: float = 0.05,
                     noise_type: str = 'gaussian') -> np.ndarray:
    """
    Добавляет случайный шум к гравитационному полю
    """
    if noise_type == 'gaussian':
        noise = np.random.normal(0, noise_level * np.std(field), field.shape)
    elif noise_type == 'uniform':
        noise = np.random.uniform(-noise_level * np.std(field),
                                  noise_level * np.std(field), field.shape)
    else:
        noise = np.zeros(field.shape)

    return field + noise


def add_trend_polynomial(field: np.ndarray, X: np.ndarray, Y: np.ndarray,
                         order: int = 2, amplitude_factor: float = 1.0) -> np.ndarray:
    """
    Добавляет полиномиальный тренд к полю
    """
    x_norm = X / 1000
    y_norm = Y / 1000

    trend = np.zeros_like(field)
    coeff_idx = 0
    np.random.seed(42)

    for i in range(order + 1):
        for j in range(order - i + 1):
            coeff = np.random.uniform(-0.5, 0.5)
            trend += coeff * (x_norm ** i) * (y_norm ** j)
            coeff_idx += 1

    trend *= amplitude_factor
    return field + trend


def apply_multiple_trends(field: np.ndarray, X: np.ndarray, Y: np.ndarray,
                          trends_config: List[Dict]) -> np.ndarray:
    """
    Применяет несколько трендов последовательно
    """
    current_field = field.copy()
    for trend_params in trends_config:
        order = trend_params.get('order', 2)
        amplitude_factor = trend_params.get('amplitude_factor', 1.0)
        current_field = add_trend_polynomial(current_field, X, Y, order, amplitude_factor)
    return current_field


# ==================== МЕТОД А: ВЫДЕЛЕНИЕ АНОМАЛИЙ ПО 3σ ====================

def detect_anomalies_3sigma(field: np.ndarray, X: np.ndarray, Y: np.ndarray,
                            n_sigma: float = 3.0, min_points: int = 3,
                            neighbor_profiles: int = 2) -> Dict[str, Any]:
    """
    Выделение аномалий по правилу "трёх сигм"

    Параметры:
    - field: гравитационное поле
    - X, Y: координатные сетки
    - n_sigma: количество стандартных отклонений
    - min_points: минимальное количество точек подряд
    - neighbor_profiles: количество соседних профилей

    Возвращает:
    - словарь с маской, многоугольниками, статистикой
    """
    mean_val = np.mean(field)
    std_val = np.std(field)

    threshold_upper = mean_val + n_sigma * std_val
    threshold_lower = mean_val - n_sigma * std_val

    # Бинарная маска
    anomaly_mask = (field > threshold_upper) | (field < threshold_lower)

    # Проверка по соседним профилям
    rows, cols = anomaly_mask.shape
    filtered_mask = np.zeros_like(anomaly_mask)

    for i in range(rows - neighbor_profiles + 1):
        for j in range(cols - min_points + 1):
            consecutive = True
            for k in range(neighbor_profiles):
                if not np.all(anomaly_mask[i + k, j:j + min_points]):
                    consecutive = False
                    break
            if consecutive:
                for k in range(neighbor_profiles):
                    filtered_mask[i + k, j:j + min_points] = True

    # Морфологическая обработка
    struct_elem = np.ones((3, 3))
    filtered_mask = binary_closing(filtered_mask, struct_elem)
    filtered_mask = binary_opening(filtered_mask, struct_elem)

    # Выделение связных областей
    labeled_mask, num_anomalies = label(filtered_mask)

    # Создание многоугольников
    x_km = X / 1000
    y_km = Y / 1000
    cell_size = X[0, 1] - X[0, 0]

    polygons = []
    ellipses = []

    for label_id in range(1, num_anomalies + 1):
        mask_label = (labeled_mask == label_id)
        contours = measure.find_contours(mask_label, 0.5)

        if not contours:
            continue

        contour = max(contours, key=len)
        contour_x = np.interp(contour[:, 1], np.arange(len(x_km[0])), x_km[0])
        contour_y = np.interp(contour[:, 0], np.arange(len(y_km[:, 0])), y_km[:, 0])

        # Упрощаем до 12 вершин
        if len(contour_x) > 12:
            indices = np.linspace(0, len(contour_x) - 1, 12, dtype=int)
            contour_x = contour_x[indices]
            contour_y = contour_y[indices]

        contour_x = np.append(contour_x, contour_x[0])
        contour_y = np.append(contour_y, contour_y[0])

        polygons.append(Polygon(np.column_stack([contour_x, contour_y]),
                                closed=True, fill=False, edgecolor='red', linewidth=2))

        # Аппроксимация эллипсом
        center_x = np.mean(contour_x[:-1])
        center_y = np.mean(contour_y[:-1])
        radius_x = np.max(np.abs(contour_x[:-1] - center_x))
        radius_y = np.max(np.abs(contour_y[:-1] - center_y))
        ellipses.append(Ellipse((center_x, center_y), 2 * radius_x, 2 * radius_y,
                                fill=False, edgecolor='blue', linewidth=2, linestyle='--'))

    return {
        'mask': filtered_mask,
        'num_anomalies': num_anomalies,
        'polygons': polygons,
        'ellipses': ellipses,
        'threshold_upper': threshold_upper,
        'threshold_lower': threshold_lower,
        'mean': mean_val,
        'std': std_val
    }


# ==================== ВИЗУАЛИЗАЦИЯ ====================

def create_custom_colormap():
    """Создаёт пользовательскую цветовую карту для гравитационного поля"""
    return LinearSegmentedColormap.from_list(
        "grav_cmap", ["darkblue", "blue", "cyan", "green", "yellow", "orange", "red"]
    )


import copy
from matplotlib.collections import PatchCollection


def plot_field_with_anomalies(field: np.ndarray, X: np.ndarray, Y: np.ndarray,
                              sources: List, z_level: float, title: str,
                              detection_result: Dict[str, Any] = None,
                              show_sources: bool = True) -> plt.Figure:
    """
    Визуализирует гравитационное поле с выделенными аномалиями
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    x_km = X / 1000
    y_km = Y / 1000

    # Левая панель: цветовая карта
    cmap = create_custom_colormap()
    contour = ax1.contourf(x_km, y_km, field, levels=50, cmap=cmap, alpha=0.9)
    cbar = plt.colorbar(contour, ax=ax1)
    cbar.set_label('Гравитационное поле, мГал', rotation=270, labelpad=20)

    # Правая панель: изолинии
    contour_lines = ax2.contour(x_km, y_km, field, levels=25, colors='black', linewidths=0.8)
    ax2.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')

    # ===== ОТОБРАЖЕНИЕ ИСТОЧНИКОВ =====
    if show_sources and sources:
        source_x = [s[0] / 1000 for s in sources]
        source_y = [s[1] / 1000 for s in sources]
        source_radii = [s[4] / 1000 for s in sources]
        sizes = np.array(source_radii) * 20

        # Левая панель - белые кружки
        ax1.scatter(source_x, source_y, c='white', s=sizes,
                    edgecolors='black', linewidth=2, zorder=6,
                    label=f'{len(sources)} источников')

        # Правая панель - красные кружки
        ax2.scatter(source_x, source_y, c='red', s=sizes,
                    edgecolors='black', linewidth=2, zorder=6,
                    label=f'{len(sources)} источников')

    # ===== ОТОБРАЖЕНИЕ ВЫДЕЛЕННЫХ АНОМАЛИЙ =====
    if detection_result:
        # Для каждого подграфика создаём свои копии полигонов
        for ax in [ax1, ax2]:
            # Копируем полигоны для текущего подграфика
            for polygon in detection_result['polygons']:
                # Создаём копию полигона с теми же вершинами
                vertices = polygon.get_xy()
                polygon_copy = Polygon(vertices, closed=True, fill=False,
                                       edgecolor='red', linewidth=2)
                ax.add_patch(polygon_copy)

            # Копируем эллипсы для текущего подграфика
            for ellipse in detection_result['ellipses']:
                # Создаём копию эллипса
                ellipse_copy = Ellipse(ellipse.center, ellipse.width, ellipse.height,
                                       fill=False, edgecolor='blue',
                                       linewidth=2, linestyle='--')
                ax.add_patch(ellipse_copy)

            # Добавляем статистику
            stats_text = (f"Аномалий: {detection_result['num_anomalies']}\n"
                          f"μ = {detection_result['mean']:.2f} мГал\n"
                          f"σ = {detection_result['std']:.2f} мГал\n"
                          f"Порог: ±{detection_result['std'] * 3:.2f} мГал")
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Настройка осей
    for ax in [ax1, ax2]:
        ax.set_xlabel('X, км')
        ax.set_ylabel('Y, км')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

    ax1.set_title(f'{title}\nЦветовая карта\nУровень измерения: {z_level} м')
    ax2.set_title(f'{title}\nИзолинии\nУровень измерения: {z_level} м')

    plt.suptitle(f'МЕТОД А: Выделение аномалий по правилу 3σ\n{title}', fontsize=14)
    plt.tight_layout()
    plt.show()

    return fig

def print_source_info(sources: List, z_level: float):
    """Выводит информацию об источниках"""
    print("\n" + "=" * 60)
    print("ИНФОРМАЦИЯ ОБ ИСТОЧНИКАХ (ШАРАХ)")
    print("=" * 60)
    print(f"Количество источников: {len(sources)}")
    print(f"Уровень измерения Z: {z_level} м")

    densities = [s[3] for s in sources]
    radii = [s[4] for s in sources]
    depths = [s[2] for s in sources]
    buffers = [s[6] for s in sources]

    print(f"\nПлотность (г/см³): средняя = {np.mean(densities):.3f}, "
          f"диапазон = [{min(densities):.3f}, {max(densities):.3f}]")
    print(f"Радиус (м): средний = {np.mean(radii):.1f}, "
          f"диапазон = [{min(radii):.1f}, {max(radii):.1f}]")
    print(f"Глубина центра (м): средняя = {np.mean(depths):.1f}, "
          f"диапазон = [{min(depths):.1f}, {max(depths):.1f}]")
    print(f"Буфер (м): средний = {np.mean(buffers):.1f}, "
          f"диапазон = [{min(buffers):.1f}, {max(buffers):.1f}]")

    # Проверка расстояния до уровня измерения
    distances = []
    for s in sources:
        z_top = s[2] + s[4]
        distances.append(z_level - z_top)
    print(f"\nРасстояние от верхней кромки до Z: "
          f"мин = {min(distances):.1f} м, макс = {max(distances):.1f} м")


# ==================== ОСНОВНАЯ ПРОГРАММА ====================

def main():
    print("=" * 60)
    print("МЕТОД А: Выделение аномалий по правилу 'трёх сигм' (3σ)")
    print("=" * 60)

    # Параметры модели
    measurement_level = 800  # м над уровнем моря

    # Конфигурация трендов
    trends_config = [
        {'order': 2, 'amplitude_factor': 0.8},
        {'order': 3, 'amplitude_factor': 0.3},
        {'order': 1, 'amplitude_factor': 0.5},
        {'order': 4, 'amplitude_factor': 0.2}
    ]

    # ===== ГЕНЕРАЦИЯ ДАННЫХ =====
    print("\n1. Генерация модельных данных...")
    X, Y, clean_field, sources, z_level = create_gravitational_field_map(
        n_sources=18,
        grid_size=250,
        density_range=(0.01, 0.3),
        radius_range=(0.5, 3000),
        z_level=measurement_level,
        smoothing_sigma=3.0
    )
    print(f"   Чистое поле: диапазон [{np.min(clean_field):.3f}, {np.max(clean_field):.3f}] мГал")

    # Добавляем шум и тренды
    print("\n2. Добавление шума и трендов...")
    noisy_field = add_random_noise(clean_field, noise_level=0.05, noise_type='gaussian')
    print(f"   После шума: диапазон [{np.min(noisy_field):.3f}, {np.max(noisy_field):.3f}] мГал")

    final_field = apply_multiple_trends(noisy_field, X, Y, trends_config)
    print(
        f"   После {len(trends_config)} трендов: диапазон [{np.min(final_field):.3f}, {np.max(final_field):.3f}] мГал")

    # Выводим информацию об источниках
    print_source_info(sources, z_level)

    # ===== ВЫДЕЛЕНИЕ АНОМАЛИЙ =====
    print("\n" + "=" * 60)
    print("ВЫДЕЛЕНИЕ АНОМАЛИЙ МЕТОДОМ 3σ")
    print("=" * 60)

    print("\n3. Анализ чистого поля (без шума и тренда)...")
    clean_result = detect_anomalies_3sigma(clean_field, X, Y, n_sigma=3.0)
    print(f"   Обнаружено аномалий: {clean_result['num_anomalies']}")
    print(f"   Среднее значение: {clean_result['mean']:.3f} мГал")
    print(f"   Стандартное отклонение: {clean_result['std']:.3f} мГал")
    print(f"   Пороговые значения: [{clean_result['threshold_lower']:.3f}, {clean_result['threshold_upper']:.3f}] мГал")

    print("\n4. Анализ поля с шумом и трендом...")
    final_result = detect_anomalies_3sigma(final_field, X, Y, n_sigma=3.0)
    print(f"   Обнаружено аномалий: {final_result['num_anomalies']}")
    print(f"   Среднее значение: {final_result['mean']:.3f} мГал")
    print(f"   Стандартное отклонение: {final_result['std']:.3f} мГал")
    print(f"   Пороговые значения: [{final_result['threshold_lower']:.3f}, {final_result['threshold_upper']:.3f}] мГал")

    # ===== ВИЗУАЛИЗАЦИЯ =====
    print("\n5. Визуализация результатов...")

    # Чистое поле с аномалиями
    plot_field_with_anomalies(
        clean_field, X, Y, sources, z_level,
        "ЧИСТОЕ ПОЛЕ (без шума и тренда)",
        detection_result=clean_result,
        show_sources=True
    )

    # Поле с шумом и трендом с аномалиями
    plot_field_with_anomalies(
        final_field, X, Y, sources, z_level,
        "ПОЛЕ С ШУМОМ И ТРЕНДАМИ",
        detection_result=final_result,
        show_sources=True
    )

    # ===== ВЫВОД СРАВНИТЕЛЬНОГО АНАЛИЗА =====
    print("\n" + "=" * 60)
    print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ")
    print("=" * 60)

    print(f"\n{'Параметр':<30} {'Чистое поле':<20} {'С шумом и трендом':<20}")
    print("-" * 70)
    print(f"{'Количество аномалий':<30} {clean_result['num_anomalies']:<20} {final_result['num_anomalies']:<20}")
    print(f"{'Среднее значение (мГал)':<30} {clean_result['mean']:<20.3f} {final_result['mean']:<20.3f}")
    print(f"{'Стд. отклонение (мГал)':<30} {clean_result['std']:<20.3f} {final_result['std']:<20.3f}")
    print(
        f"{'Нижний порог (мГал)':<30} {clean_result['threshold_lower']:<20.3f} {final_result['threshold_lower']:<20.3f}")
    print(
        f"{'Верхний порог (мГал)':<30} {clean_result['threshold_upper']:<20.3f} {final_result['threshold_upper']:<20.3f}")

    # Анализ качества выделения
    print("\n" + "=" * 60)
    print("АНАЛИЗ КАЧЕСТВА ВЫДЕЛЕНИЯ")
    print("=" * 60)

    print("\n✓ Метод 3σ основан на предположении о нормальном распределении помех")
    print("✓ Аномалии выделяются как связные области, выходящие за пределы ±3σ")
    print("✓ Добавлены проверки по соседним профилям для подавления случайных выбросов")

    if final_result['num_anomalies'] < len(sources):
        print(f"\n⚠ Предупреждение: Обнаружено {final_result['num_anomalies']} аномалий "
              f"при {len(sources)} реальных источниках")
        print("  Причины могут быть:")
        print("  - Шум и тренды маскируют слабые аномалии")
        print("  - Близкорасположенные источники сливаются в одну аномалию")
        print("  - Некоторые источники имеют малую амплитуду сигнала")
    elif final_result['num_anomalies'] > len(sources):
        print(f"\n⚠ Предупреждение: Обнаружено {final_result['num_anomalies']} аномалий "
              f"при {len(sources)} реальных источниках (ложные срабатывания)")
        print("  Возможные причины:")
        print("  - Шум превышает порог 3σ в некоторых областях")
        print("  - Тренд создаёт дополнительные экстремумы")

    print("\n" + "=" * 60)
    print("Визуализация завершена. Закройте окна для продолжения.")


if __name__ == "__main__":
    main()