"""
Как-то не очень хорошо работает метод с правилом 3 сигма сам по себе.
Чтобы он хорошо работал нужно, чтобы все тренды были сняты.
А так получается, что даже на карте исходного поля выделена только одна аномалия самая большая,
даже вторая по интенсивности аномалия (чуть выше на карте) не выделяется

Метод 3σ действительно плохо работает, когда:
- Есть региональный тренд (фон меняется по площади)
- Аномалии имеют разную амплитуду
- Распределение поля не является нормальным
Давайте улучшим метод 3σ, добавив предварительное снятие тренда и адаптивный порог:
"""

"""
МЕТОД А: Выделение аномалий по правилу "трёх сигм" (3σ)
УЛУЧШЕННАЯ ВЕРСИЯ: с предварительным снятием тренда и адаптивным порогом
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Ellipse
from scipy.ndimage import gaussian_filter, binary_closing, binary_opening, label
from scipy.signal import medfilt2d
from skimage import measure, filters, morphology
from typing import List, Tuple, Dict, Any
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures

# Подавляем предупреждения sklearn
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# ==================== КОНСТАНТЫ ====================
G = 6.67430e-11
M_TO_MGAL = 1e5
DENSITY_CONVERSION = 1000


# ==================== ГЕНЕРАЦИЯ МОДЕЛЬНЫХ ДАННЫХ ====================

def generate_non_overlapping_sources(n_sources: int, x_range: Tuple = (-4000, 4000),
                                      y_range: Tuple = (-4000, 4000),
                                      radius_range: Tuple = (0.5, 3000),
                                      density_range: Tuple = (0.01, 0.3),
                                      z_level: float = 500,
                                      min_distance_factor: float = 1.2,
                                      max_attempts: int = 100):
    """Генерирует источники-шары, которые не пересекаются друг с другом"""
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
                center_dist = np.sqrt((x_source - x_e)**2 + (y_source - y_e)**2 + (z_center - z_e)**2)
                min_dist = (radius + r_e) * min_distance_factor
                if center_dist < min_dist:
                    collision = True
                    break

            if not collision:
                density_kg_m3 = density_g_cm3 * DENSITY_CONVERSION
                volume = (4/3) * np.pi * radius**3
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
    """Создаёт карту гравитационного поля от сферических аномалий"""
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
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        r = np.maximum(r, 1e-6)
        g_vertical = G * mass * dz / r**3
        g_total += g_vertical

    g_mgal = g_total * M_TO_MGAL
    g_smoothed = gaussian_filter(g_mgal, sigma=smoothing_sigma)

    return X, Y, g_smoothed, sources, z_level


def add_random_noise(field: np.ndarray, noise_level: float = 0.05,
                     noise_type: str = 'gaussian') -> np.ndarray:
    """Добавляет случайный шум к гравитационному полю"""
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
    """Добавляет полиномиальный тренд к полю"""
    x_norm = X / 1000
    y_norm = Y / 1000

    trend = np.zeros_like(field)
    np.random.seed(42)

    for i in range(order + 1):
        for j in range(order - i + 1):
            coeff = np.random.uniform(-0.5, 0.5)
            trend += coeff * (x_norm**i) * (y_norm**j)

    trend *= amplitude_factor
    return field + trend


def apply_multiple_trends(field: np.ndarray, X: np.ndarray, Y: np.ndarray,
                          trends_config: List[Dict]) -> np.ndarray:
    """Применяет несколько трендов последовательно"""
    current_field = field.copy()
    for trend_params in trends_config:
        order = trend_params.get('order', 2)
        amplitude_factor = trend_params.get('amplitude_factor', 1.0)
        current_field = add_trend_polynomial(current_field, X, Y, order, amplitude_factor)
    return current_field


# ==================== УЛУЧШЕННЫЙ МЕТОД 3σ ====================

def remove_trend_ransac(field: np.ndarray, X: np.ndarray, Y: np.ndarray,
                        order: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Удаление тренда методом RANSAC (устойчив к аномалиям)
    Возвращает: (поле без тренда, сам тренд)
    """
    x_km = X.flatten() / 1000
    y_km = Y.flatten() / 1000
    z = field.flatten()

    # Создаём полиномиальные признаки
    poly = PolynomialFeatures(degree=order)
    X_poly = poly.fit_transform(np.column_stack([x_km, y_km]))

    # RANSAC для устойчивой аппроксимации
    ransac = RANSACRegressor(random_state=42, min_samples=0.5, residual_threshold=np.std(field) * 0.5)
    ransac.fit(X_poly, z)

    # Предсказываем тренд
    trend = ransac.predict(X_poly).reshape(field.shape)
    residual = field - trend

    return residual, trend


def remove_trend_median_filter(field: np.ndarray, window_size: int = 31) -> np.ndarray:
    """
    Удаление регионального фона медианным фильтром (сохраняет границы аномалий)
    """
    regional = medfilt2d(field, kernel_size=window_size)
    return field - regional


def remove_trend_gaussian(field: np.ndarray, sigma: float = 15.0) -> np.ndarray:
    """
    Удаление регионального фона гауссовским сглаживанием
    """
    regional = gaussian_filter(field, sigma=sigma)
    return field - regional


def detect_anomalies_adaptive_3sigma(field: np.ndarray, X: np.ndarray, Y: np.ndarray,
                                      n_sigma: float = 2.0,  # Уменьшаем порог
                                      window_size: int = 31,  # Уменьшаем окно
                                      min_area_pixels: int = 30,
                                      trend_removal_method: str = 'ransac',
                                      use_local_threshold: bool = True) -> Dict[str, Any]:
    """
    Улучшенный метод 3σ с локальными порогами и предварительным снятием тренда

    Параметры:
    - trend_removal_method: 'ransac', 'median', 'gaussian', или None
    - use_local_threshold: использовать локальные пороги или глобальный
    """
    # Шаг 1: Предварительное снятие глобального тренда
    if trend_removal_method == 'ransac':
        field_detrended, global_trend = remove_trend_ransac(field, X, Y, order=2)
    elif trend_removal_method == 'median':
        field_detrended = remove_trend_median_filter(field, window_size=31)
        global_trend = field - field_detrended
    elif trend_removal_method == 'gaussian':
        field_detrended = remove_trend_gaussian(field, sigma=15.0)
        global_trend = field - field_detrended
    else:
        field_detrended = field.copy()
        global_trend = np.zeros_like(field)

    # Шаг 2: Дополнительное усиление аномалий (нормализация)
    field_norm = (field_detrended - np.mean(field_detrended)) / np.std(field_detrended)

    if use_local_threshold:
        # Локальная пороговая обработка (скользящее окно)
        rows, cols = field.shape
        half_win = window_size // 2

        local_mask = np.zeros_like(field, dtype=bool)

        for i in range(half_win, rows - half_win):
            for j in range(half_win, cols - half_win):
                window = field_norm[i-half_win:i+half_win, j-half_win:j+half_win]
                local_mean = np.mean(window)
                local_std = np.std(window)

                # Локальный порог
                threshold_upper = local_mean + n_sigma * local_std
                threshold_lower = local_mean - n_sigma * local_std

                # Центральная точка
                if field_norm[i, j] > threshold_upper or field_norm[i, j] < threshold_lower:
                    local_mask[i, j] = True
    else:
        # Глобальный порог (проще, но хуже работает)
        threshold = n_sigma
        local_mask = np.abs(field_norm) > threshold

    # Шаг 3: Морфологическая обработка (используем новые функции)
    struct_elem = morphology.disk(3)
    # Заменяем deprecated binary_dilation на morphology.dilation
    local_mask = morphology.dilation(local_mask, struct_elem)
    # Заменяем binary_closing на morphology.closing
    local_mask = morphology.closing(local_mask, morphology.disk(5))
    # Заменяем binary_opening на morphology.opening
    local_mask = morphology.opening(local_mask, morphology.disk(3))

    # Шаг 4: Фильтрация по площади
    labeled_mask, num_labels = label(local_mask)
    for label_id in range(1, num_labels + 1):
        if np.sum(labeled_mask == label_id) < min_area_pixels:
            local_mask[labeled_mask == label_id] = False

    labeled_mask, num_anomalies = label(local_mask)

    # Шаг 5: Создание многоугольников
    x_km = X / 1000
    y_km = Y / 1000

    polygons = []
    ellipses = []
    anomaly_stats = []

    for label_id in range(1, num_anomalies + 1):
        mask_label = (labeled_mask == label_id)
        contours = measure.find_contours(mask_label, 0.5)

        if not contours:
            continue

        contour = max(contours, key=len)

        # Преобразование в координаты
        contour_x = np.interp(contour[:, 1], np.arange(len(x_km[0])), x_km[0])
        contour_y = np.interp(contour[:, 0], np.arange(len(y_km[:, 0])), y_km[:, 0])

        # Упрощение до 12 вершин
        if len(contour_x) > 12:
            indices = np.linspace(0, len(contour_x) - 1, 12, dtype=int)
            contour_x = contour_x[indices]
            contour_y = contour_y[indices]

        contour_x = np.append(contour_x, contour_x[0])
        contour_y = np.append(contour_y, contour_y[0])

        vertices = np.column_stack([contour_x, contour_y])
        polygons.append(Polygon(vertices, closed=True, fill=False, edgecolor='red', linewidth=2))

        # Статистика аномалии
        anomaly_values = field_detrended[mask_label]
        anomaly_stats.append({
            'max': np.max(anomaly_values),
            'min': np.min(anomaly_values),
            'mean': np.mean(anomaly_values),
            'std': np.std(anomaly_values),
            'area_pixels': np.sum(mask_label),
            'area_km2': np.sum(mask_label) * (X[0,1] - X[0,0])**2 / 1e6
        })

        # Эллипс
        if len(contour_x) >= 5:
            center_x = np.mean(contour_x[:-1])
            center_y = np.mean(contour_y[:-1])
            radius_x = np.max(np.abs(contour_x[:-1] - center_x))
            radius_y = np.max(np.abs(contour_y[:-1] - center_y))

            if radius_x > 0 and radius_y > 0:
                ellipses.append(Ellipse((center_x, center_y), 2*radius_x, 2*radius_y,
                                       fill=False, edgecolor='blue', linewidth=2, linestyle='--'))

    return {
        'mask': local_mask,
        'num_anomalies': num_anomalies,
        'polygons': polygons,
        'ellipses': ellipses,
        'anomaly_stats': anomaly_stats,
        'field_detrended': field_detrended,
        'field_norm': field_norm,
        'global_trend': global_trend,
        'method': trend_removal_method,
        'n_sigma': n_sigma
    }


# ==================== ВИЗУАЛИЗАЦИЯ ====================

def create_custom_colormap():
    """Создаёт пользовательскую цветовую карту"""
    return LinearSegmentedColormap.from_list(
        "grav_cmap", ["darkblue", "blue", "cyan", "green", "yellow", "orange", "red"]
    )


def plot_field_with_anomalies(field: np.ndarray, X: np.ndarray, Y: np.ndarray,
                              sources: List, z_level: float, title: str,
                              detection_result: Dict[str, Any] = None,
                              show_sources: bool = True,
                              show_detrended: bool = False) -> plt.Figure:
    """Визуализирует гравитационное поле с выделенными аномалиями"""

    if show_detrended and detection_result and 'field_detrended' in detection_result:
        field_to_plot = detection_result['field_detrended']
        plot_title = f"{title} (после снятия тренда)"
    else:
        field_to_plot = field
        plot_title = title

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    x_km = X / 1000
    y_km = Y / 1000

    # Левая панель: цветовая карта
    cmap = create_custom_colormap()
    contour = ax1.contourf(x_km, y_km, field_to_plot, levels=50, cmap=cmap, alpha=0.9)
    cbar = plt.colorbar(contour, ax=ax1)
    cbar.set_label('Гравитационное поле, мГал', rotation=270, labelpad=20)

    # Правая панель: изолинии
    contour_lines = ax2.contour(x_km, y_km, field_to_plot, levels=25, colors='black', linewidths=0.8)
    ax2.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')

    # Отображаем источники
    if show_sources and sources:
        source_x = [s[0] / 1000 for s in sources]
        source_y = [s[1] / 1000 for s in sources]
        source_radii = [s[4] / 1000 for s in sources]
        sizes = np.array(source_radii) * 20

        ax1.scatter(source_x, source_y, c='white', s=sizes,
                    edgecolors='black', linewidth=2, zorder=6,
                    label=f'{len(sources)} источников')
        ax2.scatter(source_x, source_y, c='red', s=sizes,
                    edgecolors='black', linewidth=2, zorder=6,
                    label=f'{len(sources)} источников')

    # Отображаем выделенные аномалии (СОЗДАЁМ ОТДЕЛЬНЫЕ КОЛЛЕКЦИИ ДЛЯ КАЖДОЙ ОСИ)
    if detection_result and detection_result['polygons']:
        # Для первой оси создаём свою коллекцию
        poly_collection1 = PatchCollection(detection_result['polygons'],
                                           facecolor='none', edgecolor='red', linewidth=2)
        ax1.add_collection(poly_collection1)

        # Для второй оси создаём отдельную коллекцию
        poly_collection2 = PatchCollection(detection_result['polygons'],
                                           facecolor='none', edgecolor='red', linewidth=2)
        ax2.add_collection(poly_collection2)

        # Эллипсы - тоже отдельные коллекции для каждой оси
        if detection_result['ellipses']:
            ellipse_collection1 = PatchCollection(detection_result['ellipses'],
                                                  facecolor='none', edgecolor='blue',
                                                  linewidth=2, linestyle='--')
            ax1.add_collection(ellipse_collection1)

            ellipse_collection2 = PatchCollection(detection_result['ellipses'],
                                                  facecolor='none', edgecolor='blue',
                                                  linewidth=2, linestyle='--')
            ax2.add_collection(ellipse_collection2)

        # Статистика
        stats_text = f"Аномалий: {detection_result['num_anomalies']}\n"
        stats_text += f"Метод: {detection_result.get('method', 'standard')}\n"
        stats_text += f"nσ = {detection_result.get('n_sigma', 3.0)}"

        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=9,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    for ax in [ax1, ax2]:
        ax.set_xlabel('X, км')
        ax.set_ylabel('Y, км')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

    ax1.set_title(f'{plot_title}\nЦветовая карта\nУровень измерения: {z_level} м')
    ax2.set_title(f'{plot_title}\nИзолинии\nУровень измерения: {z_level} м')

    plt.suptitle(f'МЕТОД А: Адаптивный 3σ с предварительным снятием тренда', fontsize=14)
    plt.tight_layout()
    plt.show()

    return fig


def plot_comparison(clean_field: np.ndarray, final_field: np.ndarray,
                    X: np.ndarray, Y: np.ndarray, sources: List, z_level: float,
                    result_clean: Dict, result_final: Dict) -> plt.Figure:
    """Сравнение результатов на чистом поле и поле с шумом/трендом"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    x_km = X / 1000
    y_km = Y / 1000
    cmap = create_custom_colormap()

    # Чистое поле
    ax = axes[0, 0]
    ax.contourf(x_km, y_km, clean_field, levels=50, cmap=cmap, alpha=0.9)
    ax.set_title(f'Чистое поле\n{result_clean["num_anomalies"]} аномалий')
    ax.scatter([s[0] / 1000 for s in sources], [s[1] / 1000 for s in sources],
               c='white', s=30, edgecolors='black', zorder=6)
    ax.set_xlabel('X, км')
    ax.set_ylabel('Y, км')
    ax.grid(True, alpha=0.3)

    # Чистое поле с аномалиями
    ax = axes[0, 1]
    ax.contourf(x_km, y_km, clean_field, levels=50, cmap=cmap, alpha=0.9)
    if result_clean['polygons']:
        # Создаём отдельную коллекцию для этой оси
        poly_collection = PatchCollection(result_clean['polygons'],
                                          facecolor='none', edgecolor='red', linewidth=2)
        ax.add_collection(poly_collection)
    ax.set_title(f'Чистое поле + выделенные аномалии\n{result_clean["num_anomalies"]} аномалий')
    ax.scatter([s[0] / 1000 for s in sources], [s[1] / 1000 for s in sources],
               c='white', s=30, edgecolors='black', zorder=6)
    ax.set_xlabel('X, км')
    ax.set_ylabel('Y, км')
    ax.grid(True, alpha=0.3)

    # Поле с шумом и трендом
    ax = axes[1, 0]
    ax.contourf(x_km, y_km, final_field, levels=50, cmap=cmap, alpha=0.9)
    ax.set_title(f'Поле с шумом и трендом\n{result_final["num_anomalies"]} аномалий')
    ax.scatter([s[0] / 1000 for s in sources], [s[1] / 1000 for s in sources],
               c='white', s=30, edgecolors='black', zorder=6)
    ax.set_xlabel('X, км')
    ax.set_ylabel('Y, км')
    ax.grid(True, alpha=0.3)

    # Поле с шумом и трендом + аномалии (после снятия тренда)
    ax = axes[1, 1]
    if result_final['field_detrended'] is not None:
        field_to_plot = result_final['field_detrended']
        ax.contourf(x_km, y_km, field_to_plot, levels=50, cmap=cmap, alpha=0.9)
        ax.set_title(f'Поле после снятия тренда + аномалии\n{result_final["num_anomalies"]} аномалий')
    else:
        ax.contourf(x_km, y_km, final_field, levels=50, cmap=cmap, alpha=0.9)
        ax.set_title(f'Поле с шумом/трендом + аномалии\n{result_final["num_anomalies"]} аномалий')

    if result_final['polygons']:
        # Создаём отдельную коллекцию для этой оси
        poly_collection = PatchCollection(result_final['polygons'],
                                          facecolor='none', edgecolor='red', linewidth=2)
        ax.add_collection(poly_collection)

    ax.scatter([s[0] / 1000 for s in sources], [s[1] / 1000 for s in sources],
               c='white', s=30, edgecolors='black', zorder=6)
    ax.set_xlabel('X, км')
    ax.set_ylabel('Y, км')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Сравнение метода адаптивного 3σ на разных типах полей', fontsize=16)
    plt.tight_layout()
    plt.show()

    return fig

def plot_anomaly_analysis(detection_result: Dict[str, Any], sources: List) -> plt.Figure:
    """Анализ выделенных аномалий"""
    if not detection_result['anomaly_stats']:
        print("Нет аномалий для анализа")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # График 1: Амплитуды аномалий
    ax = axes[0]
    amplitudes = [stat['max'] - stat['min'] for stat in detection_result['anomaly_stats']]
    areas = [stat['area_km2'] for stat in detection_result['anomaly_stats']]

    ax.bar(range(len(amplitudes)), amplitudes, color='steelblue', alpha=0.7)
    ax.set_xlabel('Номер аномалии')
    ax.set_ylabel('Амплитуда, мГал')
    ax.set_title('Амплитуды выделенных аномалий')
    ax.grid(True, alpha=0.3)

    # График 2: Площади аномалий
    ax = axes[1]
    ax.bar(range(len(areas)), areas, color='coral', alpha=0.7)
    ax.set_xlabel('Номер аномалии')
    ax.set_ylabel('Площадь, км²')
    ax.set_title('Площади выделенных аномалий')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Анализ выделенных аномалий', fontsize=14)
    plt.tight_layout()
    plt.show()

    return fig


# ==================== ОСНОВНАЯ ПРОГРАММА ====================

def main():
    print("="*60)
    print("МЕТОД А: Адаптивный 3σ с предварительным снятием тренда")
    print("="*60)

    measurement_level = 800

    trends_config = [
        {'order': 2, 'amplitude_factor': 0.8},
        {'order': 3, 'amplitude_factor': 0.3},
        {'order': 1, 'amplitude_factor': 0.5},
        {'order': 4, 'amplitude_factor': 0.2}
    ]

    # Генерация данных
    print("\n1. Генерация модельных данных...")
    X, Y, clean_field, sources, z_level = create_gravitational_field_map(
        n_sources=18, grid_size=250, z_level=measurement_level, smoothing_sigma=3.0
    )

    noisy_field = add_random_noise(clean_field, noise_level=0.05)
    final_field = apply_multiple_trends(noisy_field, X, Y, trends_config)

    print(f"   Чистое поле: [{np.min(clean_field):.1f}, {np.max(clean_field):.1f}] мГал")
    print(f"   С шумом и трендом: [{np.min(final_field):.1f}, {np.max(final_field):.1f}] мГал")

    # ===== ПРИМЕНЕНИЕ УЛУЧШЕННОГО МЕТОДА =====
    print("\n2. Применение адаптивного метода 3σ с RANSAC-удалением тренда...")

    # На чистом поле
    result_clean = detect_anomalies_adaptive_3sigma(
        clean_field, X, Y, n_sigma=2.0, window_size=31,
        min_area_pixels=30, trend_removal_method='ransac'
    )
    print(f"   Чистое поле: обнаружено {result_clean['num_anomalies']} аномалий")

    # На поле с шумом и трендом
    result_final = detect_anomalies_adaptive_3sigma(
        final_field, X, Y, n_sigma=2.0, window_size=31,
        min_area_pixels=30, trend_removal_method='ransac'
    )
    print(f"   С шумом и трендом: обнаружено {result_final['num_anomalies']} аномалий")

    # Вывод статистики по аномалиям
    if result_clean['anomaly_stats']:
        print(f"\n   Статистика выделенных аномалий (чистое поле):")
        for i, stat in enumerate(result_clean['anomaly_stats']):
            print(f"      Аномалия {i+1}: площадь = {stat['area_km2']:.2f} км², "
                  f"амплитуда = {stat['max'] - stat['min']:.2f} мГал")

    if result_final['anomaly_stats']:
        print(f"\n   Статистика выделенных аномалий (поле с шумом/трендом):")
        for i, stat in enumerate(result_final['anomaly_stats']):
            print(f"      Аномалия {i+1}: площадь = {stat['area_km2']:.2f} км², "
                  f"амплитуда = {stat['max'] - stat['min']:.2f} мГал")

    # Визуализация
    print("\n3. Визуализация результатов...")

    # Результат на чистом поле
    plot_field_with_anomalies(
        clean_field, X, Y, sources, z_level,
        "ЧИСТОЕ ПОЛЕ",
        detection_result=result_clean,
        show_sources=True,
        show_detrended=False
    )

    # Результат на поле с шумом и трендом (после снятия тренда)
    plot_field_with_anomalies(
        final_field, X, Y, sources, z_level,
        "ПОЛЕ С ШУМОМ И ТРЕНДАМИ",
        detection_result=result_final,
        show_sources=True,
        show_detrended=True  # Показываем поле после снятия тренда
    )

    # Сравнительная визуализация
    plot_comparison(clean_field, final_field, X, Y, sources, z_level, result_clean, result_final)

    # Анализ аномалий
    plot_anomaly_analysis(result_final, sources)

    # ===== ВЫВОД РЕКОМЕНДАЦИЙ =====
    print("\n" + "="*60)
    print("ВЫВОДЫ И РЕКОМЕНДАЦИИ")
    print("="*60)

    print(f"\n✓ Улучшенный метод 3σ с предварительным снятием тренда обнаружил:")
    print(f"  - На чистом поле: {result_clean['num_anomalies']} из {len(sources)} источников")
    print(f"  - На поле с шумом/трендом: {result_final['num_anomalies']} из {len(sources)} источников")

    print("\n✓ Ключевые улучшения:")
    print("  1. Предварительное снятие тренда методом RANSAC")
    print("  2. Локальные пороги (скользящее окно) вместо глобального")
    print("  3. Уменьшенный порог (2.0σ вместо 3σ)")
    print("  4. Фильтрация по площади для удаления шумовых выбросов")
    print("  5. Нормализация поля перед пороговой обработкой")
    print("  6. Использование morphology.disk вместо устаревших функций")

    if result_final['num_anomalies'] < len(sources):
        print("\n⚠ Некоторые аномалии не выделены. Рекомендации:")
        print("  - Уменьшить n_sigma до 1.5")
        print("  - Уменьшить window_size до 21 для более локального анализа")
        print("  - Уменьшить min_area_pixels до 15")
        print("  - Попробовать метод 'median' или 'gaussian' вместо 'ransac'")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()