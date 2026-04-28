"""
Сделай на базе этой программы (с исправлениями для функций plot_field_with_anomalies и
plot_comparison, которые ты сделал выше) программу для поиска аномалий,
подбирающую аномалии согласно предложениям в конце программы:
менять  n_sigma, window_size, min_area_pixels,
пробовать методы метод ['ransac', 'median', 'gaussian']

Лучше сделать перебор по вариантам без визуализации отдельных результатов
(но с запоминанием их параметров и результатов) и с визуализацией
наилучшего подобранного результата, т.е. такого,
когда число выделенных аномалий наиболее близко к числу шаров,
а в идеале совпадает и контуры аномалий наиболее близки к контурам шаров.
"""

"""
МЕТОД А: Автоматический подбор параметров для выделения аномалий по правилу 3σ
Программа перебирает различные комбинации параметров и выбирает наилучший результат
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Ellipse, Circle
from scipy.ndimage import gaussian_filter, label
from scipy.signal import medfilt2d
from skimage import measure, morphology
from typing import List, Tuple, Dict, Any
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
import warnings
from itertools import product
from dataclasses import dataclass, field
import json

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
        r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        r = np.maximum(r, 1e-6)
        g_vertical = G * mass * dz / r ** 3
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
            trend += coeff * (x_norm ** i) * (y_norm ** j)

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


# ==================== МЕТОД ВЫДЕЛЕНИЯ АНОМАЛИЙ ====================

def remove_trend_ransac(field: np.ndarray, X: np.ndarray, Y: np.ndarray,
                        order: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Удаление тренда методом RANSAC"""
    x_km = X.flatten() / 1000
    y_km = Y.flatten() / 1000
    z = field.flatten()

    poly = PolynomialFeatures(degree=order)
    X_poly = poly.fit_transform(np.column_stack([x_km, y_km]))

    ransac = RANSACRegressor(random_state=42, min_samples=0.5,
                             residual_threshold=np.std(field) * 0.5)
    ransac.fit(X_poly, z)

    trend = ransac.predict(X_poly).reshape(field.shape)
    residual = field - trend

    return residual, trend


def remove_trend_median_filter(field: np.ndarray, window_size: int = 31) -> np.ndarray:
    """Удаление регионального фона медианным фильтром"""
    regional = medfilt2d(field, kernel_size=window_size)
    return field - regional


def remove_trend_gaussian(field: np.ndarray, sigma: float = 15.0) -> np.ndarray:
    """Удаление регионального фона гауссовским сглаживанием"""
    regional = gaussian_filter(field, sigma=sigma)
    return field - regional


def detect_anomalies(field: np.ndarray, X: np.ndarray, Y: np.ndarray,
                     n_sigma: float = 2.0, window_size: int = 31,
                     min_area_pixels: int = 30,
                     trend_removal_method: str = 'ransac') -> Dict[str, Any]:
    """
    Выделение аномалий с заданными параметрами
    """
    # Шаг 1: Снятие тренда
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

    # Шаг 2: Нормализация
    field_norm = (field_detrended - np.mean(field_detrended)) / np.std(field_detrended)

    # Шаг 3: Локальная пороговая обработка
    rows, cols = field.shape
    half_win = window_size // 2

    local_mask = np.zeros_like(field, dtype=bool)

    for i in range(half_win, rows - half_win):
        for j in range(half_win, cols - half_win):
            window = field_norm[i - half_win:i + half_win, j - half_win:j + half_win]
            local_mean = np.mean(window)
            local_std = np.std(window)

            threshold_upper = local_mean + n_sigma * local_std
            threshold_lower = local_mean - n_sigma * local_std

            if field_norm[i, j] > threshold_upper or field_norm[i, j] < threshold_lower:
                local_mask[i, j] = True

    # Шаг 4: Морфологическая обработка
    struct_elem = morphology.disk(3)
    local_mask = morphology.dilation(local_mask, struct_elem)
    local_mask = morphology.closing(local_mask, morphology.disk(5))
    local_mask = morphology.opening(local_mask, morphology.disk(3))

    # Шаг 5: Фильтрация по площади
    labeled_mask, num_labels = label(local_mask)
    for label_id in range(1, num_labels + 1):
        if np.sum(labeled_mask == label_id) < min_area_pixels:
            local_mask[labeled_mask == label_id] = False

    labeled_mask, num_anomalies = label(local_mask)

    # Шаг 6: Создание многоугольников
    x_km = X / 1000
    y_km = Y / 1000

    polygons = []
    centers = []
    areas_km2 = []

    for label_id in range(1, num_anomalies + 1):
        mask_label = (labeled_mask == label_id)
        contours = measure.find_contours(mask_label, 0.5)

        if not contours:
            continue

        contour = max(contours, key=len)

        contour_x = np.interp(contour[:, 1], np.arange(len(x_km[0])), x_km[0])
        contour_y = np.interp(contour[:, 0], np.arange(len(y_km[:, 0])), y_km[:, 0])

        if len(contour_x) > 12:
            indices = np.linspace(0, len(contour_x) - 1, 12, dtype=int)
            contour_x = contour_x[indices]
            contour_y = contour_y[indices]

        contour_x = np.append(contour_x, contour_x[0])
        contour_y = np.append(contour_y, contour_y[0])

        vertices = np.column_stack([contour_x, contour_y])
        polygons.append(Polygon(vertices, closed=True, fill=False, edgecolor='red', linewidth=2))

        # Центр аномалии
        centers.append((np.mean(contour_x[:-1]), np.mean(contour_y[:-1])))

        # Площадь
        area_pixels = np.sum(mask_label)
        area_km2 = area_pixels * (X[0, 1] - X[0, 0]) ** 2 / 1e6
        areas_km2.append(area_km2)

    return {
        'num_anomalies': num_anomalies,
        'polygons': polygons,
        'centers': centers,
        'areas_km2': areas_km2,
        'mask': local_mask,
        'field_detrended': field_detrended,
        'params': {
            'n_sigma': n_sigma,
            'window_size': window_size,
            'min_area_pixels': min_area_pixels,
            'trend_removal_method': trend_removal_method
        }
    }


# ==================== ОЦЕНКА КАЧЕСТВА ====================

def calculate_source_centers(sources: List) -> List[Tuple[float, float]]:
    """Вычисляет центры реальных источников в км"""
    return [(s[0] / 1000, s[1] / 1000) for s in sources]


def calculate_source_radii(sources: List) -> List[float]:
    """Вычисляет радиусы реальных источников в км"""
    return [s[4] / 1000 for s in sources]


def calculate_source_areas(sources: List) -> List[float]:
    """Вычисляет площади горизонтальных сечений источников в км²"""
    return [np.pi * (s[4] / 1000) ** 2 for s in sources]


def evaluate_result(result: Dict, sources: List,
                    weight_num_anomalies: float = 0.4,
                    weight_center_distance: float = 0.3,
                    weight_area_match: float = 0.3) -> Dict:
    """
    Оценивает качество выделения аномалий

    Возвращает:
    - score: общая оценка (0-1, чем выше тем лучше)
    - num_anomalies_score: оценка за количество аномалий
    - center_distance_score: оценка за совпадение центров
    - area_score: оценка за совпадение площадей
    """
    n_sources = len(sources)
    n_detected = result['num_anomalies']

    # Оценка за количество аномалий
    if n_detected == n_sources:
        num_score = 1.0
    else:
        num_score = 1.0 - min(1.0, abs(n_detected - n_sources) / n_sources)

    # Оценка за совпадение центров
    source_centers = calculate_source_centers(sources)
    detected_centers = result['centers']

    center_score = 0.0
    if detected_centers and source_centers:
        # Для каждого обнаруженного центра находим ближайший реальный
        matched_pairs = []
        for d_center in detected_centers:
            min_dist = float('inf')
            for s_center in source_centers:
                dist = np.sqrt((d_center[0] - s_center[0]) ** 2 + (d_center[1] - s_center[1]) ** 2)
                if dist < min_dist:
                    min_dist = dist
            # Нормализуем расстояние (макс ожидаемое расстояние ~ 5 км)
            normalized_dist = min(1.0, min_dist / 5.0)
            matched_pairs.append(1.0 - normalized_dist)

        center_score = np.mean(matched_pairs) if matched_pairs else 0.0

    # Оценка за совпадение площадей
    source_areas = calculate_source_areas(sources)
    detected_areas = result['areas_km2']

    area_score = 0.0
    if detected_areas and source_areas:
        # Сортируем площади для сравнения
        source_areas_sorted = sorted(source_areas)
        detected_areas_sorted = sorted(detected_areas)

        # Сравниваем соответствующие площади
        n_compare = min(len(source_areas_sorted), len(detected_areas_sorted))
        if n_compare > 0:
            area_ratios = []
            for i in range(n_compare):
                ratio = min(detected_areas_sorted[i], source_areas_sorted[i]) / \
                        max(detected_areas_sorted[i], source_areas_sorted[i])
                area_ratios.append(ratio)
            area_score = np.mean(area_ratios)

    # Общая оценка
    total_score = (weight_num_anomalies * num_score +
                   weight_center_distance * center_score +
                   weight_area_match * area_score)

    return {
        'score': total_score,
        'num_anomalies_score': num_score,
        'center_distance_score': center_score,
        'area_score': area_score,
        'n_detected': n_detected,
        'n_sources': n_sources
    }


# ==================== ПОДБОР ПАРАМЕТРОВ ====================

@dataclass
class OptimizationResult:
    """Результат оптимизации параметров"""
    params: Dict
    result: Dict
    evaluation: Dict
    rank: int = 0


def optimize_parameters(field: np.ndarray, X: np.ndarray, Y: np.ndarray,
                        sources: List,
                        n_sigma_range: List[float] = None,
                        window_size_range: List[int] = None,
                        min_area_range: List[int] = None,
                        methods: List[str] = None,
                        verbose: bool = True) -> List[OptimizationResult]:
    """
    Перебор параметров для поиска наилучшего результата
    """
    if n_sigma_range is None:
        n_sigma_range = [1.5, 2.0, 2.5, 3.0]
    if window_size_range is None:
        window_size_range = [21, 31, 41, 51]
    if min_area_range is None:
        min_area_range = [20, 30, 40, 50, 60]
    if methods is None:
        methods = ['ransac', 'median', 'gaussian']

    total_combinations = len(n_sigma_range) * len(window_size_range) * \
                         len(min_area_range) * len(methods)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"ПОДБОР ПАРАМЕТРОВ")
        print(f"{'=' * 60}")
        print(f"Всего комбинаций: {total_combinations}")
        print(f"  n_sigma: {n_sigma_range}")
        print(f"  window_size: {window_size_range}")
        print(f"  min_area: {min_area_range}")
        print(f"  методы: {methods}")
        print(f"{'=' * 60}\n")

    results = []
    current = 0

    for n_sigma, window_size, min_area, method in product(n_sigma_range,
                                                          window_size_range,
                                                          min_area_range,
                                                          methods):
        current += 1

        if verbose and current % 10 == 0:
            print(f"  Прогресс: {current}/{total_combinations} ({100 * current / total_combinations:.1f}%)")

        try:
            # Выделение аномалий
            result = detect_anomalies(field, X, Y,
                                      n_sigma=n_sigma,
                                      window_size=window_size,
                                      min_area_pixels=min_area,
                                      trend_removal_method=method)

            # Оценка качества
            evaluation = evaluate_result(result, sources)

            results.append(OptimizationResult(
                params={
                    'n_sigma': n_sigma,
                    'window_size': window_size,
                    'min_area_pixels': min_area,
                    'trend_removal_method': method
                },
                result=result,
                evaluation=evaluation
            ))

        except Exception as e:
            if verbose:
                print(f"  Ошибка для параметров {n_sigma}, {window_size}, {min_area}, {method}: {e}")
            continue

    # Сортировка по оценке
    results.sort(key=lambda x: x.evaluation['score'], reverse=True)

    # Добавляем ранг
    for i, res in enumerate(results):
        res.rank = i + 1

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ")
        print(f"{'=' * 60}")
        print(f"\nТоп-5 лучших комбинаций параметров:\n")

        for i in range(min(5, len(results))):
            r = results[i]
            print(f"{i + 1}. Оценка: {r.evaluation['score']:.3f} "
                  f"(аномалий: {r.evaluation['n_detected']}/{r.evaluation['n_sources']})")
            print(f"   Параметры: nσ={r.params['n_sigma']}, "
                  f"win={r.params['window_size']}, "
                  f"area={r.params['min_area_pixels']}, "
                  f"method={r.params['trend_removal_method']}")
            print(f"   Детали: количество={r.evaluation['num_anomalies_score']:.3f}, "
                  f"центры={r.evaluation['center_distance_score']:.3f}, "
                  f"площади={r.evaluation['area_score']:.3f}")
            print()

    return results


# ==================== ВИЗУАЛИЗАЦИЯ ====================

def create_custom_colormap():
    """Создаёт пользовательскую цветовую карту"""
    return LinearSegmentedColormap.from_list(
        "grav_cmap", ["darkblue", "blue", "cyan", "green", "yellow", "orange", "red"]
    )


def plot_best_result(field: np.ndarray, X: np.ndarray, Y: np.ndarray,
                     sources: List, z_level: float, title: str,
                     best_result: OptimizationResult) -> plt.Figure:
    """Визуализирует наилучший результат"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    x_km = X / 1000
    y_km = Y / 1000
    cmap = create_custom_colormap()

    # 1. Исходное поле
    ax = axes[0, 0]
    ax.contourf(x_km, y_km, field, levels=50, cmap=cmap, alpha=0.9)
    ax.set_title('Исходное поле')
    ax.scatter([s[0] / 1000 for s in sources], [s[1] / 1000 for s in sources],
               c='white', s=50, edgecolors='black', linewidth=2, zorder=6,
               label='Реальные источники')
    ax.set_xlabel('X, км')
    ax.set_ylabel('Y, км')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2. Поле после снятия тренда
    ax = axes[0, 1]
    field_detrended = best_result.result['field_detrended']
    ax.contourf(x_km, y_km, field_detrended, levels=50, cmap=cmap, alpha=0.9)
    ax.set_title(f'Поле после снятия тренда\nМетод: {best_result.params["trend_removal_method"]}')
    ax.scatter([s[0] / 1000 for s in sources], [s[1] / 1000 for s in sources],
               c='white', s=50, edgecolors='black', linewidth=2, zorder=6)
    ax.set_xlabel('X, км')
    ax.set_ylabel('Y, км')
    ax.grid(True, alpha=0.3)

    # 3. Выделенные аномалии на поле
    ax = axes[1, 0]
    ax.contourf(x_km, y_km, field_detrended, levels=50, cmap=cmap, alpha=0.6)

    # Добавляем выделенные аномалии
    if best_result.result['polygons']:
        poly_collection = PatchCollection(best_result.result['polygons'],
                                          facecolor='none', edgecolor='red', linewidth=2)
        ax.add_collection(poly_collection)

    # Добавляем реальные источники (кружки)
    for s in sources:
        circle = Circle((s[0] / 1000, s[1] / 1000), s[4] / 1000,
                        fill=False, edgecolor='white', linewidth=2, linestyle='--')
        ax.add_patch(circle)

    ax.set_title(f'Выделенные аномалии (красные) и реальные источники (белые)\n'
                 f'{best_result.evaluation["n_detected"]} из {best_result.evaluation["n_sources"]} аномалий')
    ax.set_xlabel('X, км')
    ax.set_ylabel('Y, км')
    ax.grid(True, alpha=0.3)

    # 4. Информация о параметрах
    ax = axes[1, 1]
    ax.axis('off')

    info_text = f"""
    НАИЛУЧШИЙ РЕЗУЛЬТАТ
    ====================

    ПАРАМЕТРЫ:
    • nσ = {best_result.params['n_sigma']}
    • window_size = {best_result.params['window_size']}
    • min_area_pixels = {best_result.params['min_area_pixels']}
    • метод снятия тренда = {best_result.params['trend_removal_method']}

    РЕЗУЛЬТАТЫ:
    • Общая оценка: {best_result.evaluation['score']:.3f}
    • Обнаружено аномалий: {best_result.evaluation['n_detected']} / {best_result.evaluation['n_sources']}
    • Оценка количества: {best_result.evaluation['num_anomalies_score']:.3f}
    • Оценка совпадения центров: {best_result.evaluation['center_distance_score']:.3f}
    • Оценка совпадения площадей: {best_result.evaluation['area_score']:.3f}

    СТАТИСТИКА АНОМАЛИЙ:
    """

    for i, (center, area) in enumerate(zip(best_result.result['centers'],
                                           best_result.result['areas_km2'])):
        info_text += f"\n    Аномалия {i + 1}: центр=({center[0]:.1f}, {center[1]:.1f}) км, площадь={area:.2f} км²"

    ax.text(0.1, 0.95, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle(f'{title}\nОптимальное выделение аномалий', fontsize=14)
    plt.tight_layout()
    plt.show()

    return fig


def plot_optimization_summary(results: List[OptimizationResult],
                              n_top: int = 10) -> plt.Figure:
    """Визуализация результатов оптимизации"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    top_results = results[:n_top]

    # 1. Оценки топ-результатов
    ax = axes[0, 0]
    scores = [r.evaluation['score'] for r in top_results]
    n_detected = [r.evaluation['n_detected'] for r in top_results]

    x = range(len(top_results))
    ax.bar(x, scores, alpha=0.7, label='Общая оценка', color='steelblue')
    ax.set_xlabel('Ранг')
    ax.set_ylabel('Оценка')
    ax.set_title(f'Топ-{n_top} результатов по общей оценке')
    ax.grid(True, alpha=0.3)

    # 2. Количество обнаруженных аномалий
    ax = axes[0, 1]
    n_sources = top_results[0].evaluation['n_sources']
    ax.plot(x, [n_sources] * len(x), 'r--', label=f'Реальное количество ({n_sources})', linewidth=2)
    ax.bar(x, n_detected, alpha=0.7, label='Обнаружено', color='coral')
    ax.set_xlabel('Ранг')
    ax.set_ylabel('Количество аномалий')
    ax.set_title('Количество обнаруженных аномалий')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Параметры лучших результатов
    ax = axes[1, 0]
    n_sigma_vals = [r.params['n_sigma'] for r in top_results]
    window_vals = [r.params['window_size'] for r in top_results]

    ax.scatter(range(len(top_results)), n_sigma_vals, s=100, alpha=0.6,
               label='nσ', marker='o', color='green')
    ax.scatter(range(len(top_results)), window_vals, s=100, alpha=0.6,
               label='window_size', marker='s', color='purple')
    ax.set_xlabel('Ранг')
    ax.set_ylabel('Значение параметра')
    ax.set_title('Параметры лучших результатов')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Распределение методов
    ax = axes[1, 1]
    methods = [r.params['trend_removal_method'] for r in results[:50]]
    unique_methods = list(set(methods))
    method_counts = [methods.count(m) for m in unique_methods]

    ax.bar(unique_methods, method_counts, alpha=0.7, color='teal')
    ax.set_xlabel('Метод снятия тренда')
    ax.set_ylabel('Частота в топ-50')
    ax.set_title('Распределение методов снятия тренда')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Анализ результатов оптимизации параметров', fontsize=14)
    plt.tight_layout()
    plt.show()

    return fig


# ==================== ОСНОВНАЯ ПРОГРАММА ====================

def main():
    print("=" * 60)
    print("АВТОМАТИЧЕСКИЙ ПОДБОР ПАРАМЕТРОВ ДЛЯ МЕТОДА 3σ")
    print("=" * 60)

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
    print(f"   Реальное количество источников: {len(sources)}")

    # Оптимизация параметров
    print("\n2. Оптимизация параметров...")

    # На чистом поле
    print("\n--- Оптимизация для ЧИСТОГО ПОЛЯ ---")
    results_clean = optimize_parameters(
        clean_field, X, Y, sources,
        n_sigma_range=[1.5, 2.0, 2.5, 3.0],
        window_size_range=[21, 31, 41],
        min_area_range=[20, 30, 40, 50],
        methods=['ransac', 'median', 'gaussian'],
        verbose=True
    )

    # На поле с шумом и трендом
    print("\n--- Оптимизация для ПОЛЯ С ШУМОМ И ТРЕНДОМ ---")
    results_final = optimize_parameters(
        final_field, X, Y, sources,
        n_sigma_range=[1.5, 2.0, 2.5, 3.0],
        window_size_range=[21, 31, 41],
        min_area_range=[20, 30, 40, 50],
        methods=['ransac', 'median', 'gaussian'],
        verbose=True
    )

    # Визуализация лучших результатов
    print("\n3. Визуализация наилучших результатов...")

    if results_clean:
        plot_best_result(clean_field, X, Y, sources, z_level,
                         "ЧИСТОЕ ПОЛЕ", results_clean[0])

    if results_final:
        plot_best_result(final_field, X, Y, sources, z_level,
                         "ПОЛЕ С ШУМОМ И ТРЕНДОМ", results_final[0])

    # Визуализация сводки по оптимизации
    plot_optimization_summary(results_final, n_top=10)

    # Сохранение результатов в файл
    print("\n4. Сохранение результатов...")

    best = results_final[0]
    summary = {
        'best_params': best.params,
        'best_score': float(best.evaluation['score']),
        'n_detected': best.evaluation['n_detected'],
        'n_sources': best.evaluation['n_sources'],
        'all_results': [
            {
                'params': r.params,
                'score': float(r.evaluation['score']),
                'n_detected': r.evaluation['n_detected']
            }
            for r in results_final[:10]
        ]
    }

    with open('optimization_results.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("   Результаты сохранены в optimization_results.json")

    # Вывод рекомендаций
    print("\n" + "=" * 60)
    print("РЕКОМЕНДАЦИИ")
    print("=" * 60)

    print(f"\n✓ Наилучший результат достигнут с параметрами:")
    print(f"   • nσ = {best.params['n_sigma']}")
    print(f"   • window_size = {best.params['window_size']}")
    print(f"   • min_area_pixels = {best.params['min_area_pixels']}")
    print(f"   • метод снятия тренда = {best.params['trend_removal_method']}")

    print(f"\n✓ Качество выделения:")
    print(f"   • Обнаружено: {best.evaluation['n_detected']} из {best.evaluation['n_sources']} аномалий")
    print(f"   • Общая оценка: {best.evaluation['score']:.3f}")

    if best.evaluation['n_detected'] < best.evaluation['n_sources']:
        print(f"\n⚠ Для улучшения результата рекомендуется:")
        print(f"   • Уменьшить nσ до 1.5")
        print(f"   • Уменьшить window_size до 21")
        print(f"   • Уменьшить min_area_pixels до 20")
        print(f"   • Попробовать комбинацию методов")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

