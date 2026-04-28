"""
Почему не обнаружена одна аномалия (примерно X=1 км, Y=2.5 км)?
Она визуально видна на карте, но почему-то не обнаружена,
какие параметры надо изменить, чтобы можно было ее обнаружить?

Аномалия в районе X=1 км, Y=2.5 км не обнаруживается,
потому что она сливается с фоном после снятия тренда и фильтрации.
Давайте разберём причины и способы решения.

Причины необнаружения аномалии
1. Аномалия имеет малую амплитуду относительно фона
Глубоко залегающий шар или шар с малой плотностью

Сигнал слабее, чем у соседних аномалий

2. Параметр n_sigma слишком высокий
При n_sigma=2.0 порог составляет 2 стандартных отклонения

Слабая аномалия может не превышать этот порог

3. Размер окна фильтра (k) подобран неоптимально
Слишком большое окно → региональный фон "съедает" аномалию

Слишком маленькое окно → остаётся много шума

4. Порог по площади (min_area_pixels) отсекает мелкие аномалии
Если аномалия имеет маленькую площадь, она может быть отброшена
"""
import sys

"""
МЕТОД А: Выделение аномалий по правилу 3σ - РАСШИРЕННАЯ ВЕРСИЯ
Специальные параметры для обнаружения слабых аномалий
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Ellipse, Circle
from scipy.ndimage import gaussian_filter, label
from scipy.signal import convolve2d
from skimage import measure, morphology
from typing import List, Tuple, Dict, Any
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from scipy.spatial import ConvexHull
import warnings
from itertools import product
from dataclasses import dataclass
import json
import os
from datetime import datetime

warnings.filterwarnings('ignore', category=UserWarning)

# ==================== КОНСТАНТЫ ====================
G = 6.67430e-11
M_TO_MGAL = 1e5
DENSITY_CONVERSION = 1000


# ==================== ГЕНЕРАЦИЯ ИСХОДНЫХ ДАННЫХ ====================

def generate_non_overlapping_sources(n_sources: int, x_range: Tuple = (0, 5000),
                                     y_range: Tuple = (0, 5000),
                                     radius_range: Tuple = (50, 500),
                                     density_range: Tuple = (0.05, 0.3),
                                     z_level: float = 500,
                                     min_distance_factor: float = 1.5,
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
                                   density_range: Tuple = (0.05, 0.3),
                                   radius_range: Tuple = (50, 500),
                                   z_level: float = 500,
                                   smoothing_sigma: float = 2.0,
                                   min_distance_factor: float = 1.5) -> Tuple:
    """Создаёт карту гравитационного поля от сферических аномалий (БЕЗ шума и тренда)"""
    x = np.linspace(0, 5000, grid_size)
    y = np.linspace(0, 5000, grid_size)
    X, Y = np.meshgrid(x, y)

    np.random.seed(42)
    sources = generate_non_overlapping_sources(n_sources, x_range=(500, 4500),
                                               y_range=(500, 4500),
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


def add_tapering(field: np.ndarray, taper_width: int = 20) -> np.ndarray:
    """Добавляет окантовку (tapering) к полю для уменьшения краевых эффектов"""
    rows, cols = field.shape
    tapered_field = field.copy()

    taper = np.ones((rows, cols))

    for i in range(taper_width):
        weight = 0.5 * (1 - np.cos(np.pi * i / taper_width))
        taper[:, i] *= weight
        taper[:, cols - 1 - i] *= weight

    for i in range(taper_width):
        weight = 0.5 * (1 - np.cos(np.pi * i / taper_width))
        taper[i, :] *= weight
        taper[rows - 1 - i, :] *= weight

    mean_val = np.mean(field)
    tapered_field = mean_val + (field - mean_val) * taper

    return tapered_field


def generate_logspace_k_values(k_min: float = 0.005, k_max: float = 0.3,
                               n_values: int = 12) -> List[float]:
    """
    Генерирует значения k с логарифмическим шагом (расширенный диапазон)
    """
    log_min = np.log10(k_min)
    log_max = np.log10(k_max)
    log_values = np.linspace(log_min, log_max, n_values)
    k_values = 10 ** log_values
    k_values = np.round(k_values, 6)

    return list(k_values)


# ==================== МЕТОДЫ СНЯТИЯ ТРЕНДА ====================

def remove_polynomial_trend(field: np.ndarray, X: np.ndarray, Y: np.ndarray,
                            order: int = 2, use_ransac: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Снятие полиномиального тренда заданного порядка"""
    x_km = X.flatten() / 1000
    y_km = Y.flatten() / 1000
    z = field.flatten()

    poly = PolynomialFeatures(degree=order)
    X_poly = poly.fit_transform(np.column_stack([x_km, y_km]))

    if use_ransac:
        regressor = RANSACRegressor(random_state=42, min_samples=0.5,
                                    residual_threshold=np.std(field) * 0.5)
    else:
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()

    regressor.fit(X_poly, z)

    trend = regressor.predict(X_poly).reshape(field.shape)
    residual = field - trend

    if hasattr(regressor, 'estimator_'):
        coeffs = regressor.estimator_.coef_
    else:
        coeffs = regressor.coef_

    return residual, trend, coeffs


def remove_trend_polynomial_plus_averaging(field: np.ndarray, X: np.ndarray, Y: np.ndarray,
                                           polynomial_order: int, averaging_k: float,
                                           grid_size: int, use_ransac: bool = True) -> Dict[str, Any]:
    """Комбинированное снятие тренда: сначала полином, потом фильтр среднего"""
    field_no_poly, poly_trend, coeffs = remove_polynomial_trend(
        field, X, Y, order=polynomial_order, use_ransac=use_ransac
    )

    window_size = int(grid_size * averaging_k)
    if window_size % 2 == 0:
        window_size += 1
    window_size = max(3, window_size)

    kernel = np.ones((window_size, window_size)) / (window_size ** 2)
    regional_average = convolve2d(field_no_poly, kernel, mode='same', boundary='symm')
    residual = field_no_poly - regional_average

    return {
        'residual': residual,
        'poly_trend': poly_trend,
        'regional_average': regional_average,
        'polynomial_order': polynomial_order,
        'averaging_k': averaging_k,
        'window_size': window_size,
        'coeffs': coeffs
    }


# ==================== ВЫДЕЛЕНИЕ АНОМАЛИЙ ====================

def make_convex_polygon(contour_points: np.ndarray, min_vertices: int = 12) -> np.ndarray:
    """Преобразует контур в выпуклый многоугольник"""
    if len(contour_points) < 3:
        return contour_points

    hull = ConvexHull(contour_points)
    hull_points = contour_points[hull.vertices]

    if len(hull_points) < min_vertices:
        t = np.linspace(0, 1, len(hull_points))
        t_new = np.linspace(0, 1, min_vertices)

        x_coords = hull_points[:, 0]
        y_coords = hull_points[:, 1]

        x_coords = np.append(x_coords, x_coords[0])
        y_coords = np.append(y_coords, y_coords[0])
        t = np.linspace(0, 1, len(x_coords))

        x_new = np.interp(t_new, t, x_coords)
        y_new = np.interp(t_new, t, y_coords)

        hull_points = np.column_stack([x_new, y_new])

    return hull_points


def detect_anomalies_from_residual(residual: np.ndarray, X: np.ndarray, Y: np.ndarray,
                                   n_sigma: float = 1.5,  # Уменьшен порог
                                   min_area_pixels: int = 5,  # Уменьшен порог площади
                                   apply_tapering: bool = True,
                                   use_adaptive_threshold: bool = True) -> Dict[str, Any]:
    """
    Выделение аномалий из остаточного поля методом 3σ с адаптивным порогом
    """
    if apply_tapering:
        residual = add_tapering(residual, taper_width=30)

    if use_adaptive_threshold:
        # Адаптивный порог: используем локальные статистики
        rows, cols = residual.shape
        half_win = 15  # маленькое окно для локального порога
        residual_norm = np.zeros_like(residual)

        for i in range(half_win, rows - half_win):
            for j in range(half_win, cols - half_win):
                window = residual[i - half_win:i + half_win, j - half_win:j + half_win]
                local_mean = np.mean(window)
                local_std = np.std(window)
                if local_std > 0:
                    residual_norm[i, j] = (residual[i, j] - local_mean) / local_std
                else:
                    residual_norm[i, j] = 0
    else:
        # Глобальный порог
        residual_norm = (residual - np.mean(residual)) / np.std(residual)

    # Пороговая обработка
    threshold = n_sigma
    mask = np.abs(residual_norm) > threshold

    # Морфологическая обработка
    struct_elem = morphology.disk(2)
    mask = morphology.dilation(mask, struct_elem)
    mask = morphology.closing(mask, morphology.disk(3))
    mask = morphology.opening(mask, morphology.disk(2))

    # Фильтрация по площади
    labeled_mask, num_labels = label(mask)
    for label_id in range(1, num_labels + 1):
        if np.sum(labeled_mask == label_id) < min_area_pixels:
            mask[labeled_mask == label_id] = False

    labeled_mask, num_anomalies = label(mask)

    # Создание многоугольников
    x_km = X / 1000
    y_km = Y / 1000

    polygons = []
    ellipses = []
    centers = []
    areas_km2 = []
    amplitudes = []

    for label_id in range(1, num_anomalies + 1):
        mask_label = (labeled_mask == label_id)
        contours = measure.find_contours(mask_label, 0.5)

        if not contours:
            continue

        contour = max(contours, key=len)

        contour_x = np.interp(contour[:, 1], np.arange(len(x_km[0])), x_km[0])
        contour_y = np.interp(contour[:, 0], np.arange(len(y_km[:, 0])), y_km[:, 0])

        contour_points = np.column_stack([contour_x, contour_y])
        hull_points = make_convex_polygon(contour_points, min_vertices=12)
        hull_points = np.vstack([hull_points, hull_points[0]])

        polygons.append(Polygon(hull_points, closed=True, fill=False,
                                edgecolor='red', linewidth=2))

        if len(hull_points) >= 5:
            center_x = np.mean(hull_points[:-1, 0])
            center_y = np.mean(hull_points[:-1, 1])
            radius_x = np.max(np.abs(hull_points[:-1, 0] - center_x))
            radius_y = np.max(np.abs(hull_points[:-1, 1] - center_y))

            if radius_x > 0 and radius_y > 0:
                ellipses.append(Ellipse((center_x, center_y), 2 * radius_x, 2 * radius_y,
                                        fill=False, edgecolor='blue', linewidth=2, linestyle='--'))

        centers.append((np.mean(hull_points[:-1, 0]), np.mean(hull_points[:-1, 1])))

        area_pixels = np.sum(mask_label)
        area_km2 = area_pixels * (X[0, 1] - X[0, 0]) ** 2 / 1e6
        areas_km2.append(area_km2)

        # Амплитуда аномалии
        amplitudes.append(np.max(np.abs(residual_norm[mask_label])))

    return {
        'num_anomalies': num_anomalies,
        'polygons': polygons,
        'ellipses': ellipses,
        'centers': centers,
        'areas_km2': areas_km2,
        'amplitudes': amplitudes,
        'mask': mask,
        'residual_norm': residual_norm,
        'residual': residual
    }


# ==================== ОЦЕНКА КАЧЕСТВА ====================

def calculate_source_centers(sources: List) -> List[Tuple[float, float]]:
    return [(s[0] / 1000, s[1] / 1000) for s in sources]


def evaluate_result(detection_result: Dict, sources: List) -> Dict:
    """Оценивает качество выделения аномалий"""

    n_sources = len(sources)
    n_detected = detection_result['num_anomalies']

    if n_detected == n_sources:
        num_score = 1.0
    else:
        num_score = 1.0 - min(1.0, abs(n_detected - n_sources) / n_sources)

    source_centers = calculate_source_centers(sources)
    detected_centers = detection_result['centers']

    center_score = 0.0
    if detected_centers and source_centers:
        matched_pairs = []
        for d_center in detected_centers:
            min_dist = float('inf')
            for s_center in source_centers:
                dist = np.sqrt((d_center[0] - s_center[0]) ** 2 + (d_center[1] - s_center[1]) ** 2)
                if dist < min_dist:
                    min_dist = dist
            normalized_dist = min(1.0, min_dist / 3.0)
            matched_pairs.append(1.0 - normalized_dist)
        center_score = np.mean(matched_pairs) if matched_pairs else 0.0

    total_score = 0.6 * num_score + 0.4 * center_score

    return {
        'score': total_score,
        'num_score': num_score,
        'center_score': center_score,
        'n_detected': n_detected,
        'n_sources': n_sources
    }


# ==================== ОПТИМИЗАЦИЯ ПАРАМЕТРОВ ====================

@dataclass
class OptimizationResult:
    params: Dict
    residual: np.ndarray
    detection: Dict
    evaluation: Dict
    rank: int = 0


def optimize_parameters_extended(field: np.ndarray, X: np.ndarray, Y: np.ndarray,
                                 sources: List, grid_size: int,
                                 verbose: bool = True) -> List[OptimizationResult]:
    """
    Расширенный перебор параметров для обнаружения слабых аномалий
    """
    # Расширенные диапазоны параметров
    polynomial_orders = [1, 2, 3]
    averaging_k_values = generate_logspace_k_values(0.008, 0.2, 8)  # больше значений
    print(list(averaging_k_values))
    n_sigma_values = [1.0, 1.2, 1.5, 2.0]  # уменьшенные пороги
    min_area_values = [5, 10, 15, 20, 25]  # маленькие площади

    total_combinations = len(polynomial_orders) * len(averaging_k_values) * \
                         len(n_sigma_values) * len(min_area_values)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"РАСШИРЕННЫЙ ПОДБОР ПАРАМЕТРОВ (для слабых аномалий)")
        print(f"{'=' * 60}")
        print(f"Всего комбинаций: {total_combinations}")
        print(f"  Полиномы: {polynomial_orders}")
        print(f"  k: {[f'{k:.5f}' for k in averaging_k_values]}")
        print(f"  n_sigma: {n_sigma_values}")
        print(f"  min_area: {min_area_values}")
        print(f"{'=' * 60}\n")

    results = []
    current = 0

    for poly_order in polynomial_orders:
        for avg_k in averaging_k_values:
            for n_sigma in n_sigma_values:
                for min_area in min_area_values:
                    current += 1

                    if verbose and current % 20 == 0:
                        print(f"  Прогресс: {current}/{total_combinations} ({100 * current / total_combinations:.1f}%)")

                    try:
                        result_trend = remove_trend_polynomial_plus_averaging(
                            field, X, Y, poly_order, avg_k, grid_size, use_ransac=True
                        )

                        detection = detect_anomalies_from_residual(
                            result_trend['residual'], X, Y,
                            n_sigma=n_sigma, min_area_pixels=min_area,
                            apply_tapering=True, use_adaptive_threshold=True
                        )

                        evaluation = evaluate_result(detection, sources)

                        results.append(OptimizationResult(
                            params={
                                'polynomial_order': poly_order,
                                'averaging_k': avg_k,
                                'window_size': result_trend['window_size'],
                                'n_sigma': n_sigma,
                                'min_area_pixels': min_area,
                                'filter_type': 'averaging'
                            },
                            residual=result_trend['residual'],
                            detection=detection,
                            evaluation=evaluation
                        ))

                    except Exception as e:
                        continue

    results.sort(key=lambda x: x.evaluation['score'], reverse=True)

    for i, res in enumerate(results):
        res.rank = i + 1

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ")
        print(f"{'=' * 60}")
        print(f"\nТоп-10 лучших комбинаций:\n")

        for i in range(min(10, len(results))):
            r = results[i]
            print(f"{i + 1}. Оценка: {r.evaluation['score']:.3f} "
                  f"(аномалий: {r.evaluation['n_detected']}/{r.evaluation['n_sources']})")
            print(f"   Параметры: полином={r.params['polynomial_order']}, "
                  f"k={r.params['averaging_k']:.5f}, "
                  f"nσ={r.params['n_sigma']}, "
                  f"min_area={r.params['min_area_pixels']}")
            print()

    return results


# ==================== ВИЗУАЛИЗАЦИЯ ====================

def create_custom_colormap():
    return LinearSegmentedColormap.from_list(
        "grav_cmap", ["darkblue", "blue", "cyan", "green", "yellow", "orange", "red"]
    )


def save_figure(fig, filename: str, output_dir: str = "results"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   Сохранено: {filepath}")
    return filepath


def plot_results_with_analysis(original_field: np.ndarray, X: np.ndarray, Y: np.ndarray,
                               sources: List, z_level: float,
                               best_result: OptimizationResult,
                               all_results: List[OptimizationResult],
                               output_dir: str = "results") -> plt.Figure:
    """Визуализация результатов с анализом слабых аномалий"""

    fig = plt.figure(figsize=(20, 12))

    x_km = X / 1000
    y_km = Y / 1000
    cmap = create_custom_colormap()

    field_tapered = add_tapering(original_field, taper_width=30)
    residual_tapered = add_tapering(best_result.residual, taper_width=30)

    # 1. Исходное поле с контурами шаров
    ax1 = fig.add_subplot(2, 3, 1)
    contour1 = ax1.contourf(x_km, y_km, field_tapered, levels=50, cmap=cmap, alpha=0.9)
    for s in sources:
        circle = Circle((s[0] / 1000, s[1] / 1000), s[4] / 1000,
                        fill=False, edgecolor='white', linewidth=2, linestyle='--')
        ax1.add_patch(circle)
    ax1.set_title('Исходное поле\nПунктир - контуры шаров')
    ax1.set_xlabel('X, км')
    ax1.set_ylabel('Y, км')
    ax1.set_xlim(0, 5)
    ax1.set_ylim(0, 5)
    ax1.grid(True, alpha=0.3)
    plt.colorbar(contour1, ax=ax1, label='мГал')

    # 2. Поле после снятия тренда
    ax2 = fig.add_subplot(2, 3, 2)
    contour2 = ax2.contourf(x_km, y_km, residual_tapered, levels=50, cmap=cmap, alpha=0.9)
    ax2.set_title(f'После снятия тренда\n(полином={best_result.params["polynomial_order"]}, '
                  f'k={best_result.params["averaging_k"]:.5f})')
    ax2.set_xlabel('X, км')
    ax2.set_ylabel('Y, км')
    ax2.set_xlim(0, 5)
    ax2.set_ylim(0, 5)
    ax2.grid(True, alpha=0.3)
    plt.colorbar(contour2, ax=ax2, label='мГал')

    # 3. Выделенные аномалии
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.contourf(x_km, y_km, residual_tapered, levels=50, cmap=cmap, alpha=0.6)

    if best_result.detection['polygons']:
        poly_collection = PatchCollection(best_result.detection['polygons'],
                                          facecolor='none', edgecolor='red', linewidth=2)
        ax3.add_collection(poly_collection)

    for s in sources:
        circle = Circle((s[0] / 1000, s[1] / 1000), s[4] / 1000,
                        fill=False, edgecolor='white', linewidth=1.5, linestyle=':')
        ax3.add_patch(circle)

    ax3.set_title(f'Выделенные аномалии\n'
                  f'{best_result.evaluation["n_detected"]} из '
                  f'{best_result.evaluation["n_sources"]} (оценка={best_result.evaluation["score"]:.3f})')
    ax3.set_xlabel('X, км')
    ax3.set_ylabel('Y, км')
    ax3.set_xlim(0, 5)
    ax3.set_ylim(0, 5)
    ax3.grid(True, alpha=0.3)

    # 4. Амплитуды аномалий
    ax4 = fig.add_subplot(2, 3, 4)
    if best_result.detection['amplitudes']:
        amplitudes = best_result.detection['amplitudes']
        ax4.bar(range(len(amplitudes)), amplitudes, color='steelblue', alpha=0.7)
        ax4.axhline(y=best_result.params['n_sigma'], color='red', linestyle='--',
                    label=f'Порог (nσ={best_result.params["n_sigma"]})')
        ax4.set_xlabel('Номер аномалии')
        ax4.set_ylabel('Нормированная амплитуда')
        ax4.set_title('Амплитуды обнаруженных аномалий')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    # 5. Тепловая карта параметров
    ax5 = fig.add_subplot(2, 3, 5)
    # Группируем результаты по n_sigma и min_area
    scores_matrix = np.zeros((len(set(r.params['n_sigma'] for r in all_results[:50])),
                              len(set(r.params['min_area_pixels'] for r in all_results[:50]))))
    n_sigma_list = sorted(set(r.params['n_sigma'] for r in all_results[:50]))
    min_area_list = sorted(set(r.params['min_area_pixels'] for r in all_results[:50]))

    for i, ns in enumerate(n_sigma_list):
        for j, ma in enumerate(min_area_list):
            scores = [r.evaluation['score'] for r in all_results[:50]
                      if r.params['n_sigma'] == ns and r.params['min_area_pixels'] == ma]
            scores_matrix[i, j] = np.mean(scores) if scores else 0

    im = ax5.imshow(scores_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1,
                    extent=[min(min_area_list), max(min_area_list), len(n_sigma_list), 0])
    ax5.set_xlabel('min_area_pixels')
    ax5.set_ylabel('n_sigma')
    ax5.set_title('Оценка качества vs nσ и min_area')
    ax5.set_yticks(range(1, len(n_sigma_list) + 1))
    ax5.set_yticklabels(n_sigma_list)
    plt.colorbar(im, ax=ax5, label='Оценка')

    # 6. Информация о параметрах
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    info_text = f"""
    НАИЛУЧШИЙ РЕЗУЛЬТАТ (ДЛЯ СЛАБЫХ АНОМАЛИЙ)
    ========================================

    ПАРАМЕТРЫ:
    • Полином {best_result.params['polynomial_order']}-го порядка
    • k = {best_result.params['averaging_k']:.6f}
    • Размер окна: {best_result.params['window_size']}×{best_result.params['window_size']}
    • nσ = {best_result.params['n_sigma']} (уменьшен для слабых аномалий)
    • min_area = {best_result.params['min_area_pixels']} (уменьшен)
    • Адаптивный порог: ВКЛ

    РЕЗУЛЬТАТЫ:
    • Обнаружено: {best_result.evaluation['n_detected']} / {best_result.evaluation['n_sources']}
    • Общая оценка: {best_result.evaluation['score']:.3f}

    РЕКОМЕНДАЦИИ ДЛЯ СЛАБЫХ АНОМАЛИЙ:
    • Уменьшить nσ до 1.0-1.5
    • Уменьшить min_area_pixels до 5-15
    • Использовать адаптивный порог
    • Уменьшить размер окна k
    """

    ax6.text(0.1, 0.95, info_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle('Выделение аномалий (включая слабые)', fontsize=14)
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_figure(fig, f"anomaly_detection_extended_{timestamp}.png", output_dir)

    return fig


# ==================== ОСНОВНАЯ ПРОГРАММА ====================

def main():
    print("=" * 60)
    print("ВЫДЕЛЕНИЕ АНОМАЛИЙ (ВКЛЮЧАЯ СЛАБЫЕ)")
    print("РАСШИРЕННЫЙ ПОДБОР ПАРАМЕТРОВ")
    print("=" * 60)

    output_dir = "anomaly_detection_results_extended"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    measurement_level = 500
    grid_size = 250

    print("\n1. Генерация исходных данных...")
    X, Y, field, sources, z_level = create_gravitational_field_map(
        n_sources=12,
        grid_size=grid_size,
        z_level=measurement_level,
        smoothing_sigma=2.0,
        radius_range=(80, 400),
        density_range=(0.05, 0.25)
    )

    print(f"   Размер сетки: {grid_size}×{grid_size}")
    print(f"   Область: X=[0, 5] км, Y=[0, 5] км")
    print(f"   Диапазон поля: [{np.min(field):.2f}, {np.max(field):.2f}] мГал")
    print(f"   Реальное количество источников: {len(sources)}")

    # Анализ источников
    print("\n2. Информация об источниках (выявление слабых):")
    densities = [s[3] for s in sources]
    radii = [s[4] for s in sources]
    depths = [s[2] for s in sources]

    for i, (density, radius, depth) in enumerate(zip(densities, radii, depths)):
        signal_strength = density * (radius ** 3) / (depth ** 2)  # приблизительная оценка
        weak_marker = " ⚠ СЛАБАЯ" if signal_strength < np.mean(
            [d * (r ** 3) / (dp ** 2) for d, r, dp in zip(densities, radii, depths)]) * 0.7 else ""
        print(f"   Источник {i + 1}: плотность={density:.3f} г/см³, "
              f"радиус={radius:.0f} м, глубина={depth:.0f} м{weak_marker}")

    # Оптимизация с расширенными параметрами
    print("\n3. Расширенный подбор параметров (для слабых аномалий)...")
    results = optimize_parameters_extended(
        field, X, Y, sources, grid_size, verbose=True
    )

    if not results:
        print("Ошибка: не получено результатов оптимизации!")
        return

    # Визуализация
    print("\n4. Визуализация и сохранение результатов...")
    plot_results_with_analysis(
        field, X, Y, sources, z_level,
        results[0], results, output_dir=output_dir
    )

    # Сохранение JSON
    print("\n5. Сохранение данных...")
    best = results[0]
    summary = {
        'timestamp': datetime.now().isoformat(),
        'best_params': best.params,
        'best_score': float(best.evaluation['score']),
        'n_detected': best.evaluation['n_detected'],
        'n_sources': best.evaluation['n_sources']
    }

    json_path = os.path.join(output_dir, 'results_extended.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"   Сохранено: {json_path}")

    # Рекомендации
    print("\n" + "=" * 60)
    print("РЕКОМЕНДАЦИИ ДЛЯ ОБНАРУЖЕНИЯ СЛАБЫХ АНОМАЛИЙ")
    print("=" * 60)

    print(f"\n✓ Оптимальные параметры для ваших данных:")
    print(f"   • nσ = {best.params['n_sigma']} (вместо стандартного 2.0-2.5)")
    print(f"   • min_area_pixels = {best.params['min_area_pixels']} (вместо 30-50)")
    print(f"   • k = {best.params['averaging_k']:.5f} (размер окна = {best.params['window_size']})")

    print(f"\n✓ Если аномалия всё ещё не обнаружена, попробуйте:")
    print(f"   1. Уменьшить nσ до 1.0")
    print(f"   2. Уменьшить min_area_pixels до 5")
    print(f"   3. Уменьшить k до 0.008-0.015")
    print(f"   4. Использовать адаптивный порог (уже включён)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
