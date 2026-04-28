"""
Сделай полный код со всеми исправлениями фильтра с медианного на Среднее (averaging)
"""

"""
МЕТОД А: Выделение аномалий по правилу 3σ
Только для ИСХОДНЫХ ДАННЫХ (без добавления шума и тренда)
Сохранение результатов в PNG, окантовка полей, выпуклые многоугольники
Используется ФИЛЬТР СРЕДНЕГО (averaging) вместо медианного для сохранения округлой формы аномалий
k (размер окна) изменяется от 0.01 до 0.25 с логарифмическим шагом
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
    # Область 0-5 км по X и Y
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
    """
    Добавляет окантовку (tapering) к полю для уменьшения краевых эффектов

    Параметры:
    - field: исходное поле
    - taper_width: ширина окантовки в пикселях
    """
    rows, cols = field.shape
    tapered_field = field.copy()

    # Создаём весовую функцию (косинусоидальное окно)
    taper = np.ones((rows, cols))

    # Вертикальные края
    for i in range(taper_width):
        weight = 0.5 * (1 - np.cos(np.pi * i / taper_width))
        taper[:, i] *= weight
        taper[:, cols - 1 - i] *= weight

    # Горизонтальные края
    for i in range(taper_width):
        weight = 0.5 * (1 - np.cos(np.pi * i / taper_width))
        taper[i, :] *= weight
        taper[rows - 1 - i, :] *= weight

    # Применяем веса
    mean_val = np.mean(field)
    tapered_field = mean_val + (field - mean_val) * taper

    return tapered_field


def generate_logspace_k_values(k_min: float = 0.01, k_max: float = 0.25,
                               n_values: int = 8) -> List[float]:
    """
    Генерирует значения k с логарифмическим шагом

    Параметры:
    - k_min: минимальное значение (0.01)
    - k_max: максимальное значение (0.25)
    - n_values: количество значений (8)

    Возвращает:
    - список значений k
    """
    # Логарифмическая шкала
    log_min = np.log10(k_min)
    log_max = np.log10(k_max)
    log_values = np.linspace(log_min, log_max, n_values)
    k_values = 10 ** log_values

    # Округляем для удобства чтения
    k_values = np.round(k_values, 6)

    return list(k_values)


# ==================== МЕТОДЫ СНЯТИЯ ТРЕНДА ====================

def remove_polynomial_trend(field: np.ndarray, X: np.ndarray, Y: np.ndarray,
                            order: int = 2, use_ransac: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Снятие полиномиального тренда заданного порядка
    """
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
    """
    Комбинированное снятие тренда: сначала полином, потом фильтр среднего (averaging)
    Фильтр среднего сохраняет округлую форму аномалий

    Параметры:
    - averaging_k: размер окна как доля от размера карты
    """
    # Шаг 1: Снятие полиномиального тренда
    field_no_poly, poly_trend, coeffs = remove_polynomial_trend(
        field, X, Y, order=polynomial_order, use_ransac=use_ransac
    )

    # Шаг 2: Фильтр среднего (averaging) - сохраняет округлую форму
    window_size = int(grid_size * averaging_k)
    if window_size % 2 == 0:
        window_size += 1
    window_size = max(3, window_size)

    # Создаём ядро усреднения (все веса равны)
    kernel = np.ones((window_size, window_size)) / (window_size ** 2)

    # Применяем свёртку с симметричными граничными условиями
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
    """
    Преобразует контур в выпуклый многоугольник с заданным минимальным количеством вершин

    Параметры:
    - contour_points: точки контура (N x 2)
    - min_vertices: минимальное количество вершин
    """
    if len(contour_points) < 3:
        return contour_points

    # Вычисляем выпуклую оболочку
    hull = ConvexHull(contour_points)
    hull_points = contour_points[hull.vertices]

    # Если вершин меньше минимального, интерполируем
    if len(hull_points) < min_vertices:
        # Интерполяция для добавления вершин
        t = np.linspace(0, 1, len(hull_points))
        t_new = np.linspace(0, 1, min_vertices)

        # Разделяем x и y координаты
        x_coords = hull_points[:, 0]
        y_coords = hull_points[:, 1]

        # Замыкаем для интерполяции
        x_coords = np.append(x_coords, x_coords[0])
        y_coords = np.append(y_coords, y_coords[0])
        t = np.linspace(0, 1, len(x_coords))

        # Интерполируем
        x_new = np.interp(t_new, t, x_coords)
        y_new = np.interp(t_new, t, y_coords)

        hull_points = np.column_stack([x_new, y_new])

    return hull_points


def detect_anomalies_from_residual(residual: np.ndarray, X: np.ndarray, Y: np.ndarray,
                                   n_sigma: float = 2.0, min_area_pixels: int = 30,
                                   apply_tapering: bool = True) -> Dict[str, Any]:
    """
    Выделение аномалий из остаточного поля методом 3σ
    """
    # Применяем окантовку для уменьшения краевых эффектов
    if apply_tapering:
        residual = add_tapering(residual, taper_width=30)

    # Нормализация
    residual_norm = (residual - np.mean(residual)) / np.std(residual)

    # Пороговая обработка
    threshold = n_sigma
    mask = np.abs(residual_norm) > threshold

    # Морфологическая обработка
    struct_elem = morphology.disk(3)
    mask = morphology.dilation(mask, struct_elem)
    mask = morphology.closing(mask, morphology.disk(5))
    mask = morphology.opening(mask, morphology.disk(3))

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

    for label_id in range(1, num_anomalies + 1):
        mask_label = (labeled_mask == label_id)
        contours = measure.find_contours(mask_label, 0.5)

        if not contours:
            continue

        contour = max(contours, key=len)

        contour_x = np.interp(contour[:, 1], np.arange(len(x_km[0])), x_km[0])
        contour_y = np.interp(contour[:, 0], np.arange(len(y_km[:, 0])), y_km[:, 0])

        # Создаём выпуклый многоугольник
        contour_points = np.column_stack([contour_x, contour_y])
        hull_points = make_convex_polygon(contour_points, min_vertices=12)

        # Замыкаем многоугольник
        hull_points = np.vstack([hull_points, hull_points[0]])

        polygons.append(Polygon(hull_points, closed=True, fill=False,
                                edgecolor='red', linewidth=2))

        # Аппроксимация эллипсом
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

    return {
        'num_anomalies': num_anomalies,
        'polygons': polygons,
        'ellipses': ellipses,
        'centers': centers,
        'areas_km2': areas_km2,
        'mask': mask,
        'residual_norm': residual_norm,
        'residual': residual
    }


# ==================== ОЦЕНКА КАЧЕСТВА ====================

def calculate_source_centers(sources: List) -> List[Tuple[float, float]]:
    """Вычисляет центры реальных источников в км"""
    return [(s[0] / 1000, s[1] / 1000) for s in sources]


def calculate_source_areas(sources: List) -> List[float]:
    """Вычисляет площади горизонтальных сечений источников в км²"""
    return [np.pi * (s[4] / 1000) ** 2 for s in sources]


def evaluate_result(detection_result: Dict, sources: List,
                    weight_num: float = 0.4,
                    weight_centers: float = 0.3,
                    weight_areas: float = 0.3) -> Dict:
    """Оценивает качество выделения аномалий"""

    n_sources = len(sources)
    n_detected = detection_result['num_anomalies']

    # Оценка количества
    if n_detected == n_sources:
        num_score = 1.0
    else:
        num_score = 1.0 - min(1.0, abs(n_detected - n_sources) / n_sources)

    # Оценка совпадения центров
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
            normalized_dist = min(1.0, min_dist / 5.0)
            matched_pairs.append(1.0 - normalized_dist)
        center_score = np.mean(matched_pairs) if matched_pairs else 0.0

    # Оценка совпадения площадей
    source_areas = calculate_source_areas(sources)
    detected_areas = detection_result['areas_km2']

    area_score = 0.0
    if detected_areas and source_areas:
        source_areas_sorted = sorted(source_areas)
        detected_areas_sorted = sorted(detected_areas)

        n_compare = min(len(source_areas_sorted), len(detected_areas_sorted))
        if n_compare > 0:
            area_ratios = []
            for i in range(n_compare):
                ratio = min(detected_areas_sorted[i], source_areas_sorted[i]) / \
                        max(detected_areas_sorted[i], source_areas_sorted[i])
                area_ratios.append(ratio)
            area_score = np.mean(area_ratios)

    total_score = (weight_num * num_score +
                   weight_centers * center_score +
                   weight_areas * area_score)

    return {
        'score': total_score,
        'num_score': num_score,
        'center_score': center_score,
        'area_score': area_score,
        'n_detected': n_detected,
        'n_sources': n_sources
    }


# ==================== ОПТИМИЗАЦИЯ ПАРАМЕТРОВ ====================

@dataclass
class OptimizationResult:
    """Результат оптимизации"""
    params: Dict
    residual: np.ndarray
    detection: Dict
    evaluation: Dict
    rank: int = 0


def optimize_parameters(field: np.ndarray, X: np.ndarray, Y: np.ndarray,
                        sources: List, grid_size: int,
                        polynomial_orders: List[int] = None,
                        averaging_k_values: List[float] = None,
                        use_ransac: bool = True,
                        n_sigma: float = 2.0,
                        min_area_pixels: int = 30,
                        verbose: bool = True) -> List[OptimizationResult]:
    """
    Полный перебор параметров: полиномы 1-4 + фильтр среднего с логарифмическим шагом
    """
    if polynomial_orders is None:
        polynomial_orders = [1, 2, 3, 4]
    if averaging_k_values is None:
        # Генерируем значения k с логарифмическим шагом от 0.01 до 0.25 (8 значений)
        averaging_k_values = generate_logspace_k_values(0.01, 0.25, 8)

    total_combinations = len(polynomial_orders) * len(averaging_k_values)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"КОМПЛЕКСНЫЙ ПОДБОР ПАРАМЕТРОВ")
        print(f"{'=' * 60}")
        print(f"Тип фильтра: СРЕДНЕГО (averaging) - сохраняет округлую форму аномалий")
        print(f"Всего комбинаций: {total_combinations}")
        print(f"  Полиномы: {polynomial_orders}")
        print(f"  k (размер окна): {[f'{k:.6f}' for k in averaging_k_values]}")
        print(f"  (логарифмический шаг от 0.01 до 0.25, 8 значений)")
        print(f"{'=' * 60}\n")

    results = []
    current = 0

    for poly_order in polynomial_orders:
        for avg_k in averaging_k_values:
            current += 1

            if verbose and current % 5 == 0:
                print(f"  Прогресс: {current}/{total_combinations} ({100 * current / total_combinations:.1f}%)")

            try:
                # Шаг 1: Снятие полиномиального тренда + фильтр среднего
                result_trend = remove_trend_polynomial_plus_averaging(
                    field, X, Y, poly_order, avg_k, grid_size, use_ransac
                )

                # Шаг 2: Выделение аномалий из остаточного поля
                detection = detect_anomalies_from_residual(
                    result_trend['residual'], X, Y, n_sigma=n_sigma,
                    min_area_pixels=min_area_pixels, apply_tapering=True
                )

                # Шаг 3: Оценка качества
                evaluation = evaluate_result(detection, sources)

                results.append(OptimizationResult(
                    params={
                        'polynomial_order': poly_order,
                        'averaging_k': avg_k,
                        'window_size': result_trend['window_size'],
                        'filter_type': 'averaging',
                        'use_ransac': use_ransac
                    },
                    residual=result_trend['residual'],
                    detection=detection,
                    evaluation=evaluation
                ))

            except Exception as e:
                if verbose:
                    print(f"  Ошибка для poly={poly_order}, k={avg_k:.6f}: {e}")
                continue

    # Сортировка по оценке
    results.sort(key=lambda x: x.evaluation['score'], reverse=True)

    for i, res in enumerate(results):
        res.rank = i + 1

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ")
        print(f"{'=' * 60}")
        print(f"\nТоп-5 лучших комбинаций:\n")

        for i in range(min(5, len(results))):
            r = results[i]
            print(f"{i + 1}. Оценка: {r.evaluation['score']:.3f} "
                  f"(аномалий: {r.evaluation['n_detected']}/{r.evaluation['n_sources']})")
            print(f"   Параметры: полином={r.params['polynomial_order']}, "
                  f"k={r.params['averaging_k']:.6f} (размер окна={r.params['window_size']}×{r.params['window_size']})")
            print(f"   Детали: кол-во={r.evaluation['num_score']:.3f}, "
                  f"центры={r.evaluation['center_score']:.3f}, "
                  f"площади={r.evaluation['area_score']:.3f}")
            print()

    return results


# ==================== ВИЗУАЛИЗАЦИЯ ====================

def create_custom_colormap():
    """Создаёт пользовательскую цветовую карту"""
    return LinearSegmentedColormap.from_list(
        "grav_cmap", ["darkblue", "blue", "cyan", "green", "yellow", "orange", "red"]
    )


def save_figure(fig, filename: str, output_dir: str = "results"):
    """Сохраняет фигуру в файл PNG"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   Сохранено: {filepath}")
    return filepath


def plot_results(original_field: np.ndarray, X: np.ndarray, Y: np.ndarray,
                 sources: List, z_level: float, title: str,
                 best_result: OptimizationResult,
                 all_results: List[OptimizationResult],
                 output_dir: str = "results") -> plt.Figure:
    """Визуализация результатов с сохранением в PNG"""

    fig = plt.figure(figsize=(18, 12))

    x_km = X / 1000
    y_km = Y / 1000
    cmap = create_custom_colormap()

    # Применяем окантовку к исходному полю для визуализации
    field_tapered = add_tapering(original_field, taper_width=30)

    # 1. Исходное поле с контурами шаров (пунктир)
    ax1 = fig.add_subplot(2, 3, 1)
    contour1 = ax1.contourf(x_km, y_km, field_tapered, levels=50, cmap=cmap, alpha=0.9)
    # Показываем контуры шаров пунктиром
    for s in sources:
        circle = Circle((s[0] / 1000, s[1] / 1000), s[4] / 1000,
                        fill=False, edgecolor='white', linewidth=2, linestyle='--')
        ax1.add_patch(circle)
    ax1.set_title('Исходное поле (без шума и тренда)\nПунктир - контуры шаров')
    ax1.set_xlabel('X, км')
    ax1.set_ylabel('Y, км')
    ax1.set_xlim(0, 5)
    ax1.set_ylim(0, 5)
    ax1.grid(True, alpha=0.3)
    plt.colorbar(contour1, ax=ax1, label='мГал')

    # 2. Поле после снятия тренда с окантовкой
    ax2 = fig.add_subplot(2, 3, 2)
    residual_tapered = add_tapering(best_result.residual, taper_width=30)
    contour2 = ax2.contourf(x_km, y_km, residual_tapered, levels=50, cmap=cmap, alpha=0.9)
    ax2.set_title(f'После снятия тренда (фильтр СРЕДНЕГО)\n(полином={best_result.params["polynomial_order"]}, '
                  f'k={best_result.params["averaging_k"]:.6f})')
    ax2.set_xlabel('X, км')
    ax2.set_ylabel('Y, км')
    ax2.set_xlim(0, 5)
    ax2.set_ylim(0, 5)
    ax2.grid(True, alpha=0.3)
    plt.colorbar(contour2, ax=ax2, label='мГал')

    # 3. Выделенные аномалии (с окантовкой)
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.contourf(x_km, y_km, residual_tapered, levels=50, cmap=cmap, alpha=0.6)

    if best_result.detection['polygons']:
        poly_collection = PatchCollection(best_result.detection['polygons'],
                                          facecolor='none', edgecolor='red', linewidth=2)
        ax3.add_collection(poly_collection)

    if best_result.detection['ellipses']:
        ellipse_collection = PatchCollection(best_result.detection['ellipses'],
                                             facecolor='none', edgecolor='blue',
                                             linewidth=2, linestyle='--')
        ax3.add_collection(ellipse_collection)

    # Показываем реальные источники пунктиром для сравнения
    for s in sources:
        circle = Circle((s[0] / 1000, s[1] / 1000), s[4] / 1000,
                        fill=False, edgecolor='white', linewidth=1.5, linestyle=':')
        ax3.add_patch(circle)

    ax3.set_title(f'Выделенные аномалии (выпуклые 12-угольники)\n'
                  f'{best_result.evaluation["n_detected"]} из '
                  f'{best_result.evaluation["n_sources"]} (оценка={best_result.evaluation["score"]:.3f})')
    ax3.set_xlabel('X, км')
    ax3.set_ylabel('Y, км')
    ax3.set_xlim(0, 5)
    ax3.set_ylim(0, 5)
    ax3.grid(True, alpha=0.3)

    # 4. Тепловая карта оценок
    ax4 = fig.add_subplot(2, 3, 4)
    poly_orders = sorted(set(r.params['polynomial_order'] for r in all_results))
    k_values = sorted(set(r.params['averaging_k'] for r in all_results))

    score_matrix = np.zeros((len(poly_orders), len(k_values)))
    for i, poly in enumerate(poly_orders):
        for j, k in enumerate(k_values):
            for r in all_results:
                if r.params['polynomial_order'] == poly and abs(r.params['averaging_k'] - k) < 0.0001:
                    score_matrix[i, j] = r.evaluation['score']
                    break

    im = ax4.imshow(score_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1,
                    extent=[min(k_values), max(k_values), len(poly_orders), 0])
    ax4.set_xlabel('k (размер окна фильтра среднего) - логарифмическая шкала')
    ax4.set_ylabel('Порядок полинома')
    ax4.set_title('Оценка качества в зависимости от параметров')
    ax4.set_yticks(range(1, len(poly_orders) + 1))
    ax4.set_yticklabels(poly_orders)
    ax4.set_xscale('log')
    plt.colorbar(im, ax=ax4, label='Оценка качества')

    # 5. Количество обнаруженных аномалий
    ax5 = fig.add_subplot(2, 3, 5)
    n_detected_matrix = np.zeros((len(poly_orders), len(k_values)))
    for i, poly in enumerate(poly_orders):
        for j, k in enumerate(k_values):
            for r in all_results:
                if r.params['polynomial_order'] == poly and abs(r.params['averaging_k'] - k) < 0.0001:
                    n_detected_matrix[i, j] = r.evaluation['n_detected']
                    break

    im2 = ax5.imshow(n_detected_matrix, aspect='auto', cmap='viridis',
                     extent=[min(k_values), max(k_values), len(poly_orders), 0])
    ax5.set_xlabel('k (размер окна фильтра среднего) - логарифмическая шкала')
    ax5.set_ylabel('Порядок полинома')
    ax5.set_title(f'Количество обнаруженных аномалий (реально={len(sources)})')
    ax5.set_yticks(range(1, len(poly_orders) + 1))
    ax5.set_yticklabels(poly_orders)
    ax5.set_xscale('log')
    plt.colorbar(im2, ax=ax5, label='Количество аномалий')

    # 6. Информация о лучшем результате
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    info_text = f"""
    НАИЛУЧШИЙ РЕЗУЛЬТАТ
    ====================

    ПАРАМЕТРЫ:
    • Полином {best_result.params['polynomial_order']}-го порядка
    • Фильтр СРЕДНЕГО (averaging)
    • k = {best_result.params['averaging_k']:.6f}
    • Размер окна: {best_result.params['window_size']}×{best_result.params['window_size']}
    • Окантовка: применена
    • Шаг k: логарифмический (0.01 → 0.25, 8 значений)

    РЕЗУЛЬТАТЫ:
    • Общая оценка: {best_result.evaluation['score']:.3f}
    • Обнаружено: {best_result.evaluation['n_detected']} / {best_result.evaluation['n_sources']}
    • Совпадение количества: {best_result.evaluation['num_score']:.3f}
    • Совпадение центров: {best_result.evaluation['center_score']:.3f}
    • Совпадение площадей: {best_result.evaluation['area_score']:.3f}

    ПЛОЩАДИ АНОМАЛИЙ (км²):
    """

    for i, area in enumerate(best_result.detection['areas_km2']):
        info_text += f"\n    Аномалия {i + 1}: {area:.2f} км²"

    ax6.text(0.1, 0.95, info_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle(f'{title}\nВыделение аномалий методом 3σ (фильтр СРЕДНЕГО)', fontsize=14)
    plt.tight_layout()

    # Сохраняем фигуру
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_figure(fig, f"anomaly_detection_{timestamp}.png", output_dir)

    return fig


def plot_optimization_summary(all_results: List[OptimizationResult], n_top: int = 15,
                              output_dir: str = "results") -> plt.Figure:
    """Визуализация сводки результатов оптимизации с сохранением в PNG"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    top_results = all_results[:n_top]

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

    # 3. Параметры лучших результатов (на логарифмической шкале для k)
    ax = axes[1, 0]
    poly_orders = [r.params['polynomial_order'] for r in top_results]
    k_values = [r.params['averaging_k'] for r in top_results]

    ax.scatter(range(len(top_results)), poly_orders, s=100, alpha=0.6,
               label='Порядок полинома', marker='o', color='green')
    ax.scatter(range(len(top_results)), k_values, s=100, alpha=0.6,
               label='k (фильтр среднего, лог. шкала)', marker='s', color='purple')
    ax.set_yscale('log')
    ax.set_xlabel('Ранг')
    ax.set_ylabel('Значение параметра (лог. шкала для k)')
    ax.set_title('Параметры лучших результатов')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Распределение порядков полиномов
    ax = axes[1, 1]
    all_orders = [r.params['polynomial_order'] for r in all_results[:100]]
    unique_orders = sorted(set(all_orders))
    order_counts = [all_orders.count(o) for o in unique_orders]

    ax.bar([str(o) for o in unique_orders], order_counts, alpha=0.7, color='teal')
    ax.set_xlabel('Порядок полинома')
    ax.set_ylabel('Частота в топ-100')
    ax.set_title('Распределение порядков полиномов')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Анализ результатов оптимизации (фильтр СРЕДНЕГО)', fontsize=14)
    plt.tight_layout()

    # Сохраняем фигуру
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_figure(fig, f"optimization_summary_{timestamp}.png", output_dir)

    return fig


# ==================== ОСНОВНАЯ ПРОГРАММА ====================

def main():
    print("=" * 60)
    print("ВЫДЕЛЕНИЕ АНОМАЛИЙ НА ИСХОДНЫХ ДАННЫХ (БЕЗ ШУМА И ТРЕНДА)")
    print("ФИЛЬТР СРЕДНЕГО (averaging) - сохраняет округлую форму аномалий")
    print("ЛОГАРИФМИЧЕСКИЙ ШАГ ПАРАМЕТРА k (0.01 → 0.25, 8 значений)")
    print("=" * 60)

    # Создаём директорию для результатов
    output_dir = "anomaly_detection_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"\nСоздана директория: {output_dir}")

    # Параметры
    measurement_level = 500  # м над уровнем моря
    grid_size = 250

    # Генерация исходных данных (без шума и тренда)
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

    # Вывод информации об источниках
    print("\n2. Информация об источниках:")
    densities = [s[3] for s in sources]
    radii = [s[4] for s in sources]
    depths = [s[2] for s in sources]

    print(f"   Плотность: средняя = {np.mean(densities):.3f} г/см³, "
          f"диапазон = [{min(densities):.3f}, {max(densities):.3f}]")
    print(f"   Радиус: средний = {np.mean(radii):.1f} м, "
          f"диапазон = [{min(radii):.1f}, {max(radii):.1f}]")
    print(f"   Глубина центра: средняя = {np.mean(depths):.1f} м, "
          f"диапазон = [{min(depths):.1f}, {max(depths):.1f}]")

    # Генерация значений k с логарифмическим шагом
    k_values = generate_logspace_k_values(0.01, 0.25, 8)
    print(f"\n   Значения k (логарифмический шаг) для фильтра СРЕДНЕГО:")
    for i, k in enumerate(k_values, 1):
        window_size = int(grid_size * k)
        if window_size % 2 == 0:
            window_size += 1
        window_size = max(3, window_size)
        print(f"      {i}. k = {k:.6f} → размер окна: {window_size}×{window_size} пикселей")

    # Оптимизация параметров
    print("\n3. Оптимизация параметров для выделения аномалий...")
    results = optimize_parameters(
        field, X, Y, sources, grid_size,
        polynomial_orders=[1, 2, 3, 4],
        averaging_k_values=k_values,
        use_ransac=True,
        n_sigma=2.0,
        min_area_pixels=25,
        verbose=True
    )

    if not results:
        print("Ошибка: не получено результатов оптимизации!")
        return

    # Визуализация результатов с сохранением
    print("\n4. Визуализация и сохранение результатов...")
    plot_results(
        field, X, Y, sources, z_level,
        "ИСХОДНЫЕ ДАННЫЕ (без шума и тренда)",
        results[0], results,
        output_dir=output_dir
    )

    # Сводка по оптимизации
    plot_optimization_summary(results, n_top=15, output_dir=output_dir)

    # Сохранение результатов в JSON
    print("\n5. Сохранение данных в JSON...")

    best = results[0]
    summary = {
        'timestamp': datetime.now().isoformat(),
        'filter_type': 'averaging (среднего)',
        'data_info': {
            'n_sources': len(sources),
            'grid_size': grid_size,
            'field_min': float(np.min(field)),
            'field_max': float(np.max(field)),
            'measurement_level': z_level,
            'x_range': [0, 5],
            'y_range': [0, 5]
        },
        'optimization_params': {
            'k_values': k_values,
            'k_scale': 'logarithmic',
            'k_min': 0.01,
            'k_max': 0.25,
            'n_k_values': 8,
            'polynomial_orders': [1, 2, 3, 4]
        },
        'best_params': best.params,
        'best_score': float(best.evaluation['score']),
        'n_detected': best.evaluation['n_detected'],
        'n_sources': best.evaluation['n_sources'],
        'detected_anomalies': [
            {
                'center': center,
                'area_km2': area
            }
            for center, area in zip(best.detection['centers'], best.detection['areas_km2'])
        ],
        'all_results': [
            {
                'params': r.params,
                'score': float(r.evaluation['score']),
                'n_detected': r.evaluation['n_detected']
            }
            for r in results[:20]
        ]
    }

    json_path = os.path.join(output_dir, 'anomaly_detection_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"   Сохранено: {json_path}")

    # Итоговые рекомендации
    print("\n" + "=" * 60)
    print("ИТОГОВЫЕ РЕКОМЕНДАЦИИ")
    print("=" * 60)

    print(f"\n✓ Наилучший результат достигнут с параметрами:")
    print(f"   • Полином {best.params['polynomial_order']}-го порядка")
    print(f"   • Фильтр СРЕДНЕГО (averaging)")
    print(
        f"   • k = {best.params['averaging_k']:.6f} (размер окна = {best.params['window_size']}×{best.params['window_size']})")
    print(f"   • Применена окантовка для уменьшения краевых эффектов")
    print(f"   • Аномалии представлены выпуклыми 12-угольниками")
    print(f"   • Использован логарифмический шаг k от 0.01 до 0.25 (8 значений)")

    print(f"\n✓ Качество выделения:")
    print(f"   • Обнаружено: {best.evaluation['n_detected']} из {best.evaluation['n_sources']} аномалий")
    print(f"   • Общая оценка: {best.evaluation['score']:.3f}")
    print(f"   • Совпадение центров: {best.evaluation['center_score']:.3f}")
    print(f"   • Совпадение площадей: {best.evaluation['area_score']:.3f}")

    print(f"\n✓ Преимущества фильтра СРЕДНЕГО (averaging):")
    print(f"   • Сохраняет округлую форму аномалий (нет крестообразных артефактов)")
    print(f"   • Линейный фильтр, не вносит нелинейных искажений")
    print(f"   • Результат интерпретации более физически обоснован")

    if best.evaluation['n_detected'] == best.evaluation['n_sources']:
        print("\n✓ ОТЛИЧНО! Все аномалии успешно выделены!")
    elif best.evaluation['n_detected'] > best.evaluation['n_sources']:
        print("\n⚠ Внимание: обнаружено больше аномалий, чем реальных источников.")
        print("   Рекомендуется увеличить min_area_pixels или n_sigma")
    else:
        print("\n⚠ Внимание: обнаружено меньше аномалий, чем реальных источников.")
        print("   Рекомендуется уменьшить n_sigma или k")

    print(f"\n✓ Все результаты сохранены в директории: {output_dir}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()