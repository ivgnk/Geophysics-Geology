"""
Модифицируй программу:
1) Убери создание рисунков parallel_coordinates, radar_chart,
2) Восстанови рисунки: final_results_part1a_original, final_results_part1b_detrended, final_results_part2
3) По 5 худшим результатам добавь вывод QN, QC, QA
4) Карта аналогичная final_results_part2_anomalies, но для самого худшего результата
"""

"""
МЕТОД А: Выделение аномалий по правилу 3σ - КОНФИГУРИРУЕМАЯ ВЕРСИЯ
- Оптимизация параметров (полный перебор)
- Статистика: лучшие/худшие варианты, распределение Q
- Визуализация: исходное поле, поле после снятия тренда, выделенные аномалии
- Отдельная визуализация для худшего результата
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Ellipse, Circle
from scipy.ndimage import gaussian_filter, label, uniform_filter
from scipy.ndimage import binary_closing, binary_opening, generate_binary_structure
from scipy.signal import convolve2d
from skimage import measure
from typing import List, Tuple, Dict, Any
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from scipy.spatial import ConvexHull, cKDTree
import warnings
from dataclasses import dataclass
import json
import os
from datetime import datetime
import time
from multiprocessing import Pool, cpu_count

# ==================== НАСТРОЙКИ ОПТИМИЗАЦИЙ ====================

OPTIMIZATION_CONFIG = {
    'use_precomputed_filters': 1,
    'use_numba': 0,
    'use_cupy': 0,
    'use_multiprocessing': 1,
    'use_ransac_subsampling': 1,
    'use_uniform_filter': 1,
    'ransac_subsample_ratio': 0.05,
    'numba_parallel': 1,
    'multiprocessing_cores': -1,
}

# ==================== ПРОВЕРКА ДОСТУПНОСТИ БИБЛИОТЕК ====================

NUMBA_AVAILABLE = False
if OPTIMIZATION_CONFIG['use_numba']:
    try:
        from numba import jit, prange, njit

        NUMBA_AVAILABLE = True
        print("✓ Numba доступен")
    except ImportError:
        print("⚠ Numba не установлен")
        OPTIMIZATION_CONFIG['use_numba'] = 0

CUDA_AVAILABLE = False
cp = None
if OPTIMIZATION_CONFIG['use_cupy']:
    try:
        import cupy as cp

        if cp.cuda.is_available():
            CUDA_AVAILABLE = True
            print(f"✓ CuPy доступен (GPU)")
        else:
            print("⚠ CUDA не доступен")
            OPTIMIZATION_CONFIG['use_cupy'] = 0
    except ImportError:
        print("⚠ CuPy не установлен")
        OPTIMIZATION_CONFIG['use_cupy'] = 0

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ==================== КОНСТАНТЫ ====================
G = 6.67430e-11
M_TO_MGAL = 1e5
DENSITY_CONVERSION = 1000


# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def calculate_subsample_step(grid_size: int, ratio: float) -> int:
    if ratio >= 1.0:
        return 1
    n_points = grid_size * grid_size
    target_points = int(n_points * ratio)
    step = int(np.sqrt(n_points / max(1, target_points)))
    step = max(1, min(step, grid_size // 2))
    return step


# ==================== JIT-ФУНКЦИИ (Numba) ====================

if NUMBA_AVAILABLE and OPTIMIZATION_CONFIG['use_numba']:
    parallel_setting = True if OPTIMIZATION_CONFIG['numba_parallel'] else False


    @njit(parallel=parallel_setting, cache=True)
    def fast_normalize(field):
        mean = np.mean(field)
        std = np.std(field)
        if std == 0:
            return field
        return (field - mean) / std


    @njit(parallel=parallel_setting, cache=True)
    def fast_threshold(field_norm, n_sigma):
        mask = np.zeros_like(field_norm, dtype=np.bool_)
        for i in prange(field_norm.shape[0]):
            for j in range(field_norm.shape[1]):
                if abs(field_norm[i, j]) > n_sigma:
                    mask[i, j] = True
        return mask


    @njit(parallel=parallel_setting, cache=True)
    def fast_labeling(mask):
        labeled = np.zeros_like(mask, dtype=np.int32)
        current_label = 1
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j] and labeled[i, j] == 0:
                    stack = [(i, j)]
                    labeled[i, j] = current_label
                    while stack:
                        x, y = stack.pop()
                        for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < mask.shape[0] and 0 <= ny < mask.shape[1]:
                                if mask[nx, ny] and labeled[nx, ny] == 0:
                                    labeled[nx, ny] = current_label
                                    stack.append((nx, ny))
                    current_label += 1
        return labeled, current_label - 1
else:
    def fast_normalize(field):
        return (field - np.mean(field)) / np.std(field)


    def fast_threshold(field_norm, n_sigma):
        return np.abs(field_norm) > n_sigma


    def fast_labeling(mask):
        from scipy.ndimage import label
        return label(mask)


# ==================== ГЕНЕРАЦИЯ ДАННЫХ ====================

def generate_non_overlapping_sources(n_sources: int, x_range: Tuple = (0, 5000),
                                     y_range: Tuple = (0, 5000),
                                     radius_range: Tuple = (50, 500),
                                     density_range: Tuple = (0.05, 0.3),
                                     z_level: float = 500,
                                     min_distance_factor: float = 1.5,
                                     max_attempts: int = 100):
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


def create_gravitational_field_map(n_sources: int = 12, grid_size: int = 250,
                                   density_range: Tuple = (0.05, 0.3),
                                   radius_range: Tuple = (80, 400),
                                   z_level: float = 500,
                                   smoothing_sigma: float = 2.0,
                                   min_distance_factor: float = 1.5) -> Tuple:
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


def generate_logspace_k_values(k_min: float = 0.008, k_max: float = 0.25,
                               n_values: int = 12) -> List[float]:
    log_min = np.log10(k_min)
    log_max = np.log10(k_max)
    log_values = np.linspace(log_min, log_max, n_values)
    k_values = 10 ** log_values
    return [round(k, 6) for k in k_values]


# ==================== ПРЕДВЫЧИСЛЕНИЕ ФИЛЬТРОВ ====================

class PrecomputedFilters:
    def __init__(self, field: np.ndarray, grid_size: int, k_values: List[float]):
        self.field = field
        self.grid_size = grid_size
        self.k_values = k_values
        self.filters_cache = {}
        if OPTIMIZATION_CONFIG['use_precomputed_filters']:
            self._precompute_filters()

    def _precompute_filters(self):
        print("   Предвычисление фильтров...")
        for k in self.k_values:
            window_size = int(self.grid_size * k)
            if window_size % 2 == 0:
                window_size += 1
            window_size = max(3, window_size)
            self.filters_cache[k] = {'window_size': window_size}
        print(f"   Предвычислено {len(self.filters_cache)} фильтров")

    def apply_filter(self, field: np.ndarray, k: float) -> np.ndarray:
        if not OPTIMIZATION_CONFIG['use_precomputed_filters']:
            return None
        if k not in self.filters_cache:
            return None
        window_size = self.filters_cache[k]['window_size']
        if OPTIMIZATION_CONFIG['use_uniform_filter']:
            return uniform_filter(field, size=window_size, mode='reflect')
        else:
            kernel = np.ones((window_size, window_size)) / (window_size ** 2)
            return convolve2d(field, kernel, mode='same', boundary='symm')


# ==================== МЕТОДЫ СНЯТИЯ ТРЕНДА ====================

def remove_polynomial_trend_fast(field: np.ndarray, X: np.ndarray, Y: np.ndarray,
                                 order: int = 2, use_ransac: bool = True):
    grid_size = field.shape[0]
    if OPTIMIZATION_CONFIG['use_ransac_subsampling']:
        ratio = OPTIMIZATION_CONFIG['ransac_subsample_ratio']
        step = calculate_subsample_step(grid_size, ratio)
        x_km = (X[::step, ::step].flatten()) / 1000
        y_km = (Y[::step, ::step].flatten()) / 1000
        z = field[::step, ::step].flatten()
    else:
        x_km = X.flatten() / 1000
        y_km = Y.flatten() / 1000
        z = field.flatten()
    poly = PolynomialFeatures(degree=order)
    X_poly = poly.fit_transform(np.column_stack([x_km, y_km]))
    if use_ransac:
        regressor = RANSACRegressor(random_state=42, min_samples=0.3,
                                    residual_threshold=np.std(field) * 0.5,
                                    max_trials=50)
    else:
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
    regressor.fit(X_poly, z)
    X_full = X.flatten() / 1000
    Y_full = Y.flatten() / 1000
    X_full_poly = poly.transform(np.column_stack([X_full, Y_full]))
    trend = regressor.predict(X_full_poly).reshape(field.shape)
    return field - trend, trend


# ==================== ВЫДЕЛЕНИЕ АНОМАЛИЙ ====================

def detect_anomalies_fast(residual: np.ndarray, X: np.ndarray, Y: np.ndarray,
                          n_sigma: float = 1.5, min_area_pixels: int = 15) -> Dict:
    if OPTIMIZATION_CONFIG['use_numba'] and NUMBA_AVAILABLE:
        residual_norm = fast_normalize(residual)
        mask = fast_threshold(residual_norm, n_sigma)
    else:
        residual_norm = (residual - np.mean(residual)) / np.std(residual)
        mask = np.abs(residual_norm) > n_sigma
    struct = generate_binary_structure(2, 1)
    mask = binary_closing(mask, structure=struct, iterations=1)
    mask = binary_opening(mask, structure=struct, iterations=1)
    if OPTIMIZATION_CONFIG['use_numba'] and NUMBA_AVAILABLE:
        labeled_mask, num_features = fast_labeling(mask)
    else:
        from scipy.ndimage import label
        labeled_mask, num_features = label(mask)
    centers = []
    x_km = X[0, :] / 1000
    y_km = Y[:, 0] / 1000
    for i in range(1, num_features + 1):
        y_indices, x_indices = np.where(labeled_mask == i)
        if len(y_indices) >= min_area_pixels:
            center_x = np.mean(x_km[x_indices])
            center_y = np.mean(y_km[y_indices])
            centers.append((center_x, center_y))
    return {
        'num_anomalies': len(centers),
        'centers': centers
    }


def detect_anomalies_gpu(residual: np.ndarray, X: np.ndarray, Y: np.ndarray,
                         n_sigma: float = 1.5, min_area_pixels: int = 15) -> Dict:
    if not OPTIMIZATION_CONFIG['use_cupy'] or not CUDA_AVAILABLE:
        return detect_anomalies_fast(residual, X, Y, n_sigma, min_area_pixels)
    residual_gpu = cp.asarray(residual)
    mean_gpu = cp.mean(residual_gpu)
    std_gpu = cp.std(residual_gpu)
    residual_norm_gpu = (residual_gpu - mean_gpu) / std_gpu
    mask_gpu = cp.abs(residual_norm_gpu) > n_sigma
    mask = cp.asnumpy(mask_gpu)
    struct = generate_binary_structure(2, 1)
    mask = binary_closing(mask, structure=struct, iterations=1)
    mask = binary_opening(mask, structure=struct, iterations=1)
    from scipy.ndimage import label
    labeled_mask, num_features = label(mask)
    centers = []
    x_km = X[0, :] / 1000
    y_km = Y[:, 0] / 1000
    for i in range(1, num_features + 1):
        y_indices, x_indices = np.where(labeled_mask == i)
        if len(y_indices) >= min_area_pixels:
            center_x = np.mean(x_km[x_indices])
            center_y = np.mean(y_km[y_indices])
            centers.append((center_x, center_y))
    return {
        'num_anomalies': len(centers),
        'centers': centers
    }


def evaluate_result(detection: Dict, sources: List) -> Dict:
    """
    Полная оценка качества с вычислением QN, QC, QA
    """
    n_sources = len(sources)
    n_detected = detection['num_anomalies']

    # QN - оценка количества
    if n_detected == n_sources:
        num_score = 1.0
    else:
        num_score = 1.0 - min(1.0, abs(n_detected - n_sources) / n_sources)

    # QC - оценка центров
    source_centers = [(s[0] / 1000, s[1] / 1000) for s in sources]
    detected_centers = detection['centers']

    center_score = 0.0
    if detected_centers and source_centers:
        tree = cKDTree(source_centers)
        distances, _ = tree.query(detected_centers)
        center_score = 1.0 - np.mean(np.minimum(distances / 3.0, 1.0))

    # QA - оценка площадей (приближённая, по количеству аномалий)
    # Для полной оценки нужны площади, здесь используем упрощённый вариант
    if n_detected == n_sources:
        area_score = 1.0
    elif n_detected == 0:
        area_score = 0.0
    else:
        area_score = min(n_detected, n_sources) / max(n_detected, n_sources)

    # Интегральная оценка
    total_score = 0.4 * num_score + 0.3 * center_score + 0.3 * area_score

    return {
        'score': total_score,
        'num_score': num_score,
        'center_score': center_score,
        'area_score': area_score,
        'n_detected': n_detected,
        'n_sources': n_sources
    }


# ==================== ПАРАЛЛЕЛЬНАЯ ОБРАБОТКА ====================

def process_combination(args):
    field, X, Y, sources, grid_size, poly_order, avg_k, n_sigma, min_area, precomputed_filters = args
    try:
        field_no_poly, _ = remove_polynomial_trend_fast(
            field, X, Y, order=poly_order, use_ransac=True
        )
        if precomputed_filters and OPTIMIZATION_CONFIG['use_precomputed_filters']:
            regional_average = precomputed_filters.apply_filter(field_no_poly, avg_k)
        else:
            window_size = int(grid_size * avg_k)
            if window_size % 2 == 0:
                window_size += 1
            window_size = max(3, window_size)
            if OPTIMIZATION_CONFIG['use_uniform_filter']:
                regional_average = uniform_filter(field_no_poly, size=window_size, mode='reflect')
            else:
                kernel = np.ones((window_size, window_size)) / (window_size ** 2)
                regional_average = convolve2d(field_no_poly, kernel, mode='same', boundary='symm')
        residual = field_no_poly - regional_average
        detection = detect_anomalies_gpu(residual, X, Y, n_sigma, min_area)
        evaluation = evaluate_result(detection, sources)
        return {
            'params': {
                'polynomial_order': poly_order,
                'averaging_k': avg_k,
                'window_size': int(grid_size * avg_k),
                'n_sigma': n_sigma,
                'min_area_pixels': min_area,
            },
            'evaluation': evaluation,
            'detection': detection,
            'residual': residual,
            'success': True
        }
    except Exception as e:
        return {
            'params': {},
            'evaluation': {'score': 0, 'n_detected': 0, 'n_sources': len(sources)},
            'success': False
        }


def optimize_parameters_parallel(field: np.ndarray, X: np.ndarray, Y: np.ndarray,
                                 sources: List, grid_size: int,
                                 polynomial_orders: List[int],
                                 averaging_k_values: List[float],
                                 n_sigma_values: List[float],
                                 min_area_values: List[int],
                                 precomputed_filters: PrecomputedFilters = None,
                                 verbose: bool = True) -> List:
    combinations = []
    for poly_order in polynomial_orders:
        for avg_k in averaging_k_values:
            for n_sigma in n_sigma_values:
                for min_area in min_area_values:
                    combinations.append((
                        field, X, Y, sources, grid_size,
                        poly_order, avg_k, n_sigma, min_area, precomputed_filters
                    ))
    total = len(combinations)
    if verbose:
        print(f"   Всего комбинаций: {total}")
    if OPTIMIZATION_CONFIG['use_multiprocessing']:
        n_cores = OPTIMIZATION_CONFIG['multiprocessing_cores']
        if n_cores == -1:
            n_cores = max(1, cpu_count() - 1)
        print(f"   Используем ядер: {n_cores}")
        with Pool(processes=n_cores) as pool:
            results = pool.map(process_combination, combinations)
    else:
        print("   Используем последовательную обработку")
        results = []
        for i, combo in enumerate(combinations):
            if verbose and i % 50 == 0:
                print(f"   Прогресс: {i}/{total}")
            results.append(process_combination(combo))
    results = [r for r in results if r['success']]
    results.sort(key=lambda x: x['evaluation']['score'], reverse=True)
    return results


# ==================== СТАТИСТИКА И АНАЛИЗ РЕЗУЛЬТАТОВ ====================

def print_statistics(results: List):
    """
    Вывод статистики по всем результатам:
    - 5 лучших вариантов (с QN, QC, QA)
    - 5 худших вариантов (с QN, QC, QA)
    - Минимум, максимум, среднее, медиана
    """
    scores = [r['evaluation']['score'] for r in results]

    print("\n" + "=" * 80)
    print("СТАТИСТИКА РЕЗУЛЬТАТОВ ОПТИМИЗАЦИИ")
    print("=" * 80)

    # 1. 5 лучших вариантов
    print("\n🏆 ТОП-5 ЛУЧШИХ ВАРИАНТОВ:")
    print("-" * 80)
    print(
        f"{'№':<3} {'Q':<8} {'QN':<8} {'QC':<8} {'QA':<8} {'аномалий':<10} {'полином':<7} {'k':<10} {'nσ':<6} {'min_area':<8}")
    print("-" * 80)
    for i in range(min(5, len(results))):
        r = results[i]
        e = r['evaluation']
        p = r['params']
        print(f"{i + 1:<3} {e['score']:<8.4f} {e['num_score']:<8.4f} {e['center_score']:<8.4f} {e['area_score']:<8.4f} "
              f"{e['n_detected']}/{e['n_sources']:<7} {p['polynomial_order']:<7} {p['averaging_k']:<10.5f} "
              f"{p['n_sigma']:<6} {p['min_area_pixels']:<8}")

    # 2. 5 худших вариантов
    print("\n📉 ТОП-5 ХУДШИХ ВАРИАНТОВ:")
    print("-" * 80)
    print(
        f"{'№':<3} {'Q':<8} {'QN':<8} {'QC':<8} {'QA':<8} {'аномалий':<10} {'полином':<7} {'k':<10} {'nσ':<6} {'min_area':<8}")
    print("-" * 80)
    for i in range(min(5, len(results))):
        r = results[-(i + 1)]
        e = r['evaluation']
        p = r['params']
        print(f"{i + 1:<3} {e['score']:<8.4f} {e['num_score']:<8.4f} {e['center_score']:<8.4f} {e['area_score']:<8.4f} "
              f"{e['n_detected']}/{e['n_sources']:<7} {p['polynomial_order']:<7} {p['averaging_k']:<10.5f} "
              f"{p['n_sigma']:<6} {p['min_area_pixels']:<8}")

    # 3. Общая статистика
    print("\n📊 ОБЩАЯ СТАТИСТИКА ПО ВСЕМ РЕЗУЛЬТАТАМ:")
    print("-" * 80)
    print(f"   Минимальная оценка Q_min    = {np.min(scores):.4f}")
    print(f"   Максимальная оценка Q_max   = {np.max(scores):.4f}")
    print(f"   Средняя оценка Q_mean       = {np.mean(scores):.4f}")
    print(f"   Медианная оценка Q_median   = {np.median(scores):.4f}")
    print(f"   Стандартное отклонение σ    = {np.std(scores):.4f}")
    print(f"   Всего комбинаций            = {len(results)}")

    return scores


def plot_score_distribution(scores: List, output_dir: str = "results") -> plt.Figure:
    """
    Построение гистограммы (10 столбцов) и круговой диаграммы распределения оценок Q
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Гистограмма распределения оценок (10 столбцов)
    n_bins = 10
    counts, bins, patches = ax1.hist(scores, bins=n_bins, edgecolor='black',
                                     color='steelblue', alpha=0.7)

    # Цветовая градация столбцов (от красного к зелёному)
    norm = plt.Normalize(vmin=min(scores), vmax=max(scores))
    cmap = plt.cm.RdYlGn
    for patch, left, right in zip(patches, bins[:-1], bins[1:]):
        color = cmap(norm((left + right) / 2))
        patch.set_facecolor(color)

    ax1.set_xlabel('Оценка качества Q', fontsize=12)
    ax1.set_ylabel('Частота', fontsize=12)
    ax1.set_title(f'Гистограмма распределения оценок Q\n(n = {len(scores)} комбинаций, {n_bins} столбцов)',
                  fontsize=12)
    ax1.axvline(x=np.mean(scores), color='blue', linestyle='--',
                label=f'Среднее = {np.mean(scores):.3f}')
    ax1.axvline(x=np.median(scores), color='green', linestyle='--',
                label=f'Медиана = {np.median(scores):.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Круговая диаграмма (категории по диапазонам Q)
    categories = [
        (0.0, 0.2, '0.0-0.2 (очень плохо)', '#d73027'),
        (0.2, 0.4, '0.2-0.4 (плохо)', '#f46d43'),
        (0.4, 0.6, '0.4-0.6 (удовлетворительно)', '#fdae61'),
        (0.6, 0.8, '0.6-0.8 (хорошо)', '#fee08b'),
        (0.8, 1.0, '0.8-1.0 (отлично)', '#66bd63'),
    ]

    category_counts = []
    category_labels = []
    category_colors = []

    for low, high, label, color in categories:
        count = sum(1 for s in scores if low <= s < high)
        if count > 0:
            category_counts.append(count)
            category_labels.append(f'{label}\n({count})')
            category_colors.append(color)

    # Добавляем граничный случай Q = 1.0
    count_one = sum(1 for s in scores if s == 1.0)
    if count_one > 0:
        category_counts.append(count_one)
        category_labels.append(f'1.0 (идеально)\n({count_one})')
        category_colors.append('#1a9850')

    ax2.pie(category_counts, labels=category_labels, colors=category_colors,
            autopct='%1.1f%%', startangle=90, explode=[0.02] * len(category_counts))
    ax2.set_title(f'Распределение оценок качества по категориям\n(n = {len(scores)} комбинаций)',
                  fontsize=12)

    plt.suptitle('Анализ распределения оценок качества выделения аномалий', fontsize=14, fontweight='bold')
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_figure(fig, f"score_distribution_{timestamp}.png", output_dir)

    return fig


# ==================== ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ ====================

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


def make_convex_polygon(contour_points: np.ndarray, min_vertices: int = 12) -> np.ndarray:
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


def create_detailed_detection(residual: np.ndarray, X: np.ndarray, Y: np.ndarray,
                              n_sigma: float = 1.5, min_area_pixels: int = 15) -> Dict:
    """Детальное выделение аномалий с построением полигонов и эллипсов"""
    residual_norm = (residual - np.mean(residual)) / np.std(residual)
    mask = np.abs(residual_norm) > n_sigma

    struct = generate_binary_structure(2, 1)
    mask = binary_closing(mask, structure=struct, iterations=2)
    mask = binary_opening(mask, structure=struct, iterations=2)

    from scipy.ndimage import label
    labeled_mask, num_features = label(mask)

    x_km = X / 1000
    y_km = Y / 1000

    polygons = []
    ellipses = []
    centers = []
    areas_km2 = []

    for i in range(1, num_features + 1):
        mask_label = (labeled_mask == i)
        if np.sum(mask_label) < min_area_pixels:
            continue

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

    return {
        'num_anomalies': len(polygons),
        'polygons': polygons,
        'ellipses': ellipses,
        'centers': centers,
        'areas_km2': areas_km2,
        'mask': mask
    }


def get_detailed_detection(residual: np.ndarray, X: np.ndarray, Y: np.ndarray,
                           n_sigma: float = 1.5, min_area_pixels: int = 15) -> Dict:
    residual_tapered = add_tapering(residual, taper_width=30)
    detailed_detection = create_detailed_detection(
        residual_tapered, X, Y,
        n_sigma=n_sigma,
        min_area_pixels=min_area_pixels
    )
    return detailed_detection


def plot_final_results_part1a(original_field: np.ndarray, X: np.ndarray, Y: np.ndarray,
                              sources: List, z_level: float, best_params: Dict,
                              residual: np.ndarray, evaluation: Dict,
                              output_dir: str = "results") -> plt.Figure:
    fig = plt.figure(figsize=(8, 7))

    x_km = X / 1000
    y_km = Y / 1000
    cmap = create_custom_colormap()

    x_min, x_max = 0, 5
    y_min, y_max = 0, 5

    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 0.05], wspace=0.05)

    ax = fig.add_subplot(gs[0])
    contour = ax.contourf(x_km, y_km, original_field, levels=50, cmap=cmap, alpha=0.9)

    for s in sources:
        circle = Circle((s[0] / 1000, s[1] / 1000), s[4] / 1000,
                        fill=False, edgecolor='white', linewidth=2, linestyle='--')
        ax.add_patch(circle)

    ax.set_title('Исходное гравитационное поле\nс контурами шаров-источников', fontsize=12)
    ax.set_xlabel('X, км')
    ax.set_ylabel('Y, км')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    cax = fig.add_subplot(gs[1])
    cbar = plt.colorbar(contour, cax=cax, orientation='vertical')
    cbar.set_label('Δg,\nмГал', fontsize=9, linespacing=1.2)
    cax.yaxis.set_ticks_position('right')
    cax.yaxis.set_label_position('right')

    plt.suptitle('Результаты обработки гравитационного поля', fontsize=14, fontweight='bold')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_figure(fig, f"final_results_part1a_original_{timestamp}.png", output_dir)

    return fig


def plot_final_results_part1b(original_field: np.ndarray, X: np.ndarray, Y: np.ndarray,
                              sources: List, z_level: float, best_params: Dict,
                              residual: np.ndarray, evaluation: Dict,
                              output_dir: str = "results") -> plt.Figure:
    fig = plt.figure(figsize=(8, 7))

    x_km = X / 1000
    y_km = Y / 1000
    cmap = create_custom_colormap()

    x_min, x_max = 0, 5
    y_min, y_max = 0, 5

    residual_tapered = add_tapering(residual, taper_width=30)

    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 0.05], wspace=0.05)

    ax = fig.add_subplot(gs[0])
    contour = ax.contourf(x_km, y_km, residual_tapered, levels=50, cmap=cmap, alpha=0.9)

    ax.set_title(f'Поле после снятия тренда\n(полином={best_params["polynomial_order"]}, '
                 f'k={best_params["averaging_k"]:.5f})', fontsize=12)
    ax.set_xlabel('X, км')
    ax.set_ylabel('Y, км')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    cax = fig.add_subplot(gs[1])
    cbar = plt.colorbar(contour, cax=cax, orientation='vertical')
    cbar.set_label('Δg,\nмГал', fontsize=9, linespacing=1.2)
    cax.yaxis.set_ticks_position('right')
    cax.yaxis.set_label_position('right')

    plt.suptitle('Результаты обработки гравитационного поля', fontsize=14, fontweight='bold')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_figure(fig, f"final_results_part1b_detrended_{timestamp}.png", output_dir)

    return fig


def plot_final_results_part2(original_field: np.ndarray, X: np.ndarray, Y: np.ndarray,
                             sources: List, z_level: float, best_params: Dict,
                             residual: np.ndarray, evaluation: Dict,
                             output_dir: str = "results") -> plt.Figure:
    fig = plt.figure(figsize=(10, 8))

    x_km = X / 1000
    y_km = Y / 1000
    cmap = create_custom_colormap()

    x_min, x_max = 0, 5
    y_min, y_max = 0, 5

    residual_tapered = add_tapering(residual, taper_width=30)
    detailed_detection = create_detailed_detection(
        residual_tapered, X, Y,
        n_sigma=best_params['n_sigma'],
        min_area_pixels=best_params['min_area_pixels']
    )

    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 0.05], wspace=0.05)

    ax = fig.add_subplot(gs[0])
    contour = ax.contourf(x_km, y_km, residual_tapered, levels=50, cmap=cmap, alpha=0.6)

    if detailed_detection['polygons']:
        poly_collection = PatchCollection(detailed_detection['polygons'],
                                          facecolor='none', edgecolor='red', linewidth=2)
        ax.add_collection(poly_collection)

    if detailed_detection['ellipses']:
        ellipse_collection = PatchCollection(detailed_detection['ellipses'],
                                             facecolor='none', edgecolor='blue',
                                             linewidth=2, linestyle='--')
        ax.add_collection(ellipse_collection)

    for s in sources:
        circle = Circle((s[0] / 1000, s[1] / 1000), s[4] / 1000,
                        fill=False, edgecolor='white', linewidth=1.5, linestyle=':')
        ax.add_patch(circle)

    ax.set_title(f'Выделенные аномалии\n'
                 f'{detailed_detection["num_anomalies"]} из {len(sources)} '
                 f'(оценка качества={evaluation["score"]:.3f})', fontsize=12)
    ax.set_xlabel('X, км')
    ax.set_ylabel('Y, км')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    cax = fig.add_subplot(gs[1])
    cbar = plt.colorbar(contour, cax=cax, orientation='vertical')
    cbar.set_label('Δg,\nмГал', fontsize=9, linespacing=1.2)
    cax.yaxis.set_ticks_position('right')
    cax.yaxis.set_label_position('right')

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Реальные источники (шары)',
               markerfacecolor='white', markeredgecolor='black', markersize=10, linestyle='None'),
        Line2D([0], [0], marker='s', color='w', label='Выделенные аномалии (полигоны)',
               markerfacecolor='none', markeredgecolor='red', markersize=10, linestyle='-', linewidth=2),
        Line2D([0], [0], marker='s', color='w', label='Аппроксимация эллипсом',
               markerfacecolor='none', markeredgecolor='blue', markersize=10, linestyle='--', linewidth=2),
        Line2D([0], [0], marker='o', color='w', label='Контуры реальных источников',
               markerfacecolor='none', markeredgecolor='white', markersize=10, linestyle=':', linewidth=1.5)
    ]
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.15),
              ncol=2, fontsize=9, frameon=True, fancybox=True, shadow=True)

    plt.suptitle('Результаты выделения гравитационных аномалий', fontsize=14, fontweight='bold', y=0.98)
    plt.subplots_adjust(bottom=0.15)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_figure(fig, f"final_results_part2_anomalies_{timestamp}.png", output_dir)

    return fig


def plot_worst_result(original_field: np.ndarray, X: np.ndarray, Y: np.ndarray,
                      sources: List, z_level: float, worst_params: Dict,
                      residual: np.ndarray, evaluation: Dict,
                      output_dir: str = "results") -> plt.Figure:
    """
    Визуализация для худшего результата (аналогичная final_results_part2)
    """
    fig = plt.figure(figsize=(10, 8))

    x_km = X / 1000
    y_km = Y / 1000
    cmap = create_custom_colormap()

    x_min, x_max = 0, 5
    y_min, y_max = 0, 5

    residual_tapered = add_tapering(residual, taper_width=30)
    detailed_detection = create_detailed_detection(
        residual_tapered, X, Y,
        n_sigma=worst_params['n_sigma'],
        min_area_pixels=worst_params['min_area_pixels']
    )

    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 0.05], wspace=0.05)

    ax = fig.add_subplot(gs[0])
    contour = ax.contourf(x_km, y_km, residual_tapered, levels=50, cmap=cmap, alpha=0.6)

    if detailed_detection['polygons']:
        poly_collection = PatchCollection(detailed_detection['polygons'],
                                          facecolor='none', edgecolor='red', linewidth=2)
        ax.add_collection(poly_collection)

    if detailed_detection['ellipses']:
        ellipse_collection = PatchCollection(detailed_detection['ellipses'],
                                             facecolor='none', edgecolor='blue',
                                             linewidth=2, linestyle='--')
        ax.add_collection(ellipse_collection)

    for s in sources:
        circle = Circle((s[0] / 1000, s[1] / 1000), s[4] / 1000,
                        fill=False, edgecolor='white', linewidth=1.5, linestyle=':')
        ax.add_patch(circle)

    ax.set_title(f'ХУДШИЙ РЕЗУЛЬТАТ: выделенные аномалии\n'
                 f'{detailed_detection["num_anomalies"]} из {len(sources)} '
                 f'(оценка={evaluation["score"]:.3f})\n'
                 f'полином={worst_params["polynomial_order"]}, k={worst_params["averaging_k"]:.5f}, '
                 f'nσ={worst_params["n_sigma"]}, min_area={worst_params["min_area_pixels"]}',
                 fontsize=10)
    ax.set_xlabel('X, км')
    ax.set_ylabel('Y, км')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    cax = fig.add_subplot(gs[1])
    cbar = plt.colorbar(contour, cax=cax, orientation='vertical')
    cbar.set_label('Δg,\nмГал', fontsize=9, linespacing=1.2)
    cax.yaxis.set_ticks_position('right')
    cax.yaxis.set_label_position('right')

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Реальные источники (шары)',
               markerfacecolor='white', markeredgecolor='black', markersize=10, linestyle='None'),
        Line2D([0], [0], marker='s', color='w', label='Выделенные аномалии (полигоны)',
               markerfacecolor='none', markeredgecolor='red', markersize=10, linestyle='-', linewidth=2),
        Line2D([0], [0], marker='s', color='w', label='Аппроксимация эллипсом',
               markerfacecolor='none', markeredgecolor='blue', markersize=10, linestyle='--', linewidth=2),
        Line2D([0], [0], marker='o', color='w', label='Контуры реальных источников',
               markerfacecolor='none', markeredgecolor='white', markersize=10, linestyle=':', linewidth=1.5)
    ]
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.15),
              ncol=2, fontsize=9, frameon=True, fancybox=True, shadow=True)

    plt.suptitle('Худший результат выделения гравитационных аномалий', fontsize=14, fontweight='bold', y=0.98)
    plt.subplots_adjust(bottom=0.15)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_figure(fig, f"worst_result_anomalies_{timestamp}.png", output_dir)

    return fig


def plot_parameter_dependencies(results: List, polynomial_orders: List[int],
                                averaging_k_values: List[float],
                                n_sigma_values: List[float],
                                min_area_values: List[int],
                                output_dir: str = "results") -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax = axes[0, 0]
    poly_scores = {p: [] for p in polynomial_orders}
    for r in results:
        poly = r['params']['polynomial_order']
        poly_scores[poly].append(r['evaluation']['score'])
    poly_means = [np.mean(poly_scores[p]) for p in polynomial_orders]
    poly_stds = [np.std(poly_scores[p]) for p in polynomial_orders]
    ax.bar(range(len(polynomial_orders)), poly_means, yerr=poly_stds,
           capsize=5, color='steelblue', alpha=0.7)
    ax.set_xticks(range(len(polynomial_orders)))
    ax.set_xticklabels(polynomial_orders)
    ax.set_xlabel('Порядок полинома')
    ax.set_ylabel('Средняя оценка качества')
    ax.set_title('Зависимость качества от порядка полинома')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ns_matrix = np.zeros((len(n_sigma_values), len(min_area_values)))
    for i, ns in enumerate(n_sigma_values):
        for j, ma in enumerate(min_area_values):
            scores = [r['evaluation']['score'] for r in results
                      if r['params']['n_sigma'] == ns and r['params']['min_area_pixels'] == ma]
            ns_matrix[i, j] = np.mean(scores) if scores else 0
    im = ax.imshow(ns_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1,
                   extent=[min(min_area_values), max(min_area_values),
                           len(n_sigma_values), 0])
    ax.set_xlabel('min_area_pixels')
    ax.set_ylabel('n_sigma')
    ax.set_title('Оценка качества: nσ vs min_area')
    ax.set_yticks(range(1, len(n_sigma_values) + 1))
    ax.set_yticklabels(n_sigma_values)
    plt.colorbar(im, ax=ax, label='Оценка')

    ax = axes[1, 0]
    k_scores = {}
    for r in results:
        k = r['params']['averaging_k']
        if k not in k_scores:
            k_scores[k] = []
        k_scores[k].append(r['evaluation']['score'])
    k_sorted = sorted(k_scores.keys())
    k_means = [np.mean(k_scores[k]) for k in k_sorted]
    k_stds = [np.std(k_scores[k]) for k in k_sorted]
    ax.errorbar(range(len(k_sorted)), k_means, yerr=k_stds,
                fmt='o-', capsize=5, color='coral', linewidth=2, markersize=8)
    ax.set_xticks(range(len(k_sorted)))
    ax.set_xticklabels([f'{k:.4f}' for k in k_sorted], rotation=45, fontsize=8)
    ax.set_xlabel('k (размер окна фильтра)')
    ax.set_ylabel('Средняя оценка качества')
    ax.set_title('Зависимость качества от k')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    poly_k_matrix = np.zeros((len(polynomial_orders), len(k_sorted)))
    for i, poly in enumerate(polynomial_orders):
        for j, k in enumerate(k_sorted):
            scores = [r['evaluation']['score'] for r in results
                      if r['params']['polynomial_order'] == poly and r['params']['averaging_k'] == k]
            poly_k_matrix[i, j] = np.mean(scores) if scores else 0
    im2 = ax.imshow(poly_k_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1,
                    extent=[0, len(k_sorted), len(polynomial_orders), 0])
    ax.set_xlabel('k (индекс)')
    ax.set_ylabel('Порядок полинома')
    ax.set_title('Оценка качества: полином vs k')
    ax.set_yticks(range(1, len(polynomial_orders) + 1))
    ax.set_yticklabels(polynomial_orders)
    ax.set_xticks(range(len(k_sorted)))
    ax.set_xticklabels([f'{k:.3f}' for k in k_sorted], rotation=45, fontsize=7)
    plt.colorbar(im2, ax=ax, label='Оценка')

    plt.suptitle('Зависимость качества выделения аномалий от параметров', fontsize=14)
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_figure(fig, f"parameter_dependencies_{timestamp}.png", output_dir)

    return fig


# ==================== ОСНОВНАЯ ПРОГРАММА ====================

def print_optimization_status():
    print("\n" + "=" * 60)
    print("СТАТУС ОПТИМИЗАЦИЙ")
    print("=" * 60)
    status = [
        ("Предвычисление фильтров", OPTIMIZATION_CONFIG['use_precomputed_filters']),
        ("Numba JIT", OPTIMIZATION_CONFIG['use_numba'] and NUMBA_AVAILABLE),
        ("CuPy GPU", OPTIMIZATION_CONFIG['use_cupy'] and CUDA_AVAILABLE),
        ("Multiprocessing", OPTIMIZATION_CONFIG['use_multiprocessing']),
        ("RANSAC Subsampling", OPTIMIZATION_CONFIG['use_ransac_subsampling']),
        ("Uniform Filter", OPTIMIZATION_CONFIG['use_uniform_filter']),
    ]
    for name, enabled in status:
        status_str = "✓ ВКЛ" if enabled else "✗ ВЫКЛ"
        print(f"  • {name:<25} : {status_str}")
    if OPTIMIZATION_CONFIG['use_ransac_subsampling']:
        ratio = OPTIMIZATION_CONFIG['ransac_subsample_ratio']
        print(f"    └─ Доля точек: {ratio * 100:.1f}%")
    if OPTIMIZATION_CONFIG['use_multiprocessing']:
        n_cores = OPTIMIZATION_CONFIG['multiprocessing_cores']
        if n_cores == -1:
            n_cores = max(1, cpu_count() - 1)
        print(f"    └─ Ядер: {n_cores}")
    print("=" * 60)


def main():
    start_time = time.time()

    print("=" * 60)
    print("КОНФИГУРИРУЕМОЕ ВЫДЕЛЕНИЕ АНОМАЛИЙ")
    print("=" * 60)

    print_optimization_status()

    output_dir = "anomaly_detection_fast_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    measurement_level = 500
    grid_size = 250

    print("\n1. Генерация исходных данных...")
    data_start = time.time()
    X, Y, field, sources, z_level = create_gravitational_field_map(
        n_sources=12,
        grid_size=grid_size,
        z_level=measurement_level,
        smoothing_sigma=2.0,
        radius_range=(80, 400),
        density_range=(0.05, 0.25)
    )
    print(f"   Время: {time.time() - data_start:.1f} сек")
    print(f"   Размер: {grid_size}×{grid_size}")
    print(f"   Источников: {len(sources)}")

    polynomial_orders = [1, 2, 3, 4]
    averaging_k_values = generate_logspace_k_values(0.008, 0.25, 12)
    n_sigma_values = [1.0, 1.2, 1.5, 1.8, 2.0]
    min_area_values = [5, 10, 15, 20, 25]

    print(f"\n2. Параметры оптимизации:")
    print(f"   polynomial_orders: {polynomial_orders}")
    print(f"   averaging_k_values: {[f'{k:.5f}' for k in averaging_k_values]}")
    print(f"   n_sigma_values: {n_sigma_values}")
    print(f"   min_area_values: {min_area_values}")

    total_combinations = len(polynomial_orders) * len(averaging_k_values) * \
                         len(n_sigma_values) * len(min_area_values)
    print(f"   Всего комбинаций: {total_combinations}")

    print("\n3. Предвычисление фильтров...")
    precomputed_filters = PrecomputedFilters(field, grid_size, averaging_k_values)

    print("\n4. Оптимизация параметров...")
    opt_start = time.time()

    results = optimize_parameters_parallel(
        field, X, Y, sources, grid_size,
        polynomial_orders, averaging_k_values,
        n_sigma_values, min_area_values,
        precomputed_filters=precomputed_filters,
        verbose=True
    )

    opt_time = time.time() - opt_start
    print(f"\n   Время оптимизации: {opt_time:.1f} сек")
    print(f"   Скорость: {total_combinations / opt_time:.1f} комбинаций/сек")

    if not results:
        print("Ошибка: не получено результатов оптимизации!")
        return

    # Вывод статистики
    print("\n" + "=" * 60)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("=" * 60)

    scores = print_statistics(results)

    print("\n5. Визуализация распределения оценок...")
    plot_score_distribution(scores, output_dir)

    # Получаем лучший и худший результаты
    best = results[0]
    worst = results[-1]

    print("\n6. Визуализация лучшего результата...")
    plot_final_results_part1a(field, X, Y, sources, z_level, best['params'], best['residual'], best['evaluation'],
                              output_dir)
    plot_final_results_part1b(field, X, Y, sources, z_level, best['params'], best['residual'], best['evaluation'],
                              output_dir)
    plot_final_results_part2(field, X, Y, sources, z_level, best['params'], best['residual'], best['evaluation'],
                             output_dir)

    print("\n7. Визуализация худшего результата...")
    plot_worst_result(field, X, Y, sources, z_level, worst['params'], worst['residual'], worst['evaluation'],
                      output_dir)

    print("\n8. Визуализация зависимостей параметров...")
    plot_parameter_dependencies(results, polynomial_orders, averaging_k_values,
                                n_sigma_values, min_area_values, output_dir)

    # Сохранение JSON
    print("\n9. Сохранение результатов...")
    summary = {
        'timestamp': datetime.now().isoformat(),
        'optimization_config': OPTIMIZATION_CONFIG.copy(),
        'optimizations_available': {
            'numba': NUMBA_AVAILABLE,
            'cupy': CUDA_AVAILABLE,
            'cpu_cores': cpu_count()
        },
        'params_used': {
            'polynomial_orders': polynomial_orders,
            'averaging_k_values': averaging_k_values,
            'n_sigma_values': n_sigma_values,
            'min_area_values': min_area_values
        },
        'statistics': {
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores)),
            'mean_score': float(np.mean(scores)),
            'median_score': float(np.median(scores)),
            'std_score': float(np.std(scores)),
            'total_combinations': total_combinations
        },
        'best_params': best['params'],
        'best_score': float(best['evaluation']['score']),
        'best_n_detected': best['evaluation']['n_detected'],
        'best_n_sources': best['evaluation']['n_sources'],
        'worst_params': worst['params'],
        'worst_score': float(worst['evaluation']['score']),
        'worst_n_detected': worst['evaluation']['n_detected'],
        'optimization_time_sec': opt_time,
        'total_execution_time_sec': time.time() - start_time
    }

    json_path = os.path.join(output_dir, 'results_configurable.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"   Сохранено: {json_path}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("ИТОГОВЫЙ ОТЧЁТ")
    print("=" * 60)
    print(f"\n✓ Общее время выполнения: {elapsed:.1f} секунд")
    print(f"✓ Время оптимизации: {opt_time:.1f} секунд")
    print(f"✓ Обработано комбинаций: {total_combinations}")
    print(f"✓ Скорость: {total_combinations / opt_time:.1f} комбинаций/сек")

    print(f"\n✓ ЛУЧШИЙ результат:")
    print(f"   • Полином {best['params']['polynomial_order']}-го порядка")
    print(f"   • k = {best['params']['averaging_k']:.6f}")
    print(f"   • nσ = {best['params']['n_sigma']}")
    print(f"   • min_area = {best['params']['min_area_pixels']}")
    print(f"   • Обнаружено: {best['evaluation']['n_detected']} из {best['evaluation']['n_sources']}")
    print(f"   • Оценка: {best['evaluation']['score']:.4f}")
    print(
        f"   • QN = {best['evaluation']['num_score']:.4f}, QC = {best['evaluation']['center_score']:.4f}, QA = {best['evaluation']['area_score']:.4f}")

    print(f"\n✓ ХУДШИЙ результат:")
    print(f"   • Полином {worst['params']['polynomial_order']}-го порядка")
    print(f"   • k = {worst['params']['averaging_k']:.6f}")
    print(f"   • nσ = {worst['params']['n_sigma']}")
    print(f"   • min_area = {worst['params']['min_area_pixels']}")
    print(f"   • Обнаружено: {worst['evaluation']['n_detected']} из {worst['evaluation']['n_sources']}")
    print(f"   • Оценка: {worst['evaluation']['score']:.4f}")
    print(
        f"   • QN = {worst['evaluation']['num_score']:.4f}, QC = {worst['evaluation']['center_score']:.4f}, QA = {worst['evaluation']['area_score']:.4f}")

    print(f"\n✓ Статистика по всем {total_combinations} комбинациям:")
    print(f"   • Минимум: {np.min(scores):.4f}")
    print(f"   • Максимум: {np.max(scores):.4f}")
    print(f"   • Среднее: {np.mean(scores):.4f}")
    print(f"   • Медиана: {np.median(scores):.4f}")

    print(f"\n✓ Созданы визуализации:")
    print(f"   • score_distribution_*.png - гистограмма и круговая диаграмма")
    print(f"   • final_results_part1a_original_*.png - исходное поле")
    print(f"   • final_results_part1b_detrended_*.png - поле после снятия тренда")
    print(f"   • final_results_part2_anomalies_*.png - выделенные аномалии (лучший результат)")
    print(f"   • worst_result_anomalies_*.png - выделенные аномалии (худший результат)")
    print(f"   • parameter_dependencies_*.png - зависимости от параметров")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()
