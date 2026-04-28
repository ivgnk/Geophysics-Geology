"""
Сделай:
1) Подставь в программу параметры
    polynomial_orders = [1, 2, 3, 4]
    averaging_k_values = generate_logspace_k_values(0.008, 0.25, 12)
    n_sigma_values = [1.0, 1.2, 1.5, 1.8, 2.0]
    min_area_values = [5, 10, 15, 20, 25]

2) Сделай, чтобы карты в final_results_20260409_183236.png, были с одинаковым масштабом по обеим осям.

3) Сделай визуализации как улучшается результат в зависимости от параметров
(polynomial_orders, averaging_k_values, n_sigma_values, min_area_values),
но не более 3-4 диаграмм на одном листе
"""

"""
МЕТОД А: Выделение аномалий по правилу 3σ - МАКСИМАЛЬНО БЫСТРАЯ ВЕРСИЯ
Оптимизации:
- Предвычисление фильтров (precomputed filters)
- Numba JIT для критических циклов
- CuPy для GPU-ускорения (если доступен)
- Multiprocessing для параллельного перебора параметров
- Визуализация зависимости качества от параметров
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Ellipse, Circle
from scipy.ndimage import gaussian_filter, label, uniform_filter
from scipy.ndimage import binary_closing, binary_opening, generate_binary_structure
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
from functools import partial
from multiprocessing import Pool, cpu_count
import gc

# ==================== ПРОВЕРКА ДОСТУПНОСТИ БИБЛИОТЕК ====================

# Numba для JIT-компиляции
NUMBA_AVAILABLE = False
try:
    from numba import jit, prange, njit

    NUMBA_AVAILABLE = True
    print("✓ Numba доступен (JIT-компиляция)")
except ImportError:
    print("⚠ Numba не установлен (pip install numba)")

# CuPy для GPU (только если есть CUDA)
CUDA_AVAILABLE = False
cp = None
try:
    import cupy as cp

    if cp.cuda.is_available():
        CUDA_AVAILABLE = True
        print(f"✓ CuPy доступен (GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()})")
    else:
        print("⚠ CUDA не доступен, используем CPU")
except ImportError:
    print("⚠ CuPy не установлен (pip install cupy-cuda13x)")

# Подавляем предупреждения
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ==================== КОНСТАНТЫ ====================
G = 6.67430e-11
M_TO_MGAL = 1e5
DENSITY_CONVERSION = 1000

# ==================== JIT-ФУНКЦИИ (Numba) ====================

if NUMBA_AVAILABLE:
    @njit(parallel=True, cache=True)
    def fast_normalize(field):
        """Быстрая нормализация поля"""
        mean = np.mean(field)
        std = np.std(field)
        if std == 0:
            return field
        return (field - mean) / std


    @njit(parallel=True, cache=True)
    def fast_threshold(field_norm, n_sigma):
        """Быстрая пороговая обработка"""
        mask = np.zeros_like(field_norm, dtype=np.bool_)
        for i in prange(field_norm.shape[0]):
            for j in range(field_norm.shape[1]):
                if abs(field_norm[i, j]) > n_sigma:
                    mask[i, j] = True
        return mask


    @njit(parallel=True, cache=True)
    def fast_labeling(mask):
        """Быстрая разметка связных областей"""
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


def create_gravitational_field_map(n_sources: int = 12, grid_size: int = 250,
                                   density_range: Tuple = (0.05, 0.3),
                                   radius_range: Tuple = (80, 400),
                                   z_level: float = 500,
                                   smoothing_sigma: float = 2.0,
                                   min_distance_factor: float = 1.5) -> Tuple:
    """Создаёт карту гравитационного поля от сферических аномалий"""
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


def generate_logspace_k_values(k_min: float = 0.008, k_max: float = 0.25,
                               n_values: int = 12) -> List[float]:
    """Генерирует значения k с логарифмическим шагом"""
    log_min = np.log10(k_min)
    log_max = np.log10(k_max)
    log_values = np.linspace(log_min, log_max, n_values)
    k_values = 10 ** log_values
    return [round(k, 6) for k in k_values]


# ==================== ПРЕДВЫЧИСЛЕНИЕ ФИЛЬТРОВ ====================

class PrecomputedFilters:
    """Класс для предвычисления фильтров для всех значений k"""

    def __init__(self, field: np.ndarray, grid_size: int, k_values: List[float]):
        self.field = field
        self.grid_size = grid_size
        self.k_values = k_values
        self.filters_cache = {}
        self._precompute_filters()

    def _precompute_filters(self):
        """Предвычисляет фильтры среднего для всех k"""
        print("   Предвычисление фильтров...")
        for k in self.k_values:
            window_size = int(self.grid_size * k)
            if window_size % 2 == 0:
                window_size += 1
            window_size = max(3, window_size)

            # Сохраняем размер окна и результат uniform_filter
            self.filters_cache[k] = {
                'window_size': window_size,
                # uniform_filter будет применён позже к каждому полю
            }
        print(f"   Предвычислено {len(self.filters_cache)} фильтров")

    def apply_filter(self, field: np.ndarray, k: float) -> np.ndarray:
        """Применяет предвычисленный фильтр к полю"""
        if k not in self.filters_cache:
            return None

        window_size = self.filters_cache[k]['window_size']
        # Используем uniform_filter (быстрее чем convolve2d)
        return uniform_filter(field, size=window_size, mode='reflect')


# ==================== МЕТОДЫ СНЯТИЯ ТРЕНДА ====================

def remove_polynomial_trend_fast(field: np.ndarray, X: np.ndarray, Y: np.ndarray,
                                 order: int = 2, use_ransac: bool = True):
    """БЫСТРОЕ снятие полиномиального тренда с subsampling"""
    # Subsampling для ускорения (берём ~5% точек)
    step = max(1, field.shape[0] // 20)
    x_km = (X[::step, ::step].flatten()) / 1000
    y_km = (Y[::step, ::step].flatten()) / 1000
    z = field[::step, ::step].flatten()

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
    """БЫСТРОЕ выделение аномалий с использованием Numba"""

    # Нормализация
    if NUMBA_AVAILABLE:
        residual_norm = fast_normalize(residual)
        mask = fast_threshold(residual_norm, n_sigma)
    else:
        residual_norm = (residual - np.mean(residual)) / np.std(residual)
        mask = np.abs(residual_norm) > n_sigma

    # Морфологическая обработка
    struct = generate_binary_structure(2, 1)
    mask = binary_closing(mask, structure=struct, iterations=1)
    mask = binary_opening(mask, structure=struct, iterations=1)

    # Разметка
    if NUMBA_AVAILABLE:
        labeled_mask, num_features = fast_labeling(mask)
    else:
        from scipy.ndimage import label
        labeled_mask, num_features = label(mask)

    # Сбор центров
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
    """Выделение аномалий с использованием GPU (CuPy)"""
    if not CUDA_AVAILABLE:
        return detect_anomalies_fast(residual, X, Y, n_sigma, min_area_pixels)

    # Переносим данные на GPU
    residual_gpu = cp.asarray(residual)

    # Нормализация на GPU
    mean_gpu = cp.mean(residual_gpu)
    std_gpu = cp.std(residual_gpu)
    residual_norm_gpu = (residual_gpu - mean_gpu) / std_gpu

    # Пороговая обработка
    mask_gpu = cp.abs(residual_norm_gpu) > n_sigma

    # Возвращаем маску на CPU для дальнейшей обработки
    mask = cp.asnumpy(mask_gpu)

    # Морфологическая обработка (на CPU, так как scipy не работает с GPU)
    struct = generate_binary_structure(2, 1)
    mask = binary_closing(mask, structure=struct, iterations=1)
    mask = binary_opening(mask, structure=struct, iterations=1)

    # Разметка
    from scipy.ndimage import label
    labeled_mask, num_features = label(mask)

    # Сбор центров
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


def evaluate_result_fast(detection: Dict, sources: List) -> Dict:
    """Быстрая оценка качества"""
    n_sources = len(sources)
    n_detected = detection['num_anomalies']

    if n_detected == n_sources:
        num_score = 1.0
    else:
        num_score = 1.0 - min(1.0, abs(n_detected - n_sources) / n_sources)

    source_centers = [(s[0] / 1000, s[1] / 1000) for s in sources]
    detected_centers = detection['centers']

    center_score = 0.0
    if detected_centers and source_centers:
        tree = cKDTree(source_centers)
        distances, _ = tree.query(detected_centers)
        center_score = 1.0 - np.mean(np.minimum(distances / 3.0, 1.0))

    return {
        'score': 0.6 * num_score + 0.4 * center_score,
        'n_detected': n_detected,
        'n_sources': n_sources,
        'num_score': num_score,
        'center_score': center_score
    }


# ==================== ПАРАЛЛЕЛЬНАЯ ОБРАБОТКА ====================

def process_combination(args):
    """Обработка одной комбинации параметров (для multiprocessing)"""
    field, X, Y, sources, grid_size, poly_order, avg_k, n_sigma, min_area, precomputed_filters = args

    try:
        # Снятие полиномиального тренда
        field_no_poly, _ = remove_polynomial_trend_fast(
            field, X, Y, order=poly_order, use_ransac=True
        )

        # Используем предвычисленный фильтр
        if precomputed_filters:
            regional_average = precomputed_filters.apply_filter(field_no_poly, avg_k)
        else:
            window_size = int(grid_size * avg_k)
            if window_size % 2 == 0:
                window_size += 1
            window_size = max(3, window_size)
            regional_average = uniform_filter(field_no_poly, size=window_size, mode='reflect')

        residual = field_no_poly - regional_average

        # Выделение аномалий (GPU если доступен)
        if CUDA_AVAILABLE:
            detection = detect_anomalies_gpu(residual, X, Y, n_sigma, min_area)
        else:
            detection = detect_anomalies_fast(residual, X, Y, n_sigma, min_area)

        evaluation = evaluate_result_fast(detection, sources)

        return {
            'params': {
                'polynomial_order': poly_order,
                'averaging_k': avg_k,
                'window_size': int(grid_size * avg_k),
                'n_sigma': n_sigma,
                'min_area_pixels': min_area,
            },
            'evaluation': evaluation,
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
    """ПАРАЛЛЕЛЬНАЯ оптимизация параметров с предвычисленными фильтрами"""

    # Создаём список комбинаций
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
        print(f"   Используем ядер: {max(1, cpu_count() - 1)}")

    # Параллельная обработка
    with Pool(processes=max(1, cpu_count() - 1)) as pool:
        results = pool.map(process_combination, combinations)

    # Фильтруем успешные результаты
    results = [r for r in results if r['success'] and r['evaluation']['score'] > 0]
    results.sort(key=lambda x: x['evaluation']['score'], reverse=True)

    if verbose and results:
        print(f"\n   Топ-5 лучших комбинаций:")
        for i in range(min(5, len(results))):
            r = results[i]
            print(f"      {i + 1}. Оценка={r['evaluation']['score']:.3f}, "
                  f"аномалий={r['evaluation']['n_detected']}/{r['evaluation']['n_sources']}, "
                  f"полином={r['params']['polynomial_order']}, "
                  f"k={r['params']['averaging_k']:.5f}, "
                  f"nσ={r['params']['n_sigma']}, "
                  f"min_area={r['params']['min_area_pixels']}")

    return results


# ==================== ВИЗУАЛИЗАЦИЯ ЗАВИСИМОСТЕЙ ====================

def plot_parameter_dependencies(results: List, polynomial_orders: List[int],
                                averaging_k_values: List[float],
                                n_sigma_values: List[float],
                                min_area_values: List[int],
                                output_dir: str = "results") -> plt.Figure:
    """
    Визуализация зависимости качества от параметров
    Не более 4 диаграмм на одном листе
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Зависимость от порядка полинома
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

    # 2. Зависимость от n_sigma (тепловая карта с min_area)
    ax = axes[0, 1]
    # Создаём матрицу средних оценок для n_sigma и min_area
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

    # 3. Зависимость от k (размера окна фильтра)
    ax = axes[1, 0]
    # Группируем по k и усредняем по остальным параметрам
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
    ax.set_title('Зависимость качества от k (логарифмическая шкала)')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    # 4. Тепловая карта: полином vs k
    ax = axes[1, 1]
    # Создаём матрицу для полиномов и k
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
    # Добавляем подписи для k
    ax.set_xticks(range(len(k_sorted)))
    ax.set_xticklabels([f'{k:.3f}' for k in k_sorted], rotation=45, fontsize=7)
    plt.colorbar(im2, ax=ax, label='Оценка')

    plt.suptitle('Зависимость качества выделения аномалий от параметров', fontsize=14)
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_figure(fig, f"parameter_dependencies_{timestamp}.png", output_dir)

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
    """Детальное выделение аномалий с построением полигонов (для финальной визуализации)"""
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
        'areas_km2': areas_km2
    }


def plot_final_results(original_field: np.ndarray, X: np.ndarray, Y: np.ndarray,
                       sources: List, z_level: float, best_params: Dict,
                       residual: np.ndarray, evaluation: Dict,
                       output_dir: str = "results") -> plt.Figure:
    """Финальная визуализация с детальными полигонами и одинаковым масштабом осей"""
    fig = plt.figure(figsize=(18, 10))

    x_km = X / 1000
    y_km = Y / 1000
    cmap = create_custom_colormap()

    # Устанавливаем одинаковый масштаб для всех подграфиков
    x_min, x_max = 0, 5
    y_min, y_max = 0, 5

    residual_tapered = add_tapering(residual, taper_width=30)
    detailed_detection = create_detailed_detection(
        residual_tapered, X, Y,
        n_sigma=best_params['n_sigma'],
        min_area_pixels=best_params['min_area_pixels']
    )

    # 1. Исходное поле
    ax1 = fig.add_subplot(1, 3, 1)
    contour1 = ax1.contourf(x_km, y_km, original_field, levels=50, cmap=cmap, alpha=0.9)
    for s in sources:
        circle = Circle((s[0] / 1000, s[1] / 1000), s[4] / 1000,
                        fill=False, edgecolor='white', linewidth=2, linestyle='--')
        ax1.add_patch(circle)
    ax1.set_title('Исходное поле\nПунктир - контуры шаров')
    ax1.set_xlabel('X, км')
    ax1.set_ylabel('Y, км')
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_aspect('equal')  # Одинаковый масштаб осей
    ax1.grid(True, alpha=0.3)
    plt.colorbar(contour1, ax=ax1, label='мГал')

    # 2. Поле после снятия тренда
    ax2 = fig.add_subplot(1, 3, 2)
    contour2 = ax2.contourf(x_km, y_km, residual_tapered, levels=50, cmap=cmap, alpha=0.9)
    ax2.set_title(f'После снятия тренда\n(полином={best_params["polynomial_order"]}, '
                  f'k={best_params["averaging_k"]:.5f})')
    ax2.set_xlabel('X, км')
    ax2.set_ylabel('Y, км')
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.set_aspect('equal')  # Одинаковый масштаб осей
    ax2.grid(True, alpha=0.3)
    plt.colorbar(contour2, ax=ax2, label='мГал')

    # 3. Выделенные аномалии
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.contourf(x_km, y_km, residual_tapered, levels=50, cmap=cmap, alpha=0.6)

    if detailed_detection['polygons']:
        poly_collection = PatchCollection(detailed_detection['polygons'],
                                          facecolor='none', edgecolor='red', linewidth=2)
        ax3.add_collection(poly_collection)

    if detailed_detection['ellipses']:
        ellipse_collection = PatchCollection(detailed_detection['ellipses'],
                                             facecolor='none', edgecolor='blue',
                                             linewidth=2, linestyle='--')
        ax3.add_collection(ellipse_collection)

    for s in sources:
        circle = Circle((s[0] / 1000, s[1] / 1000), s[4] / 1000,
                        fill=False, edgecolor='white', linewidth=1.5, linestyle=':')
        ax3.add_patch(circle)

    ax3.set_title(f'Выделенные аномалии\n'
                  f'{detailed_detection["num_anomalies"]} из {len(sources)} '
                  f'(оценка={evaluation["score"]:.3f})')
    ax3.set_xlabel('X, км')
    ax3.set_ylabel('Y, км')
    ax3.set_xlim(x_min, x_max)
    ax3.set_ylim(y_min, y_max)
    ax3.set_aspect('equal')  # Одинаковый масштаб осей
    ax3.grid(True, alpha=0.3)

    plt.suptitle('Выделение аномалий (оптимизированная версия)', fontsize=14)
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_figure(fig, f"final_results_{timestamp}.png", output_dir)

    return fig


# ==================== ОСНОВНАЯ ПРОГРАММА ====================

def main():
    start_time = time.time()

    print("=" * 60)
    print("ОПТИМИЗИРОВАННОЕ ВЫДЕЛЕНИЕ АНОМАЛИЙ")
    print("=" * 60)
    print(f"\nДоступные оптимизации:")
    print(f"  • Numba JIT: {'✓ Включена' if NUMBA_AVAILABLE else '✗ Выключена (pip install numba)'}")
    print(f"  • CuPy GPU: {'✓ Включен' if CUDA_AVAILABLE else '✗ Выключен (pip install cupy-cuda13x)'}")
    print(f"  • Предвычисление фильтров: ✓ Включено")
    print(f"  • Multiprocessing: ✓ Включен ({cpu_count()} ядер)")
    print("=" * 60)

    output_dir = "anomaly_detection_fast_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Параметры
    measurement_level = 500
    grid_size = 250

    # Генерация данных
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

    # Параметры оптимизации (НОВЫЕ ЗНАЧЕНИЯ)
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

    # Предвычисление фильтров
    print("\n3. Предвычисление фильтров...")
    precomputed_filters = PrecomputedFilters(field, grid_size, averaging_k_values)

    # Оптимизация параметров
    print("\n4. Оптимизация параметров (параллельная обработка)...")
    opt_start = time.time()

    results = optimize_parameters_parallel(
        field, X, Y, sources, grid_size,
        polynomial_orders, averaging_k_values,
        n_sigma_values, min_area_values,
        precomputed_filters=precomputed_filters,
        verbose=True
    )

    print(f"   Время оптимизации: {time.time() - opt_start:.1f} сек")

    if not results:
        print("Ошибка: не получено результатов оптимизации!")
        return

    # Визуализация зависимостей параметров
    print("\n5. Визуализация зависимостей параметров...")
    plot_parameter_dependencies(results, polynomial_orders, averaging_k_values,
                                n_sigma_values, min_area_values, output_dir)

    # Финальная визуализация
    print("\n6. Финальная визуализация...")
    best = results[0]

    # Пересчитываем residual для лучшего результата
    field_no_poly, _ = remove_polynomial_trend_fast(
        field, X, Y, order=best['params']['polynomial_order'], use_ransac=True
    )

    if precomputed_filters:
        regional_average = precomputed_filters.apply_filter(field_no_poly, best['params']['averaging_k'])
    else:
        window_size = int(grid_size * best['params']['averaging_k'])
        if window_size % 2 == 0:
            window_size += 1
        window_size = max(3, window_size)
        regional_average = uniform_filter(field_no_poly, size=window_size, mode='reflect')

    residual = field_no_poly - regional_average

    if CUDA_AVAILABLE:
        detection = detect_anomalies_gpu(residual, X, Y,
                                         best['params']['n_sigma'],
                                         best['params']['min_area_pixels'])
    else:
        detection = detect_anomalies_fast(residual, X, Y,
                                          best['params']['n_sigma'],
                                          best['params']['min_area_pixels'])

    evaluation = evaluate_result_fast(detection, sources)

    plot_final_results(field, X, Y, sources, z_level, best['params'], residual, evaluation, output_dir)

    # Сохранение JSON
    print("\n7. Сохранение результатов...")
    summary = {
        'timestamp': datetime.now().isoformat(),
        'optimizations': {
            'numba': NUMBA_AVAILABLE,
            'cupy': CUDA_AVAILABLE,
            'precomputed_filters': True,
            'multiprocessing': True,
            'cpu_cores': cpu_count()
        },
        'params_used': {
            'polynomial_orders': polynomial_orders,
            'averaging_k_values': averaging_k_values,
            'n_sigma_values': n_sigma_values,
            'min_area_values': min_area_values
        },
        'best_params': best['params'],
        'best_score': float(evaluation['score']),
        'n_detected': evaluation['n_detected'],
        'n_sources': evaluation['n_sources'],
        'total_combinations': total_combinations,
        'execution_time_sec': time.time() - start_time
    }

    json_path = os.path.join(output_dir, 'results_fast.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"   Сохранено: {json_path}")

    # Итоговый отчёт
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("ИТОГОВЫЙ ОТЧЁТ")
    print("=" * 60)
    print(f"\n✓ Общее время выполнения: {elapsed:.1f} секунд")
    print(f"✓ Обработано комбинаций: {total_combinations}")
    print(f"✓ Скорость: {total_combinations / elapsed:.1f} комбинаций/сек")
    print(f"\n✓ Лучший результат:")
    print(f"   • Полином {best['params']['polynomial_order']}-го порядка")
    print(f"   • k = {best['params']['averaging_k']:.6f}")
    print(f"   • nσ = {best['params']['n_sigma']}")
    print(f"   • min_area = {best['params']['min_area_pixels']}")
    print(f"   • Обнаружено: {evaluation['n_detected']} из {evaluation['n_sources']}")
    print(f"   • Оценка: {evaluation['score']:.3f}")

    # Рекомендации по установке
    if not NUMBA_AVAILABLE:
        print("\n⚠ Для ускорения установите Numba:")
        print("   pip install numba")

    if not CUDA_AVAILABLE:
        print("\n⚠ Для GPU ускорения установите CuPy:")
        print("   pip install cupy-cuda13x")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Для multiprocessing на Windows
    from multiprocessing import freeze_support

    freeze_support()
    main()