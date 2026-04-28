"""
Сделай в самом начале словарь, в котором эти параметры задаются как 0 (отключено),
1 (включено), я хочу посмотреть насколько изменяется время работы у меня на компьютере
"""

"""
МЕТОД А: Выделение аномалий по правилу 3σ - КОНФИГУРИРУЕМАЯ ВЕРСИЯ
Позволяет включать/отключать различные оптимизации для оценки их эффективности

НАСТРОЙКИ ОПТИМИЗАЦИЙ (изменить в словаре OPTIMIZATION_CONFIG):
- use_precomputed_filters: предвычисление фильтров (кэширование)
- use_numba: JIT-компиляция критических циклов
- use_cupy: GPU-ускорение (требует CUDA)
- use_multiprocessing: параллельная обработка комбинаций
- use_ransac_subsampling: разреженная выборка для RANSAC
- use_uniform_filter: быстрый uniform_filter вместо convolve2d
- ransac_subsample_ratio: доля точек для RANSAC (0.01-1.0, 0.05 = 5%)

# МИНИМАЛЬНАЯ ПРОИЗВОДИТЕЛЬНОСТЬ (без оптимизаций)
OPTIMIZATION_CONFIG = {
    # === ОСНОВНЫЕ ОПТИМИЗАЦИИ ===
    'use_precomputed_filters': 0,  # Без кэширования фильтров
    'use_numba': 0,  # Без JIT-компиляции
    'use_cupy': 0,  # Без GPU (только CPU)
    'use_multiprocessing': 0,  # Без параллельной обработки (последовательно)
    'use_ransac_subsampling': 0,  # Все точки для RANSAC (медленно)
    'use_uniform_filter': 0,  # Медленный convolve2d вместо uniform_filter

    # === ДОПОЛНИТЕЛЬНЫЕ ПАРАМЕТРЫ ===
    'ransac_subsample_ratio': 1.0,  # 100% точек (все, без subsampling)
    'numba_parallel': 0,  # Без параллельных циклов
    'multiprocessing_cores': 1,  # Только 1 ядро (или 0, но лучше 1)
}

# МАКСИМАЛЬНАЯ ПРОИЗВОДИТЕЛЬНОСТЬ
OPTIMIZATION_CONFIG = {
    # === ОСНОВНЫЕ ОПТИМИЗАЦИИ ===
    'use_precomputed_filters': 1,  # Предвычисление фильтров (кэширование)
    'use_numba': 1,  # JIT-компиляция Numba
    'use_cupy': 1,  # GPU-ускорение CuPy (требует CUDA)
    'use_multiprocessing': 1,  # Параллельная обработка (multiprocessing)
    'use_ransac_subsampling': 1,  # Разреженная выборка для RANSAC
    'use_uniform_filter': 1,  # Быстрый uniform_filter вместо convolve2d

    # === ДОПОЛНИТЕЛЬНЫЕ ПАРАМЕТРЫ ===
    'ransac_subsample_ratio': 0.05,  # 5% точек для RANSAC (быстро)
    'numba_parallel': 1,  # Параллельные циклы в Numba
    'multiprocessing_cores': -1,  # Все ядра минус 1 (макс. безопасно)
}
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
from functools import partial
from multiprocessing import Pool, cpu_count
import gc

# ==================== НАСТРОЙКИ ОПТИМИЗАЦИЙ ====================
# Измените значения для включения/отключения оптимизаций
# 0 = отключено, 1 = включено

OPTIMIZATION_CONFIG = {
    # === ОСНОВНЫЕ ОПТИМИЗАЦИИ ===
    'use_precomputed_filters': 1,  # Предвычисление фильтров (кэширование)
    'use_numba': 1,  # JIT-компиляция Numba
    'use_cupy': 1,  # GPU-ускорение CuPy (требует CUDA)
    'use_multiprocessing': 1,  # Параллельная обработка (multiprocessing)
    'use_ransac_subsampling': 1,  # Разреженная выборка для RANSAC
    'use_uniform_filter': 1,  # Быстрый uniform_filter вместо convolve2d

    # === ДОПОЛНИТЕЛЬНЫЕ ПАРАМЕТРЫ ===
    'ransac_subsample_ratio': 0.05,  # Доля точек для RANSAC (0.01-1.0, 0.05 = 5%)
    'numba_parallel': 1,  # Параллельные циклы в Numba
    'multiprocessing_cores': -1,  # -1 = все ядра минус 1, иначе число ядер
}

# ==================== ПРОВЕРКА ДОСТУПНОСТИ БИБЛИОТЕК ====================

# Numba
NUMBA_AVAILABLE = False
if OPTIMIZATION_CONFIG['use_numba']:
    try:
        from numba import jit, prange, njit

        NUMBA_AVAILABLE = True
        print("✓ Numba доступен (JIT-компиляция включена)")
    except ImportError:
        print("⚠ Numba не установлен (pip install numba) - оптимизация отключена")
        OPTIMIZATION_CONFIG['use_numba'] = 0

# CuPy
CUDA_AVAILABLE = False
cp = None
if OPTIMIZATION_CONFIG['use_cupy']:
    try:
        import cupy as cp

        if cp.cuda.is_available():
            CUDA_AVAILABLE = True
            print(f"✓ CuPy доступен (GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()})")
        else:
            print("⚠ CUDA не доступен, используем CPU")
            OPTIMIZATION_CONFIG['use_cupy'] = 0
    except ImportError:
        print("⚠ CuPy не установлен (pip install cupy-cuda13x) - оптимизация отключена")
        OPTIMIZATION_CONFIG['use_cupy'] = 0

# Подавляем предупреждения
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ==================== КОНСТАНТЫ ====================
G = 6.67430e-11
M_TO_MGAL = 1e5
DENSITY_CONVERSION = 1000


# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def calculate_subsample_step(grid_size: int, ratio: float) -> int:
    """
    Расчёт шага subsampling для RANSAC

    Параметры:
    - grid_size: размер сетки (250)
    - ratio: доля точек (0.01-1.0)

    Возвращает:
    - step: шаг выборки
    """
    if ratio >= 1.0:
        return 1  # все точки

    # Количество точек в сетке
    n_points = grid_size * grid_size
    target_points = int(n_points * ratio)

    # Вычисляем шаг (чтобы получить примерно target_points точек)
    step = int(np.sqrt(n_points / max(1, target_points)))
    step = max(1, min(step, grid_size // 2))

    return step


# ==================== JIT-ФУНКЦИИ (Numba) ====================

if NUMBA_AVAILABLE and OPTIMIZATION_CONFIG['use_numba']:
    parallel_setting = True if OPTIMIZATION_CONFIG['numba_parallel'] else False


    @njit(parallel=parallel_setting, cache=True)
    def fast_normalize(field):
        """Быстрая нормализация поля"""
        mean = np.mean(field)
        std = np.std(field)
        if std == 0:
            return field
        return (field - mean) / std


    @njit(parallel=parallel_setting, cache=True)
    def fast_threshold(field_norm, n_sigma):
        """Быстрая пороговая обработка"""
        mask = np.zeros_like(field_norm, dtype=np.bool_)
        for i in prange(field_norm.shape[0]):
            for j in range(field_norm.shape[1]):
                if abs(field_norm[i, j]) > n_sigma:
                    mask[i, j] = True
        return mask


    @njit(parallel=parallel_setting, cache=True)
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
        if OPTIMIZATION_CONFIG['use_precomputed_filters']:
            self._precompute_filters()

    def _precompute_filters(self):
        """Предвычисляет фильтры среднего для всех k"""
        print("   Предвычисление фильтров...")
        for k in self.k_values:
            window_size = int(self.grid_size * k)
            if window_size % 2 == 0:
                window_size += 1
            window_size = max(3, window_size)

            self.filters_cache[k] = {
                'window_size': window_size,
            }
        print(f"   Предвычислено {len(self.filters_cache)} фильтров")

    def apply_filter(self, field: np.ndarray, k: float) -> np.ndarray:
        """Применяет предвычисленный фильтр к полю"""
        if not OPTIMIZATION_CONFIG['use_precomputed_filters']:
            return None

        if k not in self.filters_cache:
            return None

        window_size = self.filters_cache[k]['window_size']

        if OPTIMIZATION_CONFIG['use_uniform_filter']:
            return uniform_filter(field, size=window_size, mode='reflect')
        else:
            # Медленный convolve2d для сравнения
            kernel = np.ones((window_size, window_size)) / (window_size ** 2)
            return convolve2d(field, kernel, mode='same', boundary='symm')


# ==================== МЕТОДЫ СНЯТИЯ ТРЕНДА ====================

def remove_polynomial_trend_fast(field: np.ndarray, X: np.ndarray, Y: np.ndarray,
                                 order: int = 2, use_ransac: bool = True):
    """БЫСТРОЕ снятие полиномиального тренда с опциональным subsampling"""

    grid_size = field.shape[0]

    if OPTIMIZATION_CONFIG['use_ransac_subsampling']:
        # Рассчитываем шаг subsampling на основе ratio
        ratio = OPTIMIZATION_CONFIG['ransac_subsample_ratio']
        step = calculate_subsample_step(grid_size, ratio)

        x_km = (X[::step, ::step].flatten()) / 1000
        y_km = (Y[::step, ::step].flatten()) / 1000
        z = field[::step, ::step].flatten()

        n_used = len(z)
        n_total = grid_size * grid_size
        if n_used < n_total:
            print(f"      RANSAC subsampling: {n_used}/{n_total} точек ({100 * n_used / n_total:.1f}%)")
    else:
        # Используем все точки
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
    """БЫСТРОЕ выделение аномалий с опциональным использованием Numba"""

    # Нормализация
    if OPTIMIZATION_CONFIG['use_numba'] and NUMBA_AVAILABLE:
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
    if OPTIMIZATION_CONFIG['use_numba'] and NUMBA_AVAILABLE:
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
    """Выделение аномалий с опциональным использованием GPU (CuPy)"""
    if not OPTIMIZATION_CONFIG['use_cupy'] or not CUDA_AVAILABLE:
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

        # Применяем фильтр
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

        # Выделение аномалий
        detection = detect_anomalies_gpu(residual, X, Y, n_sigma, min_area)
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
    """ПАРАЛЛЕЛЬНАЯ оптимизация параметров с опциональным multiprocessing"""

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
    """Визуализация зависимости качества от параметров"""
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

    # 3. Зависимость от k
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

    # 4. Тепловая карта: полином vs k
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
    """Детальное выделение аномалий с построением полигонов"""
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


def plot_final_results_part1a(original_field: np.ndarray, X: np.ndarray, Y: np.ndarray,
                              sources: List, z_level: float, best_params: Dict,
                              residual: np.ndarray, evaluation: Dict,
                              output_dir: str = "results") -> plt.Figure:
    """
    ЧАСТЬ 1A: Исходное поле (одна карта)
    с вертикальной цветовой шкалой (высота = высоте карты, ширина = 0.05)
    """
    fig = plt.figure(figsize=(8, 7))

    x_km = X / 1000
    y_km = Y / 1000
    cmap = create_custom_colormap()

    x_min, x_max = 0, 5
    y_min, y_max = 0, 5

    # GridSpec: [карта, цветовая шкала]
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 0.05], wspace=0.05)

    # Основной график
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

    # Цветовая шкала
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
    """
    ЧАСТЬ 1B: Поле после снятия тренда (одна карта)
    с вертикальной цветовой шкалой (высота = высоте карты, ширина = 0.05)
    """
    fig = plt.figure(figsize=(8, 7))

    x_km = X / 1000
    y_km = Y / 1000
    cmap = create_custom_colormap()

    x_min, x_max = 0, 5
    y_min, y_max = 0, 5

    residual_tapered = add_tapering(residual, taper_width=30)

    # GridSpec: [карта, цветовая шкала]
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 0.05], wspace=0.05)

    # Основной график
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

    # Цветовая шкала
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
    """
    ЧАСТЬ 2: Выделенные аномалии (одна карта)
    с вертикальной цветовой шкалой (высота = высоте карты, ширина = 0.05) и легендой
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
        n_sigma=best_params['n_sigma'],
        min_area_pixels=best_params['min_area_pixels']
    )

    # GridSpec: [карта, цветовая шкала]
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 0.05], wspace=0.05)

    # Основной график
    ax = fig.add_subplot(gs[0])
    contour = ax.contourf(x_km, y_km, residual_tapered, levels=50, cmap=cmap, alpha=0.6)

    # Выделенные аномалии (полигоны)
    if detailed_detection['polygons']:
        poly_collection = PatchCollection(detailed_detection['polygons'],
                                          facecolor='none', edgecolor='red', linewidth=2)
        ax.add_collection(poly_collection)

    # Аппроксимация эллипсами
    if detailed_detection['ellipses']:
        ellipse_collection = PatchCollection(detailed_detection['ellipses'],
                                             facecolor='none', edgecolor='blue',
                                             linewidth=2, linestyle='--')
        ax.add_collection(ellipse_collection)

    # Контуры реальных источников (пунктир)
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

    # Цветовая шкала
    cax = fig.add_subplot(gs[1])
    cbar = plt.colorbar(contour, cax=cax, orientation='vertical')
    cbar.set_label('Δg,\nмГал', fontsize=9, linespacing=1.2)
    cax.yaxis.set_ticks_position('right')
    cax.yaxis.set_label_position('right')

    # Легенда
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


# ==================== ОСНОВНАЯ ПРОГРАММА ====================

def print_optimization_status():
    """Выводит текущий статус оптимизаций"""
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

    if OPTIMIZATION_CONFIG['use_numba'] and NUMBA_AVAILABLE:
        parallel = "ВКЛ" if OPTIMIZATION_CONFIG['numba_parallel'] else "ВЫКЛ"
        print(f"    └─ Parallel: {parallel}")

    print("=" * 60)


def main():
    start_time = time.time()

    print("=" * 60)
    print("КОНФИГУРИРУЕМОЕ ВЫДЕЛЕНИЕ АНОМАЛИЙ")
    print("=" * 60)

    # Выводим статус оптимизаций
    print_optimization_status()

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

    # Параметры оптимизации
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

    # Визуализация зависимостей параметров
    print("\n5. Визуализация зависимостей параметров...")
    plot_parameter_dependencies(results, polynomial_orders, averaging_k_values,
                                n_sigma_values, min_area_values, output_dir)

    # Финальная визуализация (раздельная)
    print("\n6. Финальная визуализация...")
    best = results[0]

    # Пересчитываем residual для лучшего результата
    field_no_poly, _ = remove_polynomial_trend_fast(
        field, X, Y, order=best['params']['polynomial_order'], use_ransac=True
    )

    if precomputed_filters and OPTIMIZATION_CONFIG['use_precomputed_filters']:
        regional_average = precomputed_filters.apply_filter(field_no_poly, best['params']['averaging_k'])
    else:
        window_size = int(grid_size * best['params']['averaging_k'])
        if window_size % 2 == 0:
            window_size += 1
        window_size = max(3, window_size)

        if OPTIMIZATION_CONFIG['use_uniform_filter']:
            regional_average = uniform_filter(field_no_poly, size=window_size, mode='reflect')
        else:
            kernel = np.ones((window_size, window_size)) / (window_size ** 2)
            regional_average = convolve2d(field_no_poly, kernel, mode='same', boundary='symm')

    residual = field_no_poly - regional_average

    detection = detect_anomalies_gpu(residual, X, Y,
                                     best['params']['n_sigma'],
                                     best['params']['min_area_pixels'])
    evaluation = evaluate_result_fast(detection, sources)

    # Сохраняем ЧАСТЬ 1A (исходное поле)
    plot_final_results_part1a(field, X, Y, sources, z_level, best['params'], residual, evaluation, output_dir)

    # Сохраняем ЧАСТЬ 1B (поле после снятия тренда)
    plot_final_results_part1b(field, X, Y, sources, z_level, best['params'], residual, evaluation, output_dir)

    # Сохраняем ЧАСТЬ 2 (выделенные аномалии)
    plot_final_results_part2(field, X, Y, sources, z_level, best['params'], residual, evaluation, output_dir)

    # Сохранение JSON
    print("\n7. Сохранение результатов...")
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
        'best_params': best['params'],
        'best_score': float(evaluation['score']),
        'n_detected': evaluation['n_detected'],
        'n_sources': evaluation['n_sources'],
        'total_combinations': total_combinations,
        'optimization_time_sec': opt_time,
        'total_execution_time_sec': time.time() - start_time
    }

    json_path = os.path.join(output_dir, 'results_configurable.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"   Сохранено: {json_path}")

    # Итоговый отчёт
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("ИТОГОВЫЙ ОТЧЁТ")
    print("=" * 60)
    print(f"\n✓ Общее время выполнения: {elapsed:.1f} секунд")
    print(f"✓ Время оптимизации: {opt_time:.1f} секунд")
    print(f"✓ Обработано комбинаций: {total_combinations}")
    print(f"✓ Скорость: {total_combinations / opt_time:.1f} комбинаций/сек")
    print(f"\n✓ Лучший результат:")
    print(f"   • Полином {best['params']['polynomial_order']}-го порядка")
    print(f"   • k = {best['params']['averaging_k']:.6f}")
    print(f"   • nσ = {best['params']['n_sigma']}")
    print(f"   • min_area = {best['params']['min_area_pixels']}")
    print(f"   • Обнаружено: {evaluation['n_detected']} из {evaluation['n_sources']}")
    print(f"   • Оценка: {evaluation['score']:.3f}")

    print(f"\n✓ Созданы файлы:")
    print(f"   • final_results_part1a_original_*.png - исходное поле")
    print(f"   • final_results_part1b_detrended_*.png - поле после снятия тренда")
    print(f"   • final_results_part2_anomalies_*.png - выделенные аномалии с легендой")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()