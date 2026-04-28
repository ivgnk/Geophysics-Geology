"""
DeepSeek


"""

"""
Модуль выделения аномалий на гравитационных картах
Для использования с модельными данными от сферических источников
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.signal import convolve2d, wiener
from scipy.fft import fft2, ifft2, fftshift
from scipy.ndimage import gaussian_filter, sobel, label, binary_closing, binary_opening
from sklearn.cluster import DBSCAN
from skimage import measure, morphology, filters
from matplotlib.patches import Ellipse, Polygon
from matplotlib.path import Path
import pywt
from typing import Tuple, List, Dict, Any


class AnomalyDetector:
    """Класс для выделения аномалий на гравитационных картах"""

    def __init__(self, field: np.ndarray, X: np.ndarray, Y: np.ndarray,
                 sources: List = None, z_level: float = 500):
        """
        Инициализация детектора аномалий

        Параметры:
        - field: гравитационное поле (мГал)
        - X, Y: координатные сетки (м)
        - sources: список источников (для валидации)
        - z_level: уровень измерения
        """
        self.field = field.copy()
        self.X = X
        self.Y = Y
        self.sources = sources
        self.z_level = z_level
        self.x_km = X / 1000
        self.y_km = Y / 1000
        self.cell_size = (X[0, 1] - X[0, 0])  # размер ячейки в метрах

    # ================ МЕТОД А: 3σ (вероятностно-статистический) ================

    def detect_by_3sigma(self, n_sigma: float = 3.0, min_points: int = 3,
                         neighbor_profiles: int = 2) -> Dict[str, Any]:
        """
        Выделение аномалий по правилу "трёх сигм"

        Параметры:
        - n_sigma: количество стандартных отклонений (обычно 3)
        - min_points: минимальное количество точек подряд
        - neighbor_profiles: количество соседних профилей для проверки

        Возвращает:
        - маска аномалий и метаинформация
        """
        mean_val = np.mean(self.field)
        std_val = np.std(self.field)

        threshold_upper = mean_val + n_sigma * std_val
        threshold_lower = mean_val - n_sigma * std_val

        # Создаём бинарную маску точек, превышающих порог
        anomaly_mask = (self.field > threshold_upper) | (self.field < threshold_lower)

        # Проверка по соседним профилям (по строкам - профили по Y)
        rows, cols = anomaly_mask.shape
        filtered_mask = np.zeros_like(anomaly_mask)

        for i in range(rows - neighbor_profiles + 1):
            for j in range(cols - min_points + 1):
                # Проверяем несколько последовательных профилей
                consecutive = True
                for k in range(neighbor_profiles):
                    if not np.all(anomaly_mask[i + k, j:j + min_points]):
                        consecutive = False
                        break
                if consecutive:
                    for k in range(neighbor_profiles):
                        filtered_mask[i + k, j:j + min_points] = True

        # Морфологическое замыкание для объединения близких точек
        struct_elem = np.ones((3, 3))
        filtered_mask = binary_closing(filtered_mask, struct_elem)
        filtered_mask = binary_opening(filtered_mask, struct_elem)

        # Выделение связных областей
        labeled_mask, num_anomalies = label(filtered_mask)

        # Создаём многоугольники для каждой аномалии
        polygons = self._mask_to_polygons(labeled_mask, num_anomalies)

        return {
            'method': '3σ',
            'mask': filtered_mask,
            'num_anomalies': num_anomalies,
            'polygons': polygons,
            'thresholds': (threshold_lower, threshold_upper),
            'mean': mean_val,
            'std': std_val
        }

    # ================ МЕТОД Б: Градиентные характеристики ================

    def detect_by_gradient(self, gradient_threshold: float = 0.5,
                           min_area: float = 50000) -> Dict[str, Any]:
        """
        Выделение аномалий по градиентным характеристикам

        Параметры:
        - gradient_threshold: порог для полного градиента (в мГал/км)
        - min_area: минимальная площадь аномалии (м²)
        """
        # Вычисляем градиенты
        gy, gx = np.gradient(self.field)
        dx = self.X[0, 1] - self.X[0, 0]
        dy = self.Y[1, 0] - self.Y[0, 0]

        # Полный градиент
        total_gradient = np.sqrt((gx / dx) ** 2 + (gy / dy) ** 2)

        # Нормализуем
        total_gradient_norm = total_gradient / np.max(total_gradient)

        # Пороговая обработка
        gradient_mask = total_gradient_norm > gradient_threshold

        # Морфологическая обработка
        struct_elem = np.ones((5, 5))
        gradient_mask = binary_closing(gradient_mask, struct_elem)
        gradient_mask = binary_opening(gradient_mask, struct_elem)

        # Выделение областей
        labeled_mask, num_anomalies = label(gradient_mask)

        # Фильтрация по площади
        filtered_mask = np.zeros_like(gradient_mask)
        min_pixels = min_area / (self.cell_size ** 2)

        for i in range(1, num_anomalies + 1):
            area = np.sum(labeled_mask == i)
            if area >= min_pixels:
                filtered_mask[labeled_mask == i] = True

        labeled_mask, num_anomalies = label(filtered_mask)
        polygons = self._mask_to_polygons(labeled_mask, num_anomalies)

        # Направление градиента (угол)
        gradient_direction = np.arctan2(gy, gx)

        return {
            'method': 'gradient',
            'mask': filtered_mask,
            'total_gradient': total_gradient,
            'gradient_direction': gradient_direction,
            'num_anomalies': num_anomalies,
            'polygons': polygons,
            'threshold': gradient_threshold
        }

    def detect_by_analytic_signal(self, sigma: float = 1.0) -> Dict[str, Any]:
        """
        Метод аналитического сигнала (комбинация градиентов)
        """
        # Сглаживание для уменьшения шума
        smoothed = gaussian_filter(self.field, sigma=sigma)

        # Вычисляем производные
        gy, gx = np.gradient(smoothed)
        gz = gaussian_filter(self.field, sigma=sigma) - smoothed  # аппроксимация

        dx = self.X[0, 1] - self.X[0, 0]
        dy = self.Y[1, 0] - self.Y[0, 0]

        # Аналитический сигнал
        analytic_signal = np.sqrt((gx / dx) ** 2 + (gy / dy) ** 2 + gz ** 2)

        # Нормализация
        analytic_signal_norm = analytic_signal / np.max(analytic_signal)

        # Порог (автоматический по Оцу)
        threshold = filters.threshold_otsu(analytic_signal_norm)
        mask = analytic_signal_norm > threshold

        labeled_mask, num_anomalies = label(mask)
        polygons = self._mask_to_polygons(labeled_mask, num_anomalies)

        return {
            'method': 'analytic_signal',
            'mask': mask,
            'analytic_signal': analytic_signal,
            'num_anomalies': num_anomalies,
            'polygons': polygons,
            'threshold': threshold
        }

    # ================ МЕТОД В: Вейвлет-преобразование и Фурье ================

    def detect_by_wavelet(self, wavelet: str = 'db4', level: int = 3,
                          threshold_factor: float = 2.0) -> Dict[str, Any]:
        """
        Выделение аномалий с помощью вейвлет-преобразования

        Параметры:
        - wavelet: тип вейвлета ('db4', 'sym5', 'coif3')
        - level: уровень разложения
        - threshold_factor: коэффициент порога для детализирующих коэффициентов
        """
        # 2D вейвлет-разложение
        coeffs = pywt.wavedec2(self.field, wavelet, level=level)

        # Реконструкция с подавлением шума (пороговая обработка)
        threshold = threshold_factor * np.std(coeffs[-1][0])

        # Применяем мягкий порог к детализирующим коэффициентам
        coeffs_thresh = list(coeffs)
        for i in range(1, len(coeffs_thresh)):
            if isinstance(coeffs_thresh[i], tuple):
                # Детализирующие коэффициенты (cH, cV, cD)
                coeffs_thresh[i] = tuple(pywt.threshold(c, threshold, mode='soft')
                                         for c in coeffs_thresh[i])
            else:
                coeffs_thresh[i] = pywt.threshold(coeffs_thresh[i], threshold, mode='soft')

        # Реконструкция
        denoised_field = pywt.waverec2(coeffs_thresh, wavelet)

        # Обрезаем до исходного размера
        denoised_field = denoised_field[:self.field.shape[0], :self.field.shape[1]]

        # Выделяем аномалии по порогу (3σ от denoised)
        mean_den = np.mean(denoised_field)
        std_den = np.std(denoised_field)
        threshold_val = mean_den + 2.5 * std_den

        mask = denoised_field > threshold_val

        # Морфологическая обработка
        mask = binary_closing(mask, np.ones((5, 5)))
        mask = binary_opening(mask, np.ones((3, 3)))

        labeled_mask, num_anomalies = label(mask)
        polygons = self._mask_to_polygons(labeled_mask, num_anomalies)

        return {
            'method': 'wavelet',
            'mask': mask,
            'denoised_field': denoised_field,
            'num_anomalies': num_anomalies,
            'polygons': polygons,
            'wavelet': wavelet,
            'level': level
        }

    def detect_by_fourier(self, lowpass_sigma: float = 2.0,
                          highpass_sigma: float = 0.5) -> Dict[str, Any]:
        """
        Выделение аномалий с помощью Фурье-фильтрации
        """
        # 2D FFT
        F = fft2(self.field)
        fshift = fftshift(F)

        # Создаём фильтры
        rows, cols = self.field.shape
        crow, ccol = rows // 2, cols // 2

        # Низкочастотный фильтр (региональный фон)
        x = np.arange(0, rows) - crow
        y = np.arange(0, cols) - ccol
        Xg, Yg = np.meshgrid(x, y)
        R = np.sqrt(Xg ** 2 + Yg ** 2)

        lowpass = np.exp(-R ** 2 / (2 * lowpass_sigma ** 2))
        highpass = 1 - np.exp(-R ** 2 / (2 * highpass_sigma ** 2))

        # Применяем полосовой фильтр
        bandpass = lowpass * highpass

        # Фильтрация
        F_filtered = fshift * bandpass
        field_filtered = np.real(ifft2(fftshift(F_filtered)))

        # Выделение аномалий
        mean_filt = np.mean(field_filtered)
        std_filt = np.std(field_filtered)
        threshold = mean_filt + 2.5 * std_filt

        mask = field_filtered > threshold
        mask = binary_closing(mask, np.ones((5, 5)))

        labeled_mask, num_anomalies = label(mask)
        polygons = self._mask_to_polygons(labeled_mask, num_anomalies)

        return {
            'method': 'fourier',
            'mask': mask,
            'filtered_field': field_filtered,
            'num_anomalies': num_anomalies,
            'polygons': polygons,
            'bandpass_params': (lowpass_sigma, highpass_sigma)
        }

    # ================ МЕТОД Г: Статистическое зондирование ================

    def detect_by_statistical_sounding(self, window_sizes: List[int] = None,
                                       correlation_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Статистическое зондирование - оценка радиуса корреляции в скользящих окнах

        Параметры:
        - window_sizes: список размеров окон (в пикселях)
        - correlation_threshold: порог корреляции для выделения аномалий
        """
        if window_sizes is None:
            window_sizes = [10, 20, 30, 40, 50]

        rows, cols = self.field.shape
        correlation_maps = []

        for win_size in window_sizes:
            # Скользящее окно
            half_win = win_size // 2
            corr_map = np.zeros_like(self.field)

            for i in range(half_win, rows - half_win):
                for j in range(half_win, cols - half_win):
                    window = self.field[i - half_win:i + half_win, j - half_win:j + half_win]

                    # Вычисляем автокорреляцию
                    if np.std(window) > 0:
                        # Оценка радиуса корреляции
                        corr = np.correlate(window.flatten(), window.flatten(), mode='same')
                        corr_radius = np.argmax(corr < correlation_threshold * np.max(corr))
                        corr_map[i, j] = corr_radius
                    else:
                        corr_map[i, j] = 0

            correlation_maps.append(corr_map)

        # Комбинируем карты корреляции
        combined_corr = np.mean(correlation_maps, axis=0)

        # Нормализация
        combined_corr_norm = combined_corr / np.max(combined_corr)

        # Выделяем области с высокой корреляцией (аномалии)
        mask = combined_corr_norm > 0.7
        mask = binary_closing(mask, np.ones((7, 7)))

        labeled_mask, num_anomalies = label(mask)
        polygons = self._mask_to_polygons(labeled_mask, num_anomalies)

        return {
            'method': 'statistical_sounding',
            'mask': mask,
            'correlation_maps': correlation_maps,
            'combined_correlation': combined_corr,
            'num_anomalies': num_anomalies,
            'polygons': polygons,
            'window_sizes': window_sizes
        }

    # ================ МЕТОД Д: Распознавание образов (упрощённый) ================

    def detect_by_pattern_recognition(self, template_size: int = 21,
                                      similarity_threshold: float = 0.6) -> Dict[str, Any]:
        """
        Упрощённое распознавание образов с использованием эталонных форм

        Параметры:
        - template_size: размер эталонного шаблона (пикселей)
        - similarity_threshold: порог сходства
        """
        # Создаём эталонную аномалию (гауссиан)
        center = template_size // 2
        x = np.arange(-center, center + 1)
        y = np.arange(-center, center + 1)
        Xt, Yt = np.meshgrid(x, y)
        R = np.sqrt(Xt ** 2 + Yt ** 2)
        template = np.exp(-R ** 2 / (2 * (template_size / 6) ** 2))
        template = template / np.max(template)

        # Нормализуем поле
        field_norm = (self.field - np.mean(self.field)) / np.std(self.field)

        # Корреляция с шаблоном
        correlation = convolve2d(field_norm, template, mode='same')

        # Нормализуем корреляцию
        correlation_norm = correlation / np.max(np.abs(correlation))

        # Выделяем области высокой корреляции
        mask = np.abs(correlation_norm) > similarity_threshold

        # Морфологическая обработка
        mask = binary_closing(mask, np.ones((5, 5)))
        mask = binary_opening(mask, np.ones((3, 3)))

        labeled_mask, num_anomalies = label(mask)
        polygons = self._mask_to_polygons(labeled_mask, num_anomalies)

        return {
            'method': 'pattern_recognition',
            'mask': mask,
            'correlation': correlation,
            'correlation_norm': correlation_norm,
            'num_anomalies': num_anomalies,
            'polygons': polygons,
            'template': template
        }

    # ================ ДОПОЛНИТЕЛЬНЫЕ МЕТОДЫ ================

    def detect_by_residual(self, window_size: int = 21) -> Dict[str, Any]:
        """
        Метод остаточных аномалий (вычитание регионального фона)
        """
        # Региональный фон (скользящее среднее)
        kernel = np.ones((window_size, window_size)) / (window_size ** 2)
        regional_field = convolve2d(self.field, kernel, mode='same')

        # Остаточное поле
        residual_field = self.field - regional_field

        # Выделение аномалий
        mean_res = np.mean(residual_field)
        std_res = np.std(residual_field)
        threshold = mean_res + 2.5 * std_res

        mask = np.abs(residual_field) > threshold
        mask = binary_closing(mask, np.ones((5, 5)))

        labeled_mask, num_anomalies = label(mask)
        polygons = self._mask_to_polygons(labeled_mask, num_anomalies)

        return {
            'method': 'residual',
            'mask': mask,
            'regional_field': regional_field,
            'residual_field': residual_field,
            'num_anomalies': num_anomalies,
            'polygons': polygons
        }

    def detect_by_euler(self, structural_index: float = 3.0,
                        window_size: int = 10) -> Dict[str, Any]:
        """
        Метод Эйлера для оценки глубины и положения источников

        Параметры:
        - structural_index: структурный индекс (3 для сферы)
        - window_size: размер окна для решения
        """
        # Вычисляем производные
        gy, gx = np.gradient(self.field)
        dx = self.X[0, 1] - self.X[0, 0]
        dy = self.Y[1, 0] - self.Y[0, 0]

        gx = gx / dx
        gy = gy / dy

        # Решаем уравнение Эйлера в скользящем окне
        rows, cols = self.field.shape
        half_win = window_size // 2

        x0_map = np.zeros_like(self.field)
        y0_map = np.zeros_like(self.field)
        z0_map = np.zeros_like(self.field)

        for i in range(half_win, rows - half_win):
            for j in range(half_win, cols - half_win):
                # Берём окно
                win_x = self.X[i - half_win:i + half_win, j - half_win:j + half_win].flatten()
                win_y = self.Y[i - half_win:i + half_win, j - half_win:j + half_win].flatten()
                win_g = self.field[i - half_win:i + half_win, j - half_win:j + half_win].flatten()
                win_gx = gx[i - half_win:i + half_win, j - half_win:j + half_win].flatten()
                win_gy = gy[i - half_win:i + half_win, j - half_win:j + half_win].flatten()

                # Составляем систему уравнений
                A = np.column_stack([win_gx, win_gy, np.ones_like(win_g)])
                b = structural_index * win_g + win_x * win_gx + win_y * win_gy

                try:
                    solution, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    x0_map[i, j] = solution[0]
                    y0_map[i, j] = solution[1]
                    z0_map[i, j] = solution[2]
                except:
                    pass

        # Выделяем кластеры решений
        points = np.column_stack([x0_map.flatten(), y0_map.flatten()])
        points = points[~np.isnan(points).any(axis=1)]

        # Кластеризация для выделения аномалий
        clustering = DBSCAN(eps=500, min_samples=5).fit(points)

        return {
            'method': 'euler',
            'solutions': points,
            'clusters': clustering.labels_,
            'num_anomalies': len(np.unique(clustering.labels_[clustering.labels_ >= 0])),
            'x0_map': x0_map,
            'y0_map': y0_map,
            'z0_map': z0_map
        }

    # ================ ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ================

    def _mask_to_polygons(self, labeled_mask: np.ndarray, num_labels: int,
                          min_vertices: int = 12) -> List[Dict]:
        """
        Преобразование бинарной маски в многоугольники (не менее 12 вершин)
        """
        polygons = []

        for label_id in range(1, num_labels + 1):
            # Получаем контур области
            mask_label = (labeled_mask == label_id)

            # Находим контуры
            contours = measure.find_contours(mask_label, 0.5)

            if not contours:
                continue

            # Берём самый большой контур
            contour = max(contours, key=len)

            # Преобразуем в координаты сетки
            x_coords = self.x_km[0, :]
            y_coords = self.y_km[:, 0]

            contour_x = np.interp(contour[:, 1], np.arange(len(x_coords)), x_coords)
            contour_y = np.interp(contour[:, 0], np.arange(len(y_coords)), y_coords)

            # Упрощаем контур, но сохраняем не менее min_vertices вершин
            if len(contour_x) > min_vertices:
                # Выбираем равномерно распределённые точки
                indices = np.linspace(0, len(contour_x) - 1, min_vertices, dtype=int)
                contour_x = contour_x[indices]
                contour_y = contour_y[indices]

            # Замыкаем многоугольник
            contour_x = np.append(contour_x, contour_x[0])
            contour_y = np.append(contour_y, contour_y[0])

            # Создаём эллипс аппроксимации
            if len(contour_x) >= 5:
                ellipse = self._fit_ellipse(contour_x[:-1], contour_y[:-1])
            else:
                ellipse = None

            polygons.append({
                'polygon': Polygon(np.column_stack([contour_x, contour_y]),
                                   closed=True, fill=False, edgecolor='red', linewidth=2),
                'vertices': (contour_x, contour_y),
                'ellipse': ellipse,
                'area_pixels': np.sum(mask_label),
                'area_km2': np.sum(mask_label) * (self.cell_size / 1000) ** 2
            })

        return polygons

    def _fit_ellipse(self, x: np.ndarray, y: np.ndarray) -> Ellipse:
        """
        Аппроксимирует точки эллипсом
        """
        # Центр
        center_x = np.mean(x)
        center_y = np.mean(y)

        # Радиусы
        r_x = np.max(np.abs(x - center_x))
        r_y = np.max(np.abs(y - center_y))

        # Угол (упрощённо)
        angle = 0

        return Ellipse((center_x, center_y), 2 * r_x, 2 * r_y, angle=angle,
                       fill=False, edgecolor='blue', linewidth=2, linestyle='--')

    def compare_methods(self, methods: List[str] = None) -> Dict[str, Any]:
        """
        Сравнение всех методов на одном поле
        """
        if methods is None:
            methods = ['3sigma', 'gradient', 'analytic_signal', 'wavelet',
                       'fourier', 'stat_sounding', 'pattern', 'residual']

        results = {}

        method_map = {
            '3sigma': self.detect_by_3sigma,
            'gradient': self.detect_by_gradient,
            'analytic_signal': self.detect_by_analytic_signal,
            'wavelet': self.detect_by_wavelet,
            'fourier': self.detect_by_fourier,
            'stat_sounding': self.detect_by_statistical_sounding,
            'pattern': self.detect_by_pattern_recognition,
            'residual': self.detect_by_residual
        }

        for method in methods:
            if method in method_map:
                try:
                    results[method] = method_map[method]()
                    print(f"✓ {method}: обнаружено {results[method]['num_anomalies']} аномалий")
                except Exception as e:
                    print(f"✗ {method}: ошибка - {str(e)}")
                    results[method] = None

        return results

    def visualize_results(self, results: Dict[str, Any],
                          figsize: Tuple[int, int] = (20, 15)):
        """
        Визуализация результатов всех методов
        """
        n_methods = len(results)
        n_cols = 3
        n_rows = (n_methods + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()

        for idx, (method_name, result) in enumerate(results.items()):
            if result is None:
                axes[idx].text(0.5, 0.5, f'{method_name}\nОшибка',
                               ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].set_title(f'{method_name}')
                continue

            ax = axes[idx]

            # Отображаем поле
            im = ax.imshow(self.field, extent=[self.x_km.min(), self.x_km.max(),
                                               self.y_km.min(), self.y_km.max()],
                           origin='lower', cmap='viridis', alpha=0.6)

            # Накладываем маску аномалий
            if 'mask' in result:
                mask_display = np.ma.masked_where(~result['mask'], result['mask'])
                ax.imshow(mask_display, extent=[self.x_km.min(), self.x_km.max(),
                                                self.y_km.min(), self.y_km.max()],
                          origin='lower', cmap='Reds', alpha=0.4)

            # Рисуем многоугольники
            if 'polygons' in result and result['polygons']:
                for poly_info in result['polygons']:
                    ax.add_patch(poly_info['polygon'])
                    if poly_info['ellipse']:
                        ax.add_patch(poly_info['ellipse'])

            # Отображаем реальные источники
            if self.sources:
                for src in self.sources:
                    x_src, y_src = src[0] / 1000, src[1] / 1000
                    circle = plt.Circle((x_src, y_src), src[4] / 1000,
                                        fill=False, edgecolor='white', linewidth=1)
                    ax.add_patch(circle)

            ax.set_title(f'{method_name}: {result.get("num_anomalies", 0)} аномалий')
            ax.set_xlabel('X, км')
            ax.set_ylabel('Y, км')
            ax.grid(True, alpha=0.3)

        # Скрываем неиспользуемые подграфики
        for idx in range(len(results), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.show()

        return fig


# ================ ПРИМЕР ИСПОЛЬЗОВАНИЯ ================

# Импортируем ваш модуль Sphere
from Sphere import create_gravitational_field_map, plot_gravitational_field
if __name__ == "__main__":


    # Генерируем тестовые данные (заглушка)
    print("Для использования необходимо подключить модуль Sphere")
    print("\nПример использования:")
    print("""
    # Генерация данных
    X, Y, field, sources, z_level = create_gravitational_field_map(
        n_sources=18, grid_size=250, z_level=800
    )

    # Создание детектора
    detector = AnomalyDetector(field, X, Y, sources, z_level)

    # Применение методов
    results_3sigma = detector.detect_by_3sigma()
    results_gradient = detector.detect_by_gradient()
    results_wavelet = detector.detect_by_wavelet()

    # Сравнение всех методов
    all_results = detector.compare_methods()

    # Визуализация
    detector.visualize_results(all_results)
    """)