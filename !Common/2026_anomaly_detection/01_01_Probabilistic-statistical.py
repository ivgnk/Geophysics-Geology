"""
Alice
Сделай программу на Питон, которая делает:
1) Генерация 2D данных
2) выделение площадных аномалий Вероятностно-статистическими методами
3) визуализация результатов

Шаг 1: Генерация 2D данных (generate_2d_data)
Создаёт синтетические геофизические данные с:
- случайным шумом (нормальное распределение);
- площадными аномалиями конической формы;
- настраиваемыми параметрами размера, уровня шума и количества аномалий.

Шаг 2: Выделение аномалий (detect_anomalies_statistical)
Использует вероятностно‑статистический метод:
1/ Вычисляет среднее (μ) и стандартное отклонение (σ) данных.
2/ Определяет границы аномалий по правилу μ±kσ (обычно k=3).
3/ Создаёт бинарную маску аномальных точек.
4/ Группирует соседние аномальные точки в площадные объекты.
5/ Фильтрует мелкие аномалии (менее 10 точек).

Шаг 3: Визуализация (visualize_results)
Показывает три графика:
- Исходные данные (цветовая схема terrain).
- Все аномальные точки, выявленные по правилу трёх сигм.
- Окончательные площадные аномалии после фильтрации.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


# 1. Генерация 2D данных
def generate_2d_data(size=100, noise_level=0.5, anomaly_count=5):
    """
    Генерация 2D данных с аномалиями

    Параметры:
    - size: размер сетки (size x size)
    - noise_level: уровень случайного шума
    - anomaly_count: количество аномалий
    """
    # Базовый фон — случайный шум с нормальным распределением
    background = np.random.normal(0, noise_level, (size, size))

    # Добавляем аномалии
    data = background.copy()

    for _ in range(anomaly_count):
        # Случайные параметры аномалии
        center_x = np.random.randint(10, size - 10)
        center_y = np.random.randint(10, size - 10)
        radius = np.random.randint(5, 15)
        amplitude = np.random.uniform(2, 4)  # амплитуда аномалии

        # Создаём круговую аномалию
        y_grid, x_grid = np.ogrid[:size, :size]
        distance = np.sqrt((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2)
        mask = distance <= radius

        data[mask] += amplitude * (1 - distance[mask] / radius)  # коническая форма

    return data


# 2. Выделение аномалий вероятностно-статистическим методом
def detect_anomalies_statistical(data, threshold_sigma=3):
    """
    Выделение аномалий по правилу «трёх сигм»

    Параметры:
    - data: 2D массив данных
    - threshold_sigma: порог в сигмах (обычно 3)
    """
    # Вычисляем среднее и стандартное отклонение
    mean_val = np.mean(data)
    std_val = np.std(data)

    # Определяем границы аномалий
    lower_bound = mean_val - threshold_sigma * std_val
    upper_bound = mean_val + threshold_sigma * std_val

    # Создаём маску аномалий
    anomaly_mask = (data < lower_bound) | (data > upper_bound)

    # Группируем соседние аномальные точки в площадные аномалии
    labeled_mask, num_features = ndimage.label(anomaly_mask)

    # Фильтруем малые аномалии (меньше 10 точек)
    min_size = 10
    sizes = np.bincount(labeled_mask.ravel())
    mask_sizes = sizes >= min_size
    mask_sizes[0] = False  # исключаем фон
    large_anomalies = mask_sizes[labeled_mask]

    return anomaly_mask, large_anomalies, mean_val, std_val


# 3. Визуализация результатов
def visualize_results(original_data, anomaly_mask, large_anomalies):
    """
    Визуализация исходных данных и результатов выделения аномалий
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Исходные данные
    im1 = axes[0].imshow(original_data, cmap='terrain', origin='lower')
    axes[0].set_title('Исходные геофизические данные')
    plt.colorbar(im1, ax=axes[0])

    # Все аномальные точки (по правилу 3σ)
    im2 = axes[1].imshow(anomaly_mask, cmap='hot', origin='lower')
    axes[1].set_title('Выявленные аномальные точки\n(правило трёх сигм)')
    plt.colorbar(im2, ax=axes[1])

    # Площадные аномалии после фильтрации
    im3 = axes[2].imshow(large_anomalies, cmap='Reds', origin='lower', alpha=0.7)
    axes[2].set_title('Площадные аномалии\n(после фильтрации малых объектов)')
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    plt.show()


# Основной блок программы
if __name__ == "__main__":
    # Генерация данных
    print("Генерация 2D геофизических данных...")
    geo_data = generate_2d_data(
        size=150,
        noise_level=0.8,
        anomaly_count=8
    )

    # Выделение аномалий
    print("Выделение площадных аномалий...")
    anomaly_points, area_anomalies, mean_val, std_val = detect_anomalies_statistical(
        geo_data,
        threshold_sigma=2.5
    )

    # Визуализация
    print("Визуализация результатов...")
    visualize_results(geo_data, anomaly_points, area_anomalies)

    # Дополнительная статистика
    print(f"\nСтатистика данных:")
    print(f"Среднее значение фона: {mean_val:.3f}")
    print(f"Стандартное отклонение: {std_val:.3f}")
    print(f"Порог аномалий: ±{2.5 * std_val:.3f} (2.5σ)")
    print(f"Обнаружено площадных аномалий: {np.max(ndimage.label(area_anomalies)[1])}")


