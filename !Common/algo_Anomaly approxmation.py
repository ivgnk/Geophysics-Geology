"""
Демонстрация аппроксимации аномалий: полигоны и эллипсы
Иллюстрация этапа "Аппроксимация выделенных аномалий"

Что показывает программа:
1. Слева - исходная бинарная маска аномалий (булев массив)
2. Справа - аппроксимация: выпуклые полигоны (красные) и эллипсы (синие пунктирные)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse
from scipy.ndimage import label, binary_closing, binary_opening
from scipy.spatial import ConvexHull
from skimage import measure
from typing import List, Tuple


# ==================== ФУНКЦИИ АППРОКСИМАЦИИ ====================

def make_convex_polygon(contour_points: np.ndarray, min_vertices: int = 12) -> np.ndarray:
    """
    Преобразует контур в выпуклый многоугольник с заданным минимальным количеством вершин
    """
    if len(contour_points) < 3:
        return contour_points

    # Вычисляем выпуклую оболочку
    hull = ConvexHull(contour_points)
    hull_points = contour_points[hull.vertices]

    # Если вершин меньше минимального, интерполируем
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


def fit_ellipse_simple(points: np.ndarray) -> Tuple[Tuple[float, float], float, float]:
    """
    Упрощённая аппроксимация эллипсом по выпуклому многоугольнику
    Возвращает: (центр_x, центр_y), радиус_x, радиус_y
    """
    center_x = np.mean(points[:, 0])
    center_y = np.mean(points[:, 1])

    radius_x = np.max(np.abs(points[:, 0] - center_x))
    radius_y = np.max(np.abs(points[:, 1] - center_y))

    return (center_x, center_y), radius_x, radius_y


def create_test_mask(shape: Tuple[int, int] = (200, 200)) -> np.ndarray:
    """
    Создаёт тестовую бинарную маску с несколькими аномалиями разной формы

    Возвращает:
    - mask: булев массив с аномалиями
    - centers: список центров аномалий (для информации)
    """
    mask = np.zeros(shape, dtype=bool)
    centers = []

    y, x = np.ogrid[:shape[0], :shape[1]]

    # 1. Круглая аномалия (центр)
    center1 = (100, 60)
    r1 = 25
    circle = (x - center1[1]) ** 2 + (y - center1[0]) ** 2 <= r1 ** 2
    mask[circle] = True
    centers.append(center1)

    # 2. Вытянутая аномалия (эллипс, повёрнутый)
    center2 = (150, 150)
    a2, b2 = 35, 18
    angle2 = np.radians(45)
    x_rot = (x - center2[1]) * np.cos(angle2) + (y - center2[0]) * np.sin(angle2)
    y_rot = -(x - center2[1]) * np.sin(angle2) + (y - center2[0]) * np.cos(angle2)
    ellipse = (x_rot / a2) ** 2 + (y_rot / b2) ** 2 <= 1
    mask[ellipse] = True
    centers.append(center2)

    # 3. Неправильная/вогнутая аномалия (форма «почки»)
    center3 = (50, 140)
    # Две пересекающиеся окружности
    r3a, r3b = 22, 18
    circle1 = (x - (center3[1] - 12)) ** 2 + (y - center3[0]) ** 2 <= r3a ** 2
    circle2 = (x - (center3[1] + 12)) ** 2 + (y - center3[0]) ** 2 <= r3b ** 2
    mask[circle1 | circle2] = True
    centers.append(center3)

    # 4. Кольцевая (с дыркой) — демонстрация многосвязности
    center4 = (50, 50)
    r4_outer, r4_inner = 28, 12
    outer = (x - center4[1]) ** 2 + (y - center4[0]) ** 2 <= r4_outer ** 2
    inner = (x - center4[1]) ** 2 + (y - center4[0]) ** 2 <= r4_inner ** 2
    mask[outer & ~inner] = True
    centers.append(center4)

    # 5. Маленькая аномалия (будет отфильтрована при min_area)
    center5 = (170, 50)
    r5 = 8
    small = (x - center5[1]) ** 2 + (y - center5[0]) ** 2 <= r5 ** 2
    mask[small] = True
    centers.append(center5)

    return mask, centers


def approximate_anomalies(mask: np.ndarray, min_area_pixels: int = 50) -> dict:
    """
    Аппроксимация аномалий полигонами и эллипсами

    Параметры:
    - mask: булев массив
    - min_area_pixels: минимальная площадь аномалии

    Возвращает:
    - словарь с полигонами, эллипсами, центрами и площадями
    """
    # Морфологическая обработка для сглаживания
    struct = np.ones((3, 3))
    mask_processed = binary_closing(mask, structure=struct)
    mask_processed = binary_opening(mask_processed, structure=struct)

    # Разметка связных компонент
    labeled_mask, num_features = label(mask_processed)

    polygons = []
    ellipses = []
    centers = []
    areas_pixels = []

    for label_id in range(1, num_features + 1):
        mask_label = (labeled_mask == label_id)
        area = np.sum(mask_label)

        if area < min_area_pixels:
            continue

        # Получаем контур
        contours = measure.find_contours(mask_label, 0.5)
        if not contours:
            continue

        contour = max(contours, key=len)

        # Строим выпуклый многоугольник
        hull_points = make_convex_polygon(contour, min_vertices=12)
        hull_points_closed = np.vstack([hull_points, hull_points[0]])
        polygons.append(Polygon(hull_points_closed, closed=True,
                                fill=False, edgecolor='red', linewidth=2))

        # Строим эллипс
        center, rx, ry = fit_ellipse_simple(hull_points)
        ellipses.append(Ellipse(center, 2 * rx, 2 * ry,
                                fill=False, edgecolor='blue',
                                linewidth=2, linestyle='--'))

        centers.append(center)
        areas_pixels.append(area)

    return {
        'polygons': polygons,
        'ellipses': ellipses,
        'centers': centers,
        'areas': areas_pixels,
        'num_anomalies': len(polygons)
    }


def plot_comparison(mask_original: np.ndarray, result: dict) -> plt.Figure:
    """
    Визуализация: исходная маска и результат аппроксимации
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Левая картинка: исходная бинарная маска
    ax1 = axes[0]
    ax1.imshow(mask_original, cmap='gray', origin='upper', interpolation='nearest')
    ax1.set_title('Исходная бинарная маска\n(белое — аномалии)', fontsize=12)
    ax1.set_xlabel('X (пиксели)')
    ax1.set_ylabel('Y (пиксели)')
    ax1.grid(True, alpha=0.3)

    # Правая картинка: аппроксимация полигонами и эллипсами
    ax2 = axes[1]
    ax2.imshow(mask_original, cmap='gray', alpha=0.5, origin='upper', interpolation='nearest')

    # Добавляем полигоны и эллипсы
    for polygon in result['polygons']:
        ax2.add_patch(polygon)

    for ellipse in result['ellipses']:
        ax2.add_patch(ellipse)

    ax2.set_title(f'Аппроксимация: полигоны (красные) и эллипсы (синие)\n'
                  f'Выделено аномалий: {result["num_anomalies"]}', fontsize=12)
    ax2.set_xlabel('X (пиксели)')
    ax2.set_ylabel('Y (пиксели)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return fig


def print_anomaly_info(result: dict):
    """
    Вывод информации о выделенных аномалиях
    """
    print("\n" + "=" * 50)
    print("ИНФОРМАЦИЯ О ВЫДЕЛЕННЫХ АНОМАЛИЯХ")
    print("=" * 50)

    for i, (center, area) in enumerate(zip(result['centers'], result['areas'])):
        print(f"\nАномалия {i + 1}:")
        print(f"  Центр: ({center[0]:.1f}, {center[1]:.1f}) пикселей")
        print(f"  Площадь: {area} пикселей")
        print(f"  Радиус эквивалентного круга: {np.sqrt(area / np.pi):.1f} пикселей")


# ==================== ОСНОВНАЯ ПРОГРАММА ====================

def main():
    print("=" * 50)
    print("АППРОКСИМАЦИЯ АНОМАЛИЙ: ПОЛИГОНЫ И ЭЛЛИПСЫ")
    print("=" * 50)

    # 1. Создаём тестовую бинарную маску
    print("\n1. Генерация тестовой маски...")
    mask, centers = create_test_mask((200, 200))

    total_pixels = np.sum(mask)
    print(f"   Размер маски: 200×200 пикселей")
    print(f"   Количество аномалий (исходных): {len(centers)}")
    print(f"   Общая площадь аномалий: {total_pixels} пикселей")

    # 2. Аппроксимация
    print("\n2. Аппроксимация аномалий...")
    result = approximate_anomalies(mask, min_area_pixels=30)

    print(f"   Выделено связных компонент: {result['num_anomalies']}")

    # 3. Информация об аномалиях
    print_anomaly_info(result)

    # 4. Визуализация
    print("\n3. Визуализация...")
    plot_comparison(mask, result)

    print("\n" + "=" * 50)
    print("Демонстрация завершена.")
    print("=" * 50)


if __name__ == "__main__":
    main()