"""
Alice
Программа на Питон, строящая карту потенциальной функции с несколькими источниками (15-20)

Шаг 1: Создание карты потенциала (create_potential_map)
Функция создаёт карту потенциальной функции с заданным количеством источников:
- генерирует равномерную сетку координат X и Y;
- создаёт случайные позиции источников в пределах области [−8,8] по обеим осям;
- задаёт случайную силу каждого источника в диапазоне strength_range;
- вычисляет потенциал в каждой точке сетки как сумму вкладов всех источников:
- обрабатывает случай r=0, заменяя нулевые расстояния на малое значение (10−10 ) для избежания деления на ноль.

Шаг 2: Визуализация (plot_potential_map)
Программа создаёт два графика:
- Цветовая карта потенциала (contourf): отображает распределение потенциала с цветовой градацией.
Источники отмечены чёрными точками.
- Линии уровня (contour): показывает изолинии потенциала с подписями значений.
Источники выделены красными точками с чёрной окантовкой.
- Используется пользовательская цветовая палитра от синего (низкие значения) до красного (высокие значения).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def create_potential_map(n_sources=18, grid_size=200, strength_range=(1, 3)):
    """
    Создаёт карту потенциальной функции с несколькими точечными источниками.

    Параметры:
    - n_sources: количество источников (по умолчанию 18)
    - grid_size: размер сетки (grid_size x grid_size)
    - strength_range: диапазон силы источников
    """

    # Создаём сетку координат
    x = np.linspace(-10, 10, grid_size)
    y = np.linspace(-10, 10, grid_size)
    X, Y = np.meshgrid(x, y)

    # Инициализируем потенциальное поле нулевыми значениями
    potential = np.zeros_like(X)

    # Генерируем случайные позиции и силы источников
    np.random.seed(42)  # для воспроизводимости
    sources = []

    for _ in range(n_sources):
        # Случайные координаты источника
        x_source = np.random.uniform(-8, 8)
        y_source = np.random.uniform(-8, 8)

        # Случайная сила источника
        strength = np.random.uniform(*strength_range)

        sources.append((x_source, y_source, strength))

        # Вычисляем расстояние от каждой точки сетки до источника
        r = np.sqrt((X - x_source)**2 + (Y - y_source)**2)

        # Избегаем деления на ноль
        r[r == 0] = 1e-10

        # Добавляем вклад источника в потенциальное поле
        # Для гравитационного/электростатического потенциала: V ~ 1/r
        potential += strength / r

    return X, Y, potential, sources

def plot_potential_map(X, Y, potential, sources, title="Карта потенциальной функции"):
    """
    Визуализирует карту потенциальной функции и источники.
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Первая панель: цветовая карта потенциала
    cmap = LinearSegmentedColormap.from_list(
        "custom_cmap", ["blue", "green", "yellow", "red"]
    )

    contour = ax1.contourf(X, Y, potential, levels=50, cmap=cmap, alpha=0.8)
    cbar = plt.colorbar(contour, ax=ax1)
    cbar.set_label('Значение потенциала', rotation=270, labelpad=20)

    # Отмечаем источники
    source_x, source_y, _ = zip(*sources)
    ax1.scatter(source_x, source_y, c='black', s=50, zorder=5,
               label=f'{len(sources)} источников')

    ax1.set_title('Цветовая карта потенциала')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Вторая панель: линии уровня (изолинии)
    contour_lines = ax2.contour(X, Y, potential, levels=20, colors='black', linewidths=0.8)
    ax2.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')

    # Отмечаем источники на карте изолиний
    ax2.scatter(source_x, source_y, c='red', s=60, edgecolors='black',
                zorder=5, label=f'{len(sources)} источников')

    ax2.set_title('Линии уровня потенциала')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

    return fig

# Основной блок программы
if __name__ == "__main__":
    print("Создание карты потенциальной функции...")

    # Создаём карту с 18 источниками
    X, Y, potential, sources = create_potential_map(
        n_sources=18,
        grid_size=250,
        strength_range=(0.8, 2.5)
    )

    # Визуализируем результаты
    fig = plot_potential_map(X, Y, potential, sources)

    # Дополнительная информация о источниках
    print(f"\nИнформация о источниках:")
    print(f"Количество источников: {len(sources)}")
    strengths = [s[2] for s in sources]
    print(f"Средняя сила источников: {np.mean(strengths):.2f}")
    print(f"Диапазон сил: {min(strengths):.2f} – {max(strengths):.2f}")

    # Статистика по потенциалу
    print(f"\nСтатистика потенциала:")
    print(f"Минимум: {np.min(potential):.2f}")
    print(f"Максимум: {np.max(potential):.2f}")
    print(f"Среднее: {np.mean(potential):.2f}")


