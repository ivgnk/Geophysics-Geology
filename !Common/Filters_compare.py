"""
Сравнение фильтров: медианного и среднего
"""

# Пример для демонстрации
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.ndimage import median_filter, gaussian_filter

# Создаём тестовую аномалию (круг)
size = 100
x = np.linspace(-5, 5, size)
y = np.linspace(-5, 5, size)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
anomaly = np.exp(-R**2 / (2 * 1.5**2))  # Гауссовская аномалия

# Добавляем шум
noise = np.random.normal(0, 0.1, anomaly.shape)
field = anomaly + noise

# Применяем разные фильтры
window = 11
kernel = np.ones((window, window)) / (window**2)

averaged = convolve2d(field, kernel, mode='same')
medianed = median_filter(field, size=window)
gaussian = gaussian_filter(field, sigma=window/3)

# Визуализация
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(anomaly, cmap='viridis')
axes[0, 0].set_title('Исходная аномалия (круг)')

axes[0, 1].imshow(medianed, cmap='viridis')
axes[0, 1].set_title('Медианный фильтр\n(крестообразные артефакты!)')

axes[0, 2].imshow(averaged, cmap='viridis')
axes[0, 2].set_title('Фильтр среднего\n(округлая форма)')

axes[1, 0].imshow(gaussian, cmap='viridis')
axes[1, 0].set_title('Гауссовский фильтр\n(округлая форма)')

axes[1, 1].imshow(medianed - anomaly, cmap='coolwarm')
axes[1, 1].set_title('Медианный - исходный\n(виден крест)')

axes[1, 2].imshow(averaged - anomaly, cmap='coolwarm')
axes[1, 2].set_title('Средний - исходный\n(минимальные искажения)')

plt.tight_layout()
plt.show()
