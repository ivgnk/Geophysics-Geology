"""
Программа для генерации синтетических данных гравиметрии
и фильтрации шумов, вызванных колебаниями льда и дрейфом прибора.

Методы:
- Добавление реалистичных шумов (вибрации льда, белый шум, дрейф)
- Фильтрация: High-Pass (удаление низкочастотных колебаний льда) + Low-Pass (сглаживание)
- Удаление полиномиального тренда (компенсация дрейфа)

Автор: Бот (на основе алгоритмов гравиразведки на льду)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from numpy.polynomial import Polynomial

# ------------------------------------------------------------
# 1. Генерация полезного сигнала гравитационного поля
# ------------------------------------------------------------

# Параметры съемки: время = расстояние (предполагаем профиль длиной 500 м, шаг 1 м)
x = np.linspace(0, 500, 501)  # координаты в метрах
fs = 1.0  # частота дискретизации (1 измерение на метр) = 1 Гц

# Полезный гравитационный сигнал (аномалия Буге в мкГал)
signal_clean = np.zeros_like(x)

# 1. Региональный линейный тренд (фон регионального поля)
regional_trend = 0.02 * x  # мкГал/м

# 2. Локальная аномалия от малого тела (соляной купол или рудное тело) - функция Гаусса
anomaly1 = 40 * np.exp(-((x - 150) ** 2) / (2 * 30 ** 2))

# 3. Вторая аномалия (более широкая)
anomaly2 = 25 * np.exp(-((x - 350) ** 2) / (2 * 50 ** 2))

# Суммируем полезный сигнал
useful_signal = regional_trend + anomaly1 + anomaly2

# ------------------------------------------------------------
# 2. Добавление шумов, характерных для гравиразведки на льду
# ------------------------------------------------------------

np.random.seed(42)  # для воспроизводимости

# 2.1. Шум колебаний льда (основной паразитный сигнал)
# Лед колеблется с частотами в диапазоне 0.1–0.7 Гц (низкие упругие волны)
# Моделируем как сумму 3 затухающих синусоид
t = x  # используем координату как время (при движении 1 м/с)
ice_vibration = (
    8 * np.sin(2 * np.pi * 0.6 * t) * np.exp(-0.005 * t) +
    12 * np.sin(2 * np.pi * 0.3 * t) * np.exp(-0.002 * t) +
    5 * np.sin(2 * np.pi * 0.8 * t) * np.exp(-0.008 * t)
)

# 2.2. Дрейф гравиметра (медленный экспоненциальный + линейный)
# Из-за температурной нестабильности и ползучести кварцевой пружины
gravimeter_drift = 0.05 * x + 8 * (1 - np.exp(-0.005 * x))

# 2.3. Белый гауссов шум (инструментальный, вибрации от работы оператора)
instrument_noise = np.random.normal(0, 3, len(x))  # СКО ~3 мкГал

# 2.4. Высокочастотная помеха от микротрещин (импульсные выбросы)
spike_noise = np.zeros(len(x))
spike_idx = np.random.choice(len(x), size=15, replace=False)
spike_noise[spike_idx] = np.random.normal(15, 5, 15)

# Суммарный шум
total_noise = ice_vibration + gravimeter_drift + instrument_noise + spike_noise

# Итоговый зашумленный сигнал (то, что записал бы гравиметр)
measured_signal = useful_signal + total_noise

# ------------------------------------------------------------
# 3. Обработка: устранение шумов и колебаний льда
# ------------------------------------------------------------

# 3.1. Удаление аномальных выбросов (медианный фильтр)
# Убираем одиночные скачки от трещин
signal_despiked = signal.medfilt(measured_signal, kernel_size=5)

# 3.2. Фильтрация высоких частот (High-Pass) для удаления колебаний льда
# Частота среза 0.2 Гц (колебания льда ~0.2-0.8 Гц)
nyquist = fs / 2
cutoff_hp = 0.2  # Гц
sos_hp = signal.butter(4, cutoff_hp / nyquist, btype='highpass', output='sos')
signal_hp_filtered = signal.sosfilt(sos_hp, signal_despiked)

# 3.3. Удаление дрейфа гравиметра (низкочастотного тренда)
# Аппроксимируем полиномом 2-й степени и вычитаем
poly_degree = 2
poly_coeffs = np.polyfit(x, signal_hp_filtered, poly_degree)
trend_poly = np.polyval(poly_coeffs, x)
signal_detrended = signal_hp_filtered - trend_poly

# 3.4. Сглаживание (Low-Pass фильтр) для финальной очистки
# Оставляем частоты не выше 0.05 Гц (полезный сигнал плавный)
cutoff_lp = 0.05  # Гц
sos_lp = signal.butter(4, cutoff_lp / nyquist, btype='lowpass', output='sos')
signal_cleaned = signal.sosfilt(sos_lp, signal_detrended)

# Дополнительно применим гауссовское сглаживание
signal_cleaned = gaussian_filter1d(signal_cleaned, sigma=2.0)

# ------------------------------------------------------------
# 4. Оценка качества фильтрации
# ------------------------------------------------------------

# Стандартное отклонение ошибки на остатке
residuals = useful_signal - signal_cleaned
rmse_before = np.sqrt(np.mean((useful_signal - measured_signal) ** 2))
rmse_after = np.sqrt(np.mean(residuals ** 2))
reduction = (1 - rmse_after / rmse_before) * 100

print(f"RMSE до обработки: {rmse_before:.2f} мкГал")
print(f"RMSE после обработки: {rmse_after:.2f} мкГал")
print(f"Снижение ошибки: {reduction:.1f}%")

# ------------------------------------------------------------
# 5. Визуализация результатов
# ------------------------------------------------------------

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

# Полезный сигнал vs зашумленный
axes[0].plot(x, useful_signal, 'g-', linewidth=2, label='Полезный сигнал (истинный)')
axes[0].plot(x, measured_signal, 'r-', alpha=0.6, linewidth=1, label='Измеренный (с шумами)')
axes[0].legend()
axes[0].set_ylabel('мкГал')
axes[0].set_title('Исходные данные')
axes[0].grid(True)

# Компоненты шумов
axes[1].plot(x, ice_vibration, 'c-', label='Колебания льда')
axes[1].plot(x, gravimeter_drift, 'm--', label='Дрейф прибора')
axes[1].plot(x, instrument_noise + spike_noise, 'y-', alpha=0.5, label='Прочие шумы')
axes[1].legend()
axes[1].set_ylabel('мкГал')
axes[1].set_title('Компоненты шумов')
axes[1].grid(True)

# После удаления тренда + high-pass
axes[2].plot(x, useful_signal, 'g-', linewidth=2, label='Истинный сигнал')
axes[2].plot(x, signal_cleaned, 'b-', linewidth=1.5, label='После фильтрации')
axes[2].plot(x, residuals + np.mean(useful_signal), 'r--', alpha=0.5, label='Ошибка (смещена)')
axes[2].legend()
axes[2].set_ylabel('мкГал')
axes[2].set_title('Результат фильтрации')
axes[2].grid(True)

# Спектрограммы (амплитудные спектры) до и после
f, Pxx_before = signal.welch(measured_signal, fs, nperseg=256)
f, Pxx_after = signal.welch(signal_cleaned, fs, nperseg=256)

axes[3].semilogy(f, Pxx_before, 'r-', label='До фильтрации')
axes[3].semilogy(f, Pxx_after, 'b-', label='После фильтрации')
axes[3].axvline(cutoff_hp, color='k', linestyle='--', label='High-pass срез')
axes[3].axvline(cutoff_lp, color='k', linestyle=':', label='Low-pass срез')
axes[3].legend()
axes[3].set_xlabel('Частота (Гц)')
axes[3].set_ylabel('Спектральная плотность')
axes[3].set_title('Спектральный состав сигнала')
axes[3].grid(True)

plt.tight_layout()
plt.show()