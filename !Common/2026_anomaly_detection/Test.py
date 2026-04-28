# import cupy as cp
# # Создайте простой массив на GPU для проверки
# x = cp.array([1, 2, 3])
# print(x)

import cupy as cp

# Создадим простой массив на GPU
x = cp.array([1, 2, 3])
print("Массив на GPU:", x)

# Выполним простое вычисление (например, умножение на 2)
y = x * 2
print("Результат вычисления на GPU:", y)

# Переместим результат обратно в память CPU (NumPy)
y_cpu = cp.asnumpy(y)
print("Результат в памяти CPU:", y_cpu)