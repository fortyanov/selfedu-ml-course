# Необходимо выполнить ее аппроксимацию (восстановление) на интервале [-4, 6] моделью вида:
# Вектор параметров w=[w0,w1,w2,w3]Tw=[w0​,w1​,w2​,w3​]T следует искать с помощью алгоритма стохастического градиентного спуска (SGD) с оптимизатором импульсов Нестерова:

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def func(x):
    return -0.7 * x - 0.2 * x ** 2 + 0.05 * x ** 3 - 0.2 * np.cos(3 * x) + 2

def model_a(X, w):
    return X @ w

def loss(y_pred, y_true):
    return np.mean(np.square(y_pred - y_true))

def gradient(X, y, w):
    return  (2 / X.shape[0]) * X.T @ (model_a(X, w) - y)

# Заданные значения по оси абсцисс [-4; 6] с шагом 0.1
coord_x = np.arange(-4.0, 6.0, 0.1)
coord_y = func(coord_x)  # значения функции по оси ординат
sz = len(coord_x)  # количество значений функций (точек)

# Формирование полной выборки
X_train = np.array([[1, x, x**2, x**3] for x in coord_x])
y_train = np.array(coord_y)

# Начальные параметры
eta = np.array([0.1, 0.01, 0.001, 0.0001])  # шаг обучения для каждого параметра w0, w1, w2, w3
w = np.array([0., 0., 0., 0.])              # начальные значения параметров модели
N = 500                                     # число итераций алгоритма SGD
lm = 0.02                                   # значение параметра лямбда для вычисления скользящего экспоненциального среднего
batch_size = 20                             # размер мини-батча
gamma = 0.8                                 # коэффициент гамма для вычисления импульсов Нестерова
v = np.zeros(len(w))                        # начальное значение для импульсов Нестерова

np.random.seed(0)  # генерация одинаковых последовательностей псевдослучайных чисел

Qe = loss(model_a(X_train, w), y_train)  # начальное значение среднего эмпирического риска

# Сохраняем историю параметров и потерь для визуализации
history_w = [w.copy()]
history_Qe = [Qe]
history_Qk = []

# Градиентный спуск с импульсами Нестерова
for epoch in range(N):
    # Формирование батча
    k = np.random.randint(0, sz - batch_size - 1)  # случайный выбор начального индекса для батча
    X_batch = X_train[k:k+batch_size]
    y_batch = y_train[k:k+batch_size]

    # Вычисление показателя качества на батче
    Qk = loss(model_a(X_batch, w), y_batch)
    Qe = lm * Qk + (1 - lm) * Qe  # обновление скользящего среднего

    # Коррекция весов с учетом импульса Нестерова
    v = gamma * v + (1 - gamma) * eta * gradient(X_batch, y_batch, w - gamma * v)
    w -= v

    # Сохраняем историю каждые 5 эпох (для уменьшения размера GIF)
    if epoch % 5 == 0:
        history_w.append(w.copy())
        history_Qe.append(Qe)
        history_Qk.append(Qk)

Q = loss(model_a(X_train, w), y_train)  # итоговое значение среднего эмпирического риска
print(f'w = {list(w)}\nQe = {Qe}\nQ = {Q}')


##################### Создание анимации
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# График аппроксимации
ax1.plot(coord_x, coord_y, 'b-', linewidth=2, label='Целевая функция f(x)', alpha=0.7)
line_pred, = ax1.plot([], [], 'r-', linewidth=2, label='Модель a(x)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Аппроксимация функции')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-4, 6)
ax1.set_ylim(min(coord_y) - 0.5, max(coord_y) + 0.5)

# График функции потерь
ax2.set_xlabel('Эпоха')
ax2.set_ylabel('Loss')
ax2.set_title('Сходимость функции потерь')
ax2.grid(True, alpha=0.3)
line_Qe, = ax2.plot([], [], 'g-', linewidth=2, label='Qe (скользящее среднее)')
line_Qk, = ax2.plot([], [], 'orange', linewidth=1, alpha=0.5, label='Qk (на батче)')
ax2.legend()

# Текстовая информация
info_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


def init():
    line_pred.set_data([], [])
    line_Qe.set_data([], [])
    line_Qk.set_data([], [])
    info_text.set_text('')
    return line_pred, line_Qe, line_Qk, info_text


def animate(i):
    # Аппроксимация
    y_pred = model_a(X_train, history_w[i])
    line_pred.set_data(coord_x, y_pred)

    # Графики потерь
    epochs_Qe = np.arange(i + 1) * 5
    epochs_Qk = np.arange(i) * 5

    line_Qe.set_data(epochs_Qe, history_Qe[:i + 1])
    line_Qk.set_data(epochs_Qk, history_Qk[:i])

    # Обновление границ для графика потерь
    if i > 0:
        ax2.set_xlim(0, max(epochs_Qe) + 10)
        all_losses = history_Qe[:i + 1] + history_Qk[:i]
        ax2.set_ylim(0, max(all_losses) * 1.1)

    # Текстовая информация
    info_text.set_text(f'Эпоха: {i * 5}\n'
                       f'w₀ = {history_w[i][0]:.4f}\n'
                       f'w₁ = {history_w[i][1]:.4f}\n'
                       f'w₂ = {history_w[i][2]:.4f}\n'
                       f'w₃ = {history_w[i][3]:.4f}\n'
                       f'Qe = {history_Qe[i]:.4f}')

    return line_pred, line_Qe, line_Qk, info_text


# Создание анимации
anim = FuncAnimation(fig, animate, init_func=init, frames=len(history_w),
                     interval=50, blit=True)

# Сохранение в GIF
writer = PillowWriter(fps=20)
anim.save('sgd_approximation.gif', writer=writer)
print("GIF анимация сохранена как 'sgd_approximation.gif'")

plt.show()