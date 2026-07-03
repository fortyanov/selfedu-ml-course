# Дана следующая обучающая выборка для задачи бинарной классификации
# Вам необходимо продолжить программу для вычисления вектора параметров которым описывается разделяющая линия в соответствии с выражением

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# генерация одинаковых последовательностей псевдослучайных чисел
np.random.seed(0)

data_x = [(5.3, 2.3), (5.7, 2.5), (4.0, 1.0), (5.6, 2.4), (4.5, 1.5), (5.4, 2.3), (4.8, 1.8), (4.5, 1.5), (5.1, 1.5), (6.1, 2.3), (5.1, 1.9), (4.0, 1.2), (5.2, 2.0), (3.9, 1.4), (4.2, 1.2),
          (4.7, 1.5), (4.8, 1.8), (3.6, 1.3), (4.6, 1.4), (4.5, 1.7), (3.0, 1.1), (4.3, 1.3), (4.5, 1.3), (5.5, 2.1), (3.5, 1.0), (5.6, 2.2), (4.2, 1.5), (5.8, 1.8), (5.5, 1.8), (5.7, 2.3),
          (6.4, 2.0), (5.0, 1.7), (6.7, 2.0), (4.0, 1.3), (4.4, 1.4), (4.5, 1.5), (5.6, 2.4), (5.8, 1.6), (4.6, 1.3), (4.1, 1.3), (5.1, 2.3), (5.2, 2.3), (5.6, 1.4), (5.1, 1.8), (4.9, 1.5),
          (6.7, 2.2), (4.4, 1.3), (3.9, 1.1), (6.3, 1.8), (6.0, 1.8), (4.5, 1.6), (6.6, 2.1), (4.1, 1.3), (4.5, 1.5), (6.1, 2.5), (4.1, 1.0), (4.4, 1.2), (5.4, 2.1), (5.0, 1.5), (5.0, 2.0),
          (4.9, 1.5), (5.9, 2.1), (4.3, 1.3), (4.0, 1.3), (4.9, 2.0), (4.9, 1.8), (4.0, 1.3), (5.5, 1.8), (3.7, 1.0), (6.9, 2.3), (5.7, 2.1), (5.3, 1.9), (4.4, 1.4), (5.6, 1.8), (3.3, 1.0),
          (4.8, 1.8), (6.0, 2.5), (5.9, 2.3), (4.9, 1.8), (3.3, 1.0), (3.9, 1.2), (5.6, 2.1), (5.8, 2.2), (3.8, 1.1), (3.5, 1.0), (4.5, 1.5), (5.1, 1.9), (4.7, 1.4), (5.1, 1.6), (5.1, 2.0),
          (4.8, 1.4), (5.0, 1.9), (5.1, 2.4), (4.6, 1.5), (6.1, 1.9), (4.7, 1.6), (4.7, 1.4), (4.7, 1.2), (4.2, 1.3), (4.2, 1.3)]
data_y = [1, 1, -1, 1, -1, 1, 1, -1, 1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1,
          1, -1, -1, 1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, -1]


# Вычисление среднего эмпирического риска
def QK(w, x, y):
    return 1 / x.shape[0] * (np.log2(1 + np.exp(-x @ w * y))).sum()


def gd(w, x, y):
    return - 1 / (x.shape[0] * np.log(2)) * x.T @ (y * np.exp(-x @ w * y) / (1 + np.exp(-x @ w * y)))


# данные для задачи
X_train = np.c_[np.ones(len(data_x)), data_x]
y_train = np.array(data_y)

n_train = len(X_train)  # размер обучающей выборки
w = [0.0, 0.0, 0.0]  # начальные весовые коэффициенты
nt = np.array([0.1, 0.05, 0.05])  # шаг обучения для каждого параметра w0, w1, w2
lm = 0.01  # значение параметра лямбда для вычисления скользящего экспоненциального среднего
N = 200  # число итераций алгоритма SGD
batch_size = 10  # размер мини-батча (величина K = 10)

alpha = 0.7  # параметр для RMSProp
G = np.zeros(len(w))  # параметр для RMSProp
eps = 0.01  # параметр для RMSProp

# начальное значение среднего эмпирического риска
Qe = QK(w, X_train, y_train)

# Сохраняем историю параметров и потерь для визуализации
history_w = [w.copy()]
history_Qe = [Qe]
history_Qk = []

# сам цикл спуска
for epoch in range(N):
    # выбор элементов для мини-батча из обучающего набора
    k = np.random.randint(0, n_train - batch_size - 1)
    x_batch = X_train[k: k + batch_size]
    y_batch = y_train[k: k + batch_size]

    # Вычисление ошибки на мини-батче
    Q_k = QK(w, x_batch, y_batch)

    # Вычисление градиента
    grad = gd(w, x_batch, y_batch)

    # Вычисление нормирующего множителя
    G = alpha * G + (1 - alpha) * grad ** 2

    # шаг спуска
    w = w - nt * grad / (np.sqrt(G) + eps)

    # пересчет среднего эмпирического риска
    Qe = lm * Q_k + (1 - lm) * Qe

    # Сохраняем историю каждые 2 эпохи
    if epoch % 2 == 0:
        history_w.append(w.copy())
        history_Qe.append(Qe)
        history_Qk.append(Q_k)

# Финальное вычисление Error rate
Q = 1 / n_train * (X_train @ w * y_train < 0).sum()


################ Создание визуализации
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Разделяем точки по классам
class_neg = y_train == -1
class_pos = y_train == 1

x1_neg = [data_x[i][0] for i in range(len(data_x)) if class_neg[i]]
x2_neg = [data_x[i][1] for i in range(len(data_x)) if class_neg[i]]
x1_pos = [data_x[i][0] for i in range(len(data_x)) if class_pos[i]]
x2_pos = [data_x[i][1] for i in range(len(data_x)) if class_pos[i]]

# График с точками и разделяющей линией
ax1.scatter(x1_neg, x2_neg, c='blue', marker='o', label='Класс -1', alpha=0.7, s=50)
ax1.scatter(x1_pos, x2_pos, c='red', marker='s', label='Класс +1', alpha=0.7, s=50)
line, = ax1.plot([], [], 'g-', linewidth=2, label='Разделяющая линия')
ax1.set_xlabel('Признак x₁')
ax1.set_ylabel('Признак x₂')
ax1.set_title('Бинарная классификация')
ax1.legend()
ax1.grid(True, alpha=0.3)

# График функции потерь
line_Qe, = ax2.plot([], [], 'b-', linewidth=2, label='Qe (скользящее среднее)')
line_Qk, = ax2.plot([], [], 'r-', linewidth=1, alpha=0.5, label='Qk (на батче)')
ax2.set_xlabel('Эпоха')
ax2.set_ylabel('Log Loss')
ax2.set_title('Сходимость функции потерь')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Текстовая информация
info_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


def init():
    line.set_data([], [])
    line_Qe.set_data([], [])
    line_Qk.set_data([], [])
    info_text.set_text('')
    return line, line_Qe, line_Qk, info_text


def animate(i):
    w_current = history_w[i]

    # Построение разделяющей линии: w0 + w1*x1 + w2*x2 = 0
    x1_min, x1_max = ax1.get_xlim()
    x1_vals = np.linspace(x1_min, x1_max, 100)

    if abs(w_current[2]) > 1e-10:  # Проверка, что w2 не близок к нулю
        x2_vals = -(w_current[0] + w_current[1] * x1_vals) / w_current[2]
        # Обрезаем значения для отображения в пределах графика
        y_min, y_max = ax1.get_ylim()
        mask = (x2_vals >= y_min) & (x2_vals <= y_max)
        line.set_data(x1_vals[mask], x2_vals[mask])
    else:
        line.set_data([], [])

    # Графики потерь
    epochs = np.arange(i + 1) * 2
    line_Qe.set_data(epochs, history_Qe[:i + 1])
    line_Qk.set_data(epochs[:-1] if i > 0 else [], history_Qk[:i])

    # Обновление границ для графика потерь
    if i > 0:
        ax2.set_xlim(0, max(epochs) + 10)
        all_losses = history_Qe[:i + 1] + history_Qk[:max(0, i - 1)]
        if all_losses:
            ax2.set_ylim(0, max(all_losses) * 1.1)

    # Вычисление error rate для текущих весов
    current_errors = (X_train @ w_current * y_train < 0).sum()
    current_error_rate = current_errors / n_train

    # Текстовая информация
    info_text.set_text(f'Эпоха: {i * 2}\n'
                       f'w₀ = {w_current[0]:.4f}\n'
                       f'w₁ = {w_current[1]:.4f}\n'
                       f'w₂ = {w_current[2]:.4f}\n'
                       f'Qe = {history_Qe[i]:.4f}\n'
                       f'Error rate = {current_error_rate:.4f}')

    return line, line_Qe, line_Qk, info_text


# Создание анимации
anim = FuncAnimation(fig, animate, init_func=init, frames=len(history_w),
                     interval=100, blit=True)

# Сохранение в GIF
writer = PillowWriter(fps=10)
anim.save('classification_sgd.gif', writer=writer)
print("GIF анимация сохранена как 'classification_sgd.gif'")

plt.show()