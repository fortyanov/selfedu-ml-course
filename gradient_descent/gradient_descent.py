import time
import numpy as np
import matplotlib.pyplot as plt


def compute_min():
    """
    Находим x при котором функция принимает минимальное значение
    """

    def f(x):
        return 0.5 * x + 0.2 * x ** 2 - 0.1 * x ** 3

    def df(x):
        return 0.5 + 0.2 * 2 * x - 0.1 * 3 * x ** 2

    N = 200
    x = -4
    lmd = 0.01

    x_plt = np.arange(0, 5.0, 0.1)
    f_plt = [f(x) for x in x_plt]

    plt.ion()
    fig, ax = plt.subplots()
    ax.grid(True)

    ax.plot(x_plt, f_plt)
    point = ax.scatter(x, f(x), color='red')

    for i in range(N):
        x = x - lmd * df(x)

        point.set_offsets(np.array([x, f(x)]))

        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.02)

    plt.ioff()
    print(x)
    ax.scatter(x, f(x), color='blue')
    plt.show()


def compute_weights():
    """
    Рассчитаем веса при которых функция ошибки (между аппроксимирующей моделью и реальной функцией)
    имеет минимальное значение
    """

    # исходная функция, которую нужно аппроксимировать моделью a(x)
    def func(x):
        return 5. - np.sin(x) + 0.1 * x ** 2

    def model(w, x):
        # return w[0] + w[1] * x + w[2] * x ** 2 + w[3] * x ** 3
        return w @ x

    def delta_loss(w, x_plt):
        s = np.array([[1, x, x ** 2, x ** 3] for x in x_plt])
        fnc = np.array([func(x) for x in x_plt])
        vec = (w @ s.T - fnc) @ s

        return 2 / len(x_plt) * vec

    def quality(w, x_plt):
        s = np.array([[1, x, x ** 2, x ** 3] for x in x_plt])
        fnc = np.array([func(x) for x in x_plt])
        vec = (w @ s.T - fnc) ** 2
        sum = np.sum(vec)

        return sum / len(x_plt)

    eta = np.array([0.1, 0.01, 0.001, 0.0001])  # шаг обучения для каждого параметра w0, w1, w2, w3
    w = np.array([0., 0., 0., 0.])  # начальные значения параметров модели
    N = 200  # число итераций градиентного алгоритма

    x_plt = np.arange(-5.0, 5.0, 0.1)
    f_plt = [func(x) for x in x_plt]

    for i in range(N):
        w = w - eta * delta_loss(w=w, x_plt=x_plt)
        print(w)

        q = quality(w=w, x_plt=x_plt)
        print(q)

        print()


if __name__ == '__main__':
    # compute_min()
    compute_weights()
