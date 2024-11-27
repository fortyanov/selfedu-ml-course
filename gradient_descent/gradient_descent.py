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


def compute_weights1():
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


def compute_weights_sdg():
    """
    Необходимо выполнить аппроксимацию (восстановление) функции на интервале [-5, 5] моделью
    с помощью стохастического градиентного спуска
    """

    # исходная функция, которую нужно аппроксимировать моделью model
    def func(x):
        return 0.5 * x ** 2 - 0.1 * 1 / np.exp(-x) + 0.5 * np.cos(2 * x) - 2.

    def model(x):
        return np.array([1, x, x ** 2, np.cos(2 * x), np.sin(2 * x)])

    def loss(w, x):
        return (w @ model(x) - func(x)) ** 2

    def delta_loss(w, x):
        res = 2 * (w @ model(x) - func(x)) * model(x)
        return res

    eta = np.array([0.01, 0.001, 0.0001, 0.01,
                    0.01])  # шаг обучения для каждого параметра w0, w1, w2, w3, w4 (перемножаются на коэффициенты model)
    w = np.array([0., 0., 0., 0., 0.])  # начальные значения параметров модели
    N = 500  # число итераций алгоритма SGD
    lm = 0.02  # значение параметра лямбда для вычисления скользящего экспоненциального среднего

    x_plt = np.arange(-5.0, 5.0, 0.1)
    f_plt = [func(x) for x in x_plt]

    # начальное значение среднего эмпирического риска
    Qe = np.sum([loss(w, x) for x in x_plt]) / len(x_plt)
    Qstart = Qe

    np.random.seed(0)  # генерация одинаковых последовательностей псевдослучайных чисел

    for i in range(N):
        # берем случайное значение из обучающей выборки
        rand_x = np.random.choice(x_plt)

        # обновляем веса
        w = w - eta * delta_loss(w, rand_x)
        print(w)

        # пересчитаем значение Qe - экспоненциальное скользящее среднего (легковесного аналога среднего эмпирического риска)
        Qe = lm * loss(w, rand_x) + (1 - lm) * Qe
        print(Qe)

        print(i)

    # конечное значение среднего эмпирического риска
    Q = np.sum([loss(w, x) for x in x_plt]) / len(x_plt)

    print(f'w: {w}')
    print(f'Qstart: {Qstart}')
    print(f'Qe: {Qe}')
    print(f'Q: {Q}')

    model_plt = np.array([w @ model(x) for x in x_plt])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x_plt, f_plt, color='tab:blue')
    ax.plot(x_plt, model_plt, color='tab:orange')
    plt.show()


def compute_weights_sdg_batch():
    """
    Необходимо выполнить аппроксимацию (восстановление) функции на интервале [-4, 6] моделью
    с помощью стохастического градиентного спуска с батчем
    """

    # исходная функция, которую нужно аппроксимировать моделью model
    def func(x):
        return 0.5 * x + 0.2 * x ** 2 - 0.05 * x ** 3 + 0.2 * np.sin(4 * x) - 2.5

    def model(x):
        return np.array([1, x, x ** 2, x ** 3])

    def loss(w, x):
        """
        в ответе скаляр
        """
        return (w @ model(x) - func(x)) ** 2

    def batch_loss(w, x_batch_plt):
        return np.sum([loss(w, x) for x in x_batch_plt]) / len(x_batch_plt)

    def delta_loss(w, x):
        """
        в ответе вектор дельт весов
        """
        res = 2 * (w @ model(x) - func(x)) * model(x)
        return res

    def batch_delta_loss(w, x_batch_plt):
        """
        берем среднее значение в батче для каждого веса параметра модели
        """
        delta_loss_list = [delta_loss(w, x) for x in x_batch_plt]
        res = np.array([np.sum(v) / len(x_batch_plt) for v in zip(*delta_loss_list)])

        return res

    eta = np.array([0.1, 0.01, 0.001, 0.0001])  # шаг обучения для каждого параметра w0, w1, w2, w3 (перемножаются на коэффициенты model)
    w = np.array([0., 0., 0., 0.])  # начальные значения параметров модели
    N = 500  # число итераций алгоритма SGD
    lm = 0.02  # значение параметра лямбда для вычисления скользящего экспоненциального среднего
    batch_size = 50  # размер мини-батча (величина K = 50)

    x_plt = np.arange(-4.0, 6.0, 0.1)
    f_plt = [func(x) for x in x_plt]

    # начальное значение среднего эмпирического риска
    Qe = np.sum([loss(w, x) for x in x_plt]) / len(x_plt)
    Qstart = Qe

    np.random.seed(0)  # генерация одинаковых последовательностей псевдослучайных чисел

    for i in range(N):
        rand_x = np.random.randint(0, len(x_plt) - batch_size - 1)

        w = w - eta * batch_delta_loss(w, x_plt[rand_x:rand_x + batch_size])
        print(w)

        # пересчитаем значение Qe - экспоненциальное скользящее среднего (легковесного аналога среднего эмпирического риска)
        Qe = lm * batch_loss(w, x_plt[rand_x:rand_x + batch_size]) + (1 - lm) * Qe
        print(Qe)

        print(i)

    # конечное значение среднего эмпирического риска
    Q = np.sum([loss(w, x) for x in x_plt]) / len(x_plt)

    print(f'w: {w}')
    print(f'Qstart: {Qstart}')
    print(f'Qe: {Qe}')
    print(f'Q: {Q}')

    model_plt = np.array([w @ model(x) for x in x_plt])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x_plt, f_plt, color='tab:blue')
    ax.plot(x_plt, model_plt, color='tab:orange')
    plt.show()


if __name__ == '__main__':
    # compute_min()
    # compute_weights1()
    compute_weights_sdg()
    compute_weights_sdg_batch()
