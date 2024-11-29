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
        расчет производной функции потерь, в ответе вектор дельт весов
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

    eta = np.array([0.1, 0.01, 0.001,
                    0.0001])  # шаг обучения для каждого параметра w0, w1, w2, w3 (перемножаются на коэффициенты model)
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
        # берем случайную позицию значения из обучающей выборки
        rand_x_pos = np.random.randint(0, len(x_plt) - batch_size - 1)

        w = w - eta * batch_delta_loss(w, x_plt[rand_x_pos:rand_x_pos + batch_size])
        print(w)

        # пересчитаем значение Qe - экспоненциальное скользящее среднего (легковесного аналога среднего эмпирического риска)
        Qe = lm * batch_loss(w, x_plt[rand_x_pos:rand_x_pos + batch_size]) + (1 - lm) * Qe
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


def compute_weights_sdg_binary_classification1():
    """
    Пример использования SGD при бинарной классификации образов
    """
    def loss(w, x, y):
        """
        логарифмическая функция потерь
        """
        M = np.dot(w, x) * y
        return np.log2(1.0 + np.exp(-M))

    def delta_loss(w, x, y):
        """
        производная логарифмической функции потерь по вектору w
        """
        M = np.dot(w, x) * y
        return -(np.exp(-M) * x.T * y) / ((1.0 + np.exp(-M)) * np.log(2))

    data_x = [(3.0, 4.9), (2.7, 3.9), (3.0, 5.5), (2.6, 4.0), (2.9, 4.3), (3.1, 5.1), (2.2, 4.5), (2.3, 3.3),
              (2.7, 5.1), (3.3, 5.7), (2.8, 5.1), (2.8, 4.9), (2.5, 4.5), (2.8, 4.7), (3.2, 4.7), (3.2, 5.7),
              (2.8, 6.1), (3.6, 6.1), (2.8, 4.8), (2.9, 4.5), (3.1, 4.9), (2.3, 4.4), (3.3, 6.0), (2.6, 5.6),
              (3.0, 4.4), (2.9, 4.7), (2.8, 4.0), (2.5, 5.8), (2.4, 3.3), (2.8, 6.7), (3.0, 5.1), (2.3, 4.0),
              (3.1, 5.5), (2.8, 4.8), (2.7, 5.1), (2.5, 4.0), (3.1, 4.4), (3.8, 6.7), (3.1, 5.6), (3.1, 4.7),
              (3.0, 5.8), (3.0, 5.2), (3.0, 4.5), (2.7, 4.9), (3.0, 6.6), (2.9, 4.6), (3.0, 4.6), (2.6, 3.5),
              (2.7, 5.1), (2.5, 5.0), (2.0, 3.5), (3.2, 5.9), (2.5, 5.0), (3.4, 5.6), (3.4, 4.5), (3.2, 5.3),
              (2.2, 4.0), (2.2, 5.0), (3.3, 4.7), (2.7, 4.1), (2.4, 3.7), (3.0, 4.2), (3.2, 6.0), (3.0, 4.2),
              (3.0, 4.5), (2.7, 4.2), (2.5, 3.0), (2.8, 4.6), (2.9, 4.2), (3.1, 5.4), (2.5, 4.9), (3.2, 5.1),
              (2.8, 4.5), (2.8, 5.6), (3.4, 5.4), (2.7, 3.9), (3.0, 6.1), (3.0, 5.8), (3.0, 4.1), (2.5, 3.9),
              (2.4, 3.8), (2.6, 4.4), (2.9, 3.6), (3.3, 5.7), (2.9, 5.6), (3.0, 5.2), (3.0, 4.8), (2.7, 5.3),
              (2.8, 4.1), (2.8, 5.6), (3.2, 4.5), (3.0, 5.9), (2.9, 4.3), (2.6, 6.9), (2.8, 5.1), (2.9, 6.3),
              (3.2, 4.8), (3.0, 5.5), (3.0, 5.0), (3.8, 6.4)]
    data_y = [1, -1, 1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, 1,
              1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, -1, -1,
              -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, 1, -1,
              1, -1, 1, 1, 1, -1, 1, -1, 1]

    x_train = np.array([[1, x[0], x[1]] for x in data_x])
    y_train = np.array(data_y)

    w = [0.0, 0.0, 0.0]  # начальные весовые коэффициенты
    step = np.array([0.5, 0.01, 0.01])  # шаг обучения для каждого параметра w0, w1, w2
    lm = 0.01  # значение параметра лямбда для вычисления скользящего экспоненциального среднего
    N = 1000  # число итераций алгоритма SGD

    # начальное значение экспоненциально скользящего среднего путем расчета среднего эмпирического риска
    Qe = np.mean([loss(w, x, y) for x, y in zip(x_train, y_train)])
    print(f"Qe: {Qe}")

    # генерация одинаковых последовательностей псевдослучайных чисел
    np.random.seed(0)

    for i in range(N):
        # берем случайную позицию значения из обучающей выборки
        rand_x_pos = np.random.randint(0, len(x_train) - 1)

        # обновляем веса
        w = w - step * delta_loss(w, x_train[rand_x_pos], y_train[rand_x_pos])
        print(w)

        # пересчитаем значение Qe - экспоненциальное скользящее среднего (легковесного аналога среднего эмпирического риска)
        Qe = lm * loss(w, x_train[rand_x_pos], y_train[rand_x_pos]) + (1 - lm) * Qe
        print(f"Qe: {Qe}")

    # можно сравнивать позицию точку на графике и значение ее loss
    quality = [(x[1], x[2], loss(w, x, y)) for x, y in zip(x_train, y_train)]

    # конечное значение среднего эмпирического риска путем расчета количества некорректных предсказаний класса (те у кого loss для точки >1)
    Q = np.mean([1 if q[2] > 1.0 else 0 for q in quality])
    print(f"Q: {Q}")

    positive_class = [x for i, x in enumerate(data_x) if data_y[i] > 0]
    negative_class = [x for i, x in enumerate(data_x) if data_y[i] < 0]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for x in positive_class:
        ax.scatter(x[0], x[1], color='tab:blue')

    for x in negative_class:
        ax.scatter(x[0], x[1], color='tab:red')

    # разделяющая классы прямая описывается формулой w0 + w1⋅x1 + w2⋅x2 = 0
    x1_plt = np.arange(1.0, 4.0, 0.1)
    x2 = lambda x1: (-1 * w[0] - x1 * w[1]) / w[2]
    x2_plt = [x2(x1) for x1 in x1_plt]
    ax.plot(x1_plt, x2_plt, color='tab:brown')

    plt.show()


def compute_weights_sdg_binary_classification2():
    """
    Пример использования SGD при бинарной классификации образов
    """
    def loss(w, x, y):
        """
        экспоненциальная функция потерь
        """
        M = np.dot(w, x) * y
        return np.exp(-M)

    def batch_loss(w, x_batch_plt, y_batch_plt):
        return np.mean([loss(w, x, y) for x, y in zip(x_batch_plt, y_batch_plt)])

    def delta_loss(w, x, y):
        """
        производная экспоненциальной функции потерь по вектору w
        """
        M = np.dot(w, x) * y
        return -np.exp(-M) * x.T * y

    def batch_delta_loss(w, x_batch_plt, y_batch_plt):
        """
        берем среднее значение в батче для каждого веса параметра модели
        """
        delta_loss_list = [delta_loss(w, x, y) for x, y in zip(x_batch_plt, y_batch_plt)]
        res = np.array([np.sum(v) / len(x_batch_plt) for v in zip(*delta_loss_list)])

        return res

    data_x = [(5.8, 1.2), (5.6, 1.5), (6.5, 1.5), (6.1, 1.3), (6.4, 1.3), (7.7, 2.0), (6.0, 1.8), (5.6, 1.3),
              (6.0, 1.6), (5.8, 1.9), (5.7, 2.0), (6.3, 1.5), (6.2, 1.8), (7.7, 2.3), (5.8, 1.2), (6.3, 1.8),
              (6.0, 1.0), (6.2, 1.3), (5.7, 1.3), (6.3, 1.9), (6.7, 2.5), (5.5, 1.2), (4.9, 1.0), (6.1, 1.4),
              (6.0, 1.6), (7.2, 2.5), (7.3, 1.8), (6.6, 1.4), (5.6, 2.0), (5.5, 1.0), (6.4, 2.2), (5.6, 1.3),
              (6.6, 1.3), (6.9, 2.1), (6.8, 2.1), (5.7, 1.3), (7.0, 1.4), (6.1, 1.4), (6.1, 1.8), (6.7, 1.7),
              (6.0, 1.5), (6.5, 1.8), (6.4, 1.5), (6.9, 1.5), (5.6, 1.3), (6.7, 1.4), (5.8, 1.9), (6.3, 1.3),
              (6.7, 2.1), (6.2, 2.3), (6.3, 2.4), (6.7, 1.8), (6.4, 2.3), (6.2, 1.5), (6.1, 1.4), (7.1, 2.1),
              (5.7, 1.0), (6.8, 1.4), (6.8, 2.3), (5.1, 1.1), (4.9, 1.7), (5.9, 1.8), (7.4, 1.9), (6.5, 2.0),
              (6.7, 1.5), (6.5, 2.0), (5.8, 1.0), (6.4, 2.1), (7.6, 2.1), (5.8, 2.4), (7.7, 2.2), (6.3, 1.5),
              (5.0, 1.0), (6.3, 1.6), (7.7, 2.3), (6.4, 1.9), (6.5, 2.2), (5.7, 1.2), (6.9, 2.3), (5.7, 1.3),
              (6.1, 1.2), (5.4, 1.5), (5.2, 1.4), (6.7, 2.3), (7.9, 2.0), (5.6, 1.1), (7.2, 1.8), (5.5, 1.3),
              (7.2, 1.6), (6.3, 2.5), (6.3, 1.8), (6.7, 2.4), (5.0, 1.0), (6.4, 1.8), (6.9, 2.3), (5.5, 1.3),
              (5.5, 1.1), (5.9, 1.5), (6.0, 1.5), (5.9, 1.8)]
    data_y = [-1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, 1,
              -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, 1, -1, -1, 1,
              -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, 1, 1,
              1, 1, -1, 1, 1, -1, -1, -1, -1, 1]

    x_train = np.array([[1, x[0], x[1]] for x in data_x])
    y_train = np.array(data_y)

    w = [0.0, 0.0, 0.0]  # начальные весовые коэффициенты
    step = np.array([0.5, 0.01, 0.01])  # шаг обучения для каждого параметра w0, w1, w2
    lm = 0.01  # значение параметра лямбда для вычисления скользящего экспоненциального среднего
    N = 500  # число итераций алгоритма SGD
    batch_size = 10  # размер мини-батча

    # начальное значение экспоненциально скользящего среднего путем расчета среднего эмпирического риска
    Qe = np.mean([loss(w, x, y) for x, y in zip(x_train, y_train)])
    print(f"Qe: {Qe}")

    # генерация одинаковых последовательностей псевдослучайных чисел
    np.random.seed(0)

    for i in range(N):
        # берем случайную позицию значения из обучающей выборки
        rand_x_pos = np.random.randint(0, len(x_train) - batch_size - 1)

        # обновляем веса
        w = w - step * batch_delta_loss(w, x_train[rand_x_pos:rand_x_pos + batch_size], y_train[rand_x_pos:rand_x_pos + batch_size])
        print(w)

        # пересчитаем значение Qe - экспоненциальное скользящее среднего (легковесного аналога среднего эмпирического риска)
        Qe = lm * batch_loss(w, x_train[rand_x_pos:rand_x_pos + batch_size], y_train[rand_x_pos:rand_x_pos + batch_size]) + (1 - lm) * Qe
        print(f"Qe: {Qe}")

    # можно сравнивать позицию точку на графике и значение ее loss
    quality = [(x[1], x[2], loss(w, x, y)) for x, y in zip(x_train, y_train)]

    # конечное значение среднего эмпирического риска путем расчета количества некорректных предсказаний класса (те у кого loss для точки >1)
    Q = np.mean([1 if q[2] > 1.0 else 0 for q in quality])
    print(f"Q: {Q}")

    positive_class = [x for i, x in enumerate(data_x) if data_y[i] > 0]
    negative_class = [x for i, x in enumerate(data_x) if data_y[i] < 0]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for x in positive_class:
        ax.scatter(x[0], x[1], color='tab:blue')

    for x in negative_class:
        ax.scatter(x[0], x[1], color='tab:red')

    # разделяющая классы прямая описывается формулой w0 + w1⋅x1 + w2⋅x2 = 0
    x1_plt = np.arange(5.0, 8.0, 0.1)
    x2 = lambda x1: (-1 * w[0] - x1 * w[1]) / w[2]
    x2_plt = [x2(x1) for x1 in x1_plt]
    ax.plot(x1_plt, x2_plt, color='tab:brown')

    plt.show()


if __name__ == '__main__':
    # compute_min()
    # compute_weights1()
    # compute_weights_sdg()
    # compute_weights_sdg_batch()
    # compute_weights_sdg_binary_classification1()
    compute_weights_sdg_binary_classification2()
