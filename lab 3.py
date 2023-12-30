from typing import Callable
import numpy as np


def tangent(f: Callable, df: Callable, x0: float, eps: float = 1e-6, max_iter: int = 100) -> tuple:
    """
    Функция для реализации метода касательных

    :param f: функция
    :param df: производная
    :param x0: начальная точка
    :param eps: epsilon
    :param max_iter: максимальное число итераций
    :return: tuple
    """
    x = x0
    iteration = 0
    for _ in range(max_iter):
        iteration += 1
        fx = f(x)
        dfx = df(x)

        if abs(fx) < eps:
            return x, iteration, f(x)

        if dfx == 0:
            return None, iteration, None
        x = x - fx / dfx

    return x, iteration, f(x)


def simple_iter(f: Callable, x0: float, eps: float = 1e-6, max_iter: int = 100) -> tuple:
    """
    Функция для реализации метода простой итерации

    :param f: функция
    :param x0: начальная точка
    :param eps: epsilon
    :param max_iter: максимальное число итераций
    :return: tuple
    """

    x = x0
    iteration = 0
    try:
        for _ in range(max_iter):
            iteration += 1

            fx = f(x)

            if abs(x - fx) < eps:
                return x, iteration, f(x)

            x = fx
    except:
        pass

    return None, iteration, None


def newton_system(f: Callable, J: Callable, x0: np.array, eps: float = 1e-6) -> tuple:
    """
    Метод для реализации метода Ньютона для систем нелинейных уравнений

    :param f: функция
    :param J: матрица Якоби
    :param x0: начальная точка
    :param eps: epsilon
    :return: tuple
    """

    x = x0
    fx = f(x)
    iteration = 0
    while np.linalg.norm(fx) >= eps:
        iteration += 1
        fx = f(x)
        Jx = J(x)
        if np.linalg.det(Jx) == 0:
            return None, iteration, None
        dx = np.linalg.solve(Jx, -fx)
        x = x + dx
    return x, iteration, f(x)


def run_equation(f: Callable, df: Callable, x0: float, eps: float = 1e-6, max_iter: int = 1000) -> tuple:
    """
    Функция для возвращения решения уравнения двумя методами

    :param f: функция
    :param df: производная
    :param a: начало интервала
    :param b: конец интервала
    :param x0: начальная точка
    :param eps: epsilon
    :param max_iter: максимальное число итераций
    :return: tuple
    """

    return tangent(f, df, x0, eps, max_iter), simple_iter(f, x0, eps, max_iter)


def my_func1(x):
    return np.sin(x)


def my_func_d1(x):
    return np.cos(x)


def my_func2(x):
    return np.cos(x)


def my_func_d2(x):
    return -np.sin(x)


def my_func3(x):
    return np.cos(x) - 0.2


def my_func_d3(x):
    return -np.sin(x)


def my_func4(x):
    return (2 - x ** 3) / 2


def my_func_d4(x):
    return -3 * x ** 2 / 2


def my_func5(x):
    return x ** 3 - 2 * x - 5


def my_func_d5(x):
    return 3 * x ** 2 - 2


def my_system1(x):
    return np.array([x[0] ** 2 - x[1] - 1, np.exp(x[0]) - x[0] - x[1]])


def my_Jacob1(x):
    return np.array([[2 * x[0], -1], [np.exp(x[0]) - 1, -1]])


def my_system2(x):
    return np.array([x[0] ** 2 + x[1] ** 2 - 5, x[0] ** 3 - x[1] ** 3 - 1])


def my_Jacob2(x):
    return np.array([[2 * x[0], 2 * x[1]], [3 * x[0] ** 2, -3 * x[1] ** 2]])


def my_system3(x):
    return np.array([x[0] ** 2 + x[1] ** 2 - 9, np.exp(x[0]) + x[1] - 6])


def my_Jacob3(x):
    return np.array([[2 * x[0], 2 * x[1]], [np.exp(x[0]), 1]])


def my_system4(x):
    return np.array([x[0] ** 3 - 3 * x[0] * x[1] ** 2 - 2, x[0] ** 2 + x[1] ** 2 - 5])


def my_Jacob4(x):
    return np.array([[3 * x[0] ** 2 - 3 * x[1] ** 2, -6 * x[0] * x[1]], [2 * x[0], 2 * x[1]]])


def main():
    type = input('Что нужно посчитать:\n1 - Уравнения\n2 - Системы уравнений\n0 - выйти\n')

    while True:
        if type == '1':
            equation = input(f'Какое уравнение будем считать?:\n'
                             f'1: sin(x) = 0\n'
                             f'2: cos(x) = 0\n'
                             f'3: cos(x) - 0.2 = 0\n'
                             f'4: (2 - x ^ 3) / 2 = 0\n'
                             f'5: x ^ 3 - 2x - 5 = 0\n')

            try:
                equation = int(equation)
            except:
                print('Некорректный ввод. Повторите попытку')
                continue

            if equation not in range(1, 7):
                print('Не знаю такой код, повторите ввод')
                continue

            while True:
                x_start = input('Введите начальную точку: ')

                try:
                    x_start = float(x_start)
                except:
                    print('Некорректный ввод. Повторите попытку')
                    continue

                break

            funcs = [my_func1, my_func2, my_func3, my_func4, my_func5]
            dfs = [my_func_d1, my_func_d2, my_func_d3, my_func_d4, my_func_d5]

            tangent_result, simple_iter_result = run_equation(funcs[equation - 1], dfs[equation - 1], x_start)

            print(f'Решение уравнения для метода касательных: {tangent_result[0]}. Итераций: {tangent_result[1]}. Значение функции: {tangent_result[2]}')
            print(f'Решение уравнения для метода простой итерации: {simple_iter_result[0]}. Итераций: {simple_iter_result[1]}. Значение функции: {simple_iter_result[2]}')

            print()

        elif type == '2':
            equation = input(f'Какую систему будем решать?:\n'
                             f'1:\tx ^ 2 - y - 1 = 0\n\te ^ x - x - y = 0\n'
                             f'2:\tx ^ 2 + y ^ 2 - 5 = 0\n\tx ^ 3 - y ^ 3 - 1 = 0\n'
                             f'3:\tx ^ 2 + y ^ 2 - 9 = 0\n\te ^ x + y - 6 = 0\n'
                             f'4:\tx ^ 3 - 3xy ^ 2 - 2 = 0\n\tx ^ 2 + y ^ 2 - 5 = 0\n')

            try:
                equation = int(equation)
            except:
                print('Некорректный ввод. Повторите попытку')
                continue

            if equation not in range(1, 7):
                print('Не знаю такой код, повторите ввод')
                continue

            while True:
                x_start_1 = input('Введите начальную точку по x: ')
                x_start_2 = input('Введите начальную точку по y: ')

                try:
                    x_start_1 = float(x_start_1)
                    x_start_2 = float(x_start_2)
                except:
                    print('Некорректный ввод. Повторите попытку')
                    continue

                break

            systems = [my_system1, my_system2, my_system3, my_system4]
            jacobs = [my_Jacob1, my_Jacob2, my_Jacob3, my_Jacob4]

            newton_system_result = newton_system(systems[equation - 1], jacobs[equation - 1], np.array([x_start_1, x_start_2]))

            print(f'Решение системы для метода Ньютона: {newton_system_result[0]}. Итераций: {newton_system_result[1]}. Значение функции: {newton_system_result[2]}')

            print()

        elif type == '0':
            return

        else:
            print('Не знаю такой код, повторите ввод')
            type = input('Что нужно посчитать:\n1 - Уравнения\n2 - Системы уравнений\n0 - выйти\n')
            continue

        break


if __name__ == '__main__':
    main()
