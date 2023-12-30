from typing import Callable
import numpy as np


def tangent(f: Callable, df: Callable, x0: float, eps: float = 1e-6) -> float or None:
    """
    Функция для реализации метода касательных

    :param f: функция
    :param df: производная
    :param x0: начальная точка
    :param eps: epsilon
    :return: float or None
    """
    x = x0
    fx = f(x)
    while abs(fx) >= eps:
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            return None
        x = x - fx / dfx
    return x


def simple_iter(f: Callable, x0: float, eps: float = 1e-6) -> float or None:
    """
    Функция для реализации метода простой итерации

    :param f: функция
    :param x0: начальная точка
    :param eps: epsilon
    :return: float or None
    """
    x = x0
    fx = f(x)
    while abs(x - fx) >= eps:
        x = fx
        fx = f(x)
    return x


def newton_system(f: Callable, J: Callable, x0: np.array, eps: float = 1e-6) -> np.array or None:
    """
    Метод для реализации метода Ньютона для систем нелинейных уравнений

    :param f: функция
    :param J: матрица Якоби
    :param x0: начальная точка
    :param eps: epsilon
    :return: float or None
    """

    x = x0
    fx = f(x)
    while np.linalg.norm(fx) >= eps:
        fx = f(x)
        Jx = J(x)
        if np.linalg.det(Jx) == 0:
            return None
        dx = np.linalg.solve(Jx, -fx)
        x = x + dx
    return x


def run_equation(f: Callable, df: Callable, x0: float, eps: float = 1e-6) -> tuple:
    """
    Функция для возвращения решения уравнения двумя методами

    :param f: функция
    :param df: производная
    :param x0: начальная точка
    :param eps: epsilon
    :return: tuple
    """

    return tangent(f, df, x0, eps), simple_iter(f, x0, eps)


def my_func(x):
    return np.sin(x)


def my_func_d(x):
    return np.cos(x)


def my_system(x):
    return np.array([x[0] ** 2 - x[1] - 1, np.exp(x[0]) - x[0] - x[1]])


def my_Jacob(x):
    return np.array([[2 * x[0], -1], [np.exp(x[0]) - 1, -1]])


def main():
    tangent_result, simple_iter_result = run_equation(my_func, my_func_d, 2)
    newton_system_result = newton_system(my_system, my_Jacob, np.array([1, 1]))

    print(f'Решение уравнения для метода касательных: {tangent_result}')
    print(f'Решение уравнения для метода простой итерации: {simple_iter_result}')
    print(f'Решение системы для метода Ньютона: {newton_system_result}')


if __name__ == '__main__':
    main()
