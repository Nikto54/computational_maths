import math
import warnings

warnings.filterwarnings('ignore')


def my_func1(x):
    return x ** 2


def my_func2(x):
    return x ** 3


def my_func3(x):
    try:
        return x / (x - 1)
    except ZeroDivisionError:
        return float('inf')


def my_func4(x):
    return x ** 4 - 8 * x ** 2 - 3


def my_func5(x):
    try:
        return 1 / x
    except ZeroDivisionError:
        return float('inf')


def trapezoidal(f, a, b, n, eps=0.000001):

    if not (callable(f) and all(isinstance(x, (int, float)) for x in [a, b]) and isinstance(n, int)) or a >= b or n < 1:
        raise TypeError("Invalid input arguments")

    if f(a) == float("inf") or f(b) == float("inf"):

        if f(a) == float("inf"):
            print(f"Обнаружен разрыв первого рода в точке {a}. Попытка его убрать.")
            a = a + eps

        if f(b) == float("inf"):
            print(f"Обнаружен разрыв первого рода в точке {b}. Попытка его убрать.")
            b = b - eps

    for i in range(n):
        x0 = a + i * (b - a) / n
        x1 = a + (i + 1) * (b - a) / n
        if f(x0) == float("inf") or f(x1) == float("inf"):
            if f(x0) == float("inf"):
                ValueError(f"Обнаружен разрыв второго рода в точке {x0}. Устранять его нет необходимости.")

            if f(x1) == float("inf"):
                ValueError(f"Обнаружен разрыв второго рода в точке {x1}. Устранять его нет необходимости.")

    h = (b - a) / n
    result = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        x = a + i * h
        result += f(x)
    result *= h

    if math.isnan(result) or math.isinf(result):
        raise ValueError("Невозможно убрать разрыв")

    return result


def main():
    funcs = [my_func1, my_func2, my_func3, my_func4, my_func5]
    func_string = ['x\u00B2', 'x\u00B3', 'x / (x - 1)', 'x\u2074 - 8x\u00B2 - 3', '1 / x']

    while True:
        index = input('Интеграл какой функции нужно посчитать?\n' +
                      '\n'.join([f'{ind + 1}. {f}' for ind, f in enumerate(func_string)]) + '\n' +
                      '0. Выйти\n\n')

        try:
            index = int(index)
        except Exception:
            print('Некорректный ввод. Попробуйте еще раз.\n')
            continue

        if index not in range(len(funcs) + 1):
            print('Некорректный ввод. Попробуйте еще раз.\n')
            continue

        if index == 0:
            return
        else:
            while True:
                down = input('Введите нижнюю границу интегрирования: ')
                up = input('Введите верхнюю границу интегрирования: ')
                n = input('Введите количество разбиений: ')

                try:
                    down = float(down)
                    up = float(up)
                    n = int(n)
                except:
                    print('Некорректный ввод. Попробуйте еще раз.\n\n')
                    continue

                if down >= up or n < 1:
                    print('Некорректный ввод. Попробуйте еще раз.\n\n')
                    continue

                break

            print()

            try:
                res = trapezoidal(funcs[index - 1], down, up, n)
                print(f'Значение определенного интеграла равно {res}')
            except ValueError as e:
                print(f'Ошибка: {e}')

        break


if __name__ == '__main__':
    main()
