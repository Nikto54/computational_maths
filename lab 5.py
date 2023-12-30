import numpy as np
import matplotlib.pyplot as plt


def func1(x, y):
    return np.sin(x) ** 2 * y


def analytic_func1(x, c):
    return c * np.exp((x - np.sin(x) * np.cos(x)) / 2)


def c_func1(x, y):
    return y / np.exp((x - np.sin(x) * np.cos(x)) / 2)


def func2(x, y):
    return 3 * y / x + x ** 3 + x


def analytic_func2(x, c):
    return x ** 4 - x ** 2 + c * abs(x) ** 3


def c_func2(x, y):
    return (y - x ** 4 + x ** 2) / abs(x) ** 3


texts = ["y' = sin(x)\u00B2 * y", "y' = 3y/x + x\u00B3 + x"]
functions = [func1, func2]
analytics = [analytic_func1, analytic_func2]
cs = [c_func1, c_func2]


def rungeKutta(func, x_start, y_start, x_n, h):
    x, y = [], []

    x.append(x_start)
    y.append(y_start)

    while True:
        x_h = x[-1] + h
        x_h_2 = x[-1] + h / 2

        k_1 = func(x[-1], y[-1])
        y_h_k_1 = y[-1] + h * k_1 / 2

        k_2 = func(x_h_2, y_h_k_1)
        y_h_k_2 = y[-1] + h * k_2 / 2

        k_3 = func(x_h_2, y_h_k_2)
        y_h_k_3 = y[-1] + h * k_3

        k_4 = func(x_h, y_h_k_3)

        x.append(min(x_h, x_n))
        y.append(y[-1] + (k_1 + 2 * k_2 + 2 * k_3 + k_4) * min(x_n - x[-1], h) / 6)

        if x[-1] >= x_n:
            break

    return x, y


def main():

    while True:

        print('\033[1mAvailable cases:\033[0m')

        print('\n'.join([f"{index + 1}) {text}" for index, text in enumerate(texts)]))

        func_inp = input('\n\033[1mPlease select a case or print 0 to exit\033[0m: ')

        try:
            func_inp = int(func_inp)
        except:
            print('Invalid case chosen. Try again\n\n\n')
            continue

        if func_inp < 1 or func_inp >= len(functions):
            print('Invalid case chosen. Try again\n\n\n')
            continue

        if func_inp == 0:
            return

        while True:
            x0 = input('Enter x₀: ')
            y0 = input('Enter y₀: ')
            xn = input('Enter xₙ: ')

            try:
                x0 = float(x0)
                y0 = float(y0)
                xn = float(xn)
            except:
                print('Invalid input. Try again\n\n\n')
                continue

            break

        break

    xs, ys = rungeKutta(functions[func_inp - 1], x0, y0, xn, 0.5)

    c = cs[func_inp - 1](x0, y0)

    xsa, ysa = [], []

    xsa.append(x0)
    ysa.append(analytics[func_inp - 1](x0, c))

    while True:
        xsa_new = xsa[-1] + 1 / 1000
        ysa_new = analytics[func_inp - 1](xsa[-1], c)

        xsa.append(xsa_new)
        ysa.append(ysa_new)

        if xsa[-1] >= xn:
            break

    plt.plot(xs, ys, color='red', label='Runge-Kutta')
    plt.plot(xsa, ysa, color='blue', label='Analytical solve')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
