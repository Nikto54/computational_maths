import numpy as np
from random import uniform as rand
import os


def display_matrix(matrix: list or np.array, with_x: bool = False, round_for: int or None = None) -> None:
    """
    Функция, которая выводит матрицу в читаемом для человека виде в консоль

    :param matrix: list or np.array
    :param with_x: return with x
    :param round_for: round for some digits after dot
    :return: None
    """
    k = 1
    for row in matrix:
        for elem in row:
            if round_for is not None:
                el = "%.7f" % round(elem, round_for)
            else:
                el = int(elem * 100) / 100

            if with_x:
                ret = str(el) + f' * x{k}'
            else:
                ret = str(el)

            if float(el) >= 0:
                ret = ' ' + ret

            if abs(int(float(el))) < 10:
                ret = ' ' + ret

            if abs(float(el) - int(float(el))) < 0.1:
                ret += ' '

            print(ret, end='\t')
        k += 1
        print('', end='\n')


def is_matrix_correct(matrix: list or np.array) -> bool:
    """
    Функция, проверяющая, есть ли на диагонали матрицы 0

    :param matrix: матрица
    :return: True если нет, False, если есть
    """

    for index, row in enumerate(matrix):
        if row[index] == 0:
            return False

    return True


def read_matrix(count: int) -> np.array:
    """
    Функция для ручного ввода матрицы по количеству чисел в строке

    :param count: количество чисел в строке
    :return: матрица
    """

    print(f'Введите по одной строке матрицы с {count} числами в строке, разделенных пробелом.\n' +
          'Чтобы остановить ввод используйте Ctrl+D')
    matrix = []
    k = 1
    while True:
        try:
            print(f'Введите строку {k}:')
            line = input()
            try:
                numbers = [float(n) for n in line.split()]
            except ValueError:
                print('Введено число с ошибкой. Убедитесь, что нет некорректных символов, в т.ч. десятичная часть '
                      'отделена точкой, и знак минуса не отделен пробелом от числа. Повторите ввод строки')
                continue
            if len(numbers) != count:
                print(f'В строке не {count} чисел, повторите ввод строки')
                continue
            matrix.append(numbers)
            k += 1
        except EOFError:
            return np.array(matrix)


def read_matrix_from_file(filename: str or os.path) -> np.array:
    """
    Функция для чтения матрицы из файла

    :param filename: путь к файлу
    :return: матрица
    """

    with open(filename, 'r') as file:
        content = file.read()

    lines = content.split('\n')

    if len(lines) == 0:
        print('Матрица в файле не найдена')
        return

    count = len(lines[0].split())
    matrix = []

    for line in lines:
        try:
            numbers = [float(n) for n in line.split()]
        except ValueError:
            print('Обнаружена ошибка в файле. Введено число с ошибкой. Убедитесь, что нет некорректных '
                             'символов, в т.ч. десятичная часть отделена точкой, и знак минуса не отделен '
                             'пробелом от числа.')
            return
        if len(numbers) != count:
            print(f'В строке файла не {count} чисел, как в первой строке')
            return
        matrix.append(numbers)

    return np.array(matrix)


def random_matrix(rows: int, cols: int) -> np.array:
    """
    Функция для генерации случайной матрицы размера rows * cols с десятичными числами от -100 до 100 без 0 на диагонали

    :param rows: количество строк
    :param cols: количество столбцов
    :return: матрица
    """

    result = []

    for row in range(rows):
        r = []
        for col in range(cols):
            elem = rand(-100, 100)
            while row == col and elem == 0:
                elem = rand(-100, 100)

            r.append(elem)

        result.append(r)

    return np.array(result)


def does_slay_joint(matrix: list or np.array, vector: list or np.array) -> bool:
    """
    Функция проверки совместимости матрицы

    :param matrix: матрица
    :param vector: вектор
    :return: True, если ранг матрицы и матрицы с вектором равны, False - иначе
    """

    with_vector = np.c_[matrix, vector]

    if np.linalg.matrix_rank(matrix) != np.linalg.matrix_rank(with_vector):
        return False

    return True


def determinant(matrix: list or np.array) -> float:
    """
    Функция вычисления определителя матрицы

    :param matrix: матрица
    :return: определитель матрицы
    """

    mat = np.copy(matrix)

    if len(mat) < len(mat[0]):
        mat = np.r_[mat, np.zeros((len(mat[0]) - len(mat), len(mat[0])))]

    if len(mat) > len(mat[0]):
        mat = np.c_[mat, np.zeros((len(mat), len(mat) - len(mat[0])))]

    assert len(mat.shape) == 2
    assert mat.shape[0] == mat.shape[1]
    n = mat.shape[0]

    for k in range(0, n - 1):

        for i in range(k + 1, n):
            if mat[i, k] != 0.0:
                lam = mat[i, k] / mat[k, k]
                mat[i, k:n] = mat[i, k:n] - lam * mat[k, k:n]

    return float(np.prod(np.diag(mat)))


def triangle(matrix: list or np.array) -> np.array:
    """
    Функция вычисления треугольной матрицы

    :param matrix: матрица
    :return: треугольная матрица
    """

    mat = np.copy(matrix)

    for nrow, row in enumerate(mat):

        divider = row[nrow]

        row /= divider

        for lower_row in mat[nrow + 1:]:
            factor = lower_row[nrow]
            lower_row -= factor * row

    return mat


def solve_gauss(triangle_matrix: list or np.array) -> np.array:
    """
    Функция нахождения решения по треугольной матрицы

    :param triangle_matrix: треугольная матрица
    :return: столбец решения
    """

    for nrow in range(len(triangle_matrix)-1, 0, -1):
        row = triangle_matrix[nrow]

        for upper_row in triangle_matrix[:nrow]:
            factor = upper_row[nrow]
            upper_row -= factor*row

    return np.array(triangle_matrix[:, -1])[np.newaxis].T


def discrepancy(res: np.array, vector: np.array) -> np.array:
    """
    Функция вычисления столбца невязок

    :param res: получившийся результирующий столбец
    :param vector: заданный вектор
    :return: разница векторов
    """

    return vector - res


def gauss(matrix: list or np.array, vector: list or np.array) -> None:
    """
    Функция гаусса, выводит определитель матрицы, треугольную матрицу, столбцы неизвестных и невязок

    :param matrix: матрица
    :param vector: вектор
    :return: None
    """

    if not does_slay_joint(matrix, vector):
        print('Решений нет.')
        return

    if len(matrix[0]) > np.linalg.matrix_rank(matrix):
        print('Решений бесконечное число.')
        return

    if determinant(matrix) == 0:
        print('Определитель матрицы равен нулю')
        return

    if not is_matrix_correct(matrix):
        print('На диагонали матрицы есть 0')
        return

    print('Матрица с вектором:')
    m_v = np.c_[matrix, vector]
    display_matrix(m_v)
    print()

    print(f'Определитель матрицы равен {determinant(matrix)}')
    print()

    print('Треугольная матрица с вектором:')
    display_matrix(triangle(m_v))
    print()

    print('Столбец неизвестных:')
    display_matrix(solve_gauss(triangle(m_v)), True)
    print()

    print('Столбец невязок: ')
    display_matrix(discrepancy(matrix.dot(solve_gauss(triangle(m_v))), vector), round_for=7)
    print()


def main():
    type = input('Введите код типа ввода матрицы:\n1 - Вручную\n2 - Из файла\n3 - Случайная матрица\n0 - выйти\n')

    while True:
        if type == '1':
            cnt = input('Введите количество чисел в строке матрицы (вместе с вектором): \n')

            try:
                cnt = int(cnt)
            except ValueError:
                print('Некорректный ввод. Попробуйте еще раз')
                continue

            matrix_vector = read_matrix(cnt)
            mmatrix = matrix_vector[:, :-1]
            vvector = np.array(matrix_vector[:, -1])[np.newaxis].T

        elif type == '2':
            path = input('Введите путь к файлу, содержащему матрицу с вектором: \n')

            try:
                matrix_vector = read_matrix_from_file(path)

                if matrix_vector is None:
                    return

                mmatrix = matrix_vector[:, :-1]
                vvector = np.array(matrix_vector[:, -1])[np.newaxis].T

            except FileNotFoundError:
                print('Файл не найден, попробуйте еще раз')
                continue

        elif type == '3':
            rows = input('Введите количество строк в матрице\n')
            cols = input('Введите количество столбцов в матрице (с учетом вектора)\n')

            try:
                rows = int(rows)
                cols = int(cols)
            except ValueError:
                print('Некорректный ввод. Попробуйте еще раз')
                continue

            matrix_vector = random_matrix(rows, cols)

            mmatrix = matrix_vector[:, :-1]
            vvector = np.array(matrix_vector[:, -1])[np.newaxis].T

        elif type == '0':
            return

        else:
            print('Не знаю такой код, повторите ввод')
            type = input('Введите код типа ввода матрицы:\n1 - Вручную\n2 - Из файла\n3 - Случайная матрица\n0 - выйти\n')
            continue

        break

    gauss(mmatrix, vvector)


if __name__ == '__main__':
    main()