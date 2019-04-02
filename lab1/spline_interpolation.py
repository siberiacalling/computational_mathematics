import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from scipy.linalg import solve


def h(x_nodes, i):
    return x_nodes[i + 1] - x_nodes[i]


def sum_matrixs(A, B, C):
    rows_amount = len(A)
    cols_amount = len(A[0])
    result = [[0 for x in range(rows_amount)] for y in range(cols_amount)]
    for i in range(rows_amount):
        for j in range(cols_amount):
            result[i][j] = A[i][j] + B[i][j] + C[i][j]
    return np.array(result)


# matrix A
def get_matrix_a(x_nodes):
    n = len(x_nodes)

    # matrix A
    main_diagonal = []
    for i in range(1, n + 1):
        if i == 1 or i == n:
            main_diagonal.append(1)
        else:
            main_diagonal.append(2 * (h(x_nodes, i - 1) + h(x_nodes, i - 2)))
    a = np.diag(main_diagonal, 0)

    # matrix B
    under_main_diagonal = []
    for i in range(1, n):
        if i == 1:
            under_main_diagonal.append(0)
        else:
            under_main_diagonal.append(h(x_nodes, i - 1))
    b = np.diag(under_main_diagonal, 1)

    # matrix C
    up_main_diagonal = []
    for i in range(1, n):
        if i == n - 1:
            up_main_diagonal.append(0)
        up_main_diagonal.append(h(x_nodes, i - 1))
    c = np.diag(up_main_diagonal, -1)
    result = sum_matrixs(a, b, c)
    return result


def get_matrix_b(x_nodes, y):
    n = len(x_nodes)
    b = []
    for i in range(1, n + 1):
        if i == 1 or i == n:
            b.append(0)
        else:
            b.append(3 * (y[i] - y[i - 1]) / h(x_nodes, i - 1) - 3 * (y[i - 1] - y[i - 2]) / h(x_nodes, i - 2))
    return b


def get_d_coefficient(x_nodes, c):
    d = []
    for i in range(0, len(c) - 1):
        d.append((c[i + 1] - c[i]) / (3 * h(x_nodes, i)))
    return d


def get_b_coefficient(x_nodes, c, a):
    b = []
    for i in range(0, len(c) - 1):
        b.append((a[i + 1] - a[i]) / h(x_nodes, i) + h(x_nodes, i) * (c[i + 1] + 2 * c[i]) / 3)
    return b


def cubic_spline_coeff(x_nodes, y_nodes):
    # solve Ac = b
    A = get_matrix_a(x_nodes)
    b = get_matrix_b(x_nodes, y_nodes)
    c_coef = solve(A, b)

    # calculate coefficients
    a_coef = y_nodes
    d_coef = np.array(get_d_coefficient(x_nodes, c_coef))
    b_coef = np.array(get_b_coefficient(x_nodes, c_coef, a_coef))
    return [a_coef[:-1], b_coef, c_coef[:-1], d_coef]


def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    if idx == len(array) - 1:
        idx = len(array) - 2
    return idx


def cubic_spline(x, qs_coeff, x_nodes):
    index = find_nearest_index(x_nodes, x)
    x_i = x_nodes[index]

    # coefficients
    a = qs_coeff[0][index]
    b = qs_coeff[1][index]
    c = qs_coeff[2][index]
    d = qs_coeff[3][index]

    s = a + b * (x - x_i) + c * np.power((x - x_i), 2) + d * np.power((x - x_i), 3)
    return s


def derivative_cubic_spline(x, qs_coeff, x_nodes):
    index = find_nearest_index(x_nodes, x)
    x_i = x_nodes[index]

    # coefficients
    b = qs_coeff[1][index]
    c = qs_coeff[2][index]
    d = qs_coeff[3][index]
    s = b + 2 * c * (x - x_i) + 3 * d * np.power((x - x_i), 2)
    return s


def process_csv():
    data = pd.read_csv('singapore.csv', usecols=['Data', 'T'], index_col=None)
    x_nodes = np.arange(0, 224, 1)
    y_nodes = data['T'].values
    return x_nodes, y_nodes


def draw_graph(x_nodes, y_nodes, length):
    x_draw = np.linspace(1, 28, length)
    matplotlib.rcParams.update({'font.size': 17})
    plt.plot(x_draw, y_nodes)
    plt.xlabel('День')
    plt.ylabel('Температура')
    plt.gcf().autofmt_xdate()
    plt.grid()
    plt.show()


def draw_reduced_data(x_nodes, y_nodes, step):
    cs_coeff = cubic_spline_coeff(x_nodes, y_nodes)
    x_cubic = np.arange(0, 222 / step, 0.01)
    y_cubic = [cubic_spline(x, cs_coeff, x_nodes) for x in x_cubic]
    draw_graph(x_cubic, y_cubic, len(x_cubic))


def calculate_average_day_temperature(y):
    # in original data set there are 8 temperatures for 1 day
    temp = np.zeros(28)
    for i in range(len(y)):
        temp_index = int(i / 8)
        temp[temp_index] += y[i] / 8
    return temp


def calculate_average_day_temperature_cubic(x_nodes, y_nodes, step):
    cs_coeff = cubic_spline_coeff(x_nodes[1::step], y_nodes[1::step])
    x_test = np.linspace(1, 28, len(x_nodes))
    y_cubic = [cubic_spline(x, cs_coeff, x_nodes[1::step]) for x in x_test]
    return calculate_average_day_temperature(y_cubic)


def calculate_dist_between_deleted_nodes(x_nodes, y_nodes, step):
    nodes_for_delete = np.arange(1, len(x_nodes), step)

    x_test = np.delete(x_nodes, nodes_for_delete)
    y_test = np.delete(y_nodes, nodes_for_delete)

    cs_coeff = cubic_spline_coeff(x_nodes[1::step], y_nodes[1::step])
    y_cubic = [cubic_spline(x, cs_coeff, x_nodes[1::step]) for x in x_test]
    return np.max(np.abs(y_cubic - y_test))


if __name__ == "__main__":
    # get data from csv
    x_nodes, y_nodes = process_csv()

    # drawing plot with origin data
    draw_graph(x_nodes, y_nodes, 224)

    draw_reduced_data(x_nodes, y_nodes, 1)

    # drawing plot with reduced data
    draw_reduced_data(x_nodes, y_nodes, 2)
    draw_reduced_data(x_nodes, y_nodes, 4)

    # calculate average day temperature
    average_temp = np.array(calculate_average_day_temperature(y_nodes))
    average_temp2 = np.array(calculate_average_day_temperature_cubic(x_nodes, y_nodes, 2))
    average_temp4 = np.array(calculate_average_day_temperature_cubic(x_nodes, y_nodes, 4))

    # calculate max distance between average day temperature
    dist_average = np.abs(np.max(average_temp - average_temp2))
    dist_average2 = np.abs(np.max(average_temp - average_temp4))

    print(dist_average, dist_average2)

    # calculate max distance between deleted nodes and origin nodes
    dist = calculate_dist_between_deleted_nodes(x_nodes, y_nodes, 2)
    dist2 = calculate_dist_between_deleted_nodes(x_nodes, y_nodes, 4)

    print(dist, dist2)
