import math

import matplotlib.pyplot as plt
import numpy as np


def get_x_nodes(nodes_amount):
    h = 1 / nodes_amount
    x_nodes = [h * i for i in range(1, nodes_amount)]
    return x_nodes


def get_y_nodes(nodes_amount):
    h = 1 / nodes_amount
    y_nodes = [h * j for j in range(1, nodes_amount)]
    return y_nodes


def process_index(i, j, matrix, dict, n):
    row_index = dict[(i, j)]
    matrix[row_index][row_index] = -4

    if i - 1 != 0:
        matrix[row_index][dict[(i - 1, j)]] = 1

    if i + 1 != n:
        matrix[row_index][dict[(i + 1, j)]] = 1

    if j - 1 != 0:
        matrix[row_index][dict[(i, j - 1)]] = 1

    if j + 1 != n:
        matrix[row_index][dict[(i, j + 1)]] = 1

    return matrix


def get_dictionary_of_indexes(n):
    indexes_dictionary = {}
    index_number = 0

    for i in range(n):
        for j in range(n):
            if i == 0 or j == 0 or i == n or j == n:
                pass
            else:
                indexes_dictionary[(i, j)] = index_number
                index_number += 1
    return indexes_dictionary


def get_matrix_with_coefficients(n):
    matrix_side = (n - 1) ** 2
    matrix = np.zeros((matrix_side, matrix_side))
    indexes_dictionary = get_dictionary_of_indexes(n)

    for i in range(n):
        for j in range(n):
            if i == 0 or j == 0 or i == n or j == n:
                pass
            else:
                matrix = process_index(i, j, matrix, indexes_dictionary, n)
    return matrix


def ab4(T, f, delta_t, i):
    T_next = T[i - 1]
    a_coefficient = np.array([55, -59, 37, -9])
    for j in range(4):
        T_next += (delta_t / 24) * a_coefficient[j] * ((A.dot(T[i - j - 1])) + f)
    return T_next


def rk4(T_previous, f, delta_t):
    k1 = delta_t * ((A.dot(T_previous)) + f)
    k2 = delta_t * (A.dot(T_previous + (delta_t / 2) * k1) + f)
    k3 = delta_t * (A.dot(T_previous + (delta_t / 2) * k2) + f)
    k4 = delta_t * (A.dot(T_previous + delta_t * k3) + f)
    return delta_t * 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def ode_solve(f, t_final, delta_t):
    t_max = int(t_final / delta_t) + 1
    T_result = np.zeros((t_max, len(f),))
    for i in range(1, 4):
        T_result[i] = rk4(T_result[i - 1], f, delta_t)
    for i in range(4, t_max):
        T_result[i] = ab4(T_result, f, i, delta_t)
    return T_result


def draw_average_temperature(T, t_max):
    T_avr = [np.mean(T[i]) for i in range(t_max)]
    time = np.linspace(0, t_final, t_max)
    plt.figure(1, figsize=(15, 10))
    plt.xlabel('Время')
    plt.ylabel('Температура, усредненная по области')
    plt.plot(time, T_avr)
    plt.grid(linestyle=':', linewidth=1.3)
    plt.show()


def draw_contour_graph(x_nodes, y_nodes, T_graph):
    plt.figure(2, figsize=(10, 7))
    plt.xlabel('x')
    plt.ylabel('y')
    cs = plt.contour(x_nodes, y_nodes, T_graph)
    plt.clabel(cs)
    plt.show()


if __name__ == "__main__":
    N = 18
    number_of_unknown_variables = (N - 1) * (N - 1)
    A = get_matrix_with_coefficients(N)

    f = np.ones(number_of_unknown_variables)

    t_final = 0.30
    delta_t = 0.0001
    t_max = int(t_final / delta_t) + 1

    T = ode_solve(f, t_final, delta_t)
    draw_average_temperature(T, t_max)

    x_nodes = get_x_nodes(N)
    y_nodes = get_y_nodes(N)

    T_graph = np.zeros((len(x_nodes), len(y_nodes)))

    k = 0
    for i in range(len(x_nodes)):
        for j in range(len(x_nodes)):
            T_graph[i][j] = T[k][math.trunc(t_final / delta_t)]
            k += 1
    draw_contour_graph(x_nodes, y_nodes, T_graph)
