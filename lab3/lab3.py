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


def conjugate_gradient(A, b, C_inv, eps):
    b = np.array(b)
    A_new = np.dot(C_inv, A)
    A_new = np.dot(A_new, C_inv.transpose())

    print(np.linalg.cond(A_new))

    temperature_vector = np.zeros(len(A))
    r = b - np.dot(A_new, temperature_vector)
    u = r
    r_norm = []
    number_of_iterations = 0

    while np.linalg.norm(r) > eps:
        r_norm.append(np.linalg.norm(r))
        t_k = r.dot(r) / u.dot(A_new.dot(u))
        temperature_vector = temperature_vector + t_k * u
        r_new = r - t_k * A_new.dot(u)
        s_k = r_new.dot(r_new) / r.dot(r)
        r = r_new
        u = r_new + s_k * u
        number_of_iterations += 1
    return temperature_vector, r_norm, number_of_iterations


def draw_contour_plot(x_result, n):
    x_nodes = get_x_nodes(n)
    y_nodes = get_y_nodes(n)
    x = np.zeros((len(x_nodes), len(y_nodes)))
    k = 0
    for i in range(len(x)):
        for j in range(len(x)):
            x[i][j] = x_result[k]
            k += 1
    plt.figure(1, figsize=(10, 7))
    plt.xlabel('x')
    plt.ylabel('y')
    cs = plt.contour(y_nodes, x_nodes, x)
    plt.clabel(cs)
    plt.show()


def draw_r_norm(r_norm, number_of_iterations):
    x = [i for i in range(number_of_iterations)]
    plt.figure(1, figsize=(10, 7))
    plt.xlabel('number of iterations')
    plt.ylabel('y')
    plt.xticks(x)
    plt.semilogy(x, r_norm, color='r')
    plt.grid(linestyle=':', linewidth=1.3)
    plt.show()


if __name__ == "__main__":
    N = 9
    number_of_unknown_variables = (N - 1) * (N - 1)

    # Ax = b
    A = get_matrix_with_coefficients(N)
    b = []
    for i in range(number_of_unknown_variables):
        b.append(-1 / (N ** 2))

    C_inv = np.eye(number_of_unknown_variables)
    D = (1 / np.sqrt(abs(A[0][0]))) * np.eye(number_of_unknown_variables)

    temperature_vector, r_norm, number_of_iterations = conjugate_gradient(A, b, D, eps=0.0001)
    draw_contour_plot(temperature_vector, N)
    #draw_r_norm(r_norm, number_of_iterations)
