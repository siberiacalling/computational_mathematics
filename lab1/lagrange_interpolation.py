import math
import random

import matplotlib.pyplot as plt
import numpy as np


# basis polynomial[i]
def l_i(i, x, x_nodes):
    basis_pol = 1
    for j in range(len(x_nodes)):
        if j == i:
            continue
        basis_pol *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
    return basis_pol


# lagrange polynomial
def L(x, x_nodes, y_nodes):
    lagrange_pol = 0
    for i in range(len(x_nodes)):
        lagrange_pol += y_nodes[i] * l_i(i, x, x_nodes)
    return lagrange_pol


# coefficients for pade function
def random_coefficients():
    m = random.randint(7, 15)
    n = random.randint(7, 15)
    a = []
    b = []

    for j in range(m + 1):
        a.append(random.uniform(0, 1))
    for k in range(n):
        b.append(random.uniform(0, 1))

    return {'m': m, 'n': n, 'a': a, 'b': b}


# generate pade function
def f(coefficients, x_nodes):
    m = coefficients['m']
    n = coefficients['n']
    a = coefficients['a']
    b = coefficients['b']

    sum1 = 0
    sum2 = 1
    for j in range(m + 1):
        sum1 += a[j] * np.power(x_nodes, j)
    for k in range(1, n):
        sum2 += b[k] * np.power(x_nodes, k)
    return sum1 / sum2


# generate N pade function
def f_n(x_nodes, n=100):
    functions = []
    current_function = {}
    for i in range(n):
        current_function['coefficients'] = random_coefficients()
        current_function['y'] = f(current_function['coefficients'], x_nodes)
        functions.append(current_function)
    return functions


# calculation of Chebyshev nodes
def get_chebyshev_nodes(coefficients, a, b, number_of_nodes):
    x_nodes_cheb = []
    for i in range(1, number_of_nodes + 1):
        xk = 0.5 * (a + b) + 0.5 * (b - a) * math.cos((2 * i - 1) * math.pi / (2 * number_of_nodes))
        x_nodes_cheb.append(xk)
    y_nodes_cheb = f(coefficients, x_nodes_cheb)
    return x_nodes_cheb, y_nodes_cheb


def get_lagrange_polynomial_coordinates(x_graph, x_nodes, y_nodes):
    lagrange_pol = []
    if type(x_graph) == int:
        lagrange_pol.append([x_graph, L(x_graph, x_nodes, y_nodes)])
    else:
        for i in x_graph:
            lagrange_pol.append([i, L(i, x_nodes, y_nodes)])
    lagrange_pol = np.array(lagrange_pol)
    return lagrange_pol


def draw_f(x_drawing, y_drawing, coefficients):
    uniform_nodes_amount = 10

    x_uniform_nodes = np.linspace(-1, 1, uniform_nodes_amount)
    y_uniform_nodes = f(coefficients, x_uniform_nodes)

    lagrange_pol = get_lagrange_polynomial_coordinates(x_drawing, x_uniform_nodes, y_uniform_nodes)

    x_nodes_cheb, y_nodes_cheb = get_chebyshev_nodes(coefficients, -1, 1, uniform_nodes_amount)

    # lagrange polynomial for Chebyshev nodes
    lagrange_pol_cheb = get_lagrange_polynomial_coordinates(x_drawing, x_nodes_cheb, y_nodes_cheb)

    # drawing subplots
    fig, ax = plt.subplots()
    ax.plot(x_drawing, y_drawing, 'b', label='Исходная функция')
    ax.plot(lagrange_pol[:, 0], lagrange_pol[:, 1], 'b--', label='Интерполяция равномерными узлами')
    ax.plot(lagrange_pol_cheb[:, 0], lagrange_pol_cheb[:, 1], 'r', label='Интерполяция Чебышевскими узлами')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.grid()
    ax.legend()
    plt.show()


def get_distance(x_drawing, y_drawing, coefficients, nodes_amount):
    norm_lag_graph = []
    norm_cheb_graph = []
    for current_node_amount in range(1, nodes_amount + 1):
        x_uniform_nodes = np.linspace(-1, 1, current_node_amount)
        y_uniform_nodes = f(coefficients, x_uniform_nodes)
        lagrange_pol = get_lagrange_polynomial_coordinates(x_drawing, x_uniform_nodes, y_uniform_nodes)

        x_nodes_cheb, y_nodes_cheb = get_chebyshev_nodes(coefficients, -1, 1, current_node_amount)
        lagrange_pol_cheb = get_lagrange_polynomial_coordinates(x_drawing, x_nodes_cheb, y_nodes_cheb)

        norm_lag = np.max(np.abs(y_drawing - lagrange_pol[:, 1]))
        norm_cheb = np.max(np.abs(y_drawing - lagrange_pol_cheb[:, 1]))

        norm_lag_graph.append(norm_lag)
        norm_cheb_graph.append(norm_cheb)

    node_num = [k for k in range(1, nodes_amount + 1)]

    # drawing subplots
    fig, ax = plt.subplots()
    ax.plot(node_num, norm_lag_graph, 'g', label='Равномерные узлы')
    ax.plot(node_num, norm_cheb_graph, 'r', label='Чебышевские узлы')
    ax.set_xscale("log", nonposx='clip')
    ax.set_yscale("log", nonposy='clip')
    ax.set_xlabel('Количество узлов')
    ax.set_ylabel('Значение нормы')
    ax.grid()
    ax.legend()
    plt.show()


if __name__ == "__main__":
    # drawing example
    x_drawing = np.linspace(-1, 1, 100)
    coef = random_coefficients()
    y_drawing = f(coef, x_drawing)
    draw_f(x_drawing, y_drawing, coef)

    # calculation norms
    get_distance(x_drawing, y_drawing, coef, 60)
