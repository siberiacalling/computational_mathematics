import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import truncnorm


# function returns polynomal coefficients
def poly_regression(x_nodes, y_nodes, degree):
    return np.polyfit(x_nodes, y_nodes, degree)


def f(x_nodes, x, sigma):
    y_nodes = []
    for i in range(len(x_nodes)):
        y_nodes.append(-10 * (x_nodes[i] ** 2) + 1.5 * x_nodes[i] + 1 + sigma * x[i])
    return y_nodes


def get_x_nodes(nodes_amount):
    return np.random.uniform(-1, 1, nodes_amount)


def get_x(nodes_amount):
    X = []
    for i in range(nodes_amount):
        X.append(truncnorm.rvs(-1, 1))
    return X


def get_y_nodes(x_nodes, x, sigma):
    y_nodes = []
    for i in range(len(x_nodes)):
        y_nodes.append(-10 * (x_nodes[i] ** 2) + 1.5 * x_nodes[i] + 1 + sigma * x[i])
    return y_nodes


def calculate_error(y_nodes, coefficient, x_nodes):
    regression_value = np.polyval(coefficient, x_nodes)
    error = 0
    for i in range(len(y_nodes)):
        error += (y_nodes[i] - regression_value[i]) ** 2
    return (error / len(y_nodes)) ** (1 / 2)


def draw_test_and_initial_data(coef, y_nodes, y_nodes_test, x_nodes, x_nodes_test):
    x_nodes_pol = [i for i in np.arange(-1, 1, 0.02)]
    regression_value = np.polyval(coef, x_nodes_pol)
    plt.figure(1, figsize=(10, 6))
    plt.xlabel('x', size=18)
    plt.ylabel('y', size=18)
    plt.grid(linestyle=':', linewidth=1.3)
    plt.plot(x_nodes_pol, regression_value)
    plt.scatter(x_nodes, y_nodes, color='m', label='Начальные данные')
    plt.scatter(x_nodes_test, y_nodes_test, color='c', label='Тестовые данные')
    plt.legend(loc='upper right', fontsize="x-large")
    plt.show()


def draw_relative_error(nodes_amount, coefficients):
    plt.figure(figsize=(10, 6))
    plt.xticks(nodes_amount)
    plt.xlabel('Количество узлов', size=14)
    plt.ylabel('Относительная погрешность', size=14)
    plt.plot(nodes_amount, coefficients, color='m')
    plt.grid(linestyle=':', linewidth=1.3)
    plt.show()


def draw_quadratic_error_degree(degrees, initial_data, test_data):
    plt.figure(3, figsize=(10, 6))
    plt.xticks(degrees)
    plt.xlabel('Степень полинома', size=14)
    plt.ylabel('Среднеквадратичная погрешность', size=14)
    plt.semilogy(degrees, initial_data, color='m', label='Начальные данные')
    plt.plot(degrees, test_data, color='c', label='Тестовые данные')
    plt.legend(loc='upper right', fontsize="x-large")
    plt.grid(linestyle=':', linewidth=1.3)
    plt.show()


def draw_quadratic_error_nodes(nodes_amount, initial_data_error, test_data_error):
    plt.figure(2, figsize=(10, 6))
    plt.xticks(nodes_amount)
    plt.xlabel('Количество узлов', size=14)
    plt.ylabel('Среднеквадратичная погрешность', size=14)
    plt.plot(nodes_amount, initial_data_error, color='m', label='Начальные данные')
    plt.plot(nodes_amount, test_data_error, color='c', label='Тестовые данные')
    plt.legend(loc='upper right', fontsize="x-large")
    plt.grid(linestyle=':', linewidth=1.3)
    plt.show()


def analyse():
    sigmas = [10 ** i for i in range(-2, 3)]
    nodes_amount = [2 ** i for i in range(3, 9)]
    degrees = [i for i in range(1, 6)]

    test_data_error = []
    initial_data_error = []
    coefficients_error = []

    for n in nodes_amount:
        x_nodes_initial = get_x_nodes(n)
        x = get_x(n)
        y_nodes_initial = get_y_nodes(x_nodes_initial, x, sigmas[0])
        coef = poly_regression(x_nodes_initial, y_nodes_initial, degrees[1])
        coefficients_error.append(
            (np.abs((-10 - coef[0]) / -10) + np.abs((1.5 - coef[1]) / 1.5) + np.abs(1 - coef[2])) / 3)
        regression_error = calculate_error(y_nodes_initial, coef, x_nodes_initial)
        initial_data_error.append(regression_error)

        x_nodes_test = get_x_nodes(n)
        x_test = get_x(n)
        y_nodes_test = get_y_nodes(x_nodes_test, x_test, sigmas[0])
        regression_error_test = calculate_error(y_nodes_test, coef, x_nodes_test)
        test_data_error.append(regression_error_test)
        # if n == nodes_amount[1]:
        #     draw_test_and_initial_data(coef, y_nodes_initial, y_nodes_test, x_nodes_initial, x_nodes_test)

    #draw_quadratic_error_nodes(nodes_amount, initial_data_error, test_data_error)

    initial_data_error = []
    test_data_error = []
    for degree in degrees:
        x_nodes = get_x_nodes(nodes_amount[0])
        x = get_x(nodes_amount[0])
        y_nodes = get_y_nodes(x_nodes, x, sigmas[3])
        coef = poly_regression(x_nodes, y_nodes, degree)
        regression_error = calculate_error(y_nodes, coef, x_nodes)
        initial_data_error.append(regression_error)

        x_nodes_test = get_x_nodes(nodes_amount[0])
        x_test = get_x(nodes_amount[0])
        y_nodes_test = get_y_nodes(x_nodes_test, x_test, sigmas[4])
        regression_error_test = calculate_error(y_nodes_test, coef, x_nodes_test)
        test_data_error.append(regression_error_test)

    #draw_quadratic_error_degree(degrees, initial_data_error, test_data_error)

    #fixed sigma
    #draw_relative_error(nodes_amount, coefficients_error)

    coefficients_error = []
    for i in range(len(sigmas)):
        x_nodes = get_x_nodes(nodes_amount[i])
        x = get_x(nodes_amount[i])
        y_nodes = get_y_nodes(x_nodes, x, sigmas[i])
        coef = poly_regression(x_nodes, y_nodes, degrees[1])
        coefficients_error.append(
            (np.abs((-10 - coef[0]) / -10) + np.abs((1.5 - coef[1]) / 1.5) + np.abs(1 - coef[2])) / 3)
    draw_relative_error(nodes_amount[:len(sigmas)], coefficients_error)


if __name__ == "__main__":
    analyse()
