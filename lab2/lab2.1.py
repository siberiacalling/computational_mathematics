import math
import matplotlib.pyplot as plt
import numpy as np
import sympy


# calculating array of deltas for drawing
def generate_data_for_graph(exact_integral, nodes_amount, function):
    x_nodes = [2 ** i for i in range(1, nodes_amount + 1)]
    mas_delta = []
    for i in x_nodes:
        y_apr = (spectral_integral(function, i))
        delta = math.fabs((exact_integral - y_apr) / exact_integral)
        mas_delta.append(delta)
    return x_nodes, mas_delta


def draw_graph(x, y):
    plt.figure(figsize=(10, 6))
    plt.grid(linestyle=':', linewidth=1.3)
    plt.xlabel('Количество узлов', size=14)
    plt.ylabel('Относительная погрешность', size=14)
    plt.xticks(x)
    plt.loglog(x, y, color="m")
    plt.scatter(x, y, color="c")
    plt.show()


def dft_slow(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)


def fft_coeff(y_nodes):
    """A recursive implementation of the 1D Cooley-Tukey FFT"""
    y_nodes = np.asarray(y_nodes, dtype=float)
    array_length = y_nodes.shape[0]

    if array_length % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif array_length <= 4:  # this cutoff should be optimized
        return dft_slow(y_nodes)
    else:
        x_even = fft_coeff(y_nodes[::2])
        x_odd = fft_coeff(y_nodes[1::2])
        factor = np.exp(-2j * np.pi * np.arange(array_length) / array_length)
        return np.concatenate([x_even + factor[:array_length // 2] * x_odd,
                               x_even + factor[array_length // 2:] * x_odd])


# function return True if custom fft and library fft return the same data
def test_custom_fft():
    x = np.random.random(1024)
    print(np.allclose(fft_coeff(x), np.fft.fft(x)))


def spectral_integral(func, nodes_amount):
    x_nodes = [-math.pi + j * math.pi / (nodes_amount / 2) for j in range(nodes_amount)]
    y_nodes = []
    for i in x_nodes:
        y_nodes.append(func.subs(x, i))
    A = fft_coeff(y_nodes)
    y_nodes = 5 / 4 * math.pi * A[0].real / nodes_amount
    for k in range(1, nodes_amount):
        A[k] *= (-1) ** k / nodes_amount
        y_nodes += 2 * A[k].real / k * math.sin(k * math.pi / 4) - 2 * A[k].imag / k * (
                math.cos(k * math.pi / 4) - math.cos(k * math.pi))
    return y_nodes


def calculating_integrals_and_drawing(symbol_function):
    integral_exact_value = sympy.integrate(symbol_function, (x, -math.pi / 4, math.pi))
    x_graph, y_graph = generate_data_for_graph(integral_exact_value, 8, symbol_function)
    draw_graph(x_graph, y_graph)
    print(integral_exact_value)


if __name__ == "__main__":
    x = sympy.Symbol('x')
    i_1 = x * sympy.cos(x ** 2) + math.e ** x * sympy.cos(math.e ** x)
    i_2 = sympy.Abs(x)
    calculating_integrals_and_drawing(i_2)
