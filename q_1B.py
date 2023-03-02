import numpy as np
from sympy import symbols, lambdify, diff
import matplotlib.pyplot as plt


def gradient_descent(x, y, a, b, epochs):
    # Perform gradient descent for a fixed number of iterations
    for i in range(epochs):
        error_rate = sum(E(a, b, x, y))

        grad_a, grad_b = sum(Grad_a(a, b, x, y)), sum(Grad_b(a, b, x, y))

        a -= (learning_rate * grad_a)
        b -= (learning_rate * grad_b)

        error_list.append(error_rate)
        a_list.append(a)
        b_list.append(b)

    print(f'The final a,b are: {a}, {b}')
    return a, b


def plot_linear_model(x, y, y_init, Y_pred):
    plt.scatter(x, y, label=' Data')
    plt.plot(x, y_init, 'r', label='initial model')
    plt.plot(x, Y_pred, 'g', label='linear fitted model')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Linear Regression with Gradient Descent')
    plt.grid(True, linestyle='--', linewidth=1, color='lightgray')

    plt.legend()
    plt.show()


def plot_error_function():
    plt.plot(range(epochs), error_list, 'b')
    plt.xlabel('iterations')
    plt.ylabel('error')
    plt.title('Error value per iterations')
    plt.grid(True, linestyle='--', linewidth=1, color='lightgray')
    plt.show()


def plot_error_surface(grid_size=30):
    a_grid = np.linspace(min(a_list), max(a_list), grid_size)
    b_grid = np.linspace(min(b_list), max(b_list), grid_size)

    A, B = np.meshgrid(a_grid, b_grid)
    Z = np.zeros((grid_size, grid_size))

    for i in range(grid_size):
        for j in range(grid_size):
            Z[i, j] = sum(E(A[i, j], B[i, j], x, y))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(A, B, Z, cmap='viridis', alpha=0.5)

    # Plot the parameters found during all iterations
    ax.plot(a_list, b_list, error_list, 'b', label='Parameter Values')
    ax.scatter(a_list[-1], b_list[-1], error_list[-1], s=50, c='r', label='Final Values')
    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_zlabel('error')
    ax.set_title('Error Surface and Parameter Values')
    ax.legend()
    plt.show()


if __name__ == "__main__":
    # Define symbolic variables
    x, y, a, b = symbols('x,y,a,b')

    # Set the f model in terms of symbolic variables
    f = (a * x) + b
    F = lambdify([a, b, x, y], f, 'numpy')

    # Set the error for a single figure
    e = (f - y) ** 2
    E = lambdify([a, b, x, y], e, 'numpy')

    # Set the Gradient components
    de_a = diff(e, a)
    de_b = diff(e, b)
    Grad_a = lambdify([a, b, x, y], de_a, 'numpy')
    Grad_b = lambdify([a, b, x, y], de_b, 'numpy')

    # Define the data
    x_data = np.array([-3.0, -2.0, 0.0, 1.0, 3.0, 4.0])
    y_data = np.array([-1.5, 2.0, 0.7, 5.0, 3.5, 7.5])

    # Initialize the parameters and the learning rate
    init_a = 3
    init_b = 3
    epochs = 100
    learning_rate = 0.001
    error_list = []
    a_list = []
    b_list = []

    a, b = gradient_descent(x_data, y_data, init_a, init_b, epochs)

    f_init = F(init_a, init_b, x_data, y_data)
    f_fitted = F(a, b, x_data, y_data)

    plot_linear_model(x_data, y_data, f_init, f_fitted)
    plot_error_function()
    plot_error_surface()
