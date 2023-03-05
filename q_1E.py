"""
================================
PROJECT - Question 1E - Machine learning
================================

Name: Eliezer Seror  Id: 312564776
Name: Ram Elias      Id: 205445794

"""

import numpy as np
from sympy import symbols, lambdify, diff, exp, sin, cos
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time


# Perform gradient descent for a fixed number of iterations
def gradient_descent(x, y, a, b, c, epochs, learning_rate):
    for i in range(epochs):
        error_rate = sum(E(a, b, c, x, y))

        grad_a, grad_b, grad_c = sum(Grad_a(a, b, c, x, y)), sum(Grad_b(a, b, c, x, y)), sum(Grad_c(a, b, c, x, y))

        a -= (learning_rate * grad_a)
        b -= (learning_rate * grad_b)
        c -= (learning_rate * grad_c)

        error_list.append(error_rate)
        a_list.append(a)
        b_list.append(b)
        c_list.append(c)

    print(f'The final a,b,c are: {a}, {b}, {c}')
    return a, b, c


# plot the model, the initial and fitted functions
def plot_model(x, y, y_init, Y_pred):
    plt.scatter(x, y, label=' Data')
    plt.plot(x, y_init, 'r', label='initial model')
    plt.plot(x, Y_pred, 'g', label='fitted model')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Nonlinear Regression with Gradient Descent')
    plt.grid(True, linestyle='--', linewidth=0.5, color='lightgray')

    plt.legend()
    plt.show()


# plot the error function per iteration
def plot_error_function():
    plt.plot(range(epochs), error_list, 'b')
    plt.xlabel('iterations')
    plt.ylabel('error')
    plt.title('Error value per iterations')
    plt.grid(True, linestyle='--', linewidth=0.5, color='lightgray')
    plt.show()


#
# def plot_error_surface():
#     size = 100
#     A, B, C = np.meshgrid(np.linspace(min(a_list), max(a_list), size),
#                            np.linspace(min(b_list), max(b_list), size),
#                            np.linspace(min(c_list), max(c_list), size))
#
#     Z = np.zeros((size, size))
#
#     for i in range(size):
#         for j in range(size):
#             for k in range(size):
#                 a = A[i, j, k]
#                 b = B[i, j, k]
#                 c = C[i, j, k]
#                 z = sum(E(a, b, c, x_data, y_data))
#                 Z[i, j] = z
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_surface(A, B, C, cmap='viridis', alpha=0.5, facecolors=plt.cm.viridis(Z))
#
#     # Plot the parameters found during all iterations
#     ax.scatter(a_list, b_list, c_list, s=50, c='b', label='Parameter Values')
#     ax.scatter(a_list[-1], b_list[-1], c_list[-1], s=50, c='r', label='Final Values')
#     ax.set_xlabel('a')
#     ax.set_ylabel('b')
#     ax.set_zlabel('c')
#     ax.set_title('Error Surface and Parameter Values')
#     ax.legend()
#     plt.show()
#

# section for finding the parameters using gradient descent
def find_using_gradient_descent():
    print('------Gradient descent-------')
    start = time.time()
    # Perform gradient descent and get the optimal values for a, b, and c
    a, b, c = gradient_descent(x_data, y_data, init_a, init_b, init_c, epochs, learning_rate)
    end = time.time()
    grad_time = end - start
    print(f"Total time taken with Gradient descent: {grad_time} seconds")

    # Calculate the initial and fitted models
    f_init = F(init_a, init_b, init_c, x_data, y_data)
    f_fitted = F(a, b, c, x_data, y_data)

    # Plot the initial and fitted models along with the data
    plot_model(x_data, y_data, f_init, f_fitted)

    # Plot the error value over the iterations
    plot_error_function()

    # Plot the error surface and parameter values
    plot_error_surface()


# func that return our model
def model(x, a, b, c):
    return a * np.sin(np.cos(b * x) * (c * x))


# section for finding the parameters using curve fir function
def find_using_curve_fit():
    print('------Curve fit-------')
    p0 = [1, 1, 1]  # initial guess
    start = time.time()
    popt, _ = curve_fit(model, x_data, y_data, p0)
    end = time.time()
    curve_time = end - start
    # Get the optimal values of a and b
    a_opt = popt[0]
    b_opt = popt[1]
    c_opt = popt[2]

    # Print the optimal values of a and b
    print(f'The final a,b,c are: {a_opt}, {b_opt}, {c_opt}')
    print(f"Total time taken with curve_fit : {curve_time} seconds")

    f_init = F(p0[0], p0[1], p0[2], x_data, y_data)
    f_fitted = F(a_opt, b_opt, c_opt, x_data, y_data)

    # Plot the original data, the initial guess, and the fitted model
    plt.scatter(x_data, y_data, label='Data')
    plt.plot(x_data, f_init, 'r', label='Initial guess')
    plt.plot(x_data, f_fitted, 'g', label='Fitted model')
    plt.grid(True, linestyle='--', linewidth=0.5, color='lightgray')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Fitted Model using curve_fit')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Define symbolic variables
    x, y, a, b, c = symbols('x,y,a,b,c')

    # Set the f model in terms of symbolic variables
    f = a * sin(cos(b * x) * (c * x))
    F = lambdify([a, b, c, x, y], f, 'numpy')

    # Set the error for a single figure
    e = (f - y) ** 2
    E = lambdify([a, b, c, x, y], e, 'numpy')

    # Set the Gradient components
    de_a = diff(e, a)
    de_b = diff(e, b)
    de_c = diff(e, c)
    Grad_a = lambdify([a, b, c, x, y], de_a, 'numpy')
    Grad_b = lambdify([a, b, c, x, y], de_b, 'numpy')
    Grad_c = lambdify([a, b, c, x, y], de_c, 'numpy')

    # Define the data

    x_data = np.array(
        [-5., -4.5, -4., -3.5, -3., -2.5, -2., -1.5, -1., -0.5, 0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5.])

    y_true = np.array([-2.16498306, -1.53726731, 1.67075645, 2.47647932, 4.49579917,
                       1.14600963, 0.15938811, -3.09848048, -3.67902427, -1.84892687,
                       -0.11705947, 3.14778203, 4.26365256, 2.49120585, 0.55300516,
                       -2.105836, -2.68898773, -2.39982575, -0.50261972, 1.40235643,
                       2.15371399])

    # Generate the data
    noise = np.random.normal(loc=0, scale=0.5, size=len(y_true))
    y_data = y_true + noise

    # Initialize the parameters, epochs, learning rate
    init_a = 1
    init_b = 1
    init_c = 1
    epochs = 1000
    learning_rate = 0.0001
    error_list = []
    a_list = []
    b_list = []
    c_list = []

    find_using_gradient_descent()
    find_using_curve_fit()
