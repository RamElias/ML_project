"""
================================
PROJECT - Question 1A - Machine learning
================================

Name: Eliezer Seror  Id: 312564776
Name: Ram Elias      Id: 205445794

"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sympy

# Define the linear model
def linear_model(x, a, b):
    return a * x + b


# Define the error loss function
def error_function(x, y, a, b):
    y_pred = linear_model(x, a, b)
    error = np.sum((y_pred - y) ** 2)
    return error


# Define the gradient of the loss function with respect to the parameters
def loss_gradient(x, y, a, b):
    n = len(x)
    grad_a = 2 * np.sum(x * (a * x + b - y))
    grad_b = 2 * np.sum(a * x + b - y)
    return grad_a, grad_b


# Perform gradient descent for a fixed number of iterations
def gradient_descent(x, y, a, b, epochs):
    for i in range(epochs):
        # Compute the gradient of the loss function with respect to the parameters
        grad_a, grad_b = loss_gradient(x, y, a, b)

        # Update the parameters by moving in the direction of the negative gradient
        a -= learning_rate * grad_a
        b -= learning_rate * grad_b

        error_list.append(error_function(x, y, a, b))
        a_list.append(a)
        b_list.append(b)

    print(f'The final a,b are: {a}, {b}')

    return a, b


# plot the linear model, the initial and fitted functions
def plot_linear_model():
    plt.scatter(x_data, y_data, label='Data')
    plt.plot(x_model, linear_model(x_model, init_a, init_b), 'r', label='initial model')
    plt.plot(x_model, y_model, 'g', label='linear fitted model')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Linear Regression with Gradient Descent')
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


# plot the error surface and the parameters found with the gradient descent
def plot_error_surface():
    size = 100

    A, B = np.meshgrid(np.linspace(min(a_list), max(a_list), size),
                       (np.linspace(min(b_list), max(b_list), size)))

    Z = np.zeros_like(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            Z[i, j] = error_function(x_data, y_data, A[i, j], B[i, j])

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
    # Define the data
    x_data = np.array([-3.0, -2.0, 0.0, 1.0, 3.0, 4.0])
    y_data = np.array([-1.5, 2.0, 0.7, 5.0, 3.5, 7.5])

    # Initialize the parameters, epochs, learning rate
    init_a = 3
    init_b = 3
    epochs = 100
    learning_rate = 0.01
    error_list = []
    a_list = []
    b_list = []

    a, b = gradient_descent(x_data, y_data, init_a, init_b, epochs)

    x_model = np.linspace(min(x_data), max(x_data), 100)
    y_model = linear_model(x_model, a, b)

    plot_linear_model()
    plot_error_function()
    plot_error_surface()

