"""
================================
PROJECT - ex1D - Machine learning
================================

Name: Eliezer Seror  Id: 312564776
Name: Ram Elias      Id: 205445794

"""
import numpy as np
from sympy import symbols, lambdify, sin
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def model(x, a, b):
    return a * np.sin(b * x)


if __name__ == "__main__":
    # Define symbolic variables
    x, y, a, b = symbols('x,y,a,b')

    # Set the f model in terms of symbolic variables
    f = a * sin(b * x)
    F = lambdify([a, b, x], f, 'numpy')

    # Define the data
    x_data = np.array([-5., -4.5, -4., -3.5, -3., -2.5, -2., -1.5, -1., -0.5, 0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5.])
    y_data = np.array([-2.16498306, -1.53726731, 1.67075645, 2.47647932, 4.49579917,
                       1.14600963, 0.15938811, -3.09848048, -3.67902427, -1.84892687,
                       -0.11705947, 3.14778203, 4.26365256, 2.49120585, 0.55300516,
                       -2.105836, -2.68898773, -2.39982575, -0.50261972, 1.40235643,
                       2.15371399])

    # Fit the model to the data using curve_fit
    p0 = [1, 1]  # initial guess
    popt, pcov = curve_fit(model, x_data, y_data, p0=p0)

    # Get the optimal values of a and b
    a_opt = popt[0]
    b_opt = popt[1]

    # Print the optimal values of a and b
    print(f'The optimal values of a and b are: {a_opt}, {b_opt}')

    # Evaluate the fitted model and the initial guess
    f_init = F(p0[0], p0[1], x_data)
    f_fitted = F(a_opt, b_opt, x_data)

    # Plot the original data, the initial guess, and the fitted model
    plt.scatter(x_data, y_data, label='Data')
    plt.plot(x_data, f_init, 'g', label='Initial guess')
    plt.plot(x_data, f_fitted, 'r', label='Fitted model')
    plt.grid(True, linestyle='--', linewidth=0.5, color='lightgray')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Fitted Model using curve_fit')
    plt.legend()
    plt.show()
