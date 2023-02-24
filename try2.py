import numpy as np
import matplotlib.pyplot as plt

# Define the data
x = np.array([-3.0, -2.0, 0.0, 1.0, 3.0, 4.0])
y = np.array([-1.5, 2.0, 0.7, 5.0, 3.5, 7.5])

# Define the initial parameters
a_init = 1
b_init = 1
n = len(x)

# Define the learning rate and the number of iterations
lr = 0.01
num_iter = 1000

# Define the gradient descent function
def gradient_descent(x, y, a_init, b_init, lr, num_iter):
    a = a_init
    b = b_init
    for i in range(num_iter):
        y_pred = a * x + b
        error = (y_pred - y)
        grad_a = 2 * np.sum(x * error)
        grad_b = 2 * np.sum(error)
        a -= lr * grad_a
        b -= lr * grad_b

    return a, b

# Call the gradient descent function
a, b = gradient_descent(x, y, a_init, b_init, lr, num_iter)

# Print the final parameters
print(f"Final parameters: a = {a}, b = {b}")

# Plot the data and the initial and fitted lines
plt.scatter(x, y)
plt.plot(x, a_init * x + b_init, label='Initial Line')
plt.plot(x, a * x + b, label='Fitted Line')
plt.legend()
plt.show()
