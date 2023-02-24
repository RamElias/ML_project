import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# Define the linear model
def linear_model(x, a, b):
    return a * x + b


# Define the mean squared error loss function
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


# Initialize the parameters and the learning rate
a = 3
b = 3
epochs = 100
learning_rate = 0.01
error_list = []
a_list = []
b_list = []
# a_list.append(a)
# b_list.append(b)
# Define the data
x = np.array([-3.0, -2.0, 0.0, 1.0, 3.0, 4.0])
y = np.array([-1.5, 2.0, 0.7, 5.0, 3.5, 7.5])

# Plot the linear model
x_model = np.linspace(min(x), max(x), 100)

# Plot the data
plt.scatter(x, y, label='Data')
plt.plot(x_model, linear_model(x_model, a, b), 'r', label='initial model')

# Perform gradient descent for a fixed number of iterations
for i in range(epochs):
    # Compute the gradient of the loss function with respect to the parameters
    grad_a, grad_b = loss_gradient(x, y, a, b)
    a_list.append(a)
    b_list.append(b)

    # Update the parameters by moving in the direction of the negative gradient
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b

    # Compute the error function
    error_list.append(error_function(x, y, a, b))

print(error_list)
print(f'The final a,b are: {a}, {b}')
y_model = linear_model(x_model, a, b)

plt.plot(x_model, y_model, 'g', label='linear fitted model')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression with Gradient Descent')

# Show the plot
plt.show()

plt.plot(range(epochs), error_list, 'b')
plt.xlabel('iterations')
plt.ylabel('error')
plt.title('Error value per iterations')
plt.show()


# Create a 3D plot of the error surface over the parameters
A, B = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
Z = np.zeros_like(A)
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        Z[i, j] = error_function(x, y, A[i, j], B[i, j])

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

