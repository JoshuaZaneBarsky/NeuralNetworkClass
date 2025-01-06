import numpy as np
import matplotlib.pyplot as plt
from Student import Student

# Generate data for any given function
def generate_data(func, x_range, num_samples):
    x = np.linspace(x_range[0], x_range[1], num_samples)
    y = func(x)
    return x, y

# Training parameters
learning_rate = 0.0005
num_epochs = 1000
batch_size = 4 # for epochs

# Note:
# Batch size of 4 seems to work well.

# Function to approximate
function_to_approximate = np.square
x_range = [-10, 10]
num_samples = 1000

'''
Functions like np.log10 may return an invalid value error so these may not work. 
This is likely due the need for only a positive domain.

np.cos
np.tan
np.arcsin
np.arccos
np.arctan
np.sinh
np.cosh
np.tanh
np.arcsinh
np.arccosh
np.arctanh
np.exp
np.log
np.log10
np.log2
np.abs
np.sign
np.floor
np.ceil
np.round
np.square
'''

# Generate training data
x, y = generate_data(function_to_approximate, x_range, num_samples)

# Shape data
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

# Create the student
student = Student()

# Training loop
for epoch in range(num_epochs):
    # Shuffle data for batching
    perm = np.random.permutation(x.shape[0])
    x_shuffled = x[perm]
    y_shuffled = y[perm]

    for i in range(0, x.shape[0], batch_size):
        x_batch = x_shuffled[i:i + batch_size]
        y_batch = y_shuffled[i:i + batch_size]

        # Forward pass
        outputs = student.forward(x_batch)
        loss = np.mean((outputs - y_batch) ** 2)

        # Backward pass and optimization
        student.backward(x_batch, y_batch, learning_rate)

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}')

# Evaluate the model
x_test = np.linspace(x_range[0], x_range[1], 1000).reshape(-1, 1)
y_pred = student.forward(x_test)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='True Function', color='blue')
plt.plot(x_test, y_pred, label='Model Prediction', color='red')
plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Function Approximation using Neural Network')
plt.show()