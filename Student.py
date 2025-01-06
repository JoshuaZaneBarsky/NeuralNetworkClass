import numpy as np
from ActivationFunction import Tanh, ReLu, LeakyReLu, Softmax, Swish

NEURONS = 10  # number of neurons per layer

class Student:
    def __init__(self):
        self.W1 = np.random.randn(1, NEURONS) * np.sqrt(2. / 1)  # 10 neurons in the first hidden layer
        self.b1 = np.zeros(NEURONS)
        self.W2 = np.random.randn(NEURONS, NEURONS) * np.sqrt(2. / NEURONS)  # 10 neurons in the second hidden layer
        self.b2 = np.zeros(NEURONS)
        self.W3 = np.random.randn(NEURONS, 1) * np.sqrt(2. / NEURONS)  # Output layer
        self.b3 = np.zeros(1)
        self.activation_function = Swish()

        # Note:
        # Swish seems to work very well with this model.
        # 
        # All activation functions imported from ActivationFunctions.py

    def forward(self, x):
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.activation_function.activate(self.z1)  
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.activation_function.activate(self.z2)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        return self.z3

    def backward(self, x, y, learning_rate):
        m = x.shape[0]
        dz3 = self.z3 - y
        dW3 = np.dot(self.a2.T, dz3) / m
        db3 = np.sum(dz3, axis=0) / m

        dz2 = np.dot(dz3, self.W3.T) * self.activation_function.derivative(self.z2)
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0) / m

        dz1 = np.dot(dz2, self.W2.T) * self.activation_function.derivative(self.z1)
        dW1 = np.dot(x.T, dz1) / m
        db1 = np.sum(dz1, axis=0) / m

        # Update weights and biases using gradient descent
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3
